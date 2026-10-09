#!/usr/bin/env python3
"""Secure paired image-to-image training based on pix2pix principles.

Important safety property: this program contains no code path that writes to an
original/corrected source directory.  All generated state is constrained to the
explicit output directory; optional source duplication is isolated in a gated,
verified staging directory beneath that output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from secure_pix2pix import ADAM_BETAS, IMAGE_SIZE, LAMBDA_L1, LEARNING_RATE
from secure_pix2pix.checkpoints import (
    atomic_json_save,
    atomic_torch_save,
    prune_periodic_checkpoints,
    safe_torch_load,
)
from secure_pix2pix.data import (
    PairedStyleDataset,
    PairRecord,
    build_identical_filename_pairs,
    dataset_fingerprint,
    load_verified_cache_pairs,
    split_with_frozen_validation,
)
from secure_pix2pix.metrics import denormalize_image, psnr_per_image, ssim_per_image
from secure_pix2pix.models import PatchDiscriminator, UNetGenerator, initialize_pix2pix_weights
from secure_pix2pix.security import PathSecurityError, ensure_child_path, validate_write_root
from secure_pix2pix.staging import CopyWindow, StagingWindowClosed, stage_pairs


@dataclass(frozen=True)
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


class StopController:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame: object) -> None:
        self.requested = True
        print(f"Graceful stop requested by signal {signum}; finishing the current batch.", flush=True)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch U-Net + conditional 70x70 PatchGAN on exact filename pairs."
    )
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--corrected-root", type=Path)
    parser.add_argument(
        "--cache-index",
        type=Path,
        help="Use the existing verified local replay cache without production access.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--read-only-root",
        action="append",
        default=[],
        type=Path,
        help="Additional protected production root. May be supplied more than once.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to run in this invocation.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-megapixels", type=float, default=120.0)
    parser.add_argument("--max-aspect-change", type=float, default=0.02)
    parser.add_argument(
        "--exact-cache-filenames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accept only cache records whose original production filenames matched.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-unchanged", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-psnr-delta", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-periodic-checkpoints", type=int, default=5)
    parser.add_argument("--empty-cache-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and pairing, then exit.")
    parser.add_argument("--stop-at", help="Local training cutoff: HH:MM today or YYYY-MM-DDTHH:MM.")
    parser.add_argument("--minimum-remaining-minutes", type=int, default=10)

    staging = parser.add_argument_group("controlled staging")
    staging.add_argument("--stage-data", action="store_true")
    staging.add_argument("--stage-only", action="store_true")
    staging.add_argument("--staging-dir", type=Path, default=None)
    staging.add_argument("--stage-start", default="04:00")
    staging.add_argument("--stage-end", default="10:00")
    args = parser.parse_args(argv)

    direct_source = args.original_root is not None or args.corrected_root is not None
    if args.cache_index is None and not (
        args.original_root is not None and args.corrected_root is not None
    ):
        parser.error("provide --cache-index or both --original-root and --corrected-root")
    if args.cache_index is not None and direct_source:
        parser.error("--cache-index cannot be combined with original/corrected roots")
    if args.cache_index is not None and (args.stage_data or args.stage_only):
        parser.error("the existing local cache does not require staging")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if (
        args.checkpoint_every <= 0
        or args.max_periodic_checkpoints <= 0
        or args.empty_cache_interval <= 0
    ):
        parser.error("checkpoint/cache intervals must be positive")
    if args.patience <= 0 or args.minimum_psnr_delta < 0:
        parser.error("--patience must be positive and --minimum-psnr-delta cannot be negative")
    if args.minimum_remaining_minutes < 0:
        parser.error("--minimum-remaining-minutes cannot be negative")
    if args.stage_only:
        args.stage_data = True
    return args


def deadline_today(value: str | None, now: datetime | None = None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError:
        pass
    current = now or datetime.now()
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ValueError("--stop-at must be HH:MM or YYYY-MM-DDTHH:MM") from exc
    return datetime.combine(current.date(), parsed)


def initialize_runtime(device_request: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    cuda_available = torch.cuda.is_available()
    if device_request == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    use_cuda = cuda_available and device_request != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank if distributed else 0)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    if distributed:
        backend = "nccl" if use_cuda and dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return Runtime(rank, local_rank, world_size, device, distributed)


def cleanup_runtime(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int, deterministic: bool) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def default_worker_count() -> int:
    # Conservative on shared enterprise hosts while still overlapping decode/I/O.
    return max(0, min(8, (os.cpu_count() or 2) - 1))


def make_loader(
    dataset: PairedStyleDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    sampler: DistributedSampler | None,
    cuda: bool,
) -> DataLoader:
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": cuda,
        "persistent_workers": workers > 0,
        "drop_last": False,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def reduce_sums(values: Iterable[float], runtime: Runtime) -> list[float]:
    tensor = torch.tensor(list(values), dtype=torch.float64, device=runtime.device)
    if runtime.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def update_early_stopping(
    best_psnr: float,
    current_psnr: float,
    epochs_without_improvement: int,
    minimum_delta: float,
) -> tuple[float, int, bool]:
    improved = current_psnr > best_psnr + minimum_delta
    if improved:
        return current_psnr, 0, True
    return best_psnr, epochs_without_improvement + 1, False


def unchanged_dataset_is_converged(saved: dict[str, Any], fingerprint: str) -> bool:
    return (
        saved.get("dataset_fingerprint") == fingerprint
        and bool(saved.get("dataset_converged", False))
    )


@torch.inference_mode()
def validate(
    generator: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
) -> dict[str, float]:
    generator.eval()
    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    image_count = 0
    for original, corrected, _keys in loader:
        original = original.to(runtime.device, non_blocking=True)
        corrected = corrected.to(runtime.device, non_blocking=True)
        generated = generator(original)
        batch = original.shape[0]
        l1_sum += torch.nn.functional.l1_loss(generated, corrected, reduction="mean").item() * batch
        generated_01 = denormalize_image(generated.float())
        corrected_01 = denormalize_image(corrected.float())
        psnr_sum += psnr_per_image(generated_01, corrected_01).sum().item()
        ssim_sum += ssim_per_image(generated_01, corrected_01).sum().item()
        image_count += batch

    l1_sum, psnr_sum, ssim_sum, image_count_float = reduce_sums(
        (l1_sum, psnr_sum, ssim_sum, float(image_count)), runtime
    )
    if image_count_float < 1:
        raise RuntimeError("Validation split is empty across all ranks")
    return {
        "l1": l1_sum / image_count_float,
        "psnr_db": psnr_sum / image_count_float,
        "ssim": ssim_sum / image_count_float,
    }


def checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scaler: torch.amp.GradScaler,
    discriminator_scaler: torch.amp.GradScaler,
    best_psnr: float,
    epochs_without_improvement: int,
    dataset_converged: bool,
    fingerprint: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "architecture": "from_scratch_unet256_conditional_patchgan70",
        "epoch": epoch,
        "global_step": global_step,
        "generator": unwrap(generator).state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "generator_scaler": generator_scaler.state_dict(),
        "discriminator_scaler": discriminator_scaler.state_dict(),
        "best_psnr_db": best_psnr,
        "epochs_without_improvement": epochs_without_improvement,
        "dataset_converged": dataset_converged,
        "dataset_fingerprint": fingerprint,
        "metrics": metrics,
        "fixed_hyperparameters": {
            "image_size": IMAGE_SIZE,
            "lambda_L1": LAMBDA_L1,
            "learning_rate": LEARNING_RATE,
            "adam_betas": list(ADAM_BETAS),
        },
    }


def append_json_log(output_root: Path, record: dict[str, Any]) -> None:
    # This append-only log is local model metadata; production data is untouched.
    log_file = ensure_child_path(output_root, output_root / "training_history.jsonl")
    with log_file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def train(args: argparse.Namespace) -> int:
    runtime = initialize_runtime(args.device)
    stop = StopController()
    signal.signal(signal.SIGINT, stop.request)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop.request)

    try:
        set_seed(args.seed, runtime.rank, args.deterministic)
        deadline = deadline_today(args.stop_at)
        if args.cache_index is not None:
            protected_roots = [args.cache_index.resolve(strict=True).parent, *args.read_only_root]
        else:
            protected_roots = [args.original_root, args.corrected_root, *args.read_only_root]
        output_root = validate_write_root(args.output_dir, protected_roots, create=True)
        checkpoint_root = validate_write_root(
            output_root / "checkpoints", protected_roots, create=True
        )

        report: Any
        if args.cache_index is not None:
            report = load_verified_cache_pairs(
                args.cache_index,
                exact_filename_only=args.exact_cache_filenames,
                max_aspect_change=args.max_aspect_change,
            )
            pairs = report.pairs
            source_summary = {
                "security": "local replay cache is read-only; production paths were not opened",
                "cache_records": report.cache_records,
                "accepted_records": report.accepted_records,
                "accepted_samples": len(report.pairs),
                "replay_samples": report.replay_samples,
                "rejected_match_method": report.rejected_match_method,
                "rejected_filename": report.rejected_filename,
                "rejected_geometry": report.rejected_geometry,
                "missing_files": report.missing_files,
            }
        else:
            report = build_identical_filename_pairs(args.original_root, args.corrected_root)
            pairs = report.pairs
            source_summary = {
                "security": "production inputs are read-only",
                "original_images": report.original_images,
                "corrected_images": report.corrected_images,
                "accepted_pairs": len(report.pairs),
                "unmatched_originals": report.unmatched_originals,
                "ambiguous_filenames": len(report.ambiguous_filenames),
            }
        if not pairs:
            raise RuntimeError("No secure image pairs passed the configured supervision filters")
        if runtime.primary:
            print(json.dumps(source_summary, indent=2), flush=True)

        active_original = args.original_root
        active_corrected = args.corrected_root
        if args.stage_data:
            staging_directory = args.staging_dir or (output_root / "staging")
            if runtime.primary:
                active_original, active_corrected = stage_pairs(
                    report.pairs,
                    staging_directory,
                    protected_roots,
                    CopyWindow.from_strings(args.stage_start, args.stage_end),
                )
            if runtime.distributed:
                dist.barrier()
                if not runtime.primary:
                    staging_root = Path(staging_directory).resolve(strict=True)
                    active_original = staging_root / "original"
                    active_corrected = staging_root / "corrected"
            report = build_identical_filename_pairs(active_original, active_corrected)
            pairs = report.pairs
            if len(report.pairs) < 1:
                raise RuntimeError("Verified staging did not produce any pairs")

        if args.stage_only or args.dry_run:
            if runtime.primary:
                print(
                    f"Validation complete: {len(report.pairs)} secure pairs; "
                    f"no model training performed.",
                    flush=True,
                )
            return 0

        fingerprint = dataset_fingerprint(pairs)
        frozen_path = output_root / "frozen_validation.json"
        frozen_groups: list[str] | None = None
        if frozen_path.exists():
            frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            raw_groups = frozen_payload.get("group_keys")
            if not isinstance(raw_groups, list) or not all(isinstance(item, str) for item in raw_groups):
                raise ValueError("Invalid frozen validation manifest")
            frozen_groups = raw_groups
        train_pairs, validation_pairs, selected_groups = split_with_frozen_validation(
            pairs, args.validation_fraction, args.seed, frozen_groups
        )
        if runtime.primary and frozen_groups is None:
            atomic_json_save(
                {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "group_keys": selected_groups,
                    "policy": "frozen pair groups; new groups enter training only",
                },
                frozen_path,
                output_root,
            )
        if runtime.distributed:
            # Avoid validation padding/duplication: each rank owns a unique shard.
            local_validation_pairs = validation_pairs[runtime.rank :: runtime.world_size]
        else:
            local_validation_pairs = validation_pairs

        training_dataset = PairedStyleDataset(
            train_pairs, image_size=IMAGE_SIZE, max_image_megapixels=args.max_image_megapixels
        )
        validation_dataset = PairedStyleDataset(
            local_validation_pairs,
            image_size=IMAGE_SIZE,
            max_image_megapixels=args.max_image_megapixels,
        )
        train_sampler = (
            DistributedSampler(
                training_dataset,
                num_replicas=runtime.world_size,
                rank=runtime.rank,
                shuffle=True,
                seed=args.seed,
            )
            if runtime.distributed
            else None
        )
        workers = default_worker_count() if args.num_workers is None else args.num_workers
        train_loader = make_loader(
            training_dataset,
            batch_size=args.batch_size,
            workers=workers,
            shuffle=True,
            sampler=train_sampler,
            cuda=runtime.device.type == "cuda",
        )
        validation_loader = make_loader(
            validation_dataset,
            batch_size=args.batch_size,
            workers=workers,
            shuffle=False,
            sampler=None,
            cuda=runtime.device.type == "cuda",
        )

        generator = UNetGenerator().to(runtime.device)
        discriminator = PatchDiscriminator().to(runtime.device)
        generator.apply(initialize_pix2pix_weights)
        discriminator.apply(initialize_pix2pix_weights)
        generator_optimizer = torch.optim.Adam(
            generator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS
        )
        amp_enabled = runtime.device.type == "cuda" and not args.disable_amp
        generator_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        discriminator_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        start_epoch = 1
        global_step = 0
        best_psnr = -math.inf
        epochs_without_improvement = 0
        dataset_converged = False
        dataset_changed = True
        if args.resume is not None:
            resume_path = args.resume if args.resume.is_absolute() else checkpoint_root / args.resume
            saved = safe_torch_load(resume_path, checkpoint_root, runtime.device)
            if saved.get("format_version") != 1:
                raise ValueError("Unsupported or missing checkpoint format_version")
            generator.load_state_dict(saved["generator"])
            discriminator.load_state_dict(saved["discriminator"])
            generator_optimizer.load_state_dict(saved["generator_optimizer"])
            discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
            generator_scaler.load_state_dict(saved.get("generator_scaler", {}))
            discriminator_scaler.load_state_dict(saved.get("discriminator_scaler", {}))
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved.get("global_step", 0))
            best_psnr = float(saved.get("best_psnr_db", -math.inf))
            dataset_changed = saved.get("dataset_fingerprint") != fingerprint
            if not dataset_changed:
                epochs_without_improvement = int(saved.get("epochs_without_improvement", 0))
                dataset_converged = bool(saved.get("dataset_converged", False))
            if runtime.primary:
                print(f"Resumed {resume_path}; dataset_changed={dataset_changed}", flush=True)
            if args.skip_unchanged and unchanged_dataset_is_converged(saved, fingerprint):
                if runtime.primary:
                    atomic_json_save(
                        {
                            "checked_at": datetime.now().astimezone().isoformat(),
                            "status": "skipped",
                            "reason": "dataset unchanged and early-stopping convergence already reached",
                            "dataset_fingerprint": fingerprint,
                        },
                        output_root / "shadow_status.json",
                        output_root,
                    )
                    print("Shadow training skipped: unchanged dataset is already converged.", flush=True)
                return 0

        if runtime.distributed:
            generator = DistributedDataParallel(
                generator,
                device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            )
            discriminator = DistributedDataParallel(
                discriminator,
                device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            )

        adversarial_loss = nn.BCEWithLogitsLoss()
        pixel_loss = nn.L1Loss()
        parameter_count = sum(p.numel() for p in unwrap(generator).parameters()) + sum(
            p.numel() for p in unwrap(discriminator).parameters()
        )
        end_epoch = start_epoch + args.epochs - 1
        if runtime.primary:
            metadata = {
                "created_at": datetime.now().astimezone().isoformat(),
                "architecture": "U-Net-256 generator + conditional 70x70 PatchGAN",
                "pretrained_models": False,
                "image_size": IMAGE_SIZE,
                "lambda_L1": LAMBDA_L1,
                "learning_rate": LEARNING_RATE,
                "adam_betas": list(ADAM_BETAS),
                "train_pairs": len(train_pairs),
                "validation_pairs": len(validation_pairs),
                "dataset_fingerprint": fingerprint,
                "dataset_changed": dataset_changed,
                "frozen_validation_groups": len(selected_groups),
                "early_stopping_patience": args.patience,
                "minimum_psnr_delta": args.minimum_psnr_delta,
                "scheduled_stop": deadline.isoformat() if deadline else None,
                "parameter_count": parameter_count,
                "world_size": runtime.world_size,
                "device": str(runtime.device),
                "mixed_precision": amp_enabled,
                "production_inputs_mutated": False,
            }
            atomic_json_save(metadata, output_root / "run_metadata.json", output_root)
            print(json.dumps(metadata, indent=2), flush=True)

        last_completed_epoch = start_epoch - 1
        last_metrics: dict[str, float] = {}
        scheduled_stop = False
        for epoch in range(start_epoch, end_epoch + 1):
            if stop.requested:
                break
            if deadline is not None and deadline - datetime.now() < timedelta(
                minutes=args.minimum_remaining_minutes
            ):
                scheduled_stop = True
                if runtime.primary:
                    print("No new pix2pix epoch started near the scheduled cutoff.", flush=True)
                break
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            generator.train()
            discriminator.train()
            generator_sum = discriminator_sum = adversarial_sum = l1_sum = 0.0
            image_count = 0
            epoch_started = time.monotonic()

            for batch_index, (original, corrected, _keys) in enumerate(train_loader, start=1):
                original = original.to(runtime.device, non_blocking=True)
                corrected = corrected.to(runtime.device, non_blocking=True)
                batch = original.shape[0]

                # Train the conditional PatchGAN discriminator.
                set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    generated = generator(original)
                    real_logits = discriminator(original, corrected)
                    fake_logits = discriminator(original, generated.detach())
                    d_real = adversarial_loss(real_logits, torch.ones_like(real_logits))
                    d_fake = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (d_real + d_fake)
                discriminator_scaler.scale(d_loss).backward()
                discriminator_scaler.step(discriminator_optimizer)
                discriminator_scaler.update()

                # Train G with adversarial realism plus the fixed 100x L1 fidelity term.
                set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    generated_logits = discriminator(original, generated)
                    g_adversarial = adversarial_loss(
                        generated_logits, torch.ones_like(generated_logits)
                    )
                    g_l1 = pixel_loss(generated, corrected)
                    g_loss = g_adversarial + LAMBDA_L1 * g_l1
                generator_scaler.scale(g_loss).backward()
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()
                set_requires_grad(discriminator, True)

                generator_sum += g_loss.detach().item() * batch
                discriminator_sum += d_loss.detach().item() * batch
                adversarial_sum += g_adversarial.detach().item() * batch
                l1_sum += g_l1.detach().item() * batch
                image_count += batch
                global_step += 1

                # Dynamic cache release is deliberately inside the training cycle.
                if runtime.device.type == "cuda" and global_step % args.empty_cache_interval == 0:
                    torch.cuda.empty_cache()
                if deadline is not None and datetime.now() >= deadline - timedelta(minutes=2):
                    scheduled_stop = True
                    if runtime.primary:
                        print("Scheduled cutoff approaching; closing the partial epoch safely.", flush=True)
                    break
                if stop.requested:
                    break

            generator_sum, discriminator_sum, adversarial_sum, l1_sum, count = reduce_sums(
                (generator_sum, discriminator_sum, adversarial_sum, l1_sum, float(image_count)),
                runtime,
            )
            if count < 1:
                raise RuntimeError("Training produced no batches")
            validation = validate(generator, validation_loader, runtime)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            last_completed_epoch = epoch
            best_psnr, epochs_without_improvement, improved = update_early_stopping(
                best_psnr,
                validation["psnr_db"],
                epochs_without_improvement,
                args.minimum_psnr_delta,
            )
            dataset_converged = epochs_without_improvement >= args.patience
            last_metrics = {
                "generator_loss": generator_sum / count,
                "discriminator_loss": discriminator_sum / count,
                "adversarial_loss": adversarial_sum / count,
                "pixel_l1": l1_sum / count,
                "validation_l1": validation["l1"],
                "psnr_db": validation["psnr_db"],
                "ssim": validation["ssim"],
            }
            payload = checkpoint_payload(
                epoch=epoch,
                global_step=global_step,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                best_psnr=best_psnr,
                epochs_without_improvement=epochs_without_improvement,
                dataset_converged=dataset_converged,
                fingerprint=fingerprint,
                metrics=last_metrics,
            )
            if runtime.primary:
                atomic_torch_save(payload, checkpoint_root / "last.pt", checkpoint_root)
                if improved:
                    atomic_torch_save(payload, checkpoint_root / "best.pt", checkpoint_root)
                if epoch % args.checkpoint_every == 0:
                    atomic_torch_save(
                        payload, checkpoint_root / f"epoch_{epoch:04d}.pt", checkpoint_root
                    )
                    prune_periodic_checkpoints(checkpoint_root, args.max_periodic_checkpoints)
                record = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "seconds": round(time.monotonic() - epoch_started, 3),
                    "improved": improved,
                    "epochs_without_improvement": epochs_without_improvement,
                    "dataset_converged": dataset_converged,
                    **last_metrics,
                }
                append_json_log(output_root, record)
                print(
                    f"epoch={epoch} G={last_metrics['generator_loss']:.6f} "
                    f"D={last_metrics['discriminator_loss']:.6f} "
                    f"L1={last_metrics['pixel_l1']:.6f} "
                    f"PSNR={last_metrics['psnr_db']:.3f}dB "
                    f"SSIM={last_metrics['ssim']:.6f} improved={improved}",
                    flush=True,
                )
                atomic_json_save(
                    {
                        "checked_at": datetime.now().astimezone().isoformat(),
                        "status": "converged" if dataset_converged else "training",
                        "dataset_fingerprint": fingerprint,
                        "epoch": epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "best_psnr_db": best_psnr,
                        "scheduled_stop": scheduled_stop,
                    },
                    output_root / "shadow_status.json",
                    output_root,
                )
            if dataset_converged:
                if runtime.primary:
                    print(
                        f"Early stopping: PSNR did not improve by {args.minimum_psnr_delta:.3f} dB "
                        f"for {args.patience} epochs.",
                        flush=True,
                    )
                break
            if scheduled_stop:
                break

        if runtime.primary:
            print(
                f"Training stopped cleanly after epoch {last_completed_epoch}. "
                f"Atomic checkpoint: {checkpoint_root / 'last.pt'}",
                flush=True,
            )
        return 0
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cleanup_runtime(runtime)


def main(argv: list[str] | None = None) -> int:
    try:
        return train(parse_arguments(argv))
    except (PathSecurityError, StagingWindowClosed, ValueError, RuntimeError) as exc:
        print(f"SECURE TRAINING STOPPED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
