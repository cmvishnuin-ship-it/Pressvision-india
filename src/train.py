from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from correction_ml.dataset import MaterializedDataset, PairedPatchDataset, color_moment_loss, edge_loss, gaussian_ssim_loss
from correction_ml.model import CorrectionUNetV2, count_parameters
from correction_ml.model_global import CorrectionUNetV3Global
from correction_ml.model_pretrained import CorrectionUNetV2PretrainedBackbone
from correction_ml.monitoring import pair_identifier
from correction_ml.parameters import INPUT_FEATURE_NAMES, PARAMETER_NAMES, load_input_features, load_parameter_targets, robust_normalizer
from correction_ml.pairs import discover_pairs, load_config, read_manifest, write_manifest
from correction_ml.safety import assert_not_production_output

CHECKPOINT_LOSSES = ["pixel_l1", "edge", "gaussian_ssim", "color_moments", "global_parameters"]
BACKBONE_ARCHITECTURES = {
    "scratch": "correction_unet_v2",
    "resnet18": "correction_unet_v2_pretrained_resnet18",
    "global": "correction_unet_v3_global",
}
DEFAULT_CHECKPOINT_PREFIXES = {"scratch": "v2", "resnet18": "v2_pretrained", "global": "v3_global"}
# Validation losses are only comparable under the same protocol. Version 1 is
# the first with fixed (seeded) validation crops; earlier checkpoints scored
# fresh random crops every epoch.
VALIDATION_PROTOCOL = "fixed_crops_v1"
MATERIALIZE_VALIDATION_MAX_BYTES = 1024**3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image correction model.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without validation improvement; 0 disables early stopping.")
    parser.add_argument("--balance-sets", action=argparse.BooleanOptionalAction, default=True, help="Give each production image set equal sampling influence.")
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="Cap sampled training patches per epoch; 0 uses the full logical dataset size.")
    parser.add_argument("--current-sample-fraction", type=float, default=0.60, help="Sampling share for current versus historical replay data.")
    parser.add_argument("--set-balance-exponent", type=float, default=0.0, help="With --balance-sets, each image set's sampling mass is proportional to (pair count ** exponent): 0 gives every set equal mass, 1 samples in proportion to data.")
    parser.add_argument("--patches-per-load", type=int, default=1, help="With --balance-sets, cut this many consecutive training patches from each decoded pair; decoding dominates CPU epoch time.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--backbone", choices=sorted(BACKBONE_ARCHITECTURES), default="scratch", help="'scratch' is the original from-scratch V2 encoder; 'resnet18' adds a frozen ImageNet-pretrained feature branch; 'global' adds a whole-image context branch with per-image tone curves.")
    parser.add_argument("--checkpoint-prefix", default=None, help="Checkpoint filename prefix (best_<prefix>.pt / last_<prefix>.pt). Defaults to 'v2' for scratch, 'v2_pretrained' for resnet18, 'v3_global' for global.")
    parser.add_argument("--pretrained-weights", default="artifacts/pretrained/resnet18_imagenet.pt", help="Local resnet18 ImageNet weights file, consulted only for a fresh (non-resumed) --backbone resnet18 run.")
    parser.add_argument("--init-from", help="V2 checkpoint that warm-starts a fresh (non-resumed) --backbone global run; the new model starts out identical to it.")
    parser.add_argument("--context-size", type=int, default=128, help="Side of the whole-image view given to --backbone global.")
    parser.add_argument("--global-loss-weight", type=float, default=1.0, help="For --backbone global: weight of the whole-image loss comparing the image-wide correction with the editors' whole image; 0 turns it off.")
    parser.add_argument("--train-global-only", action="store_true", help="For --backbone global: train only the whole-image layers V3 adds; the V2 layers stay as warm-started.")
    parser.add_argument("--exclude-frozen-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Keep artifacts/frozen_benchmark.json pairs out of training and validation, as its policy states.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Useful for a smoke test; 0 uses all.")
    parser.add_argument("--stop-at", help="Local cutoff: HH:MM today (such as 09:50) or YYYY-MM-DDTHH:MM. No new epoch starts near this time.")
    parser.add_argument("--cache-only", metavar="CACHE_ROOT", help="Train only on pairs whose files are inside this local cache folder: pairs with production paths are dropped and pair discovery never runs.")
    parser.add_argument("--minimum-remaining-minutes", type=int, default=5, help="Required time remaining before another epoch starts.")
    return parser.parse_args()


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
    except ValueError as error:
        raise SystemExit("--stop-at must be HH:MM (for example 09:50) or YYYY-MM-DDTHH:MM.") from error
    return datetime.combine(current.date(), parsed)


def inside_folder(path: str, folder: str) -> bool:
    """Pure string check (no filesystem access, so a production path is never touched)."""
    root = os.path.normcase(os.path.abspath(folder)).rstrip("\\/") + os.sep
    return os.path.normcase(os.path.abspath(path)).startswith(root)


def device_for_machine() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pixel_objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        functional.l1_loss(prediction, target)
        + 0.15 * edge_loss(prediction, target)
        + 0.10 * gaussian_ssim_loss(prediction, target)
        + 0.10 * color_moment_loss(prediction, target)
    )


def identity_pixel_loss(loader) -> float:
    """Pixel loss of leaving every image unchanged: the bar a useful model must beat."""
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            source, target = batch[0], batch[1]
            total += float(pixel_objective(source, target)) * source.shape[0]
            count += source.shape[0]
    return total / max(1, count)


class EpochInterrupted(Exception):
    """The hand-over time arrived during a training epoch."""


# Optional callable set by the service to publish live progress (shown in the dashboard).
progress_hook = None


def report_progress(**event) -> None:
    if progress_hook is not None:
        try:
            progress_hook(event)
        except Exception:
            pass  # progress display must never interrupt training


def run_epoch(model, loader, device, optimizer=None, global_weight: float = 0.0, deadline: datetime | None = None, on_batch=None) -> tuple[float, float, float, float]:
    """Return mean total, pixel, parameter and whole-image losses.

    With ``global_weight`` the image-wide correction of a context model, applied
    to the whole-image view of the original, is also compared with the editors'
    whole image. With ``deadline`` a training epoch raises EpochInterrupted as
    soon as it passes, so the service hands over within one batch.
    """
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    pixel_total = parameter_total = global_total = 0.0
    progress_interval = max(1, len(loader) // 10)
    for batch_number, batch in enumerate(loader, start=1):
        if training and deadline is not None and datetime.now() >= deadline:
            raise EpochInterrupted
        source, target, parameter_target = (value.to(device) for value in batch[:3])
        context = batch[3].to(device) if len(batch) > 3 else None
        target_view = batch[4].to(device) if len(batch) > 4 else None
        with torch.set_grad_enabled(training):
            if context is None:
                prediction, parameter_prediction = model.forward_with_parameters(source)
            else:
                prediction, parameter_prediction = model.forward_with_parameters(source, context)
            pixel_loss = pixel_objective(prediction, target)
            parameter_loss = functional.smooth_l1_loss(parameter_prediction, parameter_target)
            loss = pixel_loss + 0.05 * parameter_loss
            global_loss = torch.zeros((), device=source.device)
            if global_weight > 0 and target_view is not None:
                global_loss = functional.l1_loss(model.whole_image_correction(context), target_view)
                loss = loss + global_weight * global_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(loss.detach()) * source.shape[0]
        pixel_total += float(pixel_loss.detach()) * source.shape[0]
        parameter_total += float(parameter_loss.detach()) * source.shape[0]
        global_total += float(global_loss.detach()) * source.shape[0]
        count += source.shape[0]
        if training and (batch_number == 1 or batch_number % progress_interval == 0 or batch_number == len(loader)):
            percent = round(100 * batch_number / max(1, len(loader)))
            print(f"  training progress: {batch_number}/{len(loader)} batches ({percent}%)", flush=True)
            if on_batch is not None:
                on_batch(batch_number, len(loader))
    return total / max(1, count), pixel_total / max(1, count), parameter_total / max(1, count), global_total / max(1, count)


def dataset_fingerprint(manifest: Path, parameter_file: Path, extra: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for path in (manifest, parameter_file):
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    for value in extra:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def frozen_benchmark_ids(artifacts: Path) -> set[str]:
    try:
        payload = json.loads((artifacts / "frozen_benchmark.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("pair_ids", []) if isinstance(payload, dict) else []
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


class RepeatedSampler(Sampler):
    """Yield every index drawn by ``base`` ``repeats`` times in a row."""

    def __init__(self, base: Sampler, repeats: int):
        self.base = base
        self.repeats = repeats

    def __iter__(self):
        for index in self.base:
            for _ in range(self.repeats):
                yield index

    def __len__(self) -> int:
        return len(self.base) * self.repeats


def balanced_sampler(pairs, patches_per_image: int, max_samples: int = 0, current_fraction: float = 0.60, set_exponent: float = 0.0, repeats: int = 1) -> Sampler:
    def group(pair):
        age_group = "historical" if getattr(pair, "source_status", "current") == "historical" else "current"
        return age_group, pair.set_name

    counts = Counter(group(pair) for pair in pairs)
    statuses = {status for status, _ in counts}
    set_mass_totals = Counter()
    for (status, _), count in counts.items():
        set_mass_totals[status] += count**set_exponent
    if statuses == {"current", "historical"}:
        status_mass = {"current": min(1.0, max(0.0, current_fraction)), "historical": 1.0 - min(1.0, max(0.0, current_fraction))}
    else:
        status_mass = {status: 1.0 for status in statuses}
    weights = []
    for pair in pairs:
        key = group(pair)
        weight = status_mass[key[0]] * (counts[key] ** set_exponent / set_mass_totals[key[0]]) / counts[key]
        weights.extend([weight] * patches_per_image)
    sample_count = len(weights) if max_samples <= 0 else min(len(weights), max_samples)
    repeats = max(1, int(repeats))
    if repeats == 1:
        return WeightedRandomSampler(weights, num_samples=sample_count, replacement=True)
    return RepeatedSampler(WeightedRandomSampler(weights, num_samples=math.ceil(sample_count / repeats), replacement=True), repeats)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, args, pair_count, parameter_center, parameter_scale, input_feature_center, input_feature_scale, fingerprint, set_counts, epochs_without_improvement, architecture, validation_protocol=None, context_size=0, losses=CHECKPOINT_LOSSES):
    # Write then rename: the 10:00 hard stop can kill this process mid-save,
    # and a torn best_v2.pt would break every following session's resume.
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "base_channels": args.base_channels,
            "patch_size": args.patch_size,
            "training_pair_count": pair_count,
            "architecture": architecture,
            "parameter_names": PARAMETER_NAMES,
            "parameter_center": parameter_center.tolist(),
            "parameter_scale": parameter_scale.tolist(),
            "input_feature_names": INPUT_FEATURE_NAMES,
            "input_feature_center": input_feature_center.tolist(),
            "input_feature_scale": input_feature_scale.tolist(),
            "losses": list(losses),
            "dataset_fingerprint": fingerprint,
            "image_set_counts": dict(set_counts),
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_protocol": validation_protocol,
            "context_size": context_size,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> str:
    args = arguments()
    deadline = deadline_today(args.stop_at)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    artifacts = Path(args.artifacts)
    assert_not_production_output(artifacts, config)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = artifacts / "manifest.csv"
    if manifest.exists() and manifest.stat().st_size:
        pairs = read_manifest(manifest)
    elif args.cache_only:
        raise SystemExit("No local manifest.csv yet; the daily production ingest has to run first.")
    else:
        pairs, _ = discover_pairs(config)
        write_manifest(pairs, manifest)
    production_pairs_excluded = 0
    if args.cache_only:
        kept = [pair for pair in pairs if inside_folder(pair.original, args.cache_only) and inside_folder(pair.corrected, args.cache_only)]
        production_pairs_excluded = len(pairs) - len(kept)
        pairs = kept
    threshold = float(config.get("max_aspect_change_for_training", 0.02))
    pairs = [pair for pair in pairs if pair.aspect_change <= threshold]
    parameter_file = artifacts / "correction_parameters.csv"
    if not parameter_file.exists():
        raise SystemExit("Missing correction_parameters.csv. Run analyze.py before training V2.")
    parameter_targets = load_parameter_targets(parameter_file)
    input_features = load_input_features(parameter_file)
    pairs = [pair for pair in pairs if (pair.original, pair.corrected) in parameter_targets and (pair.original, pair.corrected) in input_features]
    benchmark_ids = frozen_benchmark_ids(artifacts) if args.exclude_frozen_benchmark else set()
    benchmark_excluded = 0
    if benchmark_ids:
        kept = [pair for pair in pairs if pair_identifier(pair) not in benchmark_ids]
        benchmark_excluded = len(pairs) - len(kept)
        pairs = kept
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    training_pairs = [pair for pair in pairs if pair.split == "train"]
    validation_pairs = [pair for pair in pairs if pair.split == "validation"]
    if not training_pairs:
        raise SystemExit("No training pairs found. Run analyze.py and check config.json.")
    if not validation_pairs:
        validation_pairs = training_pairs[-max(1, len(training_pairs) // 10) :]

    training_parameters = np.stack([parameter_targets[(pair.original, pair.corrected)] for pair in training_pairs])
    parameter_center, parameter_scale = robust_normalizer(training_parameters)
    training_features = np.stack([input_features[(pair.original, pair.corrected)] for pair in training_pairs])
    input_feature_center, input_feature_scale = robust_normalizer(training_features)
    # Excluded benchmark pairs change what is trained and validated even though
    # the manifest files do not change, so they are part of the fingerprint.
    fingerprint = dataset_fingerprint(manifest, parameter_file, tuple(sorted(benchmark_ids)))
    set_counts = Counter(pair.set_name for pair in training_pairs)
    source_status_counts = Counter(pair.source_status for pair in training_pairs)

    checkpoint_prefix = args.checkpoint_prefix or DEFAULT_CHECKPOINT_PREFIXES[args.backbone]
    report_progress(model=checkpoint_prefix, phase="preparing", training_pairs=len(training_pairs), validation_pairs=len(validation_pairs))
    context_size = args.context_size if args.backbone == "global" else 0
    global_weight = args.global_loss_weight if context_size > 0 else 0.0
    losses = CHECKPOINT_LOSSES + (["whole_image_global"] if global_weight > 0 else [])
    validation_per_image = max(1, args.patches_per_image // 2)
    validation_protocol = f"{VALIDATION_PROTOCOL}:patch={args.patch_size}:per_image={validation_per_image}:context={context_size}"
    training_data = PairedPatchDataset(training_pairs, args.patch_size, args.patches_per_image, True, parameter_targets, parameter_center, parameter_scale, context_size, seed)
    validation_data = PairedPatchDataset(validation_pairs, args.patch_size, validation_per_image, False, parameter_targets, parameter_center, parameter_scale, context_size, seed)
    # uint8 bytes per sample: original and corrected patch, plus the two whole-image views.
    if len(validation_data) * (6 * args.patch_size**2 + 6 * context_size**2) <= MATERIALIZE_VALIDATION_MAX_BYTES:
        print(f"Preparing {len(validation_data)} fixed validation samples once for this session.", flush=True)
        validation_data = MaterializedDataset(validation_data)
    sampler = balanced_sampler(training_pairs, args.patches_per_image, args.samples_per_epoch, args.current_sample_fraction, args.set_balance_exponent, args.patches_per_load) if args.balance_sets else None
    training_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = device_for_machine()
    architecture = BACKBONE_ARCHITECTURES[args.backbone]
    if args.backbone == "resnet18":
        pretrained_weights = Path(args.pretrained_weights) if args.pretrained_weights and not args.resume else None
        if pretrained_weights is not None and not pretrained_weights.is_file():
            pretrained_weights = None
        model = CorrectionUNetV2PretrainedBackbone(args.base_channels, len(PARAMETER_NAMES), pretrained_weights).to(device)
    elif args.backbone == "global":
        model = CorrectionUNetV3Global(args.base_channels, len(PARAMETER_NAMES), context_size).to(device)
        if args.train_global_only:
            model.freeze_v2_layers()
    else:
        model = CorrectionUNetV2(args.base_channels, len(PARAMETER_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=args.minimum_learning_rate)
    start_epoch, best_loss = 1, float("inf")
    epochs_without_improvement = 0
    data_status = "new training run"
    carried_weights = False
    initialized_from = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != architecture:
            raise SystemExit(f"Resume checkpoint architecture {checkpoint.get('architecture')!r} does not match --backbone {args.backbone!r} ({architecture!r}).")
        model.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print("Optimizer state not reused: the set of trained layers differs from the checkpoint's.", flush=True)
        carried_weights = True
        start_epoch = int(checkpoint["epoch"]) + 1
        old_fingerprint = checkpoint.get("dataset_fingerprint")
        dataset_changed = old_fingerprint != fingerprint
        losses_changed = checkpoint.get("losses") != losses
        protocol_changed = checkpoint.get("validation_protocol") != validation_protocol
        if not dataset_changed and not losses_changed and not protocol_changed:
            data_status = "unchanged since checkpoint"
            best_loss = float(checkpoint["best_validation_loss"])
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            if dataset_changed and old_fingerprint is None:
                data_status = "changed: checkpoint predates tracking; fresh validation baseline"
                old_label = "legacy"
            elif dataset_changed:
                data_status = "changed: new, removed, or re-analyzed pairs; fresh validation baseline"
                old_label = str(old_fingerprint)[:12]
            elif losses_changed:
                data_status = "changed: loss function updated; fresh validation baseline"
                old_label = "pre_gaussian_ssim" if "gaussian_ssim" not in (checkpoint.get("losses") or []) else "previous_loss"
            else:
                data_status = "changed: validation protocol updated (fixed validation crops); fresh validation baseline"
                old_label = "pre_fixed_validation"
            # Validation losses from a different dataset, loss formula, or
            # validation protocol are not comparable to this run's. Keep the
            # learned weights, re-measure them below as the baseline a new best
            # must beat, reset scheduler state, and archive the previous best.
            best_loss = float("inf")
            epochs_without_improvement = 0
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=args.minimum_learning_rate
            )
            archive = artifacts / "checkpoint_archive"
            archive.mkdir(parents=True, exist_ok=True)
            archived_checkpoint = archive / f"best_{checkpoint_prefix}_epoch_{checkpoint['epoch']}_{old_label}.pt"
            if not archived_checkpoint.exists():
                shutil.copy2(args.resume, archived_checkpoint)
            print(f"{data_status}; archived previous checkpoint at {archived_checkpoint}")
        # A later continuation sees different deterministic crops, flips, rotations,
        # and shuffle order instead of replaying the first run exactly.
        continuation_seed = seed + start_epoch
        random.seed(continuation_seed)
        np.random.seed(continuation_seed)
        torch.manual_seed(continuation_seed)
    elif args.backbone == "global" and args.init_from:
        source_checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        if source_checkpoint.get("architecture") != BACKBONE_ARCHITECTURES["scratch"]:
            raise SystemExit(f"--init-from needs a {BACKBONE_ARCHITECTURES['scratch']!r} checkpoint, not {source_checkpoint.get('architecture')!r}.")
        model.load_v2_weights(source_checkpoint["model"])
        carried_weights = True
        initialized_from = f"{Path(args.init_from).name} (epoch {source_checkpoint.get('epoch')})"
        data_status = f"new training run warm-started from {initialized_from}"

    # Measured once per session on the fixed validation samples: "unchanged"
    # shows what doing nothing scores, "starting" what the loaded weights score.
    report_progress(model=checkpoint_prefix, phase="baseline")
    identity_pixel = identity_pixel_loss(validation_loader)
    starting_loss, starting_pixel, starting_parameters, starting_global = run_epoch(model, validation_loader, device, global_weight=global_weight)
    # Checkpoints are chosen on picture quality (pixel + parameter loss). The
    # whole-image term trains the curves but does not decide which weights are kept.
    starting_loss -= global_weight * starting_global
    print(f"baseline unchanged_pixel={identity_pixel:.6f} starting_validation={starting_loss:.6f} starting_pixel={starting_pixel:.6f} starting_parameters={starting_parameters:.6f} starting_whole_image={starting_global:.6f}", flush=True)
    if carried_weights and math.isinf(best_loss):
        # A checkpoint is only replaced by weights that beat the ones this
        # session started from on the same validation samples.
        best_loss = starting_loss

    session = {"device": str(device), "architecture": architecture, "parameters": count_parameters(model), "training_pairs": len(training_pairs), "validation_pairs": len(validation_pairs), "image_sets": dict(set_counts), "source_status": dict(source_status_counts), "balanced_sampling": args.balance_sets, "set_balance_exponent": args.set_balance_exponent, "patches_per_load": args.patches_per_load, "replay_strategy": f"{args.current_sample_fraction:.0%} current / {1-args.current_sample_fraction:.0%} historical when both exist", "samples_per_epoch_cap": args.samples_per_epoch, "dataset_status": data_status, "dataset_fingerprint": fingerprint, "frozen_benchmark_pairs_excluded": benchmark_excluded, "cache_only": bool(args.cache_only), "production_pairs_excluded": production_pairs_excluded,"validation_protocol": validation_protocol, "validation_samples": len(validation_data), "unchanged_validation_pixel_loss": identity_pixel, "starting_validation_loss": starting_loss, "starting_validation_pixel_loss": starting_pixel, "best_loss_to_beat": best_loss, "whole_image_loss_weight": global_weight, "losses": losses, "initialized_from": initialized_from, "early_stopping_patience": args.patience, "scheduled_stop": deadline.isoformat() if deadline else None}
    print(json.dumps(session, indent=2))
    (artifacts / "latest_training_session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    history_path = artifacts / f"training_history_{checkpoint_prefix}.csv"
    write_header = not history_path.exists()
    last_epoch_seconds = 0.0
    # Returned to the caller: "deadline", "early_stopped" or "epochs_completed".
    outcome = "epochs_completed"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if deadline is not None:
            remaining = deadline - datetime.now()
            # An epoch must also be expected to finish before the stop time;
            # the minimum alone let a 16-minute epoch start at 09:43.
            required = max(timedelta(minutes=max(0, args.minimum_remaining_minutes)), timedelta(seconds=1.25 * last_epoch_seconds))
            if remaining <= required:
                print(f"Scheduled learning window ending; no new epoch started. Remaining time: {remaining}; required: {required}.")
                outcome = "deadline"
                break
        epoch_started = time.monotonic()
        report_progress(model=checkpoint_prefix, phase="training", epoch=epoch, batch=0, batches=len(training_loader), best=best_loss)

        def on_batch(batch: int, batches: int, epoch: int = epoch) -> None:
            report_progress(model=checkpoint_prefix, phase="training", epoch=epoch, batch=batch, batches=batches, best=best_loss)

        try:
            train_loss, train_pixel, train_parameters, train_global = run_epoch(model, training_loader, device, optimizer, global_weight, deadline, on_batch)
        except EpochInterrupted:
            print("Hand-over time reached during an epoch; it is discarded and the last saved checkpoint stays.", flush=True)
            outcome = "deadline"
            break
        validation_loss, validation_pixel, validation_parameters, validation_global = run_epoch(model, validation_loader, device, global_weight=global_weight)
        train_loss -= global_weight * train_global
        validation_loss -= global_weight * validation_global
        last_epoch_seconds = time.monotonic() - epoch_started
        improved = validation_loss < best_loss - 1e-6
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train={train_loss:.6f} validation={validation_loss:.6f} pixel={validation_pixel:.6f} parameters={validation_parameters:.6f} whole_image={validation_global:.6f} lr={learning_rate:.8f} improved={improved} seconds={last_epoch_seconds:.0f}", flush=True)
        report_progress(model=checkpoint_prefix, phase="epoch_done", epoch=epoch, validation=validation_loss, best=best_loss, improved=improved, seconds=round(last_epoch_seconds))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["epoch", "training_loss", "validation_loss", "train_pixel_loss", "validation_pixel_loss", "train_parameter_loss", "validation_parameter_loss"])
                write_header = False
            writer.writerow([epoch, train_loss, validation_loss, train_pixel, validation_pixel, train_parameters, validation_parameters])
        save_checkpoint(artifacts / f"last_{checkpoint_prefix}.pt", model, optimizer, scheduler, epoch, best_loss, args, len(pairs), parameter_center, parameter_scale, input_feature_center, input_feature_scale, fingerprint, set_counts, epochs_without_improvement, architecture, validation_protocol, context_size, losses)
        if improved:
            save_checkpoint(artifacts / f"best_{checkpoint_prefix}.pt", model, optimizer, scheduler, epoch, best_loss, args, len(pairs), parameter_center, parameter_scale, input_feature_center, input_feature_scale, fingerprint, set_counts, epochs_without_improvement, architecture, validation_protocol, context_size, losses)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping: validation did not improve for {args.patience} epochs. The best checkpoint is preserved.")
            outcome = "early_stopped"
            break
    return outcome


if __name__ == "__main__":
    main()
