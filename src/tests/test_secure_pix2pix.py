from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import pix2pix_train
from secure_pix2pix import IMAGE_SIZE, LAMBDA_L1, LEARNING_RATE
from secure_pix2pix.checkpoints import (
    atomic_torch_save,
    prune_periodic_checkpoints,
    safe_torch_load,
)
from secure_pix2pix.data import (
    PairedStyleDataset,
    build_identical_filename_pairs,
    deterministic_split,
    load_verified_cache_pairs,
    split_with_frozen_validation,
)
from secure_pix2pix.metrics import psnr_per_image, ssim_per_image
from secure_pix2pix.models import PatchDiscriminator, UNetGenerator
from secure_pix2pix.security import PathSecurityError, validate_write_root
from secure_pix2pix.staging import CopyWindow, StagingWindowClosed, stage_pairs


def _image(path: Path, value: int, *, size: tuple[int, int] = (31, 29)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_record(
    cache_root: Path,
    pair_id: str,
    *,
    value: int,
    match_method: str = "filename_and_visual",
    production_original: str | None = None,
    production_corrected: str | None = None,
    aspect_change: float = 0.0,
) -> dict:
    original_relative = Path("pairs") / pair_id / "original.png"
    corrected_relative = Path("pairs") / pair_id / "corrected.png"
    _image(cache_root / original_relative, value)
    _image(cache_root / corrected_relative, value + 1)
    filename = f"{pair_id}.png"
    return {
        "pair": {
            "match_method": match_method,
            "aspect_change": aspect_change,
        },
        "production_original": production_original or rf"C:\production\in\{filename}",
        "production_corrected": production_corrected or rf"C:\production\out\{filename}",
        "cached_original": str(original_relative),
        "cached_corrected": str(corrected_relative),
        "storage": "hot",
    }


def _write_cache_index(cache_root: Path, records: dict[str, dict]) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    index = cache_root / "cache_index.json"
    index.write_text(json.dumps({"version": 1, "pairs": records}), encoding="utf-8")
    return index


def test_fixed_required_hyperparameters() -> None:
    assert IMAGE_SIZE == 256
    assert LAMBDA_L1 == 100.0
    assert LEARNING_RATE == 0.0002


def test_early_stopping_and_unchanged_dataset_policy() -> None:
    best, stale, improved = pix2pix_train.update_early_stopping(20.0, 20.03, 4, 0.02)
    assert (best, stale, improved) == (20.03, 0, True)
    best, stale, improved = pix2pix_train.update_early_stopping(best, 20.04, stale, 0.02)
    assert (best, stale, improved) == (20.03, 1, False)
    assert pix2pix_train.unchanged_dataset_is_converged(
        {"dataset_fingerprint": "abc", "dataset_converged": True}, "abc"
    )
    assert not pix2pix_train.unchanged_dataset_is_converged(
        {"dataset_fingerprint": "old", "dataset_converged": True}, "new"
    )


def test_unet_and_patchgan_shapes() -> None:
    generator = UNetGenerator(base_channels=8).eval()
    discriminator = PatchDiscriminator(base_channels=8).eval()
    source = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        generated = generator(source)
        logits = discriminator(source, generated)
    assert generated.shape == (1, 3, 256, 256)
    assert generated.min() >= -1.0 and generated.max() <= 1.0
    assert logits.shape == (1, 1, 30, 30)


def test_pix2pix_discriminator_and_generator_backward_steps() -> None:
    generator = UNetGenerator(base_channels=8)
    discriminator = PatchDiscriminator(base_channels=8)
    source = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256).clamp(-1.0, 1.0)
    generated = generator(source)

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    discriminator_optimizer.zero_grad(set_to_none=True)
    real_logits = discriminator(source, target)
    fake_logits = discriminator(source, generated.detach())
    discriminator_loss = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
    )
    discriminator_loss.backward()
    discriminator_optimizer.step()

    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    generator_optimizer.zero_grad(set_to_none=True)
    generated_logits = discriminator(source, generated)
    generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        generated_logits, torch.ones_like(generated_logits)
    ) + LAMBDA_L1 * torch.nn.functional.l1_loss(generated, target)
    generator_loss.backward()
    generator_optimizer.step()
    assert torch.isfinite(generator_loss)


def test_identical_filename_pairing_and_readonly_dataset(tmp_path: Path) -> None:
    original_root = tmp_path / "production" / "original"
    corrected_root = tmp_path / "production" / "corrected"
    source = original_root / "day1" / "same.png"
    target = corrected_root / "another-folder" / "same.png"
    _image(source, 30)
    _image(target, 220)
    source_before = (source.stat().st_size, source.stat().st_mtime_ns, _sha256(source))
    target_before = (target.stat().st_size, target.stat().st_mtime_ns, _sha256(target))

    report = build_identical_filename_pairs(original_root, corrected_root)
    assert len(report.pairs) == 1
    dataset = PairedStyleDataset(report.pairs)
    original, corrected, key = dataset[0]

    assert original.shape == corrected.shape == (3, 256, 256)
    assert original.min() >= -1.0 and original.max() <= 1.0
    assert corrected.min() >= -1.0 and corrected.max() <= 1.0
    assert "same.png" in key
    assert source_before == (source.stat().st_size, source.stat().st_mtime_ns, _sha256(source))
    assert target_before == (target.stat().st_size, target.stat().st_mtime_ns, _sha256(target))


def test_pairing_rejects_ambiguous_duplicate_filenames(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    corrected_root = tmp_path / "corrected"
    _image(original_root / "different" / "same.png", 10)
    _image(corrected_root / "a" / "same.png", 20)
    _image(corrected_root / "b" / "same.png", 30)
    report = build_identical_filename_pairs(original_root, corrected_root)
    assert report.pairs == ()
    assert report.ambiguous_filenames == ("same.png",)


def test_deterministic_split_is_stable(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    corrected_root = tmp_path / "corrected"
    for index in range(10):
        _image(original_root / f"{index}.png", index)
        _image(corrected_root / f"{index}.png", index + 1)
    pairs = build_identical_filename_pairs(original_root, corrected_root).pairs
    first_train, first_validation = deterministic_split(pairs, 0.2, 42)
    second_train, second_validation = deterministic_split(pairs, 0.2, 42)
    assert [item.stable_key for item in first_train] == [item.stable_key for item in second_train]
    assert [item.stable_key for item in first_validation] == [
        item.stable_key for item in second_validation
    ]
    assert len(first_train) == 8 and len(first_validation) == 2


def test_verified_cache_adapter_reads_only_local_exact_pairs(tmp_path: Path) -> None:
    cache_root = tmp_path / "artifacts" / "training_cache"
    records = {
        "accepted": _cache_record(cache_root, "accepted", value=10),
        "visual-only": _cache_record(
            cache_root, "visual-only", value=20, match_method="visual_content"
        ),
        "renamed": _cache_record(
            cache_root,
            "renamed",
            value=30,
            production_original=r"C:\production\in\a.png",
            production_corrected=r"C:\production\out\b.png",
        ),
        "geometry": _cache_record(
            cache_root, "geometry", value=40, aspect_change=0.25
        ),
    }
    index = _write_cache_index(cache_root, records)
    before = {
        path: (_sha256(path), path.stat().st_mtime_ns)
        for path in cache_root.rglob("*.png")
    }
    report = load_verified_cache_pairs(index)
    assert len(report.pairs) == 1
    assert report.pairs[0].group_key == "accepted"
    assert report.rejected_match_method == 1
    assert report.rejected_filename == 1
    assert report.rejected_geometry == 1
    assert before == {
        path: (_sha256(path), path.stat().st_mtime_ns)
        for path in cache_root.rglob("*.png")
    }


def test_cache_replay_samples_stay_in_one_split_group(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    replay_samples = []
    for number in range(3):
        original = Path("replay") / "group-a" / f"{number}_original.png"
        corrected = Path("replay") / "group-a" / f"{number}_corrected.png"
        _image(cache_root / original, number)
        _image(cache_root / corrected, number + 1)
        replay_samples.append(
            {"cached_original": str(original), "cached_corrected": str(corrected)}
        )
    records = {
        "group-a": {
            "pair": {"match_method": "filename_and_visual", "aspect_change": 0.0},
            "production_original": r"C:\in\same.png",
            "production_corrected": r"C:\out\same.png",
            "storage": "replay",
            "replay_pairs": replay_samples,
        },
        "group-b": _cache_record(cache_root, "group-b", value=10),
        "group-c": _cache_record(cache_root, "group-c", value=20),
    }
    report = load_verified_cache_pairs(_write_cache_index(cache_root, records))
    training, validation, frozen = split_with_frozen_validation(report.pairs, 0.34, 42)
    training_groups = {pair.split_group for pair in training}
    validation_groups = {pair.split_group for pair in validation}
    assert training_groups.isdisjoint(validation_groups)
    assert set(frozen) == validation_groups
    assert sum(pair.split_group == "group-a" for pair in report.pairs) == 3


def test_frozen_validation_keeps_new_groups_in_training(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    records = {
        name: _cache_record(cache_root, name, value=index * 10)
        for index, name in enumerate(("a", "b", "c", "d"), start=1)
    }
    report = load_verified_cache_pairs(_write_cache_index(cache_root, records))
    _train, _validation, frozen = split_with_frozen_validation(report.pairs, 0.25, 42)
    records["new"] = _cache_record(cache_root, "new", value=60)
    expanded = load_verified_cache_pairs(_write_cache_index(cache_root, records))
    training, validation, reused = split_with_frozen_validation(
        expanded.pairs, 0.25, 42, frozen
    )
    assert reused == frozen
    assert "new" in {pair.split_group for pair in training}
    assert "new" not in {pair.split_group for pair in validation}


def test_write_root_cannot_overlap_production(tmp_path: Path) -> None:
    protected = tmp_path / "production"
    protected.mkdir()
    with pytest.raises(PathSecurityError):
        validate_write_root(protected / "artifacts", [protected])
    with pytest.raises(PathSecurityError):
        validate_write_root(tmp_path, [protected])


def test_staging_gate_rejects_copy_before_creating_directory(tmp_path: Path) -> None:
    original_root = tmp_path / "production" / "original"
    corrected_root = tmp_path / "production" / "corrected"
    _image(original_root / "a.png", 1)
    _image(corrected_root / "a.png", 2)
    pairs = build_identical_filename_pairs(original_root, corrected_root).pairs
    stage = tmp_path / "output" / "stage"
    with pytest.raises(StagingWindowClosed):
        stage_pairs(
            pairs,
            stage,
            [tmp_path / "production"],
            CopyWindow.from_strings("04:00", "10:00"),
            now=lambda: datetime(2026, 8, 21, 12, 0),
        )
    assert not stage.exists()


def test_staging_copies_locally_and_preserves_sources(tmp_path: Path) -> None:
    production = tmp_path / "production"
    original_root = production / "original"
    corrected_root = production / "corrected"
    source = original_root / "a.png"
    target = corrected_root / "a.png"
    _image(source, 1)
    _image(target, 2)
    before = (_sha256(source), source.stat().st_mtime_ns, _sha256(target), target.stat().st_mtime_ns)
    pairs = build_identical_filename_pairs(original_root, corrected_root).pairs
    original_stage, corrected_stage = stage_pairs(
        pairs,
        tmp_path / "output" / "stage",
        [production],
        CopyWindow.from_strings("04:00", "10:00"),
        now=lambda: datetime(2026, 8, 21, 5, 0),
    )
    staged = build_identical_filename_pairs(original_stage, corrected_stage)
    assert len(staged.pairs) == 1
    assert before == (
        _sha256(source),
        source.stat().st_mtime_ns,
        _sha256(target),
        target.stat().st_mtime_ns,
    )
    assert (tmp_path / "output" / "stage" / "staging_manifest.json").is_file()


def test_staging_capacity_is_checked_before_writing(tmp_path: Path) -> None:
    production = tmp_path / "production"
    original_root = production / "original"
    corrected_root = production / "corrected"
    _image(original_root / "a.png", 1)
    _image(corrected_root / "a.png", 2)
    pairs = build_identical_filename_pairs(original_root, corrected_root).pairs
    stage = tmp_path / "output" / "stage"
    with pytest.raises(RuntimeError, match="hard safety limit"):
        stage_pairs(
            pairs,
            stage,
            [production],
            CopyWindow.from_strings("04:00", "10:00"),
            now=lambda: datetime(2026, 8, 21, 5, 0),
            max_staging_bytes=1,
        )
    assert not stage.exists()


def test_staging_skips_rehash_when_source_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secure_pix2pix.staging as staging

    production = tmp_path / "production"
    original_root = production / "original"
    corrected_root = production / "corrected"
    source = original_root / "a.png"
    target = corrected_root / "a.png"
    _image(source, 1)
    _image(target, 2)
    pairs = build_identical_filename_pairs(original_root, corrected_root).pairs
    stage = tmp_path / "output" / "stage"

    stage_pairs(
        pairs, stage, [production], CopyWindow.from_strings("04:00", "10:00"),
        now=lambda: datetime(2026, 8, 21, 5, 0),
    )

    calls: list[Path] = []
    real_sha256 = staging._sha256

    def counting_sha256(path: Path, *args, **kwargs) -> str:
        calls.append(path)
        return real_sha256(path, *args, **kwargs)

    monkeypatch.setattr(staging, "_sha256", counting_sha256)

    # Nothing changed: the cheap size/mtime signature short-circuit should
    # avoid re-hashing either the staged copy or the production source.
    stage_pairs(
        pairs, stage, [production], CopyWindow.from_strings("04:00", "10:00"),
        now=lambda: datetime(2026, 8, 21, 5, 5),
    )
    assert calls == []

    # A genuinely changed production source must still be re-verified and re-copied.
    _image(source, 99)
    stage_pairs(
        pairs, stage, [production], CopyWindow.from_strings("04:00", "10:00"),
        now=lambda: datetime(2026, 8, 21, 5, 10),
    )
    assert len(calls) > 0


def test_secure_pix2pix_sets_decompression_bomb_ceiling_standalone() -> None:
    """Regression test for import-order independence: importing secure_pix2pix
    alone (no correction_ml import in this fresh process) must still raise
    Pillow's decompression-bomb ceiling to the same finite value."""
    import subprocess
    import sys

    learning_source = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import secure_pix2pix; from PIL import Image; print(Image.MAX_IMAGE_PIXELS)"],
        capture_output=True, text=True, cwd=learning_source, check=True,
    )
    assert result.stdout.strip() == "250000000"


def test_native_psnr_and_ssim() -> None:
    identical = torch.full((2, 3, 32, 32), 0.5)
    assert torch.allclose(psnr_per_image(identical, identical), torch.full((2,), 120.0))
    assert torch.allclose(ssim_per_image(identical, identical), torch.ones(2), atol=1e-5)
    changed = identical.clone()
    changed[:, :, 8:24, 8:24] = 0.0
    assert torch.all(psnr_per_image(changed, identical) < 120.0)
    assert torch.all(ssim_per_image(changed, identical) < 1.0)


def test_atomic_checkpoint_restricted_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    destination = root / "model.pt"
    atomic_torch_save({"format_version": 1, "weights": torch.arange(4)}, destination, root)
    loaded = safe_torch_load(destination, root, "cpu")
    assert loaded["format_version"] == 1
    assert torch.equal(loaded["weights"], torch.arange(4))
    with pytest.raises(PathSecurityError):
        safe_torch_load(destination, tmp_path / "different-root", "cpu")


def test_periodic_checkpoint_retention_never_removes_best_or_last(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in ("best.pt", "last.pt", "epoch_0001.pt", "epoch_0002.pt", "epoch_0003.pt"):
        (root / name).write_bytes(name.encode("ascii"))
    removed = prune_periodic_checkpoints(root, keep=2)
    assert len(removed) == 1
    assert (root / "best.pt").exists()
    assert (root / "last.pt").exists()
    assert len(list(root.glob("epoch_*.pt"))) == 2


def test_evaluate_model_comparison_end_to_end(tmp_path: Path) -> None:
    from correction_ml.model import CorrectionUNetV2
    from correction_ml.parameters import PARAMETER_NAMES

    import evaluate_model_comparison as harness

    artifacts = tmp_path / "artifacts"
    cache_root = artifacts / "training_cache"
    index = _write_cache_index(
        cache_root,
        {
            "pair-a": _cache_record(cache_root, "pair-a", value=40),
            "pair-b": _cache_record(cache_root, "pair-b", value=90),
        },
    )

    v2_model = CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES))
    torch.save(
        {
            "model": v2_model.state_dict(),
            "architecture": "correction_unet_v2",
            "base_channels": 8,
            "parameter_names": PARAMETER_NAMES,
        },
        artifacts / "best_v2.pt",
    )

    pix2pix_checkpoints = artifacts / "pix2pix" / "checkpoints"
    atomic_torch_save(
        {"format_version": 1, "generator": UNetGenerator().state_dict()},
        pix2pix_checkpoints / "best.pt",
        pix2pix_checkpoints,
    )

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"read_only_roots": [str(tmp_path / "production")]}), encoding="utf-8")

    import sys

    old_argv = sys.argv
    sys.argv = [
        "evaluate_model_comparison.py",
        "--artifacts", str(artifacts),
        "--config", str(config),
        "--cache-index", str(index),
        "--benchmark-pairs", "2",
    ]
    try:
        harness.main()
    finally:
        sys.argv = old_argv

    report = json.loads((artifacts / "model_comparison_report.json").read_text(encoding="utf-8"))
    assert report["benchmark_pair_count"] == 2
    assert "v2" in report["models_evaluated"]
    assert "pix2pix" in report["models_evaluated"]
    assert report["per_model"]["v2"]["count"] == 2
    assert report["per_model"]["pix2pix"]["count"] == 2
    assert (artifacts / "model_comparison_review.jpg").is_file()


def test_cli_dry_run_never_changes_inputs(tmp_path: Path) -> None:
    production = tmp_path / "production"
    original_root = production / "original"
    corrected_root = production / "corrected"
    for index in range(2):
        _image(original_root / f"{index}.png", index)
        _image(corrected_root / f"{index}.png", index + 1)
    before = {
        path: (_sha256(path), path.stat().st_mtime_ns)
        for path in production.rglob("*.png")
    }
    exit_code = pix2pix_train.main(
        [
            "--original-root",
            str(original_root),
            "--corrected-root",
            str(corrected_root),
            "--output-dir",
            str(tmp_path / "model-output"),
            "--read-only-root",
            str(production),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert before == {
        path: (_sha256(path), path.stat().st_mtime_ns)
        for path in production.rglob("*.png")
    }
