from __future__ import annotations

import numpy as np
import torch

from correction_ml.metrics import image_parameters, pair_parameters
from correction_ml.model import CorrectionUNet, CorrectionUNetV2, count_parameters
from correction_ml.parameters import PARAMETER_NAMES, robust_normalizer
from correction_ml.safety import assert_not_production_output
from correction_ml.pairs import normalized_stem


def test_normalized_stem():
    from pathlib import Path

    assert normalized_stem(Path("  My   Photo.JPEG")) == "my photo"


def test_model_shape_and_identity_initialization():
    model = CorrectionUNet(base_channels=8)
    value = torch.rand(2, 3, 64, 96)
    result = model(value)
    assert result.shape == value.shape
    assert torch.allclose(result, value)


def test_metrics_detect_brightness_change():
    dark = np.full((64, 64, 3), 50, dtype=np.uint8)
    bright = np.full((64, 64, 3), 100, dtype=np.uint8)
    parameters = pair_parameters(dark, bright)
    assert parameters["delta_brightness"] > 0
    assert image_parameters(bright)["brightness"] > image_parameters(dark)["brightness"]


def test_v2_predicts_pixels_and_parameters():
    model = CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES))
    value = torch.rand(2, 3, 64, 96)
    pixels, parameters = model.forward_with_parameters(value)
    assert pixels.shape == value.shape
    assert parameters.shape == (2, len(PARAMETER_NAMES))
    assert count_parameters(model) > count_parameters(CorrectionUNet(base_channels=8))


def test_gaussian_ssim_loss_is_near_zero_for_identical_images_and_rises_for_corrupted_ones():
    from correction_ml.dataset import gaussian_ssim_loss

    torch.manual_seed(0)
    reference = torch.rand(2, 3, 64, 64)
    assert float(gaussian_ssim_loss(reference, reference)) < 1e-4

    corrupted = reference.clone()
    corrupted[:, :, 16:48, 16:48] = 0.0
    assert float(gaussian_ssim_loss(corrupted, reference)) > float(gaussian_ssim_loss(reference, reference))


def _build_smoke_training_dataset(tmp_path, cached_pairs=0):
    """Six pairs (five train, one validation); the first ``cached_pairs`` live in the local cache."""
    import csv
    import json

    from PIL import Image

    from correction_ml.pairs import ImagePair, write_manifest
    from correction_ml.parameters import INPUT_FEATURE_NAMES

    production = tmp_path / "production"
    production.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    rng = np.random.default_rng(0)
    pairs = []
    for index in range(6):
        folder = artifacts / "training_cache" / "pairs" / str(index) if index < cached_pairs else production
        folder.mkdir(parents=True, exist_ok=True)
        original = folder / f"{index}_before.png"
        corrected = folder / f"{index}_after.png"
        base = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
        Image.fromarray(base).save(original)
        Image.fromarray(np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)).save(corrected)
        pairs.append(ImagePair(
            set_name="smoke", key=str(index), original=str(original), corrected=str(corrected),
            original_width=96, original_height=96, corrected_width=96, corrected_height=96,
            aspect_change=0.0, time_difference_hours=0.0, split="train" if index < 5 else "validation",
        ))
    write_manifest(pairs, artifacts / "manifest.csv")

    fields = ["original", "corrected"] + PARAMETER_NAMES + [f"original_{name}" for name in INPUT_FEATURE_NAMES]
    with (artifacts / "correction_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {"original": pair.original, "corrected": pair.corrected}
            row.update({name: 0.01 for name in PARAMETER_NAMES})
            row.update({f"original_{name}": 0.5 for name in INPUT_FEATURE_NAMES})
            writer.writerow(row)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"seed": 42, "read_only_roots": [str(production)], "learning_only": True}), encoding="utf-8")
    return artifacts, config_path


def _run_train_main(config_path, artifacts, *extra_args):
    import sys

    import train

    old_argv = sys.argv
    sys.argv = [
        "train.py", "--config", str(config_path), "--artifacts", str(artifacts),
        "--epochs", "1", "--batch-size", "2", "--patch-size", "64", "--patches-per-image", "2",
        "--base-channels", "8", "--patience", "0", *extra_args,
    ]
    try:
        return train.main()
    finally:
        sys.argv = old_argv


def test_gaussian_ssim_resume_resets_baseline_for_pre_change_checkpoint(tmp_path):
    import json

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    _run_train_main(config_path, artifacts)

    checkpoint_data = torch.load(artifacts / "last_v2.pt", map_location="cpu", weights_only=False)
    assert checkpoint_data["losses"] == ["pixel_l1", "edge", "gaussian_ssim", "color_moments", "global_parameters"]

    checkpoint_data["losses"] = ["pixel_l1", "edge", "ssim", "color_moments", "global_parameters"]
    old_checkpoint = artifacts / "pre_change.pt"
    torch.save(checkpoint_data, old_checkpoint)

    _run_train_main(config_path, artifacts, "--resume", str(old_checkpoint))
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert "loss function updated" in session["dataset_status"]
    assert list((artifacts / "checkpoint_archive").glob("*pre_gaussian_ssim*"))


def test_resnet18_backbone_trains_infers_and_resumes_independently_of_scratch(tmp_path):
    from correction_ml.inference import correct_array, load_model

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    _run_train_main(config_path, artifacts)  # scratch baseline, writes last_v2.pt/best_v2.pt
    scratch_checkpoint = torch.load(artifacts / "last_v2.pt", map_location="cpu", weights_only=False)

    _run_train_main(config_path, artifacts, "--backbone", "resnet18")
    checkpoint_data = torch.load(artifacts / "last_v2_pretrained.pt", map_location="cpu", weights_only=False)
    assert checkpoint_data["architecture"] == "correction_unet_v2_pretrained_resnet18"
    # The scratch lineage must be untouched by the resnet18 shadow run.
    assert torch.load(artifacts / "last_v2.pt", map_location="cpu", weights_only=False)["epoch"] == scratch_checkpoint["epoch"]
    assert (artifacts / "training_history_v2_pretrained.csv").exists()

    device = torch.device("cpu")
    model, _ = load_model(artifacts / "last_v2_pretrained.pt", device)
    sample = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    output = correct_array(model, sample, device, tile_size=64, overlap=8)
    assert output.shape == (48, 48, 3)

    _run_train_main(config_path, artifacts, "--backbone", "resnet18", "--resume", str(artifacts / "last_v2_pretrained.pt"))
    resumed = torch.load(artifacts / "last_v2_pretrained.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == checkpoint_data["epoch"] + 1


def test_resume_rejects_mismatched_backbone_architecture(tmp_path):
    import pytest

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    _run_train_main(config_path, artifacts)  # writes a scratch (correction_unet_v2) checkpoint

    with pytest.raises(SystemExit):
        _run_train_main(config_path, artifacts, "--backbone", "resnet18", "--resume", str(artifacts / "last_v2.pt"))


def test_pretrained_backbone_shapes_and_frozen_batchnorm():
    from correction_ml.model_pretrained import CorrectionUNetV2PretrainedBackbone

    model = CorrectionUNetV2PretrainedBackbone(base_channels=8, parameter_count=len(PARAMETER_NAMES), pretrained_weights_path=None)
    value = torch.rand(2, 3, 64, 96)
    pixels, parameters = model.forward_with_parameters(value)
    assert pixels.shape == value.shape
    assert parameters.shape == (2, len(PARAMETER_NAMES))
    assert sum(p.requires_grad for p in model.resnet_features.parameters()) == 0
    assert count_parameters(model) > 0

    running_before = model.resnet_features.stem[1].running_mean.clone()
    model.train(True)
    with torch.no_grad():
        model.forward_with_parameters(torch.rand(4, 3, 64, 96))
    assert torch.equal(running_before, model.resnet_features.stem[1].running_mean)


def test_robust_parameter_normalizer_handles_constant_columns():
    values = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 100.0]], dtype=np.float32)
    center, scale = robust_normalizer(values)
    assert np.all(scale > 0)
    assert center.tolist() == [1.0, 4.0]


def test_production_output_guard():
    import pytest

    config = {"read_only_roots": [r"P:\photo-desk", r"Q:\pictures\NEWS"]}
    with pytest.raises(ValueError):
        assert_not_production_output(r"P:\photo-desk\BUSINESS-Out", config)
    with pytest.raises(ValueError):
        assert_not_production_output(r"Q:\pictures\NEWS\Done\result.jpg", config)


def test_balanced_sampler_equalizes_sets():
    from types import SimpleNamespace

    from train import balanced_sampler

    pairs = [SimpleNamespace(set_name="large") for _ in range(3)] + [SimpleNamespace(set_name="small")]
    sampler = balanced_sampler(pairs, patches_per_image=2)
    weights = list(sampler.weights.numpy())
    assert np.isclose(sum(weights[:6]), sum(weights[6:]))


def test_replay_sampler_favors_current_data_and_caps_epoch():
    from types import SimpleNamespace

    from train import balanced_sampler

    pairs = [SimpleNamespace(set_name="set", source_status="current") for _ in range(4)]
    pairs += [SimpleNamespace(set_name="set", source_status="historical") for _ in range(8)]
    sampler = balanced_sampler(pairs, patches_per_image=2, max_samples=10, current_fraction=0.60)
    weights = list(sampler.weights.numpy())
    assert len(sampler) == 10
    assert abs(sum(weights[:8]) / sum(weights) - 0.60) < 1e-6


def test_pillow_keeps_a_finite_large_image_guard():
    from PIL import Image

    assert Image.MAX_IMAGE_PIXELS == 250_000_000


def test_input_domain_monitoring_report(tmp_path):
    from PIL import Image

    from correction_ml.inference import input_domain_report
    from correction_ml.parameters import INPUT_FEATURE_NAMES

    source = tmp_path / "sample.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(source)
    checkpoint = {
        "input_feature_names": INPUT_FEATURE_NAMES,
        "input_feature_center": [0.0] * len(INPUT_FEATURE_NAMES),
        "input_feature_scale": [1.0] * len(INPUT_FEATURE_NAMES),
    }
    report = input_domain_report(checkpoint, source)
    assert 0 <= report["heuristic_confidence_percent"] <= 100
    assert len(report["most_unusual_features"]) == 3


def test_project_configuration_is_learning_only():
    import json
    from pathlib import Path

    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    assert config["learning_only"] is True


def test_scheduled_deadline_uses_current_local_date():
    from datetime import datetime

    from train import deadline_today

    now = datetime(2026, 8, 21, 5, 0)
    assert deadline_today("09:50", now) == datetime(2026, 8, 21, 9, 50)
    # An absolute cutoff lets learning run past midnight until the next access window.
    assert deadline_today("2026-08-22T03:55", now) == datetime(2026, 8, 22, 3, 55)

    from pix2pix_train import deadline_today as pix2pix_deadline

    assert pix2pix_deadline("09:45", now) == datetime(2026, 8, 21, 9, 45)
    assert pix2pix_deadline("2026-08-22T03:55", now) == datetime(2026, 8, 22, 3, 55)


def test_pix2pix_shadow_uses_local_cache_and_respects_start_buffer(tmp_path):
    import json
    from datetime import datetime

    from learning_service import pix2pix_shadow_arguments

    artifacts = tmp_path / "artifacts"
    cache = artifacts / "training_cache"
    cache.mkdir(parents=True)
    (cache / "cache_index.json").write_text('{"version":1,"pairs":{}}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "read_only_roots": [str(tmp_path / "production")],
                "pix2pix_shadow": {
                    "enabled": True,
                    "stop_at": "09:45",
                    "minimum_remaining_minutes": 10,
                    "epochs_per_session": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = pix2pix_shadow_arguments(config, artifacts, datetime(2026, 8, 22, 3, 55))
    assert arguments is not None
    assert arguments[arguments.index("--cache-index") + 1] == str(cache / "cache_index.json")
    assert arguments[arguments.index("--stop-at") + 1] == "2026-08-22T03:55"
    assert "--skip-unchanged" in arguments
    assert str(tmp_path / "production") in arguments

    from learning_service import shadow_has_time

    settings = {"minimum_remaining_minutes": 10}
    assert shadow_has_time(settings, datetime(2026, 8, 22, 3, 40), datetime(2026, 8, 22, 3, 55))
    assert not shadow_has_time(settings, datetime(2026, 8, 22, 3, 50), datetime(2026, 8, 22, 3, 55))


def test_pretrained_backbone_shadow_reuses_session_manifest_and_respects_start_buffer(tmp_path):
    import json
    from datetime import datetime

    from learning_service import pretrained_backbone_shadow_arguments

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "pretrained_backbone_shadow": {
                    "enabled": True,
                    "stop_at": "09:55",
                    "minimum_remaining_minutes": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    deadline = datetime(2026, 8, 22, 3, 55)
    # Disabled/missing prerequisites: no manifest/parameters written yet.
    assert pretrained_backbone_shadow_arguments(config, artifacts, deadline) is None

    (artifacts / "manifest.csv").write_text("set_name\n", encoding="utf-8")
    (artifacts / "correction_parameters.csv").write_text("original\n", encoding="utf-8")

    # No last_v2_pretrained.pt to resume from, and no bundled weights available
    # in this fresh environment: cannot safely start a first-ever run.
    assert pretrained_backbone_shadow_arguments(config, artifacts, deadline) is None

    # A local weights file already exists (as if bootstrapped previously).
    weights_path = artifacts / "pretrained" / "resnet18_imagenet.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"placeholder")
    arguments = pretrained_backbone_shadow_arguments(config, artifacts, deadline)
    assert arguments is not None
    assert "--backbone" in arguments and arguments[arguments.index("--backbone") + 1] == "resnet18"
    assert arguments[arguments.index("--pretrained-weights") + 1] == str(weights_path)
    assert arguments[arguments.index("--stop-at") + 1] == "2026-08-22T03:55"
    assert arguments[arguments.index("--cache-only") + 1] == str(artifacts / "training_cache")

    (artifacts / "last_v2_pretrained.pt").write_bytes(b"placeholder")
    resumed_arguments = pretrained_backbone_shadow_arguments(config, artifacts, deadline)
    assert resumed_arguments[resumed_arguments.index("--resume") + 1] == str(artifacts / "last_v2_pretrained.pt")


def test_background_learning_window_starts_at_four():
    from datetime import datetime

    from learning_service import can_start_session, inside_access_window

    assert not inside_access_window(datetime(2026, 8, 21, 3, 59))
    assert inside_access_window(datetime(2026, 8, 21, 4, 0))
    assert can_start_session(datetime(2026, 8, 21, 9, 49))
    assert not can_start_session(datetime(2026, 8, 21, 9, 50))
    assert not inside_access_window(datetime(2026, 8, 21, 10, 0))


def test_visual_matching_pairs_renamed_corrected_image(tmp_path):
    from PIL import Image, ImageDraw, ImageEnhance

    from correction_ml.pairs import discover_pairs

    source_root = tmp_path / "IN"
    target_root = tmp_path / "OUT"
    source_root.mkdir()
    target_root.mkdir()
    image = Image.new("RGB", (160, 120), (30, 40, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 75, 90), fill=(210, 80, 30))
    draw.ellipse((85, 20, 145, 80), fill=(20, 190, 120))
    draw.line((5, 110, 155, 5), fill=(250, 250, 250), width=5)
    image.save(source_root / "original_name.png")
    corrected = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.12)).enhance(1.08)
    corrected.save(target_root / "completely_different_name.png")

    config = {
        "image_sets": [{"name": "test", "original": str(source_root), "corrected": str(target_root)}],
        "max_pair_age_days": 3,
        "validation_fraction": 0.15,
        "seed": 42,
        "visual_match_threshold": 0.70,
    }
    pairs, inventory = discover_pairs(config)
    assert len(pairs) == 1
    assert pairs[0].match_method == "visual_content"
    assert pairs[0].visual_similarity >= 0.70
    assert inventory[0]["visual_content_count"] == 1


def test_visual_matching_rejects_unrelated_content(tmp_path):
    from PIL import Image, ImageDraw

    from correction_ml.pairs import discover_pairs

    source_root = tmp_path / "IN"
    target_root = tmp_path / "OUT"
    source_root.mkdir()
    target_root.mkdir()
    first = Image.new("RGB", (160, 120), "black")
    ImageDraw.Draw(first).rectangle((10, 10, 70, 100), fill="white")
    second = Image.new("RGB", (160, 120), "black")
    ImageDraw.Draw(second).ellipse((85, 20, 150, 90), fill="white")
    first.save(source_root / "one.png")
    second.save(target_root / "two.png")
    config = {
        "image_sets": [{"name": "test", "original": str(source_root), "corrected": str(target_root)}],
        "max_pair_age_days": 3,
        "validation_fraction": 0.15,
        "seed": 42,
        "visual_match_threshold": 0.80,
    }
    pairs, _ = discover_pairs(config)
    assert pairs == []


def test_persistent_cache_survives_production_deletion(tmp_path):
    from pathlib import Path

    from PIL import Image

    from correction_ml.cache import update_training_cache
    from correction_ml.pairs import ImagePair

    production = tmp_path / "production"
    production.mkdir()
    original = production / "before.png"
    corrected = production / "after.png"
    Image.fromarray(np.full((64, 64, 3), 80, dtype=np.uint8)).save(original)
    Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8)).save(corrected)
    pair = ImagePair(
        set_name="test",
        key="sample",
        original=str(original),
        corrected=str(corrected),
        original_width=64,
        original_height=64,
        corrected_width=64,
        corrected_height=64,
        aspect_change=0.0,
        time_difference_hours=0.0,
        split="train",
        match_method="visual_content",
        visual_similarity=0.95,
    )
    cache_root = tmp_path / "artifacts" / "training_cache"
    cached, report, _ = update_training_cache([pair], cache_root, 1024**3)
    assert report["added_pair_count"] == 1
    assert Path(cached[0].original).is_file()
    assert Path(cached[0].corrected).is_file()

    original.unlink()
    corrected.unlink()
    replay, report, _ = update_training_cache([], cache_root, 1024**3)
    assert len(replay) == 1
    assert replay[0].source_status == "historical"
    assert report["historical_pair_count"] == 1
    assert Path(replay[0].original).is_file()


def test_old_full_pair_compacts_to_verified_replay_samples(tmp_path):
    from pathlib import Path

    from PIL import Image, ImageDraw

    from correction_ml.cache import update_training_cache
    from correction_ml.pairs import ImagePair

    production = tmp_path / "production"
    production.mkdir()
    original = production / "before.png"
    corrected = production / "after.png"
    before = Image.new("RGB", (320, 240), (40, 60, 80))
    ImageDraw.Draw(before).rectangle((30, 30, 200, 180), fill=(180, 90, 30))
    after = Image.new("RGB", (320, 240), (55, 75, 95))
    ImageDraw.Draw(after).rectangle((30, 30, 200, 180), fill=(210, 110, 45))
    before.save(original)
    after.save(corrected)
    pair = ImagePair(
        set_name="test", key="compact", original=str(original), corrected=str(corrected),
        original_width=320, original_height=240, corrected_width=320, corrected_height=240,
        aspect_change=0.0, time_difference_hours=0.0, split="train",
    )
    cache_root = tmp_path / "cache"
    replay, report, aliases = update_training_cache(
        [pair], cache_root, 1024**3, hot_days=30, hot_max_bytes=0,
        replay_max_bytes=1024**3, replay_patch_size=96, replay_patches_per_pair=4,
    )
    assert report["compacted_record_count"] == 1
    assert report["hot_pair_record_count"] == 0
    assert report["replay_pair_record_count"] == 1
    assert len(replay) == 5  # global view plus four spatial patches
    assert len(aliases) == 5
    assert all(Path(item.original).is_file() and Path(item.corrected).is_file() for item in replay)
    assert not list((cache_root / "pairs").rglob("original.png"))


def test_frozen_benchmark_never_exceeds_requested_count(tmp_path):
    from types import SimpleNamespace

    from correction_ml.monitoring import ensure_frozen_benchmark

    pairs = [
        SimpleNamespace(set_name=f"set-{index}", key="k", original=f"o{index}", corrected=f"c{index}")
        for index in range(5)
    ]
    selected = ensure_frozen_benchmark(pairs, tmp_path, count=3, seed=42)
    assert len(selected) == 3
    # Never replaced automatically: re-calling returns the identical set.
    assert ensure_frozen_benchmark(pairs, tmp_path, count=3, seed=42) == selected


def test_capacity_admission_benefits_from_same_run_compaction(tmp_path):
    import json

    from PIL import Image

    from correction_ml.cache import update_training_cache
    from correction_ml.pairs import ImagePair

    production = tmp_path / "production"
    production.mkdir()
    original_a = production / "a_before.png"
    corrected_a = production / "a_after.png"
    Image.fromarray(np.full((32, 32, 3), 40, dtype=np.uint8)).save(original_a)
    Image.fromarray(np.full((32, 32, 3), 60, dtype=np.uint8)).save(corrected_a)
    pair_a = ImagePair(
        set_name="test", key="a", original=str(original_a), corrected=str(corrected_a),
        original_width=32, original_height=32, corrected_width=32, corrected_height=32,
        aspect_change=0.0, time_difference_hours=0.0, split="train",
    )
    cache_root = tmp_path / "cache"
    update_training_cache(
        [pair_a], cache_root, max_bytes=1024**3, hot_max_bytes=1024**3,
        replay_max_bytes=1024**3, replay_patch_size=16, replay_patches_per_pair=1,
    )

    # Simulate pair A currently counting as a large hot record (e.g. a big
    # production photo) even though its backing files on disk are tiny --
    # this isolates the capacity/compaction interaction from real PNG sizes.
    index_path = cache_root / "cache_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    pair_id = next(iter(index["pairs"]))
    index["pairs"][pair_id]["bytes"] = 50_000_000
    index_path.write_text(json.dumps(index), encoding="utf-8")

    original_b = production / "b_before.png"
    corrected_b = production / "b_after.png"
    Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8)).save(original_b)
    Image.fromarray(np.full((32, 32, 3), 110, dtype=np.uint8)).save(corrected_b)
    bytes_b = original_b.stat().st_size + corrected_b.stat().st_size
    pair_b = ImagePair(
        set_name="test", key="b", original=str(original_b), corrected=str(corrected_b),
        original_width=32, original_height=32, corrected_width=32, corrected_height=32,
        aspect_change=0.0, time_difference_hours=0.0, split="train",
    )

    # Only room for the (inflated) A plus B if A gets compacted down first;
    # hot_max_bytes=0 forces every hot record (A here, then B once admitted)
    # to compact immediately, so both end up compacted -- the key assertions
    # are that B was still admitted and capacity never wrongly rejected it.
    _, report, _ = update_training_cache(
        [pair_b], cache_root, max_bytes=50_000_000 + bytes_b - 1, hot_days=30, hot_max_bytes=0,
        replay_max_bytes=1024**3, replay_patch_size=16, replay_patches_per_pair=1,
    )
    assert report["compacted_record_count"] == 2
    assert report["skipped_capacity_count"] == 0
    assert report["added_pair_count"] == 1


def test_frozen_pix2pix_validation_pair_is_not_compacted(tmp_path):
    import json

    from PIL import Image

    from analyze import frozen_pix2pix_validation_groups
    from correction_ml.cache import update_training_cache
    from correction_ml.pairs import ImagePair

    production = tmp_path / "production"
    production.mkdir()
    original = production / "same.png"
    corrected = production / "same-corrected.png"
    Image.fromarray(np.full((96, 96, 3), 60, dtype=np.uint8)).save(original)
    Image.fromarray(np.full((96, 96, 3), 80, dtype=np.uint8)).save(corrected)
    pair = ImagePair(
        set_name="test", key="protected", original=str(original), corrected=str(corrected),
        original_width=96, original_height=96, corrected_width=96, corrected_height=96,
        aspect_change=0.0, time_difference_hours=0.0, split="validation",
    )
    artifacts = tmp_path / "artifacts"
    cache_root = artifacts / "training_cache"
    update_training_cache([pair], cache_root, 1024**3)
    index = json.loads((cache_root / "cache_index.json").read_text(encoding="utf-8"))
    pair_id = next(iter(index["pairs"]))
    frozen = artifacts / "pix2pix" / "frozen_validation.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"group_keys": [pair_id]}), encoding="utf-8")

    protected = frozen_pix2pix_validation_groups(artifacts)
    replay, report, _ = update_training_cache(
        [], cache_root, 1024**3, hot_days=0, hot_max_bytes=0,
        replay_max_bytes=1024**3, protected_pair_ids=protected,
    )
    assert report["protected_validation_record_count"] == 1
    assert report["compacted_record_count"] == 0
    assert report["hot_pair_record_count"] == 1
    assert len(replay) == 1


def test_checkpoint_archive_retention_is_bounded(tmp_path):
    import os
    from datetime import datetime, timedelta

    from learning_service import prune_checkpoint_archive

    archive = tmp_path / "checkpoint_archive"
    archive.mkdir()
    now = datetime(2026, 8, 21)
    for number in range(20):
        path = archive / f"checkpoint_{number}.pt"
        path.write_bytes(bytes([number]))
        timestamp = (now - timedelta(days=number * 8)).timestamp()
        os.utime(path, (timestamp, timestamp))
    report = prune_checkpoint_archive(tmp_path)
    # Seven recent files plus no more than one older file per represented month.
    assert report["retained"] < 20
    assert report["removed"] > 0
    assert len(list(archive.glob("*.pt"))) == report["retained"]


def _patch_pairs(tmp_path, count=3, size=96):
    from PIL import Image

    from correction_ml.pairs import ImagePair

    rng = np.random.default_rng(1)
    pairs = []
    for index in range(count):
        original = tmp_path / f"{index}_before.png"
        corrected = tmp_path / f"{index}_after.png"
        base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(base).save(original)
        Image.fromarray(np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)).save(corrected)
        pairs.append(ImagePair(
            set_name="set", key=str(index), original=str(original), corrected=str(corrected),
            original_width=size, original_height=size, corrected_width=size, corrected_height=size,
            aspect_change=0.0, time_difference_hours=0.0, split="validation",
        ))
    return pairs


def test_validation_crops_are_fixed_across_epochs_and_instances(tmp_path):
    from correction_ml.dataset import PairedPatchDataset

    pairs = _patch_pairs(tmp_path)
    first = PairedPatchDataset(pairs, 32, 2, False, seed=42)
    second = PairedPatchDataset(pairs, 32, 2, False, seed=42)
    for index in range(len(first)):
        assert torch.equal(first[index][0], first[index][0])
        assert torch.equal(first[index][0], second[index][0])
    # Different patch slots of one image are different crops.
    assert not torch.equal(first[0][0], first[1][0])


def test_materialized_validation_is_exact(tmp_path):
    from correction_ml.dataset import MaterializedDataset, PairedPatchDataset

    pairs = _patch_pairs(tmp_path)
    targets = {(pair.original, pair.corrected): np.full(len(PARAMETER_NAMES), 0.5, dtype=np.float32) for pair in pairs}
    center, scale = np.zeros(len(PARAMETER_NAMES), np.float32), np.ones(len(PARAMETER_NAMES), np.float32)
    dataset = PairedPatchDataset(pairs, 32, 1, False, targets, center, scale, context_size=16, seed=42)
    cached = MaterializedDataset(dataset)
    assert len(cached) == len(dataset)
    for index in range(len(dataset)):
        live, stored = dataset[index], cached[index]
        assert len(live) == len(stored) == 5  # patches, parameters, original and editors' whole-image views
        for value, copy in zip(live, stored):
            assert torch.equal(value, copy)


def test_repeated_sampler_cuts_several_patches_from_each_decode(tmp_path, monkeypatch):
    import correction_ml.dataset as dataset_module
    from correction_ml.dataset import PairedPatchDataset
    from train import balanced_sampler

    pairs = _patch_pairs(tmp_path)
    reads = []
    original_read = dataset_module.read_rgb
    monkeypatch.setattr(dataset_module, "read_rgb", lambda path, *a, **k: reads.append(path) or original_read(path, *a, **k))
    dataset = PairedPatchDataset(pairs, 32, 4, True)
    sampler = balanced_sampler(pairs, 4, max_samples=12, repeats=4)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 12
    assert all(indices[start] == indices[start + offset] for start in range(0, 12, 4) for offset in range(4))
    for index in indices:
        dataset[index]
    assert len(reads) <= 2 * 3  # one original+corrected decode per group of four patches


def test_set_balance_exponent_tempers_tiny_image_sets():
    from types import SimpleNamespace

    from train import balanced_sampler

    pairs = [SimpleNamespace(set_name="large") for _ in range(100)] + [SimpleNamespace(set_name="tiny")]

    def tiny_share(exponent):
        weights = balanced_sampler(pairs, 1, set_exponent=exponent).weights.numpy()
        return weights[-1] / weights.sum()

    assert np.isclose(tiny_share(0.0), 0.5)
    assert np.isclose(tiny_share(0.5), 1.0 / 11.0)
    assert np.isclose(tiny_share(1.0), 1.0 / 101.0)


def test_global_model_reproduces_v2_at_warm_start_and_then_uses_context():
    from correction_ml.model_global import CorrectionUNetV3Global

    torch.manual_seed(0)
    v2 = CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES)).eval()
    torch.nn.init.normal_(v2.residual.weight, std=0.1)  # a non-identity "trained" V2
    v3 = CorrectionUNetV3Global(base_channels=8, parameter_count=len(PARAMETER_NAMES), context_size=32).eval()
    v3.load_v2_weights(v2.state_dict())
    value = torch.rand(2, 3, 64, 64)
    dark, bright = torch.full((2, 3, 32, 32), 0.1), torch.full((2, 3, 32, 32), 0.9)
    with torch.no_grad():
        expected_pixels, expected_parameters = v2.forward_with_parameters(value)
        for context in (dark, bright, None):
            pixels, parameters = v3.forward_with_parameters(value, context)
            assert torch.allclose(pixels, expected_pixels, atol=1e-5)
            assert torch.allclose(parameters, expected_parameters, atol=1e-5)
        torch.nn.init.normal_(v3.curve_head.weight, std=0.5)
        assert not torch.allclose(v3(value, dark), v3(value, bright))


def test_global_only_training_leaves_the_v2_layers_untouched():
    from correction_ml.model_global import NEW_LAYER_PREFIXES, CorrectionUNetV3Global

    torch.manual_seed(0)
    v2 = CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES))
    model = CorrectionUNetV3Global(base_channels=8, parameter_count=len(PARAMETER_NAMES), context_size=32)
    model.load_v2_weights(v2.state_dict())
    model.freeze_v2_layers()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith(NEW_LAYER_PREFIXES) for name in trainable)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    value, context = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 32, 32)
    for _ in range(3):
        optimizer.zero_grad()
        model(value, context).mean().backward()
        optimizer.step()
    for name, tensor in v2.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name


def test_global_tone_curve_is_identity_at_init_and_monotonic():
    from correction_ml.model_global import CorrectionUNetV3Global

    model = CorrectionUNetV3Global(base_channels=8, parameter_count=len(PARAMETER_NAMES), context_size=32)
    ramp = torch.linspace(0, 1, 101).view(1, 1, 1, 101).expand(1, 3, 1, 101).contiguous()
    vector = torch.randn(1, 128)
    with torch.no_grad():
        assert torch.allclose(model.global_tone(ramp, vector), ramp, atol=1e-6)
        torch.nn.init.normal_(model.curve_head.weight, std=1.0)
        curved = model.global_tone(ramp, vector)
    assert torch.all(curved[..., 1:] >= curved[..., :-1])


def test_replay_patch_context_uses_sibling_global_view(tmp_path):
    from PIL import Image

    from correction_ml.dataset import PairedPatchDataset, replay_whole_image_paths
    from correction_ml.pairs import ImagePair

    replay = tmp_path / "replay" / "abc"
    replay.mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(replay / "patch_1_original.png")
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(replay / "patch_1_corrected.png")
    Image.fromarray(np.full((96, 128, 3), 255, np.uint8)).save(replay / "global_original.png")
    Image.fromarray(np.full((96, 128, 3), 51, np.uint8)).save(replay / "global_corrected.png")
    pair = ImagePair(
        set_name="set", key="k#replay-patch_1", original=str(replay / "patch_1_original.png"),
        corrected=str(replay / "patch_1_corrected.png"), original_width=64, original_height=64,
        corrected_width=64, corrected_height=64, aspect_change=0.0, time_difference_hours=0.0, split="train",
    )
    assert replay_whole_image_paths(pair) == (replay / "global_original.png", replay / "global_corrected.png")
    targets = {(pair.original, pair.corrected): np.zeros(len(PARAMETER_NAMES), np.float32)}
    center, scale = np.zeros(len(PARAMETER_NAMES), np.float32), np.ones(len(PARAMETER_NAMES), np.float32)
    source, _, _, context, target_view = PairedPatchDataset([pair], 32, 1, False, targets, center, scale, context_size=16)[0]
    assert float(source.mean()) == 0.0  # the patch itself is black...
    assert float(context.mean()) == 1.0  # ...but its whole-image view is the white global image
    assert abs(float(target_view.mean()) - 0.2) < 1e-6  # and the editors' whole image is the global corrected one


def test_frozen_benchmark_pairs_are_excluded_from_training(tmp_path):
    import json

    from correction_ml.monitoring import pair_identifier
    from correction_ml.pairs import read_manifest

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    first_training_pair = read_manifest(artifacts / "manifest.csv")[0]
    (artifacts / "frozen_benchmark.json").write_text(json.dumps({"pair_ids": [pair_identifier(first_training_pair)]}), encoding="utf-8")
    _run_train_main(config_path, artifacts)
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert session["frozen_benchmark_pairs_excluded"] == 1
    assert session["training_pairs"] == 4

    _run_train_main(config_path, artifacts, "--no-exclude-frozen-benchmark")
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert session["frozen_benchmark_pairs_excluded"] == 0
    assert session["training_pairs"] == 5


def test_resumed_best_checkpoint_is_kept_unless_training_beats_its_starting_weights(tmp_path):
    import csv
    import json

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    _run_train_main(config_path, artifacts)
    assert torch.load(artifacts / "best_v2.pt", map_location="cpu", weights_only=False)["epoch"] == 1

    # New analysis results change the dataset fingerprint, as a new morning does.
    parameter_file = artifacts / "correction_parameters.csv"
    rows = list(csv.DictReader(parameter_file.open(newline="", encoding="utf-8")))
    rows[0][PARAMETER_NAMES[0]] = "0.02"
    with parameter_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A learning rate of zero leaves the weights unchanged: nothing may replace best.
    _run_train_main(config_path, artifacts, "--resume", str(artifacts / "best_v2.pt"), "--learning-rate", "0")
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert "changed" in session["dataset_status"]
    assert session["best_loss_to_beat"] == session["starting_validation_loss"]
    assert session["unchanged_validation_pixel_loss"] > 0
    assert torch.load(artifacts / "best_v2.pt", map_location="cpu", weights_only=False)["epoch"] == 1
    assert torch.load(artifacts / "last_v2.pt", map_location="cpu", weights_only=False)["epoch"] == 2


def test_global_backbone_warm_starts_from_v2_trains_infers_and_resumes(tmp_path):
    import csv
    import json

    from correction_ml.inference import correct_array, load_model

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    _run_train_main(config_path, artifacts)
    with (artifacts / "training_history_v2.csv").open(newline="", encoding="utf-8") as handle:
        v2_validation_pixel = float(list(csv.DictReader(handle))[-1]["validation_pixel_loss"])

    _run_train_main(config_path, artifacts, "--backbone", "global", "--context-size", "32", "--init-from", str(artifacts / "last_v2.pt"))
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert session["architecture"] == "correction_unet_v3_global"
    assert session["initialized_from"].startswith("last_v2.pt")
    # Same fixed validation crops, and the warm-started model is V2 exactly (its
    # total also carries the whole-image term, which V2 does not have).
    assert abs(session["starting_validation_pixel_loss"] - v2_validation_pixel) < 1e-5
    assert "whole_image_global" in session["losses"]
    checkpoint_data = torch.load(artifacts / "last_v3_global.pt", map_location="cpu", weights_only=False)
    assert checkpoint_data["architecture"] == "correction_unet_v3_global"
    assert checkpoint_data["context_size"] == 32
    assert (artifacts / "training_history_v3_global.csv").exists()
    assert not list(artifacts.glob("*.partial"))  # checkpoints are written then renamed

    device = torch.device("cpu")
    model, _ = load_model(artifacts / "last_v3_global.pt", device)
    sample = np.random.randint(0, 255, (80, 72, 3), dtype=np.uint8)
    assert correct_array(model, sample, device, tile_size=64, overlap=8).shape == (80, 72, 3)

    _run_train_main(config_path, artifacts, "--backbone", "global", "--context-size", "32", "--resume", str(artifacts / "last_v3_global.pt"))
    resumed = torch.load(artifacts / "last_v3_global.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == checkpoint_data["epoch"] + 1

    import pytest

    with pytest.raises(SystemExit):
        _run_train_main(config_path, artifacts, "--backbone", "global", "--init-from", str(artifacts / "last_v3_global.pt"))


def test_global_context_shadow_warm_starts_from_v2_then_resumes_its_own_best(tmp_path):
    import json
    from datetime import datetime

    from learning_service import global_context_shadow_arguments

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"global_context_shadow": {"enabled": True}}), encoding="utf-8")
    deadline = datetime(2026, 8, 22, 3, 55)

    assert global_context_shadow_arguments(config, artifacts, deadline) is None  # no local manifest yet
    (artifacts / "manifest.csv").write_text("set_name\n", encoding="utf-8")
    (artifacts / "correction_parameters.csv").write_text("original\n", encoding="utf-8")
    assert global_context_shadow_arguments(config, artifacts, deadline) is None  # nothing to warm-start from

    (artifacts / "best_v2.pt").write_bytes(b"placeholder")
    arguments = global_context_shadow_arguments(config, artifacts, deadline)
    assert arguments[arguments.index("--backbone") + 1] == "global"
    assert arguments[arguments.index("--init-from") + 1] == str(artifacts / "best_v2.pt")
    assert arguments[arguments.index("--stop-at") + 1] == "2026-08-22T03:55"
    assert arguments[arguments.index("--cache-only") + 1] == str(artifacts / "training_cache")
    assert arguments[arguments.index("--global-loss-weight") + 1] == "1.0"
    assert "--resume" not in arguments

    (artifacts / "best_v3_global.pt").write_bytes(b"placeholder")
    arguments = global_context_shadow_arguments(config, artifacts, deadline)
    assert arguments[arguments.index("--resume") + 1] == str(artifacts / "best_v3_global.pt")
    assert "--init-from" not in arguments

    config.write_text(json.dumps({"global_context_shadow": {"enabled": False}}), encoding="utf-8")
    assert global_context_shadow_arguments(config, artifacts, deadline) is None


def test_controller_reads_production_only_in_its_window_and_learns_the_rest_of_the_day():
    from datetime import datetime

    from learning_service import next_action

    data = "manifest.csv:1:1|correction_parameters.csv:1:1"
    day = datetime(2026, 9, 28)

    def at(hour, minute=0):
        return day.replace(hour=hour, minute=minute)

    fresh = {}
    done_today = {"ingest_completed_date": "2026-09-28"}

    # Before the window: learn from the cache and hand over at 03:55.
    action = next_action(at(1), fresh, False, data)
    assert action.kind == "learn" and action.until == at(3, 55)
    assert next_action(at(3, 45), fresh, False, data).kind == "wait"  # too little time for a useful run
    # Window open: stop learning first, then read production.
    assert next_action(at(4), fresh, True, data).kind == "stop_learning"
    assert next_action(at(4), fresh, False, data).kind == "ingest"
    # After the ingest, learning runs until 03:55 tomorrow.
    action = next_action(at(4, 20), done_today, False, data)
    assert action.kind == "learn" and action.until == datetime(2026, 9, 29, 3, 55)
    # A failed ingest is retried after its delay; learning waits meanwhile.
    failed_once = {"ingest_attempt_date": "2026-09-28", "ingest_attempts": 1, "ingest_retry_after": "2026-09-28T04:30:00"}
    assert next_action(at(4, 20), failed_once, False, data).kind == "wait"
    assert next_action(at(4, 31), failed_once, False, data).kind == "ingest"
    # After three failures, or past the last start time, learning uses the cached data.
    failed_three = {"ingest_attempt_date": "2026-09-28", "ingest_attempts": 3}
    assert next_action(at(5), failed_three, False, data).kind == "learn"
    assert next_action(at(9, 55), fresh, False, data).kind == "learn"
    assert next_action(at(23), fresh, False, data).until == datetime(2026, 9, 29, 3, 55)
    # Converged on this data: idle until an ingest changes it.
    learned = {**done_today, "learned_data": data}
    assert next_action(at(10, 30), learned, False, data).kind == "wait"
    assert next_action(at(10, 30), learned, False, "changed").kind == "learn"
    # A failed run backs off, a running one is left alone, and no dataset means nothing to learn.
    assert next_action(at(11), {**done_today, "learning_retry_after": "2026-09-28T11:30:00"}, False, data).kind == "wait"
    assert next_action(at(11), done_today, True, data).kind == "wait"
    assert next_action(at(11), done_today, False, None).kind == "wait"


def test_access_window_and_learning_settings_come_from_config():
    import json
    from datetime import time
    from pathlib import Path

    import pytest

    from learning_service import DEFAULT_WINDOW, PRIORITY_CLASSES, access_window_from_config, learning_priority_flag, learning_thread_count

    assert access_window_from_config({}) == DEFAULT_WINDOW
    window = access_window_from_config({"production_access": {"start": "19:00", "last_start": "19:40", "end": "19:50"}})
    assert (window.start, window.last_start, window.end) == (time(19), time(19, 40), time(19, 50))
    with pytest.raises(ValueError):
        access_window_from_config({"production_access": {"start": "10:00", "end": "04:00"}})
    assert learning_thread_count({}, cpu_count=12) == 6
    assert learning_thread_count({"max_threads": 3}, cpu_count=12) == 3
    assert learning_priority_flag({}) == PRIORITY_CLASSES["below_normal"] != 0
    with pytest.raises(ValueError):
        learning_priority_flag({"cpu_priority": "realtime"})

    shipped = json.loads(Path("config.server.json").read_text(encoding="utf-8"))
    assert access_window_from_config(shipped) == DEFAULT_WINDOW
    assert learning_priority_flag(shipped["learning"]) == PRIORITY_CLASSES["below_normal"]


def test_dataset_signature_changes_when_an_ingest_rewrites_the_dataset(tmp_path):
    from learning_service import dataset_signature

    assert dataset_signature(tmp_path) is None
    (tmp_path / "manifest.csv").write_text("a\n", encoding="utf-8")
    (tmp_path / "correction_parameters.csv").write_text("b\n", encoding="utf-8")
    first = dataset_signature(tmp_path)
    assert first
    (tmp_path / "manifest.csv").write_text("a\nnew pair\n", encoding="utf-8")
    assert dataset_signature(tmp_path) != first


def test_cache_only_training_never_opens_a_production_path(tmp_path, monkeypatch):
    import json

    import pytest

    import correction_ml.dataset as dataset_module
    import train

    artifacts, config_path = _build_smoke_training_dataset(tmp_path, cached_pairs=4)
    cache_root = artifacts / "training_cache"
    opened = []
    original_read = dataset_module.read_rgb
    monkeypatch.setattr(dataset_module, "read_rgb", lambda path, *a, **k: opened.append(str(path)) or original_read(path, *a, **k))
    _run_train_main(config_path, artifacts, "--cache-only", str(cache_root))
    session = json.loads((artifacts / "latest_training_session.json").read_text())
    assert session["cache_only"] is True
    assert session["production_pairs_excluded"] == 2
    assert session["training_pairs"] == 4
    assert opened and all(path.startswith(str(cache_root)) for path in opened)

    # Without a local manifest, cache-only training stops instead of scanning production.
    (artifacts / "manifest.csv").unlink()
    monkeypatch.setattr(train, "discover_pairs", lambda config: pytest.fail("production discovery must not run"))
    with pytest.raises(SystemExit):
        _run_train_main(config_path, artifacts, "--cache-only", str(cache_root))


def test_training_reports_whether_the_deadline_or_early_stopping_ended_it(tmp_path):
    from datetime import datetime, timedelta

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    soon = (datetime.now() + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M")
    assert _run_train_main(config_path, artifacts, "--stop-at", soon) == "deadline"
    assert not (artifacts / "last_v2.pt").exists()
    # Weights that cannot change stop improving after the first epoch.
    assert _run_train_main(config_path, artifacts, "--epochs", "5", "--patience", "1", "--learning-rate", "0") == "early_stopped"


def test_manifest_is_written_then_renamed(tmp_path, monkeypatch):
    import correction_ml.pairs as pairs_module
    from correction_ml.pairs import read_manifest, write_manifest

    pairs = _patch_pairs(tmp_path)
    replaced = []
    original_replace = pairs_module.os.replace
    monkeypatch.setattr(pairs_module.os, "replace", lambda source, target: replaced.append((str(source), str(target))) or original_replace(source, target))
    target = tmp_path / "manifest.csv"
    write_manifest(pairs, target)
    assert replaced == [(str(tmp_path / "manifest.csv.partial"), str(target))]
    assert read_manifest(target) == pairs
    assert not list(tmp_path.glob("*.partial"))


def test_whole_images_of_any_size_can_be_corrected():
    from correction_ml.inference import correct_array
    from correction_ml.model_global import CorrectionUNetV3Global

    device = torch.device("cpu")
    image = np.random.default_rng(0).integers(0, 255, (90, 91, 3), dtype=np.uint8)
    for model in (CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES)).eval(),
                  CorrectionUNetV3Global(base_channels=8, parameter_count=len(PARAMETER_NAMES), context_size=32).eval()):
        assert correct_array(model, image, device, tile_size=64, overlap=8).shape == (90, 91, 3)
        assert correct_array(model, image, device, tile_size=512).shape == (90, 91, 3)


def test_global_model_learns_an_image_wide_curve_from_whole_image_views():
    from correction_ml.model_global import CorrectionUNetV3Global

    torch.manual_seed(0)
    model = CorrectionUNetV3Global(base_channels=8, parameter_count=len(PARAMETER_NAMES), context_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    views = torch.rand(16, 3, 32, 32)
    edited = views**0.7  # an editor's brightening tone curve
    with torch.no_grad():
        initial = float(torch.nn.functional.l1_loss(model.whole_image_correction(views), edited))
    for _ in range(150):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(model.whole_image_correction(views), edited)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.25 * initial


def test_whole_image_report_scores_models_and_the_learned_pattern(tmp_path):
    import csv
    import json

    import evaluate_whole_images

    artifacts, config_path = _build_smoke_training_dataset(tmp_path, cached_pairs=6)
    _run_train_main(config_path, artifacts, "--patch-size", "64")
    report = evaluate_whole_images.main(["--artifacts", str(artifacts), "--max-side", "96", "--tile-size", "64"])
    assert report["images"] == 1  # the smoke dataset's validation pair
    for group in ("ALL", "smoke"):
        models = report["publications"][group]["models"]
        assert {"unchanged", "v2", "ceiling_curves", "ceiling_colour_map"} <= set(models)
        assert models["unchanged"]["reproduced"] == 0.0
        assert models["ceiling_colour_map"]["mae"] < models["unchanged"]["mae"]
        assert set(report["publications"][group]["editors_change"]) == set(models["v2"]["effects"])
    # The editors brightened every pixel by 10 levels.
    assert report["publications"]["ALL"]["editors_change"]["brightness"] > 0.03
    saved = json.loads((artifacts / "whole_image_report.json").read_text(encoding="utf-8"))
    assert saved["images"] == 1
    with (artifacts / "whole_image_history.csv").open(newline="", encoding="utf-8") as handle:
        assert {row["model"] for row in csv.DictReader(handle)} >= {"unchanged", "v2"}


def test_watchdog_restarts_only_when_a_session_overruns(tmp_path, monkeypatch):
    import time
    from datetime import datetime, timedelta

    import learning_service
    from learning_service import Watchdog

    monkeypatch.setattr(learning_service, "LOG_DIR", tmp_path)  # keep the real service log clean
    fired = []
    with Watchdog(datetime.now() + timedelta(seconds=0.3), "Test session", on_expire=lambda: fired.append(True)):
        time.sleep(1.0)
    assert fired == [True]

    fired.clear()
    with Watchdog(datetime.now() + timedelta(seconds=2), "Test session", on_expire=lambda: fired.append(True)):
        time.sleep(0.1)  # finishes in time
    time.sleep(2.5)
    assert fired == []


def test_fresh_install_without_a_checkpoint_trains_from_scratch(tmp_path, monkeypatch):
    import learning_service

    monkeypatch.setattr(learning_service, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(learning_service, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(learning_service, "resource_dir", lambda: tmp_path / "bundle")
    assert learning_service.bootstrap_checkpoint() is None
    # The self-test still checks every model runtime, on untrained V2 weights.
    assert learning_service.self_test() == 0
    log = next((tmp_path / "logs").glob("service_*.log")).read_text(encoding="utf-8")
    assert "Self-test passed: untrained V2" in log

    bundled = tmp_path / "bundle" / "artifacts" / "best_v2.pt"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"placeholder")
    assert learning_service.bootstrap_checkpoint() == tmp_path / "artifacts" / "best_v2.pt"
    assert (tmp_path / "artifacts" / "best_v2.pt").read_bytes() == b"placeholder"


def test_only_one_service_process_can_run():
    import ctypes
    import threading

    from learning_service import acquire_single_instance

    name = "ImageCorrectionLearningServiceTest"
    first = acquire_single_instance(name=name)
    assert first
    assert acquire_single_instance(name=name) is False
    # A restarted copy waits until the previous one has closed.
    threading.Timer(1.0, lambda: ctypes.windll.kernel32.CloseHandle(first)).start()
    second = acquire_single_instance(wait_seconds=10, name=name)
    assert second
    ctypes.windll.kernel32.CloseHandle(second)


def test_learning_lowers_the_process_priority_and_restores_it():
    import ctypes
    from ctypes import wintypes

    from learning_service import PRIORITY_CLASSES, set_process_priority

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    try:
        set_process_priority(PRIORITY_CLASSES["below_normal"])
        assert kernel32.GetPriorityClass(kernel32.GetCurrentProcess()) == PRIORITY_CLASSES["below_normal"]
    finally:
        set_process_priority(PRIORITY_CLASSES["normal"])
    assert kernel32.GetPriorityClass(kernel32.GetCurrentProcess()) == PRIORITY_CLASSES["normal"]


def test_ingest_stops_cleanly_when_the_access_window_closes(tmp_path):
    import json
    from datetime import datetime, timedelta

    import pytest
    from PIL import Image

    from correction_ml.cache import update_training_cache
    from correction_ml.pairs import ImagePair, discover_pairs
    from correction_ml.safety import AccessWindowClosed

    source_root, target_root = tmp_path / "IN", tmp_path / "OUT"
    source_root.mkdir()
    target_root.mkdir()
    Image.fromarray(np.full((64, 64, 3), 80, np.uint8)).save(source_root / "a.png")
    Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(target_root / "a.png")
    config = {"image_sets": [{"name": "test", "original": str(source_root), "corrected": str(target_root)}]}
    closed = datetime.now() - timedelta(minutes=1)
    with pytest.raises(AccessWindowClosed):
        discover_pairs(config, deadline=closed)

    pair = ImagePair(
        set_name="test", key="a", original=str(source_root / "a.png"), corrected=str(target_root / "a.png"),
        original_width=64, original_height=64, corrected_width=64, corrected_height=64,
        aspect_change=0.0, time_difference_hours=0.0, split="train",
    )
    cache_root = tmp_path / "cache"
    update_training_cache([], cache_root, 1024**3)
    index_before = json.loads((cache_root / "cache_index.json").read_text(encoding="utf-8"))["pairs"]
    with pytest.raises(AccessWindowClosed):
        update_training_cache([pair], cache_root, 1024**3, deadline=closed)
    assert json.loads((cache_root / "cache_index.json").read_text(encoding="utf-8"))["pairs"] == index_before


def test_training_stops_within_one_batch_at_the_hand_over_time():
    from datetime import datetime, timedelta

    import pytest

    from train import EpochInterrupted, run_epoch

    model = CorrectionUNetV2(base_channels=8, parameter_count=len(PARAMETER_NAMES))
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), torch.zeros(2, len(PARAMETER_NAMES)))] * 3
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(EpochInterrupted):
        run_epoch(model, batches, torch.device("cpu"), optimizer, deadline=datetime.now() - timedelta(seconds=1))
    # Validation (no optimizer) is not interrupted, and without a deadline training runs.
    assert len(run_epoch(model, batches, torch.device("cpu"), deadline=datetime.now() - timedelta(seconds=1))) == 4
    assert len(run_epoch(model, batches, torch.device("cpu"), optimizer)) == 4


def test_training_publishes_live_progress(tmp_path, monkeypatch):
    import train

    artifacts, config_path = _build_smoke_training_dataset(tmp_path)
    events = []
    monkeypatch.setattr(train, "progress_hook", events.append)
    _run_train_main(config_path, artifacts, "--epochs", "2")
    phases = [event["phase"] for event in events]
    assert phases[:2] == ["preparing", "baseline"]
    assert "training" in phases and phases.count("epoch_done") == 2
    assert all(event["model"] == "v2" for event in events)
    assert any(event["phase"] == "training" and event.get("batch") == event.get("batches") for event in events)


def test_service_writes_live_status_for_the_dashboard(tmp_path, monkeypatch):
    import json
    import os

    import learning_service
    from learning_service import describe_learning_progress, write_live_status

    monkeypatch.setattr(learning_service, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(learning_service, "LIVE_STATUS_FILE", tmp_path / "live_status.json")
    write_live_status("learning", describe_learning_progress({"model": "v3_global", "phase": "training", "epoch": 3, "batch": 10, "batches": 250}), until="2026-09-28T03:55")
    status = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert status["pid"] == os.getpid() and status["activity"] == "learning"
    assert status["detail"] == "V3 whole-picture: epoch 3, batch 10 of 250"
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_tool_marks_the_service_stopped_for_the_dashboard(tmp_path):
    import json

    from stop_service import mark_stopped

    mark_stopped(tmp_path / "missing")  # no artifacts folder: nothing to write
    mark_stopped(tmp_path)
    status = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert status["activity"] == "stopped" and status["pid"] is None


def test_stop_tool_handles_processes_that_are_not_running():
    from stop_service import running_process_ids, stop_processes

    assert running_process_ids("not-an-image-correction-process.exe") == []
    assert stop_processes("not-an-image-correction-process.exe") == (0, [])
