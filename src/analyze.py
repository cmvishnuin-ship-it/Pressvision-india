from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from correction_ml.metrics import pair_parameters, read_rgb
from correction_ml.pairs import discover_pairs, load_config, write_manifest
from correction_ml.safety import assert_not_production_output, check_access_deadline


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover pairs and measure production corrections.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--limit", type=int, default=0, help="Analyze only N pairs (0 means every pair).")
    parser.add_argument("--refresh-all", action="store_true", help="Recalculate parameters already present in the local cache.")
    parser.add_argument("--stop-at", type=datetime.fromisoformat, help="End of the production access window (YYYY-MM-DDTHH:MM); the ingest stops cleanly before reading past it.")
    return parser.parse_args()


def existing_parameter_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists() or not path.stat().st_size:
        return {}
    rows = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row: dict = dict(raw)
            for key, value in list(row.items()):
                if key in {"set_name", "key", "original", "corrected", "split", "match_method", "source_status", "geometry_class"}:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows[(raw["original"], raw["corrected"])] = row
    return rows


def summarize(rows: list[dict], inventory: list[dict]) -> dict:
    numeric_columns = sorted(
        key for key, value in rows[0].items() if isinstance(value, (int, float))
    ) if rows else []
    statistics = {}
    for name in numeric_columns:
        values = np.asarray([float(row[name]) for row in rows if np.isfinite(float(row[name]))])
        if values.size:
            statistics[name] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "standard_deviation": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return {"inventory": inventory, "analyzed_pair_count": len(rows), "statistics": statistics}


def frozen_pix2pix_validation_groups(artifacts: Path) -> set[str]:
    """Read local cache IDs that must remain exact for comparable shadow metrics."""

    path = artifacts / "pix2pix" / "frozen_validation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("group_keys", [])
        if not isinstance(raw, list):
            return set()
        # Cache IDs are 24-character lowercase hexadecimal hashes.
        return {
            value
            for value in raw
            if isinstance(value, str)
            and len(value) == 24
            and all(character in "0123456789abcdef" for character in value)
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    artifacts = Path(args.artifacts)
    assert_not_production_output(artifacts, config)
    artifacts.mkdir(parents=True, exist_ok=True)
    pairs, inventory = discover_pairs(config, deadline=args.stop_at)
    if bool(config.get("persistent_training_cache", True)):
        from correction_ml.cache import update_training_cache

        maximum_bytes = int(float(config.get("training_cache_max_gb", 300)) * 1024**3)
        protected_validation = frozen_pix2pix_validation_groups(artifacts)
        pairs, cache_report, parameter_aliases = update_training_cache(
            pairs,
            artifacts / "training_cache",
            maximum_bytes,
            hot_days=int(config.get("training_cache_hot_days", 30)),
            hot_max_bytes=int(float(config.get("training_cache_hot_max_gb", 50)) * 1024**3),
            replay_max_bytes=int(float(config.get("training_cache_replay_max_gb", 150)) * 1024**3),
            replay_patch_size=int(config.get("training_cache_replay_patch_size", 192)),
            replay_patches_per_pair=int(config.get("training_cache_replay_patches_per_pair", 4)),
            protected_pair_ids=protected_validation,
            deadline=args.stop_at,
        )
        inventory.append({"set_name": "PERSISTENT-TRAINING-CACHE", **cache_report})
        print(json.dumps({"persistent_training_cache": cache_report}, indent=2))
    write_manifest(pairs, artifacts / "manifest.csv")
    selected = pairs[: args.limit] if args.limit > 0 else pairs
    output_csv = artifacts / "correction_parameters.csv"
    reusable = {} if args.refresh_all else existing_parameter_rows(output_csv)
    parameter_aliases = locals().get("parameter_aliases", {})
    rows = []
    for number, pair in enumerate(selected, start=1):
        old_row = reusable.get((pair.original, pair.corrected))
        if old_row is None and (pair.original, pair.corrected) in parameter_aliases:
            old_row = reusable.get(parameter_aliases[(pair.original, pair.corrected)])
        if old_row is not None:
            old_row.update(asdict(pair))
            rows.append(old_row)
            print(f"[{number}/{len(selected)}] REUSE {pair.set_name}: {Path(pair.original).name}")
            continue
        check_access_deadline(args.stop_at)
        try:
            original = read_rgb(pair.original)
            corrected = read_rgb(pair.corrected)
            row = asdict(pair)
            row["scale_x"] = pair.corrected_width / pair.original_width
            row["scale_y"] = pair.corrected_height / pair.original_height
            row["geometry_class"] = "crop_or_reframe" if pair.aspect_change > config["max_aspect_change_for_training"] else "same_frame"
            row.update(pair_parameters(original, corrected))
            rows.append(row)
            print(f"[{number}/{len(selected)}] {pair.set_name}: {Path(pair.original).name}")
        except Exception as error:
            print(f"SKIP {pair.original}: {error}")

    if rows:
        temporary_csv = output_csv.with_suffix(".tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(output_csv)
    summary = summarize(rows, inventory)
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Matched {len(pairs)} pairs; analyzed {len(rows)}. Reports: {artifacts.resolve()}")


if __name__ == "__main__":
    main()
