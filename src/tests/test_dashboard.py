from __future__ import annotations

import json
from datetime import datetime, timedelta

import native_dashboard
from native_dashboard import load_snapshot, service_status

NOW = datetime(2026, 9, 28, 2, 0)


def live(activity: str, pid: int | None = 42, minutes_ago: float = 1, **extra) -> dict:
    return {"pid": pid, "activity": activity, "detail": f"{activity} detail", "updated_at": (NOW - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds"), **extra}


def test_service_status_covers_every_state():
    assert service_status(live("learning"), NOW, [42])["label"] == "Learning"
    assert service_status(live("ingest"), NOW, [42])["label"] == "Reading new images"
    assert service_status(live("idle"), NOW, [42])["label"] == "Idle"
    stale = service_status(live("learning", minutes_ago=45), NOW, [42])
    assert stale["state"] == "learning" and "no update for 45 min" in stale["detail"]
    # A process is running but the report is from an earlier process (or not written yet).
    assert service_status(live("idle", pid=7), NOW, [42])["state"] == "running"
    assert service_status({}, NOW, [42])["state"] == "running"
    assert service_status(live("stopped", pid=None), NOW, [])["label"] == "Stopped"
    assert service_status(live("error"), NOW, [])["state"] == "error"
    # The report says learning, but no process is alive: it was killed or crashed.
    assert service_status(live("learning"), NOW, [])["state"] == "not_running"
    assert service_status(live("learning"), NOW, [42, 43])["processes"] == 2


def _project(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "training_cache").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (artifacts / "live_status.json").write_text(json.dumps(live("learning", model="v3_global", batch=125, batches=250, until="2026-09-28T03:55")), encoding="utf-8")
    (artifacts / "service_state.json").write_text(json.dumps({"ingest_completed_date": "2026-09-28", "ingest_attempts": 1, "learning_status": "running", "learning_started": "2026-09-28T01:05:00", "learning_until": "2026-09-28T03:55"}), encoding="utf-8")
    (artifacts / "training_history_v2.csv").write_text("epoch,training_loss,validation_loss,train_pixel_loss,validation_pixel_loss,train_parameter_loss,validation_parameter_loss\n1,0.2,0.11,0.1,0.08,0.5,0.55\n2,0.19,0.105,0.1,0.075,0.5,0.54\n", encoding="utf-8")
    report = {"generated_at": "2026-09-28T01:40:00+05:30", "images": 60, "publications": {
        "ALL": {"editors_change": {"brightness": 0.03, "sharpening_log2": 1.6},
                "models": {"unchanged": {"images": 60, "mae": 0.078, "psnr": 20, "delta_e": 15, "reproduced": 0.0, "effects": {"brightness": {"change": 0}, "sharpening_log2": {"change": 0}}},
                           "v2": {"images": 60, "mae": 0.054, "psnr": 23, "delta_e": 10, "reproduced": 0.30, "effects": {"brightness": {"change": 0.019}, "sharpening_log2": {"change": 1.4}}},
                           "v3_global": {"images": 60, "mae": 0.053, "psnr": 23, "delta_e": 10, "reproduced": 0.32, "effects": {"brightness": {"change": 0.025}, "sharpening_log2": {"change": 1.45}}},
                           "ceiling_curves": {"images": 60, "mae": 0.034, "psnr": 27, "delta_e": 7, "reproduced": 0.57, "effects": {}}}},
        "NEWS": {"editors_change": {}, "models": {}}}}
    (artifacts / "whole_image_report.json").write_text(json.dumps(report), encoding="utf-8")
    (artifacts / "whole_image_history.csv").write_text("generated_at,publication,model,images,mae,psnr,delta_e,reproduced\nx,ALL,v2,60,0.055,23,10,0.29\nx,ALL,v3_global,60,0.054,23,10,0.30\ny,ALL,v2,60,0.054,23,10,0.30\ny,ALL,v3_global,60,0.053,23,10,0.32\n", encoding="utf-8")
    (artifacts / "training_cache" / "cache_status.json").write_text(json.dumps({"used_gigabytes": 17.9, "maximum_bytes": 300 * 2**30, "hot_pair_record_count": 3442, "hot_gigabytes": 17.0, "replay_pair_record_count": 1079, "replay_gigabytes": 0.9, "added_pair_count": 89}), encoding="utf-8")
    (artifacts / "summary.json").write_text(json.dumps({"inventory": [{"set_name": "NEWS", "original_count": 10, "corrected_count": 9, "matched_count": 8}, {"set_name": "PERSISTENT-TRAINING-CACHE"}]}), encoding="utf-8")
    (tmp_path / "config.server.json").write_text(json.dumps({"production_access": {"start": "04:00", "end": "10:00"}}), encoding="utf-8")
    (tmp_path / "logs" / f"service_{NOW.date().isoformat()}.log").write_text("[2026-09-28 01:05:46] Service started (one process).\n", encoding="utf-8")
    return tmp_path


def test_publications_that_add_nothing_are_flagged():
    from native_dashboard import inventory_note

    assert inventory_note({"original_count": 0, "corrected_count": 30, "matched_count": 0}) == "No originals: check folder"
    assert inventory_note({"original_count": 135, "corrected_count": 0, "matched_count": 0}) == "No corrected images: check folder"
    assert inventory_note({"original_count": 0, "corrected_count": 0, "matched_count": 0}) == "Both folders empty"
    assert inventory_note({"original_count": 314, "corrected_count": 312, "matched_count": 299}) == ""
    assert inventory_note({"original_count": 1211, "corrected_count": 798, "matched_count": 753}) == "458 originals not paired yet"


def test_snapshot_reads_every_source(tmp_path):
    snapshot = load_snapshot(_project(tmp_path), now=NOW, running_pids=[42])
    assert snapshot["service"]["label"] == "Learning"
    assert snapshot["state"]["learning_status"] == "running"
    assert len(snapshot["history"]["v2"]) == 2 and snapshot["history"]["v3_global"] == []
    assert snapshot["report"]["images"] == 60 and len(snapshot["report_history"]) == 4
    assert snapshot["cache"]["added_pair_count"] == 89
    assert [row["set_name"] for row in snapshot["inventory"]] == ["NEWS"]
    assert "Service started" in snapshot["service_log"]
    assert native_dashboard.best_epoch(snapshot["history"]["v2"])["epoch"] == "2"


def test_empty_project_does_not_break_the_snapshot(tmp_path):
    snapshot = load_snapshot(tmp_path, now=NOW, running_pids=[])
    assert snapshot["service"]["state"] == "not_running"
    assert snapshot["report"] == {} and snapshot["inventory"] == []


def test_dashboard_window_builds_and_refreshes(tmp_path, monkeypatch):
    monkeypatch.setattr(native_dashboard, "service_pids", lambda: [42])
    for root in (_project(tmp_path / "full"), tmp_path / "empty"):
        root.mkdir(parents=True, exist_ok=True)
        window = native_dashboard.Dashboard(root=root, auto_refresh=False)
        try:
            window.withdraw()
            window.update_idletasks()
            window.refresh()
            window.publication.set("NEWS")
            window._show_quality()
            window.update_idletasks()
        finally:
            window.destroy()
