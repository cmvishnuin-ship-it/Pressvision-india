"""Stop every Image Correction Learning process and switch off login auto-start.

Built as StopImageCorrectionLearning.exe and shipped next to
ImageCorrectionLearningService.exe: run that EXE to start learning, run this
one to stop it. Checkpoints, reports and production images are not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import winreg
from datetime import datetime
from pathlib import Path

PROCESS_NAMES = ("ImageCorrectionLearningService.exe", "ImageCorrectionLearningDashboard.exe")
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "ImageCorrectionLearningService"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def running_process_ids(name: str) -> list[int]:
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, errors="replace", creationflags=NO_WINDOW,
    )
    return [int(row[1]) for row in csv.reader(result.stdout.splitlines()) if len(row) > 1 and row[0].casefold() == name.casefold()]


def stop_processes(name: str) -> tuple[int, list[int]]:
    """Return how many were running and which are still running afterwards."""
    found = running_process_ids(name)
    remaining = found
    for _ in range(10):
        if not remaining:
            break
        # The learner has no window, so only a forced stop (/F) ends it; /T includes
        # any child process. Repeating catches a copy started meanwhile.
        subprocess.run(["taskkill", "/IM", name, "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        time.sleep(0.5)
        remaining = running_process_ids(name)
    return len(found), remaining


def project_artifacts() -> Path:
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    return base / "artifacts"


def mark_stopped(artifacts: Path) -> None:
    """Tell the dashboard the service was stopped on purpose (it reads live_status.json)."""
    if not artifacts.is_dir():
        return
    payload = {
        "pid": None, "updated_at": datetime.now().isoformat(timespec="seconds"),
        "activity": "stopped", "detail": "Stopped with StopImageCorrectionLearning.exe",
    }
    try:
        temporary = artifacts / "live_status.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, artifacts / "live_status.json")
    except OSError:
        pass


def remove_login_autostart() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, RUN_VALUE)
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop every Image Correction Learning process.")
    parser.add_argument("--no-pause", action="store_true", help="Close without waiting for Enter.")
    args = parser.parse_args()

    print("Stopping Image Correction Learning...\n")
    still_running: dict[str, list[int]] = {}
    for name in PROCESS_NAMES:
        found, remaining = stop_processes(name)
        if remaining:
            still_running[name] = remaining
            print(f"  {name}: {len(remaining)} of {found} could NOT be stopped")
        elif found:
            print(f"  {name}: stopped {found} process(es)")
        else:
            print(f"  {name}: was not running")
    if remove_login_autostart():
        print("  Login auto-start: removed")
    if PROCESS_NAMES[0] not in still_running:
        mark_stopped(project_artifacts())

    if still_running:
        print("\nSome processes belong to another account. Right-click StopImageCorrectionLearning.exe and choose 'Run as administrator'.")
        code = 1
    else:
        print("\nEverything is stopped. Run ImageCorrectionLearningService.exe to start again.")
        print("Checkpoints, reports and production images were not touched.")
        code = 0
    if not args.no_pause:
        input("\nPress Enter to close...")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
