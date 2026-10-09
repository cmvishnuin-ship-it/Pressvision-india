from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from ctypes import wintypes
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path


SERVICE_NAME = "ImageCorrectionLearningService"
# Production images may be read only inside this window (config: "production_access").
WINDOW_START = clock_time(4, 0)
TRAINING_CUTOFF = clock_time(9, 50)  # last moment a production ingest may start
WINDOW_END = clock_time(10, 0)
RETRY_DELAY_MINUTES = 15
MAX_ATTEMPTS_PER_DAY = 3
RECENT_CHECKPOINT_ARCHIVES = 7
# Learning reads only the local cache and may run at any time. It hands over
# this long before the next production access window opens.
LEARNING_HANDOFF_MINUTES = 5
LEARNING_MINIMUM_MINUTES = 15
LEARNING_RETRY_MINUTES = 30
LEARN_EXIT_DEADLINE = 3
REPORT_MINIMUM_MINUTES = 30  # the whole-image report runs only with this much time left
CONTROLLER_POLL_SECONDS = 30
# Sessions stop themselves; the watchdog restarts the service only if one is
# still running this long after its deadline (for example a hung network read).
WATCHDOG_GRACE_MINUTES = 5
RESTART_EXIT_CODE = 4
RESTART_MUTEX_WAIT_SECONDS = 120
PRIORITY_CLASSES = {
    "normal": getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0x00000020),
    "below_normal": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000),
    "idle": getattr(subprocess, "IDLE_PRIORITY_CLASS", 0x00000040),
}
V2_TRAINING_ARGUMENTS = [
    "--epochs", "1000",
    "--patch-size", "64",
    "--patches-per-image", "1",
    "--patches-per-load", "4",
    "--batch-size", "8",
    "--samples-per-epoch", "2000",
    "--current-sample-fraction", "0.60",
    "--set-balance-exponent", "0.5",
    "--minimum-remaining-minutes", "5",
]


@dataclass(frozen=True)
class AccessWindow:
    start: clock_time = WINDOW_START
    last_start: clock_time = TRAINING_CUTOFF
    end: clock_time = WINDOW_END


DEFAULT_WINDOW = AccessWindow()


@dataclass(frozen=True)
class Action:
    kind: str  # "ingest", "learn", "stop_learning" or "wait"
    until: datetime | None = None
    reason: str = ""


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", application_dir()))


APP_DIR = application_dir()
LOG_DIR = APP_DIR / "logs"
ARTIFACT_DIR = APP_DIR / "artifacts"
STATE_FILE = ARTIFACT_DIR / "service_state.json"
LIVE_STATUS_FILE = ARTIFACT_DIR / "live_status.json"
MODEL_LABELS = {"v2": "V2", "v3_global": "V3 whole-picture", "v2_pretrained": "V2 pretrained", "pix2pix": "Pix2pix"}


def write_live_status(activity: str, detail: str, **details: object) -> None:
    """Publish what the service is doing right now for the dashboard (best effort)."""
    payload = {"pid": os.getpid(), "updated_at": datetime.now().isoformat(timespec="seconds"), "activity": activity, "detail": detail, **details}
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        temporary = LIVE_STATUS_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, LIVE_STATUS_FILE)
    except OSError:
        pass  # e.g. the dashboard is reading it this instant; the next update follows shortly


def describe_learning_progress(event: dict) -> str:
    model = MODEL_LABELS.get(str(event.get("model")), str(event.get("model", "model")))
    phase = event.get("phase")
    if phase == "preparing":
        return f"{model}: preparing the fixed validation samples"
    if phase == "baseline":
        return f"{model}: measuring the starting model"
    if phase == "training":
        return f"{model}: epoch {event.get('epoch')}, batch {event.get('batch')} of {event.get('batches')}"
    if phase == "epoch_done":
        return f"{model}: epoch {event.get('epoch')} done" + (" (new best)" if event.get("improved") else "")
    return f"{model}: {phase}"


def append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (LOG_DIR / f"service_{date.today().isoformat()}.log").open(
        "a", encoding="utf-8", buffering=1
    ) as handle:
        handle.write(f"[{stamp}] {message}\n")


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(**updates: object) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.update(updates)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def acquire_single_instance(wait_seconds: float = 0.0, name: str = SERVICE_NAME):
    """Return a mutex handle, or False if another copy holds it for longer than ``wait_seconds``."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    give_up = time.monotonic() + wait_seconds
    while True:
        handle = kernel32.CreateMutexW(None, False, f"Local\\{name}")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() != 183:  # not ERROR_ALREADY_EXISTS
            return handle
        kernel32.CloseHandle(handle)
        if time.monotonic() >= give_up:
            return False
        time.sleep(1)


def configuration_path() -> Path:
    local = APP_DIR / "config.server.json"
    if local.exists():
        return local
    bundled = resource_dir() / "config.server.json"
    if bundled.exists():
        return bundled
    raise FileNotFoundError("config.server.json is missing")


def bootstrap_checkpoint() -> Path | None:
    """Return the V2 checkpoint to resume from, installing a bundled one on first use.

    None means there is no V2 checkpoint yet: V2 then trains from scratch.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    local = ARTIFACT_DIR / "best_v2.pt"
    if local.exists():
        return local
    bundled = resource_dir() / "artifacts" / "best_v2.pt"
    if not bundled.exists():
        return None
    shutil.copy2(bundled, local)
    append_log("Installed the bundled initial best_v2.pt checkpoint locally.")
    return local


def bootstrap_pretrained_weights(artifacts: Path = ARTIFACT_DIR) -> Path | None:
    """Install the bundled ImageNet resnet18 weights locally on first use only.

    A resumed run never needs this file (the checkpoint already contains the
    resnet18 submodule's weights), so a missing bundled file only blocks a
    brand-new, never-before-resumed resnet18 shadow run, not the primary V2
    training path.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    local = artifacts / "pretrained" / "resnet18_imagenet.pt"
    if local.exists():
        return local
    bundled = resource_dir() / "artifacts" / "pretrained" / "resnet18_imagenet.pt"
    if not bundled.exists():
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled, local)
    append_log("Installed the bundled initial resnet18_imagenet.pt weights locally.")
    return local


def prune_checkpoint_archive(artifacts: Path = ARTIFACT_DIR) -> dict:
    """Keep seven recent archives plus one older milestone per calendar month."""
    archive = artifacts / "checkpoint_archive"
    files = sorted(archive.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True) if archive.exists() else []
    keep = set(files[:RECENT_CHECKPOINT_ARCHIVES])
    retained_months = set()
    for path in files[RECENT_CHECKPOINT_ARCHIVES:]:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        month = (modified.year, modified.month)
        if month not in retained_months:
            retained_months.add(month)
            keep.add(path)
    removed = 0
    recovered = 0
    for path in files:
        if path in keep:
            continue
        recovered += path.stat().st_size
        path.unlink()
        removed += 1
    return {"retained": len(keep), "removed": removed, "recovered_bytes": recovered}


def access_window_from_config(config: dict) -> AccessWindow:
    settings = config.get("production_access") or {}
    if not isinstance(settings, dict):
        raise ValueError("production_access must be a JSON object")

    def clock(name: str, default: clock_time) -> clock_time:
        value = settings.get(name)
        if value is None:
            return default
        try:
            return datetime.strptime(str(value), "%H:%M").time()
        except ValueError as exc:
            raise ValueError(f"production_access.{name} must use HH:MM") from exc

    window = AccessWindow(clock("start", WINDOW_START), clock("last_start", TRAINING_CUTOFF), clock("end", WINDOW_END))
    if not window.start < window.last_start <= window.end:
        raise ValueError("production_access needs start < last_start <= end within one day")
    return window


def inside_access_window(moment: datetime, window: AccessWindow = DEFAULT_WINDOW) -> bool:
    return window.start <= moment.time() < window.end


def can_start_session(moment: datetime, window: AccessWindow = DEFAULT_WINDOW) -> bool:
    return window.start <= moment.time() < window.last_start


def next_window_start(moment: datetime, window: AccessWindow = DEFAULT_WINDOW) -> datetime:
    today_start = datetime.combine(moment.date(), window.start)
    return today_start if moment < today_start else today_start + timedelta(days=1)


def seconds_until_next_start(moment: datetime, window: AccessWindow = DEFAULT_WINDOW) -> float:
    return max(1.0, (next_window_start(moment, window) - moment).total_seconds())


def stop_at_argument(deadline: datetime) -> str:
    return deadline.strftime("%Y-%m-%dT%H:%M")


def dataset_signature(artifacts: Path) -> str | None:
    """Cheap identity of the local dataset; it changes whenever an ingest rewrites it."""
    parts = []
    for name in ("manifest.csv", "correction_parameters.csv"):
        try:
            status = (artifacts / name).stat()
        except OSError:
            return None
        if not status.st_size:
            return None
        parts.append(f"{name}:{status.st_size}:{status.st_mtime_ns}")
    return "|".join(parts)


def learning_thread_count(settings: dict, cpu_count: int | None = None) -> int:
    requested = int(settings.get("max_threads", 0) or 0)
    if requested > 0:
        return requested
    return max(1, (cpu_count or os.cpu_count() or 2) // 2)


def learning_priority_flag(settings: dict) -> int:
    name = str(settings.get("cpu_priority", "below_normal")).casefold()
    if name not in PRIORITY_CLASSES:
        raise ValueError(f"learning.cpu_priority must be one of {sorted(PRIORITY_CLASSES)}")
    return PRIORITY_CLASSES[name]


def _state_time(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def ingest_attempts_today(state: dict, today: str) -> int:
    return int(state.get("ingest_attempts", 0)) if state.get("ingest_attempt_date") == today else 0


def ingest_pending(now: datetime, state: dict, window: AccessWindow = DEFAULT_WINDOW) -> bool:
    today = now.date().isoformat()
    return (
        can_start_session(now, window)
        and state.get("ingest_completed_date") != today
        and ingest_attempts_today(state, today) < MAX_ATTEMPTS_PER_DAY
    )


def next_production_read(now: datetime, state: dict, window: AccessWindow = DEFAULT_WINDOW) -> datetime:
    """When the service will next read production images."""
    if ingest_pending(now, state, window):
        retry_after = _state_time(state.get("ingest_retry_after"))
        return retry_after if retry_after is not None and retry_after > now else now
    return next_window_start(now, window)


def next_action(now: datetime, state: dict, learning_running: bool, data_signature: str | None,
                window: AccessWindow = DEFAULT_WINDOW) -> Action:
    """Decide the controller's next step; pure so every case can be tested."""
    if ingest_pending(now, state, window):
        if learning_running:
            return Action("stop_learning", reason="the production access window is open")
        retry_after = _state_time(state.get("ingest_retry_after"))
        if retry_after is not None and now < retry_after:
            return Action("wait", reason="waiting to retry today's image read")
        return Action("ingest", reason="the production access window is open")
    if learning_running:
        return Action("wait", reason="learning in progress")
    if data_signature is None:
        return Action("wait", reason="no images in the local cache yet; waiting for the first image read")
    if state.get("learned_data") == data_signature:
        return Action("wait", reason="learning is complete on the current images; waiting for new images")
    retry_after = _state_time(state.get("learning_retry_after"))
    if retry_after is not None and now < retry_after:
        return Action("wait", reason="waiting before retrying learning after an error")
    until = next_window_start(now, window) - timedelta(minutes=LEARNING_HANDOFF_MINUTES)
    if until - now < timedelta(minutes=LEARNING_MINIMUM_MINUTES):
        return Action("wait", reason="too close to the next image read to start learning")
    return Action("learn", until=until)


def run_python_main(main_function, arguments: list[str]):
    old_argv = sys.argv
    try:
        sys.argv = arguments
        return main_function()
    finally:
        sys.argv = old_argv


def load_service_configuration(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Service configuration must be a JSON object")
    return payload


def shadow_settings(config_path: Path, section: str) -> dict:
    settings = load_service_configuration(config_path).get(section, {})
    return settings if isinstance(settings, dict) else {}


def shadow_has_time(settings: dict, now: datetime, deadline: datetime) -> bool:
    return deadline - now >= timedelta(minutes=int(settings.get("minimum_remaining_minutes", 10)))


def pix2pix_shadow_arguments(config_path: Path, artifacts: Path, deadline: datetime) -> list[str] | None:
    """Build a cache-only shadow command, or return None when it is disabled or has no cache."""

    config = load_service_configuration(config_path)
    settings = config.get("pix2pix_shadow", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None
    cache_index = artifacts / "training_cache" / "cache_index.json"
    if not cache_index.is_file():
        return None
    minimum_minutes = int(settings.get("minimum_remaining_minutes", 10))

    output = artifacts / "pix2pix"
    arguments = [
        "--cache-index", str(cache_index),
        "--output-dir", str(output),
        "--epochs", str(int(settings.get("epochs_per_session", 20))),
        "--batch-size", str(int(settings.get("batch_size", 1))),
        "--num-workers", str(int(settings.get("num_workers", 0))),
        "--validation-fraction", str(float(settings.get("validation_fraction", 0.10))),
        "--max-aspect-change", str(float(settings.get("max_aspect_change", 0.02))),
        "--patience", str(int(settings.get("patience", 10))),
        "--minimum-psnr-delta", str(float(settings.get("minimum_psnr_delta", 0.02))),
        "--checkpoint-every", str(int(settings.get("checkpoint_every", 5))),
        "--max-periodic-checkpoints", str(int(settings.get("max_periodic_checkpoints", 5))),
        "--stop-at", stop_at_argument(deadline),
        "--minimum-remaining-minutes", str(minimum_minutes),
        "--skip-unchanged",
    ]
    if bool(settings.get("exact_filename_only", True)):
        arguments.append("--exact-cache-filenames")
    else:
        arguments.append("--no-exact-cache-filenames")
    for root in config.get("read_only_roots", []):
        arguments.extend(("--read-only-root", str(root)))
    resume = output / "checkpoints" / "last.pt"
    if resume.is_file():
        arguments.extend(("--resume", "last.pt"))
    return arguments


def pretrained_backbone_shadow_arguments(config_path: Path, artifacts: Path, deadline: datetime) -> list[str] | None:
    """Build a cache-only resnet18-backbone shadow command, or None when it is disabled or unready.

    Unlike the pix2pix shadow (which reads the local cache index directly),
    this reuses train.py's normal manifest.csv/correction_parameters.csv
    inputs, restricted to files inside the local cache.
    """

    config = load_service_configuration(config_path)
    settings = config.get("pretrained_backbone_shadow", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None
    manifest = artifacts / "manifest.csv"
    parameters = artifacts / "correction_parameters.csv"
    if not manifest.is_file() or not parameters.is_file():
        return None  # no production ingest has produced them yet
    minimum_minutes = int(settings.get("minimum_remaining_minutes", 10))

    arguments = [
        "--config", str(config_path),
        "--artifacts", str(artifacts),
        "--backbone", "resnet18",
        "--epochs", "1000",
        "--patch-size", str(int(settings.get("patch_size", 64))),
        "--patches-per-image", str(int(settings.get("patches_per_image", 1))),
        "--batch-size", str(int(settings.get("batch_size", 8))),
        "--samples-per-epoch", str(int(settings.get("samples_per_epoch", 2000))),
        "--current-sample-fraction", str(float(settings.get("current_sample_fraction", 0.60))),
        "--patience", str(int(settings.get("patience", 5))),
        "--stop-at", stop_at_argument(deadline),
        "--minimum-remaining-minutes", str(minimum_minutes),
        "--cache-only", str(artifacts / "training_cache"),
    ]
    resume = artifacts / "last_v2_pretrained.pt"
    if resume.is_file():
        arguments.extend(("--resume", str(resume)))
    else:
        weights = bootstrap_pretrained_weights(artifacts)
        if weights is None:
            return None  # no prior checkpoint and no bundled weights: cannot start safely
        arguments.extend(("--pretrained-weights", str(weights)))
    return arguments


def global_context_shadow_arguments(config_path: Path, artifacts: Path, deadline: datetime) -> list[str] | None:
    """Build the whole-image-context (V3) shadow command, or None when it is disabled or unready.

    Like the pretrained-backbone shadow it reuses the local manifest and
    correction parameters, restricted to files inside the local cache. A first
    run is warm-started from best_v2.pt, so it begins exactly where V2 is and
    its first saved checkpoint must already beat V2 on the fixed validation set.
    """

    config = load_service_configuration(config_path)
    settings = config.get("global_context_shadow", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None
    manifest = artifacts / "manifest.csv"
    parameters = artifacts / "correction_parameters.csv"
    if not manifest.is_file() or not parameters.is_file():
        return None
    minimum_minutes = int(settings.get("minimum_remaining_minutes", 10))

    arguments = [
        "--config", str(config_path),
        "--artifacts", str(artifacts),
        "--backbone", "global",
        "--epochs", "1000",
        "--patch-size", str(int(settings.get("patch_size", 64))),
        "--patches-per-image", str(int(settings.get("patches_per_image", 1))),
        "--patches-per-load", str(int(settings.get("patches_per_load", 4))),
        "--batch-size", str(int(settings.get("batch_size", 8))),
        "--samples-per-epoch", str(int(settings.get("samples_per_epoch", 2000))),
        "--current-sample-fraction", str(float(settings.get("current_sample_fraction", 0.60))),
        "--set-balance-exponent", str(float(settings.get("set_balance_exponent", 0.5))),
        "--context-size", str(int(settings.get("context_size", 128))),
        "--global-loss-weight", str(float(settings.get("global_loss_weight", 1.0))),
        "--learning-rate", str(float(settings.get("learning_rate", 2e-4))),
        "--patience", str(int(settings.get("patience", 5))),
        "--stop-at", stop_at_argument(deadline),
        "--minimum-remaining-minutes", str(minimum_minutes),
        "--cache-only", str(artifacts / "training_cache"),
    ]
    if bool(settings.get("train_global_only", False)):
        arguments.append("--train-global-only")
    resume = artifacts / "best_v3_global.pt"
    if resume.is_file():
        arguments.extend(("--resume", str(resume)))
    else:
        warm_start = artifacts / "best_v2.pt"
        if not warm_start.is_file():
            return None
        arguments.extend(("--init-from", str(warm_start)))
    return arguments


def run_global_context_shadow(config_path: Path, artifacts: Path, deadline: datetime, now: datetime | None = None) -> str:
    """Run the whole-image-context V3 variant as a cache-only shadow worker.

    It writes only its own best_v3_global.pt / last_v3_global.pt lineage; a
    failure here cannot invalidate the primary V2 checkpoint. Returns the
    training outcome ("early_stopped", "deadline", ...) or "skipped".
    """

    arguments = global_context_shadow_arguments(config_path, artifacts, deadline)
    if arguments is None:
        print("Global-context shadow training skipped: disabled or missing prerequisites.", flush=True)
        return "skipped"
    if not shadow_has_time(shadow_settings(config_path, "global_context_shadow"), now or datetime.now(), deadline):
        print("Global-context shadow training not started: too close to the next production access window.", flush=True)
        return "deadline"
    import train

    print("Starting global-context (V3) shadow training from the local cache.", flush=True)
    return run_python_main(train.main, ["train.py", *arguments])


def run_pretrained_backbone_shadow(config_path: Path, artifacts: Path, deadline: datetime, now: datetime | None = None) -> str:
    """Run the experimental pretrained-backbone V2 variant as a cache-only shadow worker.

    It writes to its own last_v2_pretrained.pt / best_v2_pretrained.pt lineage;
    a failure here cannot invalidate the primary V2 checkpoint.
    """

    arguments = pretrained_backbone_shadow_arguments(config_path, artifacts, deadline)
    if arguments is None:
        print("Pretrained-backbone shadow training skipped: disabled or missing prerequisites.", flush=True)
        return "skipped"
    if not shadow_has_time(shadow_settings(config_path, "pretrained_backbone_shadow"), now or datetime.now(), deadline):
        print("Pretrained-backbone shadow training not started: too close to the next production access window.", flush=True)
        return "deadline"
    import train

    print("Starting pretrained-backbone shadow training from the local cache.", flush=True)
    return run_python_main(train.main, ["train.py", *arguments])


def run_pix2pix_shadow(config_path: Path, artifacts: Path, deadline: datetime, now: datetime | None = None) -> str:
    """Run the experimental model only against the verified local cache."""

    arguments = pix2pix_shadow_arguments(config_path, artifacts, deadline)
    if arguments is None:
        print("Pix2pix shadow training skipped: disabled or no cache.", flush=True)
        return "skipped"
    settings = shadow_settings(config_path, "pix2pix_shadow")
    if not shadow_has_time(settings, now or datetime.now(), deadline):
        print("Pix2pix shadow training not started: too close to the next production access window.", flush=True)
        return "deadline"
    import pix2pix_train

    print("Starting cache-only pix2pix shadow training; production paths will not be opened.", flush=True)
    return_code = int(pix2pix_train.main(arguments))
    if return_code != 0:
        return f"failed:{return_code}"
    return "deadline" if not shadow_has_time(settings, datetime.now(), deadline) else "finished"


def _run_logged(label: str, body) -> int:
    """Run a worker body with its output appended to today's training log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"training_{date.today().isoformat()}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            try:
                return body()
            except BaseException:
                print(f"{label} failed:", flush=True)
                traceback.print_exc(file=output)
                output.flush()
                return 1


def ingest_session(window: AccessWindow) -> int:
    """Read production images (the only step that does) and refresh the local cache and dataset.

    It stops cleanly between files when the access window ends (0 = done, 2 = not
    allowed now or stopped at the window end, 1 = failed).
    """

    def body() -> int:
        now = datetime.now()
        print(f"[{now.isoformat(timespec='seconds')}] Production ingest started", flush=True)
        if not can_start_session(now, window):
            print(f"Ingest cancelled: production images may only be read from {window.start:%H:%M} (start before {window.last_start:%H:%M}).", flush=True)
            return 2
        import analyze
        from correction_ml.safety import AccessWindowClosed

        end = datetime.combine(now.date(), window.end)
        try:
            run_python_main(analyze.main, [
                "analyze.py", "--config", str(configuration_path()), "--artifacts", str(ARTIFACT_DIR),
                "--stop-at", end.isoformat(timespec="minutes"),
            ])
        except AccessWindowClosed as closed:
            print(f"{closed} The cache and dataset keep their previous state; the next ingest continues.", flush=True)
            return 2
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Production ingest completed", flush=True)
        return 0

    return _run_logged("Production ingest", body)


def learn_session(until: datetime) -> int:
    """Train every enabled model from the local cache only, until they converge or ``until`` nears.

    Returns 0 when every model finished (converged or skipped), LEARN_EXIT_DEADLINE
    when the deadline cut the chain short, and 1 on failure.
    """

    def body() -> int:
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] Learning started from the local cache; "
            f"hands over at {until:%Y-%m-%d %H:%M}. Production paths are not opened.",
            flush=True,
        )
        config = configuration_path()
        checkpoint = bootstrap_checkpoint()
        threads = learning_thread_count(load_service_configuration(config).get("learning", {}))
        import torch

        torch.set_num_threads(threads)
        print(f"Learning uses {threads} CPU threads.", flush=True)
        if dataset_signature(ARTIFACT_DIR) is None:
            print("No local dataset yet; waiting for the first production ingest.", flush=True)
            return 0
        import train

        started = datetime.now().isoformat(timespec="seconds")
        until_text = until.isoformat(timespec="minutes")
        train.progress_hook = lambda event: write_live_status(
            "learning", describe_learning_progress(event), until=until_text, started=started, threads=threads, **event
        )
        if checkpoint is None:
            print("No V2 checkpoint yet; V2 trains from scratch.", flush=True)
        outcomes = {
            "v2": run_python_main(train.main, [
                "train.py",
                "--config", str(config),
                "--artifacts", str(ARTIFACT_DIR),
                *(["--resume", str(checkpoint)] if checkpoint is not None else []),
                "--cache-only", str(ARTIFACT_DIR / "training_cache"),
                "--stop-at", stop_at_argument(until),
                *V2_TRAINING_ARGUMENTS,
            ])
        }
        print(f"Checkpoint archive retention: {prune_checkpoint_archive()}", flush=True)
        for name, runner in (
            ("v3_global", run_global_context_shadow),
            ("v2_pretrained", run_pretrained_backbone_shadow),
            ("pix2pix", run_pix2pix_shadow),
        ):
            write_live_status("learning", f"{MODEL_LABELS[name]}: starting", model=name, until=until_text, started=started, threads=threads)
            try:
                outcomes[name] = runner(config, ARTIFACT_DIR, until)
            except BaseException:
                outcomes[name] = "failed"
                print(f"{name} shadow worker failed; V2 results remain valid and unchanged:\n{traceback.format_exc()}", flush=True)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Learning ended: {outcomes}", flush=True)
        if "deadline" in outcomes.values():
            return LEARN_EXIT_DEADLINE
        if until - datetime.now() >= timedelta(minutes=REPORT_MINIMUM_MINUTES):
            write_live_status("learning", "Scoring whole images for the quality report", phase="report", until=until_text, started=started)
            try:
                import evaluate_whole_images

                run_python_main(evaluate_whole_images.main, ["evaluate_whole_images.py", "--artifacts", str(ARTIFACT_DIR)])
            except BaseException:
                print(f"Whole-image report failed; the learned models are unaffected:\n{traceback.format_exc()}", flush=True)
        return 0

    return _run_logged("Learning", body)


def set_process_priority(priority_class: int) -> None:
    """Lower this process's CPU priority while learning; restore it for the ingest."""
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class)


def restart_service() -> None:
    """Start a fresh copy of the service, then end this one at once (closing any open file)."""
    script = [] if getattr(sys, "frozen", False) else [str(Path(__file__).resolve())]
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        creationflags = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [sys.executable, *script, "--restarted"], cwd=APP_DIR,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        startupinfo=startupinfo, creationflags=creationflags,
    )
    os._exit(RESTART_EXIT_CODE)


class Watchdog:
    """Last resort when a session does not stop itself in time (for example a hung
    network read): restart the service, which ends every open file handle.

    Sessions normally stop on their own (between files, or within one batch), so
    this only fires if something is stuck.
    """

    def __init__(self, deadline: datetime, activity: str, on_expire=None):
        self.deadline = deadline
        self.activity = activity
        self.on_expire = on_expire or restart_service
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> "Watchdog":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self._cancelled.set()
        self._thread.join(timeout=5)
        return False

    def _watch(self) -> None:
        while True:
            remaining = (self.deadline - datetime.now()).total_seconds()
            if remaining <= 0:
                append_log(f"{self.activity} was still running at {self.deadline:%H:%M}; restarting the service to enforce the schedule.")
                self.on_expire()
                return
            if self._cancelled.wait(timeout=min(30.0, remaining)):
                return


def status_once() -> int:
    now = datetime.now()
    APP_DIR.mkdir(parents=True, exist_ok=True)
    append_log(
        "Status-only check: executable is healthy; "
        f"production access permitted now={inside_access_window(now)}; no production path was accessed."
    )
    return 0


def self_test() -> int:
    """Verify the packaged ML runtime and checkpoint without production access."""
    try:
        config = load_service_configuration(configuration_path())
        access_window_from_config(config)
        learning_priority_flag(config.get("learning", {}))
        checkpoint_path = bootstrap_checkpoint()
        import torch
        import analyze  # noqa: F401
        import pix2pix_train  # noqa: F401
        import train  # noqa: F401
        from correction_ml.model import CorrectionUNetV2
        from correction_ml.model_global import CorrectionUNetV3Global
        from correction_ml.model_pretrained import CorrectionUNetV2PretrainedBackbone
        from correction_ml.parameters import PARAMETER_NAMES
        from secure_pix2pix import IMAGE_SIZE, LAMBDA_L1, LEARNING_RATE
        from secure_pix2pix.models import PatchDiscriminator, UNetGenerator

        # Without a checkpoint (a fresh install) the checks run on untrained V2 weights.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else {}
        model = CorrectionUNetV2(
            int(checkpoint.get("base_channels", 32)),
            len(PARAMETER_NAMES),
        )
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            prediction, parameters = model.forward_with_parameters(torch.zeros(1, 3, 64, 64))
        if tuple(prediction.shape) != (1, 3, 64, 64) or parameters.shape[1] != len(PARAMETER_NAMES):
            raise RuntimeError("Unexpected model output dimensions")

        # The global-context shadow is warm-started from this checkpoint and
        # must reproduce V2 exactly before its first update.
        global_model = CorrectionUNetV3Global(int(checkpoint.get("base_channels", 32)), len(PARAMETER_NAMES)).eval()
        global_model.load_v2_weights(model.state_dict())
        with torch.no_grad():
            probe = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
            reference, _ = model.forward_with_parameters(probe)
            global_prediction, global_parameters = global_model.forward_with_parameters(probe, torch.rand(1, 3, 128, 128))
        if global_parameters.shape[1] != len(PARAMETER_NAMES) or not torch.allclose(global_prediction, reference, atol=1e-5):
            raise RuntimeError("Global-context model does not reproduce V2 at warm start")

        # Proves the packaged torchvision dependency (needed only for this
        # experimental shadow variant) actually loads in the frozen EXE; no
        # real weights are needed for this shape/import check.
        pretrained_model = CorrectionUNetV2PretrainedBackbone(8, len(PARAMETER_NAMES), pretrained_weights_path=None).eval()
        with torch.no_grad():
            pretrained_prediction, pretrained_parameters = pretrained_model.forward_with_parameters(torch.zeros(1, 3, 64, 64))
        if tuple(pretrained_prediction.shape) != (1, 3, 64, 64) or pretrained_parameters.shape[1] != len(PARAMETER_NAMES):
            raise RuntimeError("Unexpected pretrained-backbone model output dimensions")

        shadow_generator = UNetGenerator(base_channels=8).eval()
        shadow_discriminator = PatchDiscriminator(base_channels=8).eval()
        with torch.no_grad():
            source = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            shadow_output = shadow_generator(source)
            shadow_patch = shadow_discriminator(source, shadow_output)
        if tuple(shadow_output.shape) != (1, 3, 256, 256) or tuple(shadow_patch.shape) != (1, 1, 30, 30):
            raise RuntimeError("Unexpected pix2pix shadow model output dimensions")
        if LAMBDA_L1 != 100.0 or LEARNING_RATE != 0.0002:
            raise RuntimeError("Pix2pix fixed hyperparameters changed unexpectedly")
        append_log(
            f"Self-test passed: {'V2 checkpoint' if checkpoint else 'untrained V2 (no checkpoint yet)'}, "
            "global-context shadow warm start, pretrained-backbone shadow runtime, and cache-only "
            "pix2pix shadow runtime loaded; no production path was accessed."
        )
        return 0
    except BaseException:
        append_log("Self-test failed:\n" + traceback.format_exc())
        return 1


def run_ingest(now: datetime, state: dict, window: AccessWindow) -> None:
    today = now.date().isoformat()
    attempts = ingest_attempts_today(state, today) + 1
    save_state(ingest_attempt_date=today, ingest_attempts=attempts, ingest_status="running", ingest_started=now.isoformat(timespec="seconds"))
    append_log(f"Starting daily production ingest attempt {attempts}/{MAX_ATTEMPTS_PER_DAY}.")
    set_process_priority(PRIORITY_CLASSES["normal"])
    window_end = datetime.combine(now.date(), window.end)
    write_live_status(
        "ingest", f"Reading new images from production (attempt {attempts} of {MAX_ATTEMPTS_PER_DAY})",
        started=now.isoformat(timespec="seconds"), window_end=window_end.isoformat(timespec="minutes"),
    )
    with Watchdog(window_end + timedelta(minutes=WATCHDOG_GRACE_MINUTES), "The production ingest"):
        return_code = ingest_session(window)
    if return_code == 0:
        save_state(ingest_completed_date=today, ingest_status="completed", ingest_return_code=0, ingest_retry_after=None)
        append_log("Daily production ingest completed; learning continues from the local cache.")
    else:
        retry_after = datetime.now() + timedelta(minutes=RETRY_DELAY_MINUTES)
        save_state(ingest_status="failed", ingest_return_code=return_code, ingest_retry_after=retry_after.isoformat(timespec="seconds"))
        append_log(f"Production ingest attempt failed with code {return_code}; details are in the training log.")


def run_learning(now: datetime, until: datetime, priority_class: int) -> None:
    data = dataset_signature(ARTIFACT_DIR)
    write_live_status("learning", "Starting to learn from the local cache", started=now.isoformat(timespec="seconds"), until=until.isoformat(timespec="minutes"))
    save_state(learning_status="running", learning_started=now.isoformat(timespec="seconds"), learning_until=until.isoformat(timespec="minutes"))
    append_log(f"Learning started from the local cache; it hands over at {until:%Y-%m-%d %H:%M}.")
    set_process_priority(priority_class)
    try:
        backstop = until + timedelta(minutes=LEARNING_HANDOFF_MINUTES + WATCHDOG_GRACE_MINUTES)
        with Watchdog(backstop, "Learning"):
            return_code = learn_session(until)
    finally:
        set_process_priority(PRIORITY_CLASSES["normal"])
    record_learning_result(data, return_code)


def record_learning_result(data: str | None, return_code: int) -> None:
    if return_code == 0:
        save_state(learning_status="finished", learned_data=data, learning_retry_after=None)
        append_log("Learning finished: every model has stopped improving on the current data. Waiting for new images.")
    elif return_code == LEARN_EXIT_DEADLINE:
        save_state(learning_status="handed_over")
        append_log("Learning paused for the production access window; it resumes after the ingest.")
    else:
        retry_after = datetime.now() + timedelta(minutes=LEARNING_RETRY_MINUTES)
        save_state(learning_status="failed", learning_return_code=return_code, learning_retry_after=retry_after.isoformat(timespec="seconds"))
        append_log(f"Learning failed with code {return_code}; retrying after {retry_after:%H:%M}. Details are in the training log.")


def controller(restarted: bool = False) -> int:
    """The whole service in one process: it schedules, ingests and learns itself."""
    # After a watchdog restart the previous process may still be closing.
    mutex = acquire_single_instance(wait_seconds=RESTART_MUTEX_WAIT_SECONDS if restarted else 0)
    if mutex is False:
        return 0
    try:
        config = load_service_configuration(configuration_path())
        window = access_window_from_config(config)
        priority = learning_priority_flag(config.get("learning", {}))
        append_log(
            f"Service {'restarted' if restarted else 'started'} (one process). Production images are read only between "
            f"{window.start:%H:%M} and {window.end:%H:%M}; learning uses the local cache at any other time."
        )
        while True:
            now = datetime.now()
            state = load_state()
            action = next_action(now, state, False, dataset_signature(ARTIFACT_DIR), window)
            if action.kind == "ingest":
                run_ingest(now, state, window)
            elif action.kind == "learn":
                run_learning(now, action.until, priority)
            else:
                write_live_status(
                    "idle", action.reason[:1].upper() + action.reason[1:],
                    next_production_read=next_production_read(now, state, window).isoformat(timespec="minutes"),
                )
                time.sleep(CONTROLLER_POLL_SECONDS)
    except BaseException:
        append_log("Service stopped unexpectedly:\n" + traceback.format_exc())
        write_live_status("error", "The service stopped unexpectedly; see the service log")
        return 1
    finally:
        if mutex not in (None, False) and os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(mutex)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restarted", action="store_true")
    parser.add_argument("--status-once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    os.chdir(APP_DIR)
    if args.status_once:
        return status_once()
    if args.self_test:
        return self_test()
    return controller(restarted=args.restarted)


if __name__ == "__main__":
    raise SystemExit(main())
