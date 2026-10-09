"""Native monitoring window for Image Correction Learning.

Local files only: it reads the project's artifacts and logs, checks whether the
service process is running, and refreshes itself. It never starts, stops or
changes anything, and never opens production images.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

SERVICE_EXE = "ImageCorrectionLearningService.exe"
REFRESH_MS = 10_000
STALE_MINUTES = 20  # a running service normally reports at least this often
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

COLOURS = {
    "header": "#0f172a", "page": "#f1f5f9", "card": "#ffffff", "border": "#e2e8f0",
    "text": "#0f172a", "muted": "#64748b", "v2": "#2563eb", "v3_global": "#ea580c",
    "unchanged": "#94a3b8", "grid": "#e2e8f0",
}
STATES = {
    "learning": ("Learning", "#15803d"),
    "ingest": ("Reading new images", "#2563eb"),
    "idle": ("Idle", "#0e7490"),
    "running": ("Running", "#15803d"),
    "stopped": ("Stopped", "#6b7280"),
    "not_running": ("Not running", "#b45309"),
    "error": ("Stopped after an error", "#b91c1c"),
}
MODEL_LABELS = {
    "unchanged": "Unchanged original", "v2": "V2 (patch model)", "v3_global": "V3 (whole-picture)",
    "v2_pretrained": "V2 pretrained", "ceiling_curves": "Ceiling: best tone curves",
    "ceiling_colour_map": "Ceiling: best colour map",
}
EFFECT_LABELS = {
    "brightness": "Brightness", "contrast": "Contrast", "saturation": "Saturation",
    "temperature_proxy": "Warmth", "tint_proxy": "Tint", "shadow_mean": "Shadows",
    "highlight_mean": "Highlights", "sharpening_log2": "Sharpening (log2)",
}


# ----------------------------------------------------------------- data ----

def project_root() -> Path:
    """The folder holding artifacts/ and logs/, as the service sees it.

    Packaged, the dashboard lives in <service folder>/Dashboard; from source both
    scripts share a folder (learning_service.py uses its own folder too).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_rows(path: Path, limit: int | None = None) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    return rows[-limit:] if limit else rows


def read_tail(path: Path, max_bytes: int = 40_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text if size <= max_bytes else text.split("\n", 1)[-1]


def parse_time(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def service_pids() -> list[int]:
    """Process IDs of every running service process."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {SERVICE_EXE}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", creationflags=NO_WINDOW, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(row[1]) for row in csv.reader(result.stdout.splitlines()) if len(row) > 1 and row[0].casefold() == SERVICE_EXE.casefold()]


def service_status(live: dict, now: datetime, running_pids: list[int]) -> dict:
    """Combine the process list with the service's own live report."""
    activity = str(live.get("activity", ""))
    updated = parse_time(live.get("updated_at"))
    age_minutes = (now - updated).total_seconds() / 60 if updated else None
    if running_pids:
        if live.get("pid") in running_pids and activity in ("learning", "ingest", "idle"):
            state, detail = activity, str(live.get("detail", ""))
            if age_minutes is not None and age_minutes > STALE_MINUTES:
                detail += f"  (no update for {age_minutes:.0f} min)"
        else:
            state, detail = "running", "Running; waiting for its first status update"
    elif activity in ("stopped", "error"):
        state, detail = activity, str(live.get("detail", ""))
    else:
        state, detail = "not_running", f"Start {SERVICE_EXE} to begin learning"
    label, colour = STATES[state]
    return {"state": state, "label": label, "colour": colour, "detail": detail, "updated": updated,
            "age_minutes": age_minutes, "processes": len(running_pids)}


def best_epoch(rows: list[dict]) -> dict | None:
    scored = [row for row in rows if number(row.get("validation_loss")) is not None]
    return min(scored, key=lambda row: number(row["validation_loss"])) if scored else None


def load_snapshot(root: Path, now: datetime | None = None, running_pids: list[int] | None = None) -> dict:
    now = now or datetime.now()
    artifacts, logs = root / "artifacts", root / "logs"
    live = read_json(artifacts / "live_status.json")
    config = read_json(root / "config.server.json")
    summary = read_json(artifacts / "summary.json")
    checkpoints = {}
    for name in ("best_v2", "last_v2", "best_v3_global", "last_v3_global"):
        path = artifacts / f"{name}.pt"
        if path.is_file():
            stat = path.stat()
            checkpoints[name] = {"modified": datetime.fromtimestamp(stat.st_mtime), "size_mb": stat.st_size / 2**20}
    cache_path = artifacts / "training_cache" / "cache_status.json"
    today = now.date().isoformat()
    return {
        "now": now,
        "root": root,
        "service": service_status(live, now, service_pids() if running_pids is None else running_pids),
        "live": live,
        "state": read_json(artifacts / "service_state.json"),
        "config": config,
        "session": read_json(artifacts / "latest_training_session.json"),
        "history": {model: read_rows(artifacts / f"training_history_{model}.csv") for model in ("v2", "v3_global")},
        "report": read_json(artifacts / "whole_image_report.json"),
        "report_history": read_rows(artifacts / "whole_image_history.csv"),
        "cache": read_json(cache_path),
        "cache_updated": datetime.fromtimestamp(cache_path.stat().st_mtime) if cache_path.is_file() else None,
        "inventory": [row for row in summary.get("inventory", []) if isinstance(row, dict) and row.get("set_name") != "PERSISTENT-TRAINING-CACHE"],
        "outliers": read_rows(artifacts / "pair_outliers.csv", limit=300),
        "checkpoints": checkpoints,
        "service_log": read_tail(logs / f"service_{today}.log"),
        "training_log": read_tail(logs / f"training_{today}.log"),
    }


# ----------------------------------------------------------- formatting ----

def clock(value: datetime | None, now: datetime) -> str:
    if value is None:
        return "—"
    return value.strftime("%H:%M") if value.date() == now.date() else value.strftime("%d %b %H:%M")


def ago(value: datetime | None, now: datetime) -> str:
    if value is None:
        return "never"
    seconds = max(0, (now - value).total_seconds())
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def percent(value: object) -> str:
    result = number(value)
    return "—" if result is None else f"{result:.0%}"


def ingest_summary(state: dict, config: dict, now: datetime) -> tuple[str, list[str]]:
    window = config.get("production_access", {}) if isinstance(config.get("production_access"), dict) else {}
    window_text = f"Allowed {window.get('start', '04:00')}–{window.get('end', '10:00')}"
    today = now.date().isoformat()
    if state.get("ingest_completed_date") == today:
        return "Done", [window_text, f"Finished after {state.get('ingest_attempts', 1)} attempt(s)"]
    if state.get("ingest_attempt_date") == today:
        status = state.get("ingest_status", "")
        attempts = state.get("ingest_attempts", 0)
        if status == "running":
            return "Reading now", [window_text, f"Attempt {attempts} of 3"]
        retry = parse_time(state.get("ingest_retry_after"))
        return f"Failed ({attempts} of 3 tries)", [window_text, f"Next try {clock(retry, now)}" if retry and retry > now else "No more tries today"]
    return "Not yet today", [window_text]


def inventory_note(row: dict) -> str:
    """Flag publications that add nothing to learning, which usually means a folder problem."""
    originals, corrected, paired = (int(number(row.get(key)) or 0) for key in ("original_count", "corrected_count", "matched_count"))
    if not originals and not corrected:
        return "Both folders empty"
    if not originals:
        return "No originals: check folder"
    if not corrected:
        return "No corrected images: check folder"
    if not paired:
        return "Nothing paired: check folders"
    unpaired = originals - paired
    return f"{unpaired} originals not paired yet" if unpaired > max(10, originals // 5) else ""


def learning_summary(state: dict, now: datetime) -> tuple[str, list[str]]:
    status = state.get("learning_status", "")
    started, until = parse_time(state.get("learning_started")), parse_time(state.get("learning_until"))
    lines = [f"Last started {clock(started, now)}"] if started else []
    if status == "running":
        return "Running", lines + [f"Hands over {clock(until, now)}"]
    if status == "finished":
        return "Finished", lines + ["Models stopped improving; waits for new images"]
    if status == "handed_over":
        return "Paused", lines + ["Paused for the daily image read"]
    if status == "failed":
        return "Error", lines + [f"Retry after {clock(parse_time(state.get('learning_retry_after')), now)}", "See the Logs tab"]
    return "Not started", lines


# --------------------------------------------------------------- widgets ----

class LineChart(tk.Canvas):
    """Minimal line chart drawn with Tk (no extra libraries)."""

    def __init__(self, parent, title: str, as_percent: bool = False, height: int = 260):
        super().__init__(parent, background=COLOURS["card"], highlightthickness=0, height=height)
        self.title, self.as_percent = title, as_percent
        self.series: list[tuple[str, str, list[float]]] = []
        self.empty_text = "No data yet"
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_series(self, series: list[tuple[str, str, list[float]]], empty_text: str = "No data yet") -> None:
        self.series = [item for item in series if item[2]]
        self.empty_text = empty_text
        self.redraw()

    def _label(self, value: float) -> str:
        return f"{value:.0%}" if self.as_percent else f"{value:.4f}"

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 200), max(self.winfo_height(), 120)
        left, right, top, bottom = 64, 16, 34, 28
        self.create_text(left, 14, text=self.title, anchor="w", font=("Segoe UI", 10, "bold"), fill=COLOURS["text"])
        if not self.series:
            self.create_text(width / 2, height / 2, text=self.empty_text, fill=COLOURS["muted"], font=("Segoe UI", 10))
            return
        values = [value for _, _, points in self.series for value in points]
        low, high = min(values), max(values)
        pad = (high - low) * 0.08 or abs(high) * 0.05 or 0.01
        low, high = low - pad, high + pad
        length = max(len(points) for _, _, points in self.series)
        plot_w, plot_h = width - left - right, height - top - bottom

        def x_at(index: int, count: int) -> float:
            offset = length - count  # align the latest points on the right
            return left + plot_w * ((index + offset) / max(1, length - 1))

        def y_at(value: float) -> float:
            return top + plot_h * (1 - (value - low) / (high - low))

        for step in range(5):
            value = low + (high - low) * step / 4
            y = y_at(value)
            self.create_line(left, y, width - right, y, fill=COLOURS["grid"])
            self.create_text(left - 6, y, text=self._label(value), anchor="e", fill=COLOURS["muted"], font=("Segoe UI", 8))
        legend_x = width - right
        for label, colour, points in reversed(self.series):
            coordinates = []
            for index, value in enumerate(points):
                coordinates += [x_at(index, len(points)), y_at(value)]
            if len(points) == 1:
                x, y = coordinates
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colour, outline=colour)
            else:
                self.create_line(*coordinates, fill=colour, width=2)
            x, y = coordinates[-2:]
            self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colour, outline=colour)
            text = self.create_text(legend_x, 14, text=f"{label}: {self._label(points[-1])}", anchor="e", fill=colour, font=("Segoe UI", 9, "bold"))
            legend_x = self.bbox(text)[0] - 14
        self.create_text(left, height - 10, text="older", anchor="w", fill=COLOURS["muted"], font=("Segoe UI", 8))
        self.create_text(width - right, height - 10, text="latest", anchor="e", fill=COLOURS["muted"], font=("Segoe UI", 8))


class Card(ttk.Frame):
    def __init__(self, parent, title: str, wraplength: int = 360):
        super().__init__(parent, style="Card.TFrame", padding=14)
        ttk.Label(self, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w")
        self.value = ttk.Label(self, text="—", style="CardValue.TLabel")
        self.value.pack(anchor="w", pady=(4, 2))
        self.bar = ttk.Progressbar(self, mode="determinate", maximum=100, style="Green.Horizontal.TProgressbar")
        self.details = ttk.Label(self, text="", style="Card.TLabel", justify="left", wraplength=wraplength)
        self.details.pack(anchor="w")

    def show(self, value: str, lines: list[str], bar: float | None = None) -> None:
        self.value.configure(text=value)
        self.details.configure(text="\n".join(lines))
        if bar is None:
            self.bar.pack_forget()
        else:
            self.bar.configure(value=max(0.0, min(100.0, bar)))
            self.bar.pack(fill="x", pady=(2, 6), before=self.details)


def make_table(parent, columns: list[tuple[str, int]], height: int = 8) -> ttk.Treeview:
    frame = ttk.Frame(parent, style="Card.TFrame")
    table = ttk.Treeview(frame, columns=[name for name, _ in columns], show="headings", height=height)
    for name, width in columns:
        table.heading(name, text=name)
        table.column(name, width=width, minwidth=60, stretch=True, anchor="w")
    scroll = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scroll.set)
    table.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    table.frame = frame
    return table


def fill_table(table: ttk.Treeview, rows: list[list[object]], empty: str = "Nothing to show yet") -> None:
    table.delete(*table.get_children())
    if not rows:
        table.insert("", "end", values=[empty])
        return
    for row in rows:
        table.insert("", "end", values=["" if value is None else value for value in row])


# ------------------------------------------------------------ dashboard ----

def enable_sharp_text() -> None:
    """Opt out of Windows bitmap scaling so text stays crisp on scaled (125–200 %) displays."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


class Dashboard(tk.Tk):
    def __init__(self, root: Path | None = None, auto_refresh: bool = True) -> None:
        enable_sharp_text()
        super().__init__()
        self.root_dir = root or project_root()
        self.snapshot: dict = {}
        self.title("Image Correction Learning — Monitor")
        # Pixel sizes below are for a 100 % display; scale them to this screen.
        self.scale = max(1.0, self.winfo_fpixels("1i") / 96.0)
        width = min(int(1280 * self.scale), int(self.winfo_screenwidth() * 0.95))
        height = min(int(840 * self.scale), int(self.winfo_screenheight() * 0.9))
        self.geometry(f"{width}x{height}")
        self.minsize(int(1040 * self.scale * 0.8), int(700 * self.scale * 0.8))
        self.configure(background=COLOURS["page"])
        self.auto = tk.BooleanVar(value=auto_refresh)
        self.publication = tk.StringVar(value="ALL")
        self._style()
        self._build()
        self.refresh()
        self.after(REFRESH_MS, self._tick)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        base = ("Segoe UI", 10)
        style.configure(".", font=base, background=COLOURS["page"], foreground=COLOURS["text"])
        style.configure("Header.TFrame", background=COLOURS["header"])
        style.configure("Header.TLabel", background=COLOURS["header"], foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("HeaderSub.TLabel", background=COLOURS["header"], foreground="#cbd5e1")
        style.configure("Header.TCheckbutton", background=COLOURS["header"], foreground="#ffffff")
        style.map("Header.TCheckbutton", background=[("active", COLOURS["header"])])
        style.configure("Card.TFrame", background=COLOURS["card"], relief="solid", borderwidth=1, bordercolor=COLOURS["border"])
        style.configure("Card.TLabel", background=COLOURS["card"], foreground=COLOURS["muted"])
        style.configure("CardTitle.TLabel", background=COLOURS["card"], foreground=COLOURS["muted"], font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background=COLOURS["card"], foreground=COLOURS["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", background=COLOURS["card"], font=("Segoe UI", 20, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground=COLOURS["muted"])
        style.configure("Treeview", rowheight=int(24 * self.scale), fieldbackground=COLOURS["card"], background=COLOURS["card"])
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook.Tab", padding=(14, 6))
        style.configure("Green.Horizontal.TProgressbar", background="#15803d", troughcolor=COLOURS["border"],
                        bordercolor=COLOURS["border"], lightcolor="#15803d", darkcolor="#15803d")

    def _build(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 12))
        header.pack(fill="x")
        titles = ttk.Frame(header, style="Header.TFrame")
        titles.pack(side="left")
        ttk.Label(titles, text="Image Correction Learning", style="Header.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Learns what editors change between original and corrected photos · learning only, never edits images", style="HeaderSub.TLabel").pack(anchor="w")
        ttk.Button(header, text="Refresh now", command=self.refresh).pack(side="right")
        ttk.Checkbutton(header, text="Auto-refresh (10 s)", variable=self.auto, style="Header.TCheckbutton").pack(side="right", padx=12)
        self.updated = ttk.Label(header, text="", style="HeaderSub.TLabel")
        self.updated.pack(side="right", padx=12)

        banner = ttk.Frame(self, style="Card.TFrame", padding=(18, 12))
        banner.pack(fill="x", padx=16, pady=(14, 8))
        self.status_dot = tk.Canvas(banner, width=22, height=22, background=COLOURS["card"], highlightthickness=0)
        self.status_dot.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        self.status_label = ttk.Label(banner, text="", style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky="w")
        self.status_detail = ttk.Label(banner, text="", style="Card.TLabel", font=("Segoe UI", 11))
        self.status_detail.grid(row=0, column=2, sticky="w", padx=(14, 0))
        self.status_meta = ttk.Label(banner, text="", style="Card.TLabel")
        self.status_meta.grid(row=1, column=1, columnspan=2, sticky="w")
        self.status_bar = ttk.Progressbar(banner, mode="determinate", maximum=100, length=int(260 * self.scale), style="Green.Horizontal.TProgressbar")
        self.status_bar.grid(row=0, column=3, rowspan=2, sticky="e")
        banner.columnconfigure(2, weight=1)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._build_overview(tabs)
        self._build_quality(tabs)
        self._build_training(tabs)
        self._build_data(tabs)
        self._build_logs(tabs)

    def _page(self, tabs: ttk.Notebook, title: str) -> ttk.Frame:
        page = ttk.Frame(tabs, padding=12)
        tabs.add(page, text=title)
        return page

    def _build_overview(self, tabs) -> None:
        page = self._page(tabs, "Overview")
        titles = ["Today's image read", "Learning", "Models", "Editors' change reproduced", "Training data", "Storage"]
        self.cards = {}
        for index, title in enumerate(titles):
            card = Card(page, title, wraplength=int(360 * self.scale))
            card.grid(row=index // 3, column=index % 3, sticky="nsew", padx=6, pady=6)
            self.cards[title] = card
        for column in range(3):
            page.columnconfigure(column, weight=1, uniform="cards")
        page.rowconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

    def _build_quality(self, tabs) -> None:
        page = self._page(tabs, "Learning quality")
        top = ttk.Frame(page)
        top.pack(fill="x")
        ttk.Label(top, text="How much of the editors' change each model reproduces on whole validation images", style="Section.TLabel").pack(side="left")
        self.publication_box = ttk.Combobox(top, textvariable=self.publication, state="readonly", width=24)
        self.publication_box.pack(side="right")
        self.publication_box.bind("<<ComboboxSelected>>", lambda _event: self._show_quality())
        ttk.Label(top, text="Publication:").pack(side="right", padx=6)
        self.report_note = ttk.Label(page, text="", style="Muted.TLabel")
        self.report_note.pack(anchor="w", pady=(2, 6))
        tables = ttk.Frame(page)
        tables.pack(fill="x")
        self.models_table = make_table(tables, [("Model", 210), ("Reproduced", 90), ("Error", 80), ("Colour ΔE", 80), ("Images", 60)], height=6)
        self.models_table.frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.pattern_table = make_table(tables, [("Effect", 130), ("Editors", 80), ("V2", 80), ("V3", 80), ("Closer", 70)], height=8)
        self.pattern_table.frame.pack(side="left", fill="both", expand=True)
        self.trend_chart = LineChart(page, "Share of the editors' change reproduced, per report", as_percent=True)
        self.trend_chart.pack(fill="both", expand=True, pady=(10, 0))

    def _build_training(self, tabs) -> None:
        page = self._page(tabs, "Training")
        self.training_chart = LineChart(page, "Validation loss per epoch (lower is better)", height=320)
        self.training_chart.pack(fill="both", expand=True)
        ttk.Label(page, text="Latest epochs", style="Section.TLabel").pack(anchor="w", pady=(10, 4))
        self.epochs_table = make_table(page, [("Model", 150), ("Epoch", 70), ("Validation", 100), ("Pixel", 100), ("Parameters", 100), ("Best so far", 90)], height=8)
        self.epochs_table.frame.pack(fill="both", expand=True)

    def _build_data(self, tabs) -> None:
        page = self._page(tabs, "Data")
        ttk.Label(page, text="Latest image read, per publication", style="Section.TLabel").pack(anchor="w")
        self.inventory_table = make_table(page, [("Publication", 150), ("Originals", 70), ("Corrected", 70), ("Paired", 60), ("By name", 60), ("By picture", 70), ("Ambiguous", 70), ("Unreadable", 70), ("Note", 260)], height=7)
        self.inventory_table.frame.pack(fill="x", pady=(4, 10))
        ttk.Label(page, text="Pairs worth a human look (outliers)", style="Section.TLabel").pack(anchor="w")
        self.outlier_table = make_table(page, [("Publication", 120), ("Key", 220), ("Reasons", 220), ("Similarity", 80), ("Pixel change", 90)], height=10)
        self.outlier_table.frame.pack(fill="both", expand=True, pady=(4, 0))

    def _build_logs(self, tabs) -> None:
        page = self._page(tabs, "Logs")
        panes = ttk.PanedWindow(page, orient="vertical")
        panes.pack(fill="both", expand=True)
        self.log_views = {}
        for name, weight in (("Service log (today)", 1), ("Training log (today, latest part)", 3)):
            frame = ttk.Frame(panes)
            ttk.Label(frame, text=name, style="Section.TLabel").pack(anchor="w")
            text = tk.Text(frame, wrap="none", font=("Consolas", 9), height=8, background="#0b1220", foreground="#e2e8f0", insertbackground="#e2e8f0", state="disabled")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            text.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            panes.add(frame, weight=weight)
            self.log_views[name] = text

    # ------------------------------------------------------------ refresh --

    def _tick(self) -> None:
        if self.auto.get():
            self.refresh()
        self.after(REFRESH_MS, self._tick)

    def refresh(self) -> None:
        self.snapshot = load_snapshot(self.root_dir)
        now = self.snapshot["now"]
        self.updated.configure(text=f"Updated {now:%H:%M:%S}")
        self._show_status()
        self._show_overview()
        self._show_quality()
        self._show_training()
        self._show_data()
        self._show_logs()

    def _show_status(self) -> None:
        snap, now = self.snapshot, self.snapshot["now"]
        status, live = snap["service"], snap["live"]
        self.status_dot.delete("all")
        self.status_dot.create_oval(3, 3, 19, 19, fill=status["colour"], outline=status["colour"])
        self.status_label.configure(text=status["label"], foreground=status["colour"])
        self.status_detail.configure(text=status["detail"])
        meta = []
        if status["state"] in ("learning", "ingest", "idle", "running"):
            if live.get("started"):
                meta.append(f"Started {clock(parse_time(live.get('started')), now)}")
            if live.get("until"):
                meta.append(f"Hands over {clock(parse_time(live.get('until')), now)}")
            if live.get("next_production_read"):
                meta.append(f"Next image read {clock(parse_time(live.get('next_production_read')), now)}")
            if live.get("threads"):
                meta.append(f"{live['threads']} CPU threads, low priority")
        if status["updated"]:
            meta.append(f"Last update {ago(status['updated'], now)}")
        if status["processes"] > 1:
            meta.append(f"Warning: {status['processes']} service processes running")
        self.status_meta.configure(text="   ·   ".join(meta))
        batch, batches = number(live.get("batch")), number(live.get("batches"))
        if status["state"] == "learning" and batch is not None and batches:
            self.status_bar.configure(value=100 * batch / batches)
            self.status_bar.grid()
        else:
            self.status_bar.grid_remove()

    def _show_overview(self) -> None:
        snap, now = self.snapshot, self.snapshot["now"]
        state, config, session, cache = snap["state"], snap["config"], snap["session"], snap["cache"]

        value, lines = ingest_summary(state, config, now)
        if snap["cache_updated"] and snap["cache_updated"].date() == now.date() and cache:
            lines.append(f"New pairs today: {cache.get('added_pair_count', 0)}")
        self.cards["Today's image read"].show(value, lines)

        value, lines = learning_summary(state, now)
        self.cards["Learning"].show(value, lines)

        lines, trained = [], []
        for model in ("v2", "v3_global"):
            best = best_epoch(snap["history"][model])
            checkpoint = snap["checkpoints"].get(f"best_{model}")
            if best or checkpoint:
                trained.append(MODEL_LABELS[model].split(" ")[0])
            if best:
                saved = f", saved {clock(checkpoint['modified'], now)}" if checkpoint else ""
                lines.append(f"{MODEL_LABELS[model]}: best {number(best['validation_loss']):.4f} at epoch {best.get('epoch')}{saved}")
            else:
                lines.append(f"{MODEL_LABELS[model]}: not trained yet")
        self.cards["Models"].show(" + ".join(trained) if trained else "Not trained yet", lines)

        overall = snap["report"].get("publications", {}).get("ALL", {}).get("models", {})
        if overall:
            value = "  ·  ".join(f"{name.upper().replace('_GLOBAL', '')} {percent(overall[name]['reproduced'])}" for name in ("v2", "v3_global") if name in overall)
            ceilings = sorted(number(overall[name]["reproduced"]) or 0.0 for name in ("ceiling_curves", "ceiling_colour_map") if name in overall)
            lines = [f"Reachable by a perfect whole-picture fix: {'–'.join(dict.fromkeys(percent(value) for value in ceilings))}", f"{snap['report'].get('images', 0)} validation images, {clock(parse_time(snap['report'].get('generated_at', '')[:19]), now)}"]
            self.cards["Editors' change reproduced"].show(value, lines)
        else:
            self.cards["Editors' change reproduced"].show("No report yet", ["Written after a learning run finishes"])

        if session:
            sets = session.get("image_sets", {})
            status_counts = session.get("source_status", {})
            lines = [f"Held back for validation: {session.get('validation_pairs', '—')}",
                     f"Still in production folders: {status_counts.get('current', 0)} · kept from earlier days: {status_counts.get('historical', 0)}",
                     "Per publication: " + ", ".join(f"{name} {count}" for name, count in sorted(sets.items()))]
            self.cards["Training data"].show(f"{session.get('training_pairs', '—')} training samples", lines)
        else:
            self.cards["Training data"].show("—", ["No training session recorded yet"])

        if cache:
            used, limit = number(cache.get("used_gigabytes")) or 0.0, (number(cache.get("maximum_bytes")) or 0) / 2**30
            lines = [f"Full pairs: {cache.get('hot_pair_record_count', 0)} ({number(cache.get('hot_gigabytes')) or 0:.1f} GB)",
                     f"Compact replay: {cache.get('replay_pair_record_count', 0)} ({number(cache.get('replay_gigabytes')) or 0:.1f} GB)"]
            self.cards["Storage"].show(f"{used:.1f} of {limit:.0f} GB", lines, bar=100 * used / limit if limit else None)
        else:
            self.cards["Storage"].show("—", ["No cache report yet"])

    def _show_quality(self) -> None:
        report = self.snapshot.get("report", {})
        publications = report.get("publications", {})
        names = ["ALL"] + sorted(name for name in publications if name != "ALL")
        self.publication_box.configure(values=names)
        if self.publication.get() not in names:
            self.publication.set("ALL")
        group = publications.get(self.publication.get(), {})
        models = group.get("models", {})
        if report:
            self.report_note.configure(text=f"Report from {report.get('generated_at', '')[:16].replace('T', ' ')} · {report.get('images', 0)} images · reproduced = 1 − error ÷ unchanged error. Ceilings are fitted to the editors' own result.")
        else:
            self.report_note.configure(text="No whole-image report yet. The service writes it after a learning run finishes.")
        order = ["unchanged", "v2", "v3_global", "v2_pretrained", "ceiling_curves", "ceiling_colour_map"]
        fill_table(self.models_table, [
            [MODEL_LABELS.get(name, name), percent(models[name].get("reproduced")), f"{models[name].get('mae', 0):.4f}", f"{models[name].get('delta_e', 0):.2f}", models[name].get("images")]
            for name in order if name in models
        ])
        editors = group.get("editors_change", {})
        rows = []
        for effect, target in editors.items():
            changes = {name: models.get(name, {}).get("effects", {}).get(effect, {}).get("change") for name in ("v2", "v3_global")}
            known = {name: value for name, value in changes.items() if value is not None}
            closer = min(known, key=lambda name: abs(known[name] - target)) if known else None
            rows.append([EFFECT_LABELS.get(effect, effect), f"{target:+.3f}",
                         *[("—" if changes[name] is None else f"{changes[name]:+.3f}") for name in ("v2", "v3_global")],
                         {"v2": "V2", "v3_global": "V3"}.get(closer, "")])
        fill_table(self.pattern_table, rows)
        history = [row for row in self.snapshot.get("report_history", []) if row.get("publication") == self.publication.get()]
        self.trend_chart.set_series([
            (label, COLOURS[name], [number(row["reproduced"]) for row in history if row.get("model") == name and number(row.get("reproduced")) is not None])
            for name, label in (("v2", "V2"), ("v3_global", "V3"))
        ], empty_text="The trend appears after the first reports")

    def _show_training(self) -> None:
        history = self.snapshot["history"]
        self.training_chart.set_series([
            (label, COLOURS[name], [number(row["validation_loss"]) for row in history[name][-150:] if number(row.get("validation_loss")) is not None])
            for name, label in (("v2", "V2"), ("v3_global", "V3"))
        ], empty_text="No training history yet")
        rows = []
        for name in ("v2", "v3_global"):
            best = float("inf")
            recent = history[name][-40:]
            for row in history[name][:-40]:
                best = min(best, number(row.get("validation_loss")) or best)
            for row in recent:
                value = number(row.get("validation_loss"))
                is_best = value is not None and value < best
                best = min(best, value) if value is not None else best
                rows.append((MODEL_LABELS[name], row.get("epoch"), f"{value:.5f}" if value is not None else "—",
                             f"{number(row.get('validation_pixel_loss')) or 0:.5f}", f"{number(row.get('validation_parameter_loss')) or 0:.5f}", "yes" if is_best else ""))
        fill_table(self.epochs_table, [list(row) for row in reversed(rows[-60:])])

    def _show_data(self) -> None:
        fill_table(self.inventory_table, [
            [row.get("set_name"), row.get("original_count"), row.get("corrected_count"), row.get("matched_count"),
             row.get("filename_and_visual_count"), row.get("visual_content_count"), row.get("ambiguous_visual_count"),
             row.get("unreadable_count"), inventory_note(row)]
            for row in self.snapshot["inventory"]
        ], empty="No image read recorded yet")
        fill_table(self.outlier_table, [
            [row.get("set_name"), row.get("key"), (row.get("outlier_reasons") or "").replace(";", ", ").replace("_", " "),
             f"{number(row.get('visual_similarity')) or 0:.2f}", f"{number(row.get('mae')) or 0:.3f}"]
            for row in self.snapshot["outliers"]
        ], empty="No outliers reported")

    def _show_logs(self) -> None:
        for name, key in (("Service log (today)", "service_log"), ("Training log (today, latest part)", "training_log")):
            text = self.log_views[name]
            at_end = text.yview()[1] > 0.98
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", self.snapshot[key] or "No log written today yet.")
            text.configure(state="disabled")
            if at_end:
                text.see("end")


if __name__ == "__main__":
    Dashboard().mainloop()
