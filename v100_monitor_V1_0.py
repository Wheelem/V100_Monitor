# Main script for the V100 monitor GUI.
#
# The application connects to a remote NVIDIA Tesla V100 via SSH, periodically
# running `nvidia-smi` to gather GPU metrics and processes.  The data is
# visualised with pyqtgraph in four real‑time plots:
#   * GPU utilisation (%)
#   * VRAM usage (MB)
#   * Temperature (°C)
#   * Power draw (W)
#
# Users can adjust refresh rate, time window, smoothing and theme.  A side
# panel shows the latest values and a table of compute processes.  Event logs
# record when a GPU becomes hot or recovers.
#
# The code is intentionally written with readability in mind; comments
# annotate the key parts of the program and clarify non‑obvious logic.
#
# ---------------------------------------------------------------------------
"""V100 Monitor v1.0 – a remote GPU monitoring GUI.

Copyright (c) 2026 Odiun

This project is licensed under the MIT license.  See the accompanying
``LICENSE`` file for details.

Version: 1.0.0
"""

__version__ = "1.0.0"

import sys
import time
import csv
from collections import deque
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTableWidget, QTableWidgetItem, QSlider, QCheckBox,
                             QComboBox, QFileDialog, QHeaderView, QPlainTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings, QTimer, QEvent
from PyQt6.QtGui import QColor
import pyqtgraph as pg
from PyQt6.QtWidgets import QTextEdit
import paramiko
import numpy as np

# --- Configuration & Constants -------------------------------------------------
# Default time window (in seconds) displayed on the x‑axis of the live plots.
DEFAULT_TIME_WINDOW = 60

# Maximum VRAM reported by a V100; increased automatically if the card reports more.
VRAM_DEFAULT_MAX_MB = 16384

# Power limit (W) for a V100; keeps a safety margin beyond the 250 W TDP.
POWER_MAX_W = 300

# Temperature limits used for visual warnings.
TEMP_MAX_C = 100
TEMP_THRESHOLD_C = 85  # V100 starts throttling near 85 °C.

# Marker used to split the combined `nvidia-smi` output into metrics and processes.
PROC_MARKER = "===PROCS==="

# Colors used for GPU curves (cycled if more than 7 GPUs).
GPU_COLORS = ['r', 'b', 'g', 'm', 'c', 'y', '#800080']

# Moving‑average window size when smoothing is enabled.
SMOOTH_WINDOW = 5

# Frame‑rate for the smooth scrolling animation.
ANIM_FPS = 30

# --- Threshold-aware coloring: (warn_at, hot_at) per metric; vram uses a ratio of total ---
ZONE_THRESHOLDS = {
    'gpu_util':  (80.0, 95.0),
    'temp':      (70.0, 85.0),
    'power':     (200.0, 250.0),
    'vram_used': (0.80, 0.95),   # fraction of vram_total
}
ZONE_COLORS = {'safe': '#2e7d32', 'warn': '#f9a825', 'hot': '#c62828'}

# Single GPU -> one color per plot, zone-colored by value.
# Multiple GPUs -> keep the classic per-GPU colors for identification.
SINGLE_GPU_ZONE = True

# --- Themes: selectable Light / Dark (persisted between sessions) ---
THEMES = {
    'light': dict(
        window='#f0f0f0', text='black', input_bg='white', border='#9e9e9e',
        grid='#c0c0c0', header_bg='#dcdcdc', plot_bg='w', minimap_bg='#fafafa',
        minimap_line='#888888', grid_alpha=0.3, title_fg='#374151',
    ),
    'dark': dict(
        window='#121212', text='#e0e0e0', input_bg='#1e1e1e', border='#4a4a4a',
    __version__ = "1.0.0"


        grid='#3c3c3c', header_bg='#2a2a2a', plot_bg='#181818', minimap_bg='#161616',
        minimap_line='#aaaaaa', grid_alpha=0.5, title_fg='#e0e0e0',
    ),
}

    if "--version" in sys.argv:
        print(__version__)
        sys.exit(0)



class MonitorThread(QThread):
    """Background thread that handles SSH communication.

    The thread connects to the remote host, retrieves a single ``nvidia-smi``
    snapshot, and emits the parsed data via ``data_received``.  It also
    implements exponential back‑off for transient connection failures.
    """
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, host, user, password, port, ssh_key, initial_refresh):
        """Create the thread.

        Parameters are passed directly from the GUI.  ``port`` may be a
        string, so we coerce it to ``int`` and fall back to the default
        SSH port (22) on failure.
        """
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        try:
            self.port = int(port)
        except (ValueError, TypeError):
            self.port = 22
        self.ssh_key = ssh_key
        self.refresh_rate = initial_refresh
        self.running = True

    def _sleep(self, secs):
        """Sleep for ``secs`` but wake early if the thread is stopped.

        The 0.2 s granularity keeps the UI responsive without spamming the
        kernel with syscalls.
        """
        end = time.time() + secs
        while self.running and time.time() < end:
            time.sleep(0.2)

    def _connect(self):
        """Create and return an authenticated SSH client.

        The host key policy automatically accepts new keys (useful for
        first‑time connections).  If a private key path is provided, the
        local agent is bypassed.
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(
            hostname=self.host,
            username=self.user,
            port=self.port,
            timeout=10,
            banner_timeout=15,
            auth_timeout=15,
        )
        if self.ssh_key:
            kwargs["key_filename"] = self.ssh_key
            kwargs["allow_agent"] = False
        if self.password:
            kwargs["password"] = self.password
        client.connect(**kwargs)
        return client

    def _static_info(self, client):
        """Retrieve static GPU metadata.

        This command is run once per connection and returns information
        such as GPU model, driver version and maximum clocks.  The output
        is parsed into a dictionary.
        """
        cmd = (
            "nvidia-smi --query-gpu=name,driver_version,"
            "clocks.max.mem,clocks.max.gr,power.limit,"
            "memory.total --format=csv,noheader,nounits"
        )
        try:
            stdin, stdout, stderr = client.exec_command(cmd)
            raw = stdout.read().decode(errors="replace")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if not lines:
                return {}
            p = [x.strip() for x in lines[0].split(',')]
            def fnum(s):
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None
            return {
                'name': p[0] if len(p) > 0 else '?',
                'driver': p[1] if len(p) > 1 else '?',
                'max_mem_clock_mhz': fnum(p[2]) if len(p) > 2 else None,
                'max_gr_clock_mhz': fnum(p[3]) if len(p) > 3 else None,
                'power_limit_w': fnum(p[4]) if len(p) > 4 else None,
                'mem_total_mb': fnum(p[5]) if len(p) > 5 else None,
            }
        except Exception:
            return {}

    def _collect(self, client):
        """Collect the live metrics and running processes.

        The command concatenates two ``nvidia-smi`` queries.  The
        ``PROC_MARKER`` line separates the GPU metrics from the process
        list so we can split the output reliably.
        """
        cmd = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits;"
            f" echo {PROC_MARKER};"
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits"
        )
        stdin, stdout, stderr = client.exec_command(cmd)
        raw = stdout.read().decode(errors="replace")

        lines = [ln.strip() for ln in raw.splitlines()]
        if PROC_MARKER in lines:
            mi = lines.index(PROC_MARKER)
            metric_lines, proc_lines = lines[:mi], lines[mi + 1:]
        else:
            metric_lines, proc_lines = lines, []

        gpus = []
        for ln in metric_lines:
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) >= 6:
                try:
                    gpus.append({
                        'gpu_util': float(parts[0]),
                        'vram_used': float(parts[1]),
                        'vram_total': float(parts[2]),
                        'temp': float(parts[3]),
                        'power': float(parts[4]),
                        'clock': float(parts[5]),
                    })
                except ValueError:
                    pass

        processes = []
        for ln in proc_lines:
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) >= 3:
                processes.append((parts[0], parts[1], parts[2]))

        def vram_key(p):
            try:
                return float(p[2])
            except ValueError:
                return -1.0
        processes.sort(key=vram_key, reverse=True)   # biggest VRAM consumer first
        return gpus, processes

    def run(self):
        """Main thread loop.

        The algorithm is:
        * Try to connect.  On failure emit ``error_occurred`` and retry
          with exponential back‑off.
        * On success, emit static GPU info and then repeatedly collect live
          metrics until ``self.running`` becomes ``False``.
        * Any exception during collection is reported and triggers a
          reconnection.
        """
        backoff = 2.0
        while self.running:
            try:
                client = self._connect()
            except Exception as e:
                self.error_occurred.emit(str(e))
                if not self.running:
                    break
                self._sleep(backoff)
                backoff = min(backoff * 2, 15.0)
                continue

            backoff = 2.0   # connection succeeded — reset backoff
            static_info = self._static_info(client)
            try:
                while self.running:
                    gpus, processes = self._collect(client)
                    self.data_received.emit(
                        {
                            't': time.time(),
                            'gpus': gpus,
                            'processes': processes,
                            'info': static_info,
                        }
                    )
                    self._sleep(self.refresh_rate / 1000.0)
            except Exception as e:   # connection dropped mid-session
                self.error_occurred.emit(str(e))
                if not self.running:
                    break
                self._sleep(backoff)
                backoff = min(backoff * 2, 15.0)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def stop(self):
        """Signal the thread to terminate and wait for it to finish."""
        self.running = False
        self.wait(5000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_title = "Remote NVIDIA Tesla V100 Monitor"
        self.setWindowTitle(self.base_title)
        self.resize(1280, 920)
        self.theme = 'light'   # set from saved settings before first paint

        self.settings = QSettings("Odiun", "V100Monitor")
        self.samples = deque(maxlen=DEFAULT_TIME_WINDOW * 10)   # rebuilt in update_settings
        self.window_seconds = DEFAULT_TIME_WINDOW
        self.refresh_ms = 1000
        self.paused = False
        self.smoothing_enabled = False
        self._hot_state = {}          # gpu index -> bool (for edge-triggered alerts)
        self._event_state = {}        # (metric,gpu) -> bool, for edge-triggered event log
        self.y_locked = [False] * 4   # per-plot: y auto-scale pinned by double-click
        self.live = True              # False while scrubbing back through history
        self._last_info = None       # last static info dict shown (avoid re-render)
        self.minimap_curves = []      # full-session lines in the overview strip
        self._flash_timer = None     # window-flash blink while a hot alert is active
        self.monitor_thread = None

        # --- LEFT SIDEBAR ---
        sidebar = QVBoxLayout()

        conn_group = QVBoxLayout()
        conn_group.addWidget(QLabel("<b>SSH Connection</b>"))
        self.host_input = QLineEdit("192.168.5.38")
        self.user_input = QLineEdit("odiun")
        self.port_input = QLineEdit("22")
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Optional SSH Key Path")

        conn_group.addWidget(QLabel("Host IP:"))
        conn_group.addWidget(self.host_input)
        conn_group.addWidget(QLabel("Username:"))
        conn_group.addWidget(self.user_input)
        conn_group.addWidget(QLabel("Port:"))
        conn_group.addWidget(self.port_input)
        conn_group.addWidget(QLabel("Password:"))
        conn_group.addWidget(self.pass_input)
        conn_group.addWidget(QLabel("SSH Key:"))
        conn_group.addWidget(self.key_input)

        self.connect_btn = QPushButton("Connect to V100")
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.connect_btn.clicked.connect(self.start_monitoring)
        conn_group.addWidget(self.connect_btn)
        sidebar.addLayout(conn_group)

        metrics_group = QVBoxLayout()
        metrics_group.addWidget(QLabel("<b>Current Metrics</b>"))
        self.status_label = QLabel("Status: Disconnected")
        metrics_group.addWidget(self.status_label)

        self.gpu_val = QLineEdit()
        self.vram_val = QLineEdit()
        metrics_group.addWidget(QLabel("GPU Core Usage:"))
        metrics_group.addWidget(self.gpu_val)
        metrics_group.addWidget(QLabel("VRAM Usage (MB):"))
        metrics_group.addWidget(self.vram_val)

        self.temp_label = QLabel("Temp: -- °C | Power: -- W | Clock: -- MHz")
        self.temp_label.setWordWrap(True)
        metrics_group.addWidget(self.temp_label)

        sidebar.addLayout(metrics_group)

        # Pause + Export buttons
        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause Plots")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.png_btn = QPushButton("Save PNG")
        self.png_btn.setToolTip("Save all four plots as one PNG image")
        self.png_btn.clicked.connect(self.export_png)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.png_btn)
        sidebar.addLayout(btn_row)

        self.info_label = QLabel("GPU Info: --")
        self.info_label.setWordWrap(True)
        sidebar.addWidget(self.info_label)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["PID", "Process", "VRAM (MB)"])
        self.table.setMinimumHeight(90)
        self.table.setMaximumHeight(120)   # ~3 rows of data
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 70)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 90)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        sidebar.addWidget(self.table)

        # --- Event Log (connection + alert events, newest first) ---
        ev_label = QLabel("<b>Event Log</b>")
        ev_label.setContentsMargins(0, 6, 0, 0)
        sidebar.addWidget(ev_label)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMinimumHeight(45)
        self.event_log.setMaximumHeight(75)
        self._events = []          # newest-first HTML lines, capped in _log_event
        sidebar.addWidget(self.event_log)

        # --- Controls Group ---
        controls_group = QVBoxLayout()

        refresh_row = QHBoxLayout()
        refresh_row.addWidget(QLabel("<b>Refresh Rate</b>"))
        self.refresh_val_label = QLabel("1000 ms")
        refresh_row.addWidget(self.refresh_val_label)
        refresh_row.addStretch(1)
        controls_group.addLayout(refresh_row)
        self.refresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.refresh_slider.setRange(100, 5000)
        self.refresh_slider.setValue(1000)
        self.refresh_slider.valueChanged.connect(self.update_settings)
        controls_group.addWidget(self.refresh_slider)

        self.smooth_check = QCheckBox("Smooth lines")
        self.smooth_check.setToolTip(
            "Apply a moving-average filter to soften the curves (visual only; readouts stay raw).")
        self.smooth_check.toggled.connect(self._toggle_smoothing)
        controls_group.addWidget(self.smooth_check)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("<b>Theme</b>"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['Light', 'Dark'])
        self.theme_combo.currentIndexChanged.connect(self._toggle_theme)
        theme_row.addWidget(self.theme_combo)
        controls_group.addLayout(theme_row)

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("<b>Time Window</b>"))
        self.window_val_label = QLabel("60 s")
        window_row.addWidget(self.window_val_label)
        window_row.addStretch(1)
        controls_group.addLayout(window_row)
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setRange(10, 300)
        self.window_slider.setValue(DEFAULT_TIME_WINDOW)
        self.window_slider.valueChanged.connect(self.update_settings)
        controls_group.addWidget(self.window_slider)

        sidebar.addLayout(controls_group)

        # --- Main Layout: sidebar | minimap+plots stacked vertically on the right ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.addLayout(sidebar, 1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(2)
        layout.addLayout(right_layout, 3)

        # --- Session minimap: full-history overview strip above the live plots. ---
        self.minimap = pg.PlotWidget()
        self.minimap.setBackground('#fafafa')
        self.minimap.setFixedHeight(70)
        self.minimap.setMouseEnabled(False)
        self.minimap.hideButtons()
        self.minimap.getPlotItem().hideAxis('left')
        self.minimap.getPlotItem().setAxisItems({'bottom': pg.DateAxisItem()})   # clock times, not epoch
        self.minimap.getPlotItem().setLabel('bottom', 'Session overview (full history)')
        right_layout.addWidget(self.minimap)

        # --- RIGHT SIDE (Plots in Rows) ---
        graph_layout = QVBoxLayout()
        graph_layout.setSpacing(2)
        titles = ["GPU Utilization (%)", "VRAM Usage (MB)", "Temperature (°C)", "Power Draw (W)"]
        self.plot_widgets = []
        self.curves = [[], [], [], []]     # per plot: one curve line per GPU
        self.legend_chips = [[], [], [], []]
        self.stats_labels = []

        self.plot_titles = list(titles)   # kept for theme-aware re-styling
        for i, title in enumerate(titles):
            pw = pg.PlotWidget(title=title)
            pw.setBackground('w')
            pw.setAntialiasing(True)
            pw.setDownsampling(True)
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setMouseEnabled(x=True, y=False)   # pan/zoom in time only; Y auto-fits
            pw.getPlotItem().setAxisItems({'bottom': pg.DateAxisItem()})   # real clock times on x-axis
            pw.installEventFilter(self)   # double-click to pin/unpin Y auto-scale

            # Y-axis auto-scales to fit the data on every plot.
            pw.enableAutoRange(axis='y')

            if i == 2:   # temperature plot keeps its throttle reference line
                pw.addLine(TEMP_THRESHOLD_C, pen=pg.mkPen('red', width=2,
                          style=Qt.PenStyle.DashLine),
                          label=f"throttle ~{TEMP_THRESHOLD_C}°C",
                          labelOpts={'color': 'red'})

            graph_layout.addWidget(pw)
            self.plot_widgets.append(pw)

            # Per-GPU color legend for this metric (one chip per GPU).
            leg_row = QHBoxLayout()
            self.legend_chips[i] = []
            for g in range(8):   # room for up to 8 GPUs; extras hidden as needed
                chip = QLabel(f"GPU {g}")
                chip.setStyleSheet("padding: 1px 6px; border-radius: 3px;")
                chip.hide()
                leg_row.addWidget(chip)
                self.legend_chips[i].append(chip)
            leg_row.addStretch(1)
            graph_layout.addLayout(leg_row)

            lbl = QLabel("min -- | avg -- | max --")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            graph_layout.addWidget(lbl)
            self.stats_labels.append(lbl)

        right_layout.addLayout(graph_layout)

        # Continuous scroll: advance the visible time window every tick so the
        # plots glide smoothly instead of jumping once per sample.
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(int(1000 / ANIM_FPS))
        self.anim_timer.timeout.connect(self._animate_scroll)
        self.anim_timer.start()

        # --- Restore saved settings (never the password) ---
        s = self.settings
        self.theme = s.value("theme", "light") if s.value("theme") in THEMES else "light"
        self.apply_theme()
        self.host_input.setText(s.value("host", "192.168.5.38"))
        self.user_input.setText(s.value("user", "odiun"))
        self.port_input.setText(s.value("port", "22"))
        self.key_input.setText(s.value("key", ""))
        self.refresh_slider.setValue(int(s.value("refresh", 1000)))
        self.window_slider.setValue(int(s.value("window", DEFAULT_TIME_WINDOW)))
        self.smooth_check.setChecked(s.value("smooth", "false").lower() == "true")

    # ------------------------------------------------------------------ theme
    def _toggle_theme(self, index):
        self.theme = 'dark' if index == 1 else 'light'
        self.apply_theme()

    def apply_theme(self):
        """Restyle the whole window + all plots for Light or Dark."""
        t = THEMES[self.theme]
        dark = (self.theme == 'dark')
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {t['window']}; font-family: Arial; }}
            QLabel      {{ color: {t['text']}; }}
            QLineEdit   {{ color: {t['text']}; background-color: {t['input_bg']}; border: 1px solid {t['border']}; padding: 4px; }}
            QComboBox   {{ color: {t['text']}; background-color: {t['input_bg']}; border: 1px solid {t['border']}; padding: 3px; }}
            QTableWidget{{ color: {t['text']}; background-color: {t['input_bg']}; gridline-color: {t['grid']}; }}
            QTextEdit   {{ color: {t['text']}; background-color: {t['input_bg']}; border: 1px solid {t['border']}; font-family: 'Courier New', monospace; font-size: 10pt; }}
            QHeaderView::section {{ background-color: {t['header_bg']}; color: {t['text']}; border: 1px solid {t['border']}; }}
        """V100 Monitor
        =================

        This module implements a lightweight GUI that polls a remote NVIDIA Tesla V100
        via ``nvidia‑smi`` over SSH and displays utilisation, VRAM, temperature and
        power consumption.  It is intended to be run as a standalone script but can
        also be imported as a module for further testing.

        Version: ``__version__``
        -----------------------
        Use ``python v100_monitor_V1_0.py --version`` to print the current
        release.
        """

        # Copyright (c) 2026, The V100 Monitor Authors
        #
        # SPDX-License-Identifier: MIT
        __version__ = "1.0.0"
        for i, pw in enumerate(self.plot_widgets):
            pw.setBackground(t['plot_bg'])
            ax = pw.getPlotItem().getAxis('bottom')
            ax.setPen(pg.mkPen(t['text'], width=1))
            ax.setTextPen(pg.mkPen(t['text']))
            ax.setTickFont(pg.QtGui.QFont("Arial", 8))
            pw.showGrid(x=True, y=True, alpha=t['grid_alpha'])
            pw.getPlotItem().setTitle(self.plot_titles[i], color=t['title_fg'], size='10pt')
        self.minimap.setBackground(t['minimap_bg'])

    # ------------------------------------------------------------------ helpers
    def _save_settings(self):
        s = self.settings
        s.setValue("host", self.host_input.text())
        s.setValue("user", self.user_input.text())
        s.setValue("port", self.port_input.text())
        s.setValue("key", self.key_input.text())
        s.setValue("theme", self.theme)
        s.setValue("refresh", self.refresh_slider.value())
        s.setValue("window", self.window_slider.value())
        s.setValue("smooth", str(self.smooth_check.isChecked()).lower())

    def update_settings(self, *args):
        self.refresh_ms = self.refresh_slider.value()
        self.window_seconds = self.window_slider.value()
        self.refresh_val_label.setText(f"{self.refresh_ms} ms")
        self.window_val_label.setText(f"{self.window_seconds} s")

        # Resize the history buffer without losing the newest samples.
        new_max = max(1, int(self.window_seconds * 1000 / self.refresh_ms)) + 5
        if self.samples.maxlen != new_max:
            self.samples = deque(list(self.samples), maxlen=new_max)

        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.refresh_rate = self.refresh_ms

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("Resume Plots" if self.paused else "Pause Plots")
        if not self.paused:
            self._redraw()

    # ------------------------------------------------------------------ connect
    def start_monitoring(self):
        self._save_settings()
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()

        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")

        self.monitor_thread = MonitorThread(
            self.host_input.text(), self.user_input.text(),
            self.pass_input.text(), self.port_input.text(),
            self.key_input.text().strip(), self.refresh_slider.value())
        self.monitor_thread.data_received.connect(self.update_ui)
        self.monitor_thread.error_occurred.connect(self.handle_error)
        self.monitor_thread.start()

    def handle_error(self, err):
        # No modal popups — just the status label; the thread auto-retries.
        self.status_label.setText(f"Status: Reconnecting… ({err})")
        self.status_label.setStyleSheet("color: red;")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect to V100")

    # ------------------------------------------------------------------ drawing
    def _ensure_curves(self, plot_idx, n_gpus):
        pw = self.plot_widgets[plot_idx]
        t = THEMES[self.theme]
        while len(self.curves[plot_idx]) < n_gpus:
            g = len(self.curves[plot_idx])
            col = GPU_COLORS[g % len(GPU_COLORS)]
            if self.theme == 'dark':
                # brighten per-GPU colors so they pop on the dark plot bg
                c = QColor(col)
                col = c.lighter(150).name()
            pen = pg.mkPen(col, width=2 if n_gpus == 1 else 1.5)
            self.curves[plot_idx].append(pw.plot(pen=pen))
        for c in self.curves[plot_idx][n_gpus:]:
            pw.removeItem(c)
        del self.curves[plot_idx][n_gpus:]

    def _zone_color(self, key, value, total=None):
        """Return (color, label) for a metric value based on its thresholds."""
        warn_at, hot_at = ZONE_THRESHOLDS[key]
        if key == 'vram_used' and total:
            ratio = value / max(total, 1.0)
            level = 'hot' if ratio >= hot_at else ('warn' if ratio >= warn_at else 'safe')
        else:
            level = 'hot' if value >= hot_at else ('warn' if value >= warn_at else 'safe')
        return ZONE_COLORS[level], level

    def _update_legend(self, i, n_gpus):
        chips = self.legend_chips[i]
        zone_mode = SINGLE_GPU_ZONE and n_gpus == 1
        for g in range(len(chips)):
            chip = chips[g]
            if g >= n_gpus:
                chip.hide()
                continue
            chip.show()
            if zone_mode:
                col = ZONE_COLORS['safe']
                tip = "GPU 0 — color tracks the value: green < amber < red"
            else:
                col = QColor(GPU_COLORS[g % len(GPU_COLORS)]).name()
                tip = f"GPU {g}"
            chip.setText(f"GPU {g}")
            chip.setStyleSheet(
                f"background-color:{col}; color:white; padding:1px 6px; border-radius:3px;")
            chip.setToolTip(tip)

    def _redraw(self):
        now = time.time()
        t0 = now - self.window_seconds
        rows = [s for s in self.samples if s['t'] >= t0]
        n_gpus = max((len(s['gpus']) for s in rows), default=0)

        keys = ['gpu_util', 'vram_used', 'temp', 'power']
        zone_mode = SINGLE_GPU_ZONE and n_gpus == 1
        for i, key in enumerate(keys):
            self._ensure_curves(i, n_gpus)
            all_vals = []
            for g in range(n_gpus):
                xs, ys, totals = [], [], []
                for s in rows:
                    if g < len(s['gpus']):
                        gl = s['gpus'][g]
                        xs.append(s['t'])
                        ys.append(gl[key])
                        totals.append(gl.get('vram_total', 0))
                # Optional visual smoothing (moving average); stats stay on raw values.
                disp = self._smooth_ys(ys)
                curve = self.curves[i][g]

                if zone_mode:
                    # Color the line by the latest value's zone (idle / working / hot).
                    col, _lvl = self._zone_color(key, ys[-1] if ys else 0,
                                                 totals[-1] if totals else None)
                    curve.setData(xs, disp, pen=pg.mkPen(col, width=2))
                else:
                    curve.setData(xs, disp)   # per-GPU color set in _ensure_curves

                all_vals.extend(ys)
            self._update_legend(i, n_gpus)
            if all_vals:
                self.stats_labels[i].setText(
                    f"min {min(all_vals):.1f} | avg {sum(all_vals)/len(all_vals):.1f} | max {max(all_vals):.1f}")

        # X window always advances; Y auto-scales unless the user pinned it
        # for this plot with a double-click (self.y_locked[i]).
        # Re-fit Y every sample; the 30 fps scroll timer only advances X,
        # so a pinned axis (double-click) stays frozen between samples.
        for i, pw in enumerate(self.plot_widgets):
            if rows:
                pw.setXRange(t0, now, padding=0)
            # Manually adjust the Y‑range based on the current data instead of
            # relying on pyqtgraph’s auto‑range each sample.  This prevents
            # jittering when a single spike would temporarily change the
            # bounds.  When the plot is pinned (y_locked[i] == True) we keep
            # the range fixed.
            if not self.y_locked[i]:
                all_y = []
                for curve in self.curves[i]:
                    if curve is not None:
                        data = curve.getData()
                        if data and len(data) == 2:
                            all_y.extend(data[1])
                if all_y:
                    min_y, max_y = min(all_y), max(all_y)
                    # add a small padding for visual breathing room
                    pw.setYRange(min_y, max_y, padding=0.05)
                else:
                    # No data yet – keep the default range
                    pw.enableAutoRange(axis='y')
            else:
                pw.disableAutoRange(axis='y')

        # --- Session minimap: one line per metric across the whole history. ---
        if rows and len(self.minimap_curves) < 4:
            while len(self.minimap_curves) < 4:
                self.minimap_curves.append(
                    self.minimap.plot(pen=pg.mkPen(THEMES[self.theme]['minimap_line'], width=1)))
        for i, key in enumerate(keys):
            mxs = [s['t'] for s in self.samples]
            mys = [max((g[key] for g in s['gpus']), default=0) for s in self.samples]
            if mxs:
                self.minimap_curves[i].setData(mxs, mys)
        if self.samples:
            self.minimap.setXRange(self.samples[0]['t'], now, padding=0)

    def _smooth_ys(self, ys):
        """Centered moving-average filter used only for display when smoothing is on."""
        if not self.smoothing_enabled or len(ys) < 3:
            return ys
        half = SMOOTH_WINDOW // 2
        n = len(ys)
        return [sum(ys[max(0, i - half):min(n, i + half + 1)]) /
                (min(n, i + half + 1) - max(0, i - half)) for i in range(n)]

    def _toggle_smoothing(self, on):
        self.smoothing_enabled = bool(on)
        # re-color zone lines in case smoothing was off when the theme applied
        for i, pw in enumerate(self.plot_widgets):
            for c in self.curves[i]:
                if not self.smoothing_enabled and SINGLE_GPU_ZONE:
                    key = ['gpu_util', 'vram_used', 'temp', 'power'][i]
                    vals = [g[key] for s in self.samples for g in s['gpus'][:1]]
                    col, _ = self._zone_color(key, vals[-1] if vals else 0)
                    c.setPen(pg.mkPen(col, width=2))
        self._redraw()

    def _animate_scroll(self):
        """Slide the time window continuously so scrolling looks smooth regardless
        of how often new samples arrive. Frozen while paused."""
        if self.paused or not self.samples:
            return
        now = time.time()
        t0 = now - self.window_seconds
        for pw in self.plot_widgets:
            pw.setXRange(t0, now, padding=0)

    def update_ui(self, data):
        t = data['t']
        gpus = data['gpus']
        processes = data['processes']

        # --- static hardware/driver facts (shown once per connection) ---
        info = data.get('info') or {}
        if info and info != getattr(self, '_last_info', None):
            self._last_info = info
            parts = []
            if info.get('name'):
                parts.append(f"Model: {info['name']}")
            if info.get('driver'):
                parts.append(f"Driver: {info['driver']}")
            if info.get('max_mem_clock_mhz'):
                parts.append(f"Max Mem Clock: {info['max_mem_clock_mhz']:.0f} MHz")
            if info.get('max_gr_clock_mhz'):
                parts.append(f"Max GPU Clock: {info['max_gr_clock_mhz']:.0f} MHz")
            if info.get('voltage_v'):
                parts.append(f"Max Voltage: {info['voltage_v']:.3f} V")
            if info.get('power_limit_w'):
                parts.append(f"Power Limit: {info['power_limit_w']:.0f} W")
            self.info_label.setText(("GPU Info:\n" + "\n".join(parts)) if parts else "GPU Info: --")

        # --- history + plots (frozen while paused) ---
        if gpus and not self.paused:
            self.samples.append({'t': t, 'gpus': gpus})
            self._redraw()

        # --- sidebar current values (always live) ---
        if gpus:
            g0 = gpus[0]
            self.gpu_val.setText(f"{g0['gpu_util']:.0f} %")
            self.vram_val.setText(f"{g0['vram_used']:.0f} / {g0['vram_total']:.0f}")

            parts = [f"GPU{i}: {g['temp']:.0f}°C | {g['power']:.1f} W | {g['clock']:.0f} MHz"
                     for i, g in enumerate(gpus)]
            self.temp_label.setText("  •  ".join(parts))

            # Compute hot state (simple check — no crash risk).
            any_hot = False
            for i, g in enumerate(gpus):
                hot = g['temp'] >= TEMP_THRESHOLD_C
                self._hot_state[i] = hot
                any_hot = any_hot or hot
            self._hot_state = {k: v for k, v in self._hot_state.items() if k < len(gpus)}

            # --- status + button (always update, even if alerts fail) ---
            if any_hot:
                self.setWindowTitle(f"🔥 HIGH TEMP — {self.base_title}")
                self.status_label.setText("Status: Connected — ⚠ HIGH TEMPERATURE")
                self.status_label.setStyleSheet("color: #ff9800;")
            else:
                self.setWindowTitle(self.base_title)
                self.status_label.setText("Status: Connected")
                self.status_label.setStyleSheet("color: green;")
            self.connect_btn.setText("Connected")

        # --- process table (sorted by VRAM desc, done in the thread) ---
        self.table.setRowCount(0)
        for pid, name, vram in processes:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(pid))
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(name)          # full path on hover
            self.table.setItem(row, 1, name_item)
            self.table.setItem(row, 2, QTableWidgetItem(vram))

        # --- edge-triggered alerts (never block UI) ---
        try:
            for i, g in enumerate(gpus):
                hot = g['temp'] >= TEMP_THRESHOLD_C
                was = getattr(self, '_hot_prev', {}).get(i, False)
                if hot and not was:
                    QApplication.beep()
                    self._log_event('⚠', 'hot', f"GPU{i} temp {g['temp']:.0f}°C ≥ throttle {TEMP_THRESHOLD_C}°C")
                elif not hot and was:
                    self._log_event('✓', 'safe', f"GPU{i} temp back to normal ({g['temp']:.0f}°C)")
            if hasattr(self, '_hot_prev'):
                self._hot_prev = {k: v for k, v in self._hot_state.items()}
            else:
                self._hot_prev = dict(self._hot_state)

            for i, g in enumerate(gpus):
                ratio = g['vram_used'] / max(g['vram_total'], 1.0)
                v_hot = ratio >= ZONE_THRESHOLDS['vram_used'][1]
                was_v = self._event_state.get(('vram', i), False)
                if v_hot and not was_v:
                    self._log_event('⚠', 'hot',
                                    f"GPU{i} VRAM {ratio*100:.0f}% of total ({g['vram_used']:.0f}/{g['vram_total']:.0f} MB)")
                elif not v_hot and was_v:
                    self._log_event('✓', 'safe', f"GPU{i} VRAM pressure eased ({ratio*100:.0f}%)")
                self._event_state[('vram', i)] = v_hot
        except Exception as e:
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ export
    def export_csv(self):
        rows = list(self.samples)
        if not rows:
            self.status_label.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export history", "v100_history.csv", "CSV files (*.csv)")
        if not path:
            return
        n_gpus = max(len(s['gpus']) for s in rows)
        keys = ['gpu_util', 'vram_used', 'vram_total', 'temp', 'power', 'clock']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['timestamp']
            for g in range(n_gpus):
                header += [f"gpu{g}_{k}" for k in keys]
            w.writerow(header)
            for s in rows:
                row = [datetime.fromtimestamp(s['t']).isoformat(timespec='seconds')]
                for g in range(n_gpus):
                    if g < len(s['gpus']):
                        row += [s['gpus'][g][k] for k in keys]
                    else:
                        row += [''] * len(keys)
                w.writerow(row)
        self.status_label.setText(f"Exported {len(rows)} samples to CSV")

    # ------------------------------------------------------------------ events
    def _log_event(self, icon, level, msg):
        """Append a timestamped line to the event log (kept newest-first)."""
        ts = datetime.now().strftime('%H:%M:%S')
        color = ZONE_COLORS.get(level, '#000')
        self._events.insert(0, f'<div style="color:{color}">[{ts}] {icon} {msg}</div>')
        del self._events[200:]
        self.event_log.setHtml(''.join(self._events))

    def eventFilter(self, obj, event):
        """Double-click a plot to pin/unpin its Y auto-scale."""
        if event.type() == QEvent.Type.MouseButtonDblClick:
            for i, pw in enumerate(self.plot_widgets):
                if obj is pw:
                    self.y_locked[i] = not self.y_locked[i]
                    if self.y_locked[i]:
                        pw.enableAutoRange(axis='y')   # one last fit, then freeze
                        self.status_label.setText(f"Y-axis pinned on plot {i+1} (double-click to release)")
                    else:
                        pw.enableAutoRange(axis='y')
                        self.status_label.setText("Status: Connected")
                    return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ export
    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plots as image", "v100_plots.png", "PNG files (*.png)")
        if not path:
            return
        try:
            from PyQt6.QtGui import QImage
            # Composite the four live plots into one tall image.
            imgs = [pw.grab() for pw in self.plot_widgets]
            w = max(im.width() for im in imgs)
            h = sum(im.height() for im in imgs) + 3 * (w // 400)
            canvas = QImage(w, h, QImage.Format.Format_ARGB32)
            canvas.fill(QColor(THEMES[self.theme]['plot_bg']))
            painter = pg.QtGui.QPainter(canvas)
            y = 0
            for im in imgs:
                painter.drawImage(0, y, im)
                y += im.height() + w // 400
            painter.end()
            canvas.save(path)
            self.status_label.setText(f"Saved plots to PNG")
        except Exception as e:
            self.status_label.setText(f"PNG save failed: {e}")

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, e):
        self._save_settings()
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent look regardless of environment
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
