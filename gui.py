import argparse
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from api import CozmoInterface
from helpers import *
from map_wander import BOARD_WIDTH_MM, BOARD_HEIGHT_MM, MapWanderController
from marker_search import MarkerSearchController
from audio import (
    get_audio_status,
    init_audio_controller,
    is_audio_playing,
    play_audio as play_speech_audio,
    set_volume as set_speech_volume,
    stop_audio as stop_speech_audio,
    toggle_audio_enabled,
    toggle_mute as toggle_speech_mute,
)


class CozmoGui:
    MAP_SIZE_MM = 500.0
    TRAJECTORY_FADE_S = 8.0
    UI_UPDATE_MS = 50
    ICON_BASE_HEADING_DEG = 0.0
    TELEOP_MAX_MMPS = 120.0
    TELEOP_TURN_SCALE = 0.9
    GO_TO_KP_TURN = 4.0
    GO_TO_ANGLE_DEADBAND_RAD = 0.05
    GO_TO_DISTANCE_STOP_MM = 25.0
    TARGET_DOT_RADIUS_PX = 2
    TARGET_ANIM_START_RADIUS_PX = 8
    TARGET_ANIM_DURATION_S = 0.35
    WINDOW_WIDTH_RATIO = 0.95
    MAP_WIDTH_RATIO = 0.35
    CAMERA_WIDTH_RATIO = 0.3
    WINDOW_HEIGHT_EXTRA_PX = 180

    MODE_TELEOP = "teleop"
    MODE_ROUTINE = "routine"
    MODE_CUBE_SEARCH = "cube_search"
    MODE_MAP_WANDER = "map_wander"
    MODE_IDLE = "idle"
    MODES = (MODE_TELEOP, MODE_ROUTINE, MODE_CUBE_SEARCH, MODE_MAP_WANDER, MODE_IDLE)

    MAP_WANDER_MAP_FILE = "map.png"   # white=ground, black=obstacle, red dot=start

    CANVAS_SIZE_PX = 500

    def __init__(self, root: tk.Tk, cozmo: CozmoInterface):
        self.root = root
        self.cozmo = cozmo

        self.state = RobotState()
        self.mode = self.MODE_IDLE
        self.routine = None
        self.cube_search = None
        self.map_wander = None

        self._map_bg_photo = None    # cached PhotoImage of the board map PNG
        self._map_bg_size_px = 0     # canvas size at which it was rendered
        self._board_map_pil = None   # PIL image loaded at startup for display
        self._board_start_bx_mm = 2.0 * 25.4   # mm from left — overridden by red dot
        self._board_start_by_mm = 2.0 * 25.4   # mm from top  — overridden by red dot

        self.last_tick_s = time.perf_counter()
        self.trajectory = []
        self.last_drawn_pose = None
        self.keys_down = set()
        self.teleop_target_mm = None
        self._last_active_target_mm = None
        self._target_anim_center_mm = None
        self._target_anim_start_s = None
        self._camera_photo = None
        self._configure_window_geometry()

        self.data_dir = Path(__file__).resolve().parent / "data"
        self.robot_icon_path = self.data_dir / "robot_icon.png"

        self.status_mode = tk.StringVar(value=self.mode)
        self.status_pose = tk.StringVar(value="x=0.0, y=0.0, theta=0.0")
        self.status_wheels = tk.StringVar(value="L=0.0 mm/s, R=0.0 mm/s")
        self.status_target = tk.StringVar(value="target=none")
        self.status_routine = tk.StringVar(value="routine=inactive")
        self.status_cube_search = tk.StringVar(value="cube search=inactive")
        self.status_map_wander = tk.StringVar(value="map wander=inactive")
        self.status_conn = tk.StringVar(value=self._connection_status_text())
        self.status_audio = tk.StringVar(value="audio: init")

        self.map_size_var = tk.DoubleVar(value=self.MAP_SIZE_MM)
        self.teleop_speed_var = tk.DoubleVar(value=self.TELEOP_MAX_MMPS)
        self.routine_update_ms_var = tk.IntVar(value=RoutineController.UPDATE_INTERVAL_MS)
        self.routine_dist_stop_var = tk.DoubleVar(value=RoutineController.DISTANCE_TO_STOP_MM)
        self.routine_goal_dist_var = tk.DoubleVar(value=RoutineController.NEW_GOAL_MAX_DISTANCE_MM)
        self.routine_vmax_var = tk.DoubleVar(value=RoutineController.V_MAX_MMPS)
        self.routine_return_ms_var = tk.IntVar(value=RoutineController.TIME_BEFORE_GOING_BACK_MS)
        self.go_to_kp_var = tk.DoubleVar(value=self.GO_TO_KP_TURN)
        self.go_to_angle_deadband_var = tk.DoubleVar(value=self.GO_TO_ANGLE_DEADBAND_RAD)
        self.go_to_distance_stop_var = tk.DoubleVar(value=self.GO_TO_DISTANCE_STOP_MM)

        self.audio_files = []
        self.audio_var = tk.StringVar(value="")
        self.audio_volume_var = tk.IntVar(value=60000)

        self._build_ui()
        self._load_audio_files()
        self._load_robot_icon()
        self._load_board_map_image()
        init_audio_controller(self.cozmo.cli, data_dir=self.data_dir)
        set_speech_volume(self.audio_volume_var.get())
        self.cozmo.enable_camera_stream(color=True)

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.set_mode(self.MODE_IDLE, speak=False)
        self.root.after(self.UI_UPDATE_MS, self._tick)

    def _connection_status_text(self):
        if self.cozmo.test_mode and self.cozmo.connection_error:
            return f"test_mode=True; robot unavailable: {self.cozmo.connection_error}"
        return f"test_mode={self.cozmo.test_mode}"

    def _build_ui(self):
        self.root.title("Cozmo Controller")

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main.columnconfigure(0, weight=0, minsize=self.map_panel_width_px)
        main.columnconfigure(1, weight=0, minsize=self.camera_panel_width_px)
        main.columnconfigure(2, weight=0, minsize=self.sidebar_width_px)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            main,
            width=self.map_panel_size_px,
            height=self.map_panel_size_px,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.canvas.bind("<Button-1>", self._on_map_click)

        camera_frame = ttk.Frame(main)
        camera_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        camera_frame.grid_propagate(False)
        camera_frame.configure(width=self.camera_panel_width_px, height=self.main_panel_height_px)
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(1, weight=1)

        ttk.Label(camera_frame, text="Camera Feed").grid(row=0, column=0, sticky="w")
        self.camera_label = ttk.Label(camera_frame, text="No camera frame yet", anchor="center")
        self.camera_label.grid(row=1, column=0, sticky="nsew")

        side_outer = ttk.Frame(main, width=self.sidebar_width_px, height=self.main_panel_height_px)
        side_outer.grid(row=0, column=2, sticky="nsew")
        side_outer.grid_propagate(False)
        side_outer.columnconfigure(0, weight=1)
        side_outer.rowconfigure(0, weight=1)

        self.sidebar_canvas = tk.Canvas(side_outer, highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar = ttk.Scrollbar(side_outer, orient="vertical", command=self.sidebar_canvas.yview)
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        side = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window_id = self.sidebar_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", self._on_sidebar_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        side.columnconfigure(0, weight=1)
        side.columnconfigure(1, weight=1)

        button_style = {
            "font": ("Segoe UI", 9),
            "fg": "#111111",
            "bg": "#e6e6e6",
            "activeforeground": "#111111",
            "activebackground": "#d9d9d9",
            "relief": "raised",
            "bd": 1,
            "highlightthickness": 0,
        }

        row = 0
        ttk.Label(side, text="Mode").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        mode_buttons = tk.Frame(side)
        mode_buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
        mode_buttons.columnconfigure(0, weight=1)
        mode_buttons.columnconfigure(1, weight=1)
        mode_buttons.columnconfigure(2, weight=1)
        mode_buttons.columnconfigure(3, weight=1)
        tk.Button(
            mode_buttons, text="Map Wander", command=lambda: self.set_mode(self.MODE_MAP_WANDER), **button_style
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 2))
        tk.Button(
            mode_buttons, text="Find Cube", command=lambda: self.set_mode(self.MODE_CUBE_SEARCH), **button_style
        ).grid(row=0, column=2, sticky="ew", padx=2)
        tk.Button(
            mode_buttons, text="Idle", command=lambda: self.set_mode(self.MODE_IDLE), **button_style
        ).grid(row=0, column=3, sticky="ew", padx=(2, 0))
        tk.Button(
            mode_buttons, text="Teleop", command=lambda: self.set_mode(self.MODE_TELEOP), **button_style
        ).grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(2, 0))
        tk.Button(
            mode_buttons, text="Routine", command=lambda: self.set_mode(self.MODE_ROUTINE), **button_style
        ).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(2, 0), pady=(2, 0))
        row += 1

        ttk.Separator(side).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(side, text="Status").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(side, textvariable=self.status_mode, font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_pose, font=("Segoe UI", 9)).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(side, textvariable=self.status_wheels, font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_target, font=("Segoe UI", 9)).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(side, textvariable=self.status_routine, font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_conn, font=("Segoe UI", 9)).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(side, textvariable=self.status_cube_search, font=("Segoe UI", 9)).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(side, textvariable=self.status_map_wander, font=("Segoe UI", 9)).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        ttk.Label(side, textvariable=self.status_audio, font=("Segoe UI", 9)).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Separator(side).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(side, text="Commands").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        commands_left = (
            "W/A/S/D: Drive\n"
            "Mouse Click: Go-to"
        )
        commands_right = (
            "Q/E: Head\n"
            "R/F: Lift\n"
            "P: Play/Stop speech\n"
            "O: Mute speech"
        )
        tk.Message(
            side,
            text=commands_left,
            justify="left",
            width=max(80, (self.sidebar_width_px // 2) - 16),
            anchor="w",
            fg="#111111",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w")
        tk.Message(
            side,
            text=commands_right,
            justify="left",
            width=max(80, (self.sidebar_width_px // 2) - 16),
            anchor="w",
            fg="#111111",
            font=("Segoe UI", 9),
        ).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Separator(side).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(side, text="Audio").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.audio_menu = ttk.OptionMenu(side, self.audio_var, "")
        self.audio_menu.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        audio_buttons = tk.Frame(side)
        audio_buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
        audio_buttons.columnconfigure(0, weight=1)
        audio_buttons.columnconfigure(1, weight=1)
        audio_buttons.columnconfigure(2, weight=1)
        tk.Button(
            audio_buttons, text="Play", command=self._play_selected_audio, **button_style
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tk.Button(
            audio_buttons, text="Toggle", command=self._toggle_audio_enabled, **button_style
        ).grid(row=0, column=1, sticky="ew", padx=2)
        tk.Button(
            audio_buttons, text="Mute", command=self._toggle_mute, **button_style
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        row += 1
        ttk.Label(side, text="Volume").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Scale(
            side,
            from_=0,
            to=65535,
            orient="horizontal",
            variable=self.audio_volume_var,
            command=self._on_volume_change,
        ).grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Separator(side).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(side, text="Settings").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        settings = ttk.Frame(side)
        settings.grid(row=row, column=0, columnspan=2, sticky="ew")
        settings.columnconfigure(0, weight=0)
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(2, weight=0)
        settings.columnconfigure(3, weight=1)

        settings_items = [
            ("Map Size (mm)", self.map_size_var),
            ("Teleop Max (mm/s)", self.teleop_speed_var),
            ("Go-to Kp", self.go_to_kp_var),
            ("Angle Deadband (rad)", self.go_to_angle_deadband_var),
            ("Distance Stop (mm)", self.go_to_distance_stop_var),
            ("Routine dt (ms)", self.routine_update_ms_var),
            ("Routine Stop (mm)", self.routine_dist_stop_var),
            ("Routine Goal Max (mm)", self.routine_goal_dist_var),
            ("Routine Vmax (mm/s)", self.routine_vmax_var),
            ("Routine Return (ms)", self.routine_return_ms_var),
        ]
        srow = 0
        for i in range(0, len(settings_items), 2):
            left_label, left_var = settings_items[i]
            self._add_setting(settings, srow, left_label, left_var, base_col=0)
            if i + 1 < len(settings_items):
                right_label, right_var = settings_items[i + 1]
                self._add_setting(settings, srow, right_label, right_var, base_col=2)
            srow += 1

        tk.Button(side, text="Apply Settings", command=self._apply_settings, **button_style).grid(
            row=row + 1, column=0, columnspan=2, sticky="ew"
        )

    def _add_setting(self, parent, row, label, variable, base_col=0):
        ttk.Label(parent, text=label).grid(row=row, column=base_col, sticky="w", padx=(0, 6), pady=1)
        ttk.Entry(parent, textvariable=variable, width=9).grid(row=row, column=base_col + 1, sticky="ew", pady=1)

    def _load_robot_icon(self):
        self.robot_icon_original = None
        self.robot_icon_tk = None
        if self.robot_icon_path.exists():
            self.robot_icon_original = Image.open(self.robot_icon_path).convert("RGBA")

    def _load_audio_files(self):
        self.audio_files = self.cozmo.list_audio_files()
        menu = self.audio_menu["menu"]
        menu.delete(0, "end")

        if self.audio_files:
            self.audio_var.set(self.audio_files[0])
            for item in self.audio_files:
                menu.add_command(label=item, command=lambda x=item: self.audio_var.set(x))
        else:
            self.audio_var.set("")

    def _apply_settings(self):
        self.MAP_SIZE_MM = max(100.0, float(self.map_size_var.get()))
        self.TELEOP_MAX_MMPS = max(10.0, float(self.teleop_speed_var.get()))
        self.GO_TO_KP_TURN = max(0.1, float(self.go_to_kp_var.get()))
        self.GO_TO_ANGLE_DEADBAND_RAD = max(0.001, float(self.go_to_angle_deadband_var.get()))
        self.GO_TO_DISTANCE_STOP_MM = max(1.0, float(self.go_to_distance_stop_var.get()))

        RoutineController.UPDATE_INTERVAL_MS = max(20, int(self.routine_update_ms_var.get()))
        RoutineController.DISTANCE_TO_STOP_MM = max(5.0, float(self.routine_dist_stop_var.get()))
        RoutineController.NEW_GOAL_MAX_DISTANCE_MM = max(10.0, float(self.routine_goal_dist_var.get()))
        RoutineController.V_MAX_MMPS = max(10.0, float(self.routine_vmax_var.get()))
        RoutineController.TIME_BEFORE_GOING_BACK_MS = max(1000, int(self.routine_return_ms_var.get()))

        if self.mode == self.MODE_ROUTINE:
            self.routine = RoutineController(self.state, self.cozmo.max_wheel_speed_mmps, self.cozmo.track_width_mm)

    def _on_sidebar_frame_configure(self, _event):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event):
        self.sidebar_canvas.itemconfigure(self.sidebar_window_id, width=event.width)

    def set_mode(self, mode: str, speak: bool = True):
        print(f"[GUI] set_mode: {self.mode} → {mode}")
        self.mode = mode
        self.routine = None
        self.cube_search = None
        self.map_wander = None

        if mode == self.MODE_MAP_WANDER:
            map_path = str(self.data_dir / self.MAP_WANDER_MAP_FILE)
            self.map_wander = MapWanderController(
                map_path=map_path,
                max_wheel_mmps=self.cozmo.max_wheel_speed_mmps,
                track_width_mm=self.cozmo.track_width_mm,
            )
            self.teleop_target_mm = None
            if speak:
                play_speech_audio(SpeechCategory.NEW_COMMAND)
        elif mode == self.MODE_ROUTINE:
            self.routine = RoutineController(self.state, self.cozmo.max_wheel_speed_mmps, self.cozmo.track_width_mm)
            self.teleop_target_mm = None
            if speak:
                play_speech_audio(SpeechCategory.NEW_COMMAND)
        elif mode == self.MODE_CUBE_SEARCH:
            try:
                self.cube_search = MarkerSearchController(
                    max_wheel_mmps=self.cozmo.max_wheel_speed_mmps,
                    track_width_mm=self.cozmo.track_width_mm,
                    on_lift_up=self.cozmo.set_lift_up,
                    on_lift_down=self.cozmo.set_lift_down,
                    get_cliff_detected=self.cozmo.get_cliff_detected,
                    get_lift_height_mm=self.cozmo.get_lift_height_mm,
                    on_head_level=lambda: self.cozmo.set_head_angle_abs(-0.3),
                    on_head_approach=self.cozmo.set_head_min,
                )
            except RuntimeError as exc:
                self.mode = self.MODE_IDLE
                self.cube_search = None
                self.cozmo.stop()
                self.status_cube_search.set(f"cube search=error: {exc}")
                return
            self.teleop_target_mm = None
            if speak:
                play_speech_audio(SpeechCategory.NEW_COMMAND)
        elif mode == self.MODE_IDLE:
            self.teleop_target_mm = None
            self.cozmo.stop()
            if speak:
                play_speech_audio(SpeechCategory.IDLE_COMMENTS)
        elif speak:
            play_speech_audio(SpeechCategory.NEW_COMMAND)

    def _on_key_press(self, event):
        key = event.keysym.lower()
        self.keys_down.add(key)
        if key == "p":
            self._play_selected_audio()
        elif key == "o":
            self._toggle_mute()
        elif key == "q":
            self.cozmo.set_head_up()
        elif key == "e":
            self.cozmo.set_head_down()
        elif key == "r":
            self.cozmo.set_lift_up()
        elif key == "f":
            self.cozmo.set_lift_down()

    def _on_key_release(self, event):
        self.keys_down.discard(event.keysym.lower())

    def _on_map_click(self, event):
        if self.mode == self.MODE_TELEOP:
            self.teleop_target_mm = self._canvas_to_world(event.x, event.y)
            self._start_target_animation(self.teleop_target_mm)
            play_speech_audio(SpeechCategory.NEW_COMMAND)

    def _play_selected_audio(self):
        if is_audio_playing():
            ok, msg = stop_speech_audio()
            self.status_conn.set(msg if ok else f"audio error: {msg}")
            return

        line = self.audio_var.get() if self.audio_var.get() else None
        ok, msg = play_speech_audio(SpeechCategory.NEW_COMMAND, always_play=True, line_override=line)
        self.status_conn.set(msg if ok else f"audio error: {msg}")

    def _toggle_mute(self):
        ok, msg = toggle_speech_mute()
        self.status_conn.set(msg if ok else f"audio error: {msg}")

    def _toggle_audio_enabled(self):
        ok, msg = toggle_audio_enabled()
        self.status_conn.set(msg if ok else f"audio error: {msg}")

    def _on_volume_change(self, _value):
        ok, msg = set_speech_volume(self.audio_volume_var.get())
        if not ok:
            self.status_conn.set(f"audio error: {msg}")

    def _tick(self):
        now_s = time.perf_counter()
        dt_s = now_s - self.last_tick_s
        self.last_tick_s = now_s

        left_mmps, right_mmps = self.cozmo.get_wheel_speeds()
        self.state.update_from_wheels(left_mmps, right_mmps, dt_s)

        self._append_trajectory(now_s)
        self._run_mode(now_s)
        self._refresh_status()
        self._draw_map(now_s)
        self._draw_camera()

        self.root.after(self.UI_UPDATE_MS, self._tick)

    def _append_trajectory(self, now_s):
        x_mm, y_mm = self.state.x_mm, self.state.y_mm
        if self.last_drawn_pose is None:
            self.trajectory.append((now_s, x_mm, y_mm))
            self.last_drawn_pose = (x_mm, y_mm)
            return

        lx, ly = self.last_drawn_pose
        if ((x_mm - lx) ** 2 + (y_mm - ly) ** 2) ** 0.5 >= 3.0:
            self.trajectory.append((now_s, x_mm, y_mm))
            self.last_drawn_pose = (x_mm, y_mm)

        min_time = now_s - self.TRAJECTORY_FADE_S
        self.trajectory = [p for p in self.trajectory if p[0] >= min_time]

    def _run_mode(self, now_s):
        if self.mode == self.MODE_IDLE:
            self.cozmo.drive_wheels(0.0, 0.0)
            return

        if self.mode == self.MODE_MAP_WANDER:
            if self.map_wander is None:
                self.map_wander = MapWanderController(
                    map_path=str(self.data_dir / self.MAP_WANDER_MAP_FILE),
                    max_wheel_mmps=self.cozmo.max_wheel_speed_mmps,
                    track_width_mm=self.cozmo.track_width_mm,
                )
            cmd = self.map_wander.update(
                now_s, self.state, image=self.cozmo.latest_camera_image
            )
            if self.map_wander.found_marker_id is not None:
                print(f"[GUI] AprilTag {self.map_wander.found_marker_id} detected during wander → switching to cube search")
                self.set_mode(self.MODE_CUBE_SEARCH)
                return
            self.cozmo.drive_wheels(cmd[0], cmd[1])
            return

        if self.mode == self.MODE_ROUTINE:
            if self.routine is None:
                self.routine = RoutineController(self.state, self.cozmo.max_wheel_speed_mmps, self.cozmo.track_width_mm)
            cmd = self.routine.update(now_s)
            if self.routine.last_event == "new_goal":
                play_speech_audio(SpeechCategory.NEW_GOAL)
            elif self.routine.last_event == "returning_home":
                play_speech_audio(SpeechCategory.RETURNING_HOME)
            if cmd is not None:
                self.cozmo.drive_wheels(cmd[0], cmd[1])
            return

        if self.mode == self.MODE_CUBE_SEARCH:
            if self.cube_search is None:
                try:
                    self.cube_search = MarkerSearchController(
                        max_wheel_mmps=self.cozmo.max_wheel_speed_mmps,
                        track_width_mm=self.cozmo.track_width_mm,
                        on_lift_up=self.cozmo.set_lift_up,
                        on_lift_down=self.cozmo.set_lift_down,
                        get_cliff_detected=self.cozmo.get_cliff_detected,
                    )
                except RuntimeError as exc:
                    self.status_cube_search.set(f"cube search=error: {exc}")
                    self.set_mode(self.MODE_IDLE, speak=False)
                    return
            try:
                cmd = self.cube_search.update(now_s, self.cozmo.latest_camera_image, self.state)
            except RuntimeError as exc:
                self.status_cube_search.set(f"cube search=error: {exc}")
                self.set_mode(self.MODE_IDLE, speak=False)
                return
            if cmd is not None:
                self.cozmo.drive_wheels(cmd[0], cmd[1])
            return

        manual_cmd = self._teleop_manual_command()
        if manual_cmd is not None:
            self.teleop_target_mm = None
            self.cozmo.drive_wheels(manual_cmd[0], manual_cmd[1])
            if "s" in self.keys_down and "w" not in self.keys_down:
                play_speech_audio(SpeechCategory.GOING_BACK)
            elif "a" in self.keys_down and "d" not in self.keys_down:
                play_speech_audio(SpeechCategory.GOING_LEFT)
            elif "d" in self.keys_down and "a" not in self.keys_down:
                play_speech_audio(SpeechCategory.GOING_RIGHT)
            else:
                play_speech_audio(SpeechCategory.GOING_STRAIGHT)
            return

        if self.teleop_target_mm is not None:
            tx, ty = self.teleop_target_mm
            dist_mm = ((tx - self.state.x_mm) ** 2 + (ty - self.state.y_mm) ** 2) ** 0.5
            if dist_mm <= self.GO_TO_DISTANCE_STOP_MM:
                self.teleop_target_mm = None
                self.cozmo.drive_wheels(0.0, 0.0)
                return

            omega_max = 0.6 * self.cozmo.max_wheel_speed_mmps * 2.0 / self.cozmo.track_width_mm
            left_mmps, right_mmps = calculate_wheel_speeds_for_point(
                x_mm=self.state.x_mm,
                y_mm=self.state.y_mm,
                theta_rad=self.state.theta_rad,
                target_x_mm=tx,
                target_y_mm=ty,
                max_wheel_mmps=self.cozmo.max_wheel_speed_mmps,
                track_width_mm=self.cozmo.track_width_mm,
                v_max_mmps=self.TELEOP_MAX_MMPS,
                kp_turn=self.GO_TO_KP_TURN,
                angle_deadband_rad=self.GO_TO_ANGLE_DEADBAND_RAD,
                omega_max_radps=omega_max,
            )
            self.cozmo.drive_wheels(left_mmps, right_mmps)
            play_speech_audio(SpeechCategory.GOING_STRAIGHT)
            return

        self.cozmo.drive_wheels(0.0, 0.0)
        play_speech_audio(SpeechCategory.IDLE_COMMENTS)

    def _teleop_manual_command(self):
        max_speed = self.TELEOP_MAX_MMPS
        turn_speed = max_speed * self.TELEOP_TURN_SCALE

        forward = "w" in self.keys_down
        backward = "s" in self.keys_down
        left = "a" in self.keys_down
        right = "d" in self.keys_down

        if not any((forward, backward, left, right)):
            return None

        if forward and not backward:
            base = max_speed
        elif backward and not forward:
            base = -max_speed
        else:
            base = 0.0

        turn = 0.0
        if left and not right:
            turn = turn_speed
        elif right and not left:
            turn = -turn_speed

        left_mmps = base - turn
        right_mmps = base + turn

        clamp = self.cozmo.max_wheel_speed_mmps
        left_mmps = max(-clamp, min(clamp, left_mmps))
        right_mmps = max(-clamp, min(clamp, right_mmps))
        return left_mmps, right_mmps

    def _board_to_canvas(self, bx_mm: float, by_mm: float):
        """Board-frame mm → canvas pixel for map_wander display (y-down)."""
        map_size_px, ox, oy = self._get_map_geometry()
        scale = min(map_size_px / BOARD_WIDTH_MM, map_size_px / BOARD_HEIGHT_MM)
        disp_w = BOARD_WIDTH_MM * scale
        disp_h = BOARD_HEIGHT_MM * scale
        x_off = ox + (map_size_px - disp_w) / 2.0
        y_off = oy + (map_size_px - disp_h) / 2.0
        return x_off + bx_mm * scale, y_off + by_mm * scale

    def _load_board_map_image(self) -> None:
        """Load map.png at startup for canvas display and red-dot start detection."""
        map_path = self.data_dir / self.MAP_WANDER_MAP_FILE
        if not map_path.exists():
            print(f"[GUI] Map file not found: {map_path}")
            return
        try:
            import numpy as np
            img = Image.open(map_path).convert("RGB")
            self._board_map_pil = img
            arr = np.array(img, dtype=np.uint8)
            h, w = arr.shape[:2]
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            red_mask = (r >= 235) & (g <= 20) & (b <= 20)
            if red_mask.any():
                ys, xs = np.where(red_mask)
                self._board_start_bx_mm = float(xs.mean()) / w * BOARD_WIDTH_MM
                self._board_start_by_mm = float(ys.mean()) / h * BOARD_HEIGHT_MM
                print(f"[GUI] Map loaded: {w}×{h}px  red-dot start=({self._board_start_bx_mm:.1f}, {self._board_start_by_mm:.1f}) mm")
            else:
                print(f"[GUI] Map loaded: {w}×{h}px  no red dot found, using default start ({self._board_start_bx_mm:.1f}, {self._board_start_by_mm:.1f}) mm")
            free_pct = float((arr.max(axis=2) > 200).mean()) * 100.0
            print(f"[GUI] Map free area: {free_pct:.1f}%")
        except Exception as exc:
            print(f"[GUI] Map load error: {exc}")

    def _refresh_map_bg(self, map_size_px: int) -> None:
        """Cache the map raster scaled to current canvas size."""
        if self._map_bg_size_px == map_size_px and self._map_bg_photo is not None:
            return
        # Prefer the active controller's image (it may have been processed differently)
        img = None
        if self.map_wander is not None:
            img = self.map_wander.raster_image_pil
        if img is None:
            img = self._board_map_pil
        if img is None:
            self._map_bg_photo = None
            return
        scale = min(map_size_px / img.width, map_size_px / img.height)
        new_w = max(1, int(img.width  * scale))
        new_h = max(1, int(img.height * scale))
        self._map_bg_photo = ImageTk.PhotoImage(
            img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
        )
        self._map_bg_size_px = map_size_px

    def _draw_map(self, now_s):
        self.canvas.delete("all")

        map_size_px, ox, oy = self._get_map_geometry()
        if map_size_px < 8:
            return

        # ── PNG obstacle map background (always shown) ───────────────────────
        self._refresh_map_bg(map_size_px)
        scale = min(map_size_px / BOARD_WIDTH_MM, map_size_px / BOARD_HEIGHT_MM)
        disp_w = BOARD_WIDTH_MM * scale
        disp_h = BOARD_HEIGHT_MM * scale
        x_off = ox + (map_size_px - disp_w) / 2.0
        y_off = oy + (map_size_px - disp_h) / 2.0

        if self._map_bg_photo is not None:
            self.canvas.create_image(
                int(x_off + disp_w / 2), int(y_off + disp_h / 2),
                image=self._map_bg_photo, anchor="center",
            )
        else:
            # No map file — draw a plain white board outline
            self.canvas.create_rectangle(
                x_off, y_off, x_off + disp_w, y_off + disp_h,
                fill="white", outline="#888",
            )

        # Board boundary
        self.canvas.create_rectangle(
            x_off, y_off, x_off + disp_w, y_off + disp_h,
            outline="#333", width=2,
        )

        # ── Trajectory ────────────────────────────────────────────────────────
        for i in range(1, len(self.trajectory)):
            _, x0, y0 = self.trajectory[i - 1]
            t1, x1, y1 = self.trajectory[i]
            age = now_s - t1
            alpha = max(0.0, 1.0 - age / self.TRAJECTORY_FADE_S)
            color = self._fade_color(alpha)
            c0 = self._world_to_canvas(x0, y0)
            c1 = self._world_to_canvas(x1, y1)
            self.canvas.create_line(c0[0], c0[1], c1[0], c1[1], fill=color, width=2)

        # ── Active target dot ─────────────────────────────────────────────────
        active_target_mm = self._get_active_target_mm()
        if active_target_mm is not None and not self._target_equal(active_target_mm, self._last_active_target_mm):
            self._start_target_animation(active_target_mm)
        self._last_active_target_mm = active_target_mm

        if active_target_mm is not None:
            tx, ty = self._world_to_canvas(*active_target_mm)
            r = self.TARGET_DOT_RADIUS_PX
            self.canvas.create_oval(tx - r, ty - r, tx + r, ty + r, fill="#ff8c1a", outline="#ff8c1a")
            self._draw_target_animation(now_s)

        # ── Robot icon ────────────────────────────────────────────────────────
        rx, ry = self._world_to_canvas(self.state.x_mm, self.state.y_mm)
        self._draw_robot_icon(rx, ry, self.state.theta_rad)

    def _draw_robot_icon(self, x, y, theta_rad):
        if self.robot_icon_original is not None:
            icon = self.robot_icon_original.resize((30, 30), resample=Image.Resampling.BILINEAR)
            icon = icon.rotate(theta_rad * 180.0 / 3.141592653589793 + self.ICON_BASE_HEADING_DEG, expand=True)
            self.robot_icon_tk = ImageTk.PhotoImage(icon)
            self.canvas.create_image(x, y, image=self.robot_icon_tk)
        else:
            size = 12
            self.canvas.create_oval(x - size, y - size, x + size, y + size, fill="#7ad1ff", outline="#ffffff")

        # Heading indicator in map coordinates (small front dot).
        heading_len = 18
        dot_r = 1
        end_x = x + heading_len * math.cos(theta_rad)
        end_y = y - heading_len * math.sin(theta_rad)
        self.canvas.create_oval(
            end_x - dot_r,
            end_y - dot_r,
            end_x + dot_r,
            end_y + dot_r,
            fill="#ffffff",
            outline="#ffffff",
        )

    def _draw_camera(self):
        image = self.cozmo.latest_camera_image
        if self.mode == self.MODE_CUBE_SEARCH and self.cube_search is not None:
            image = self.cube_search.annotated_image or image
        if image is None:
            if self.cozmo.test_mode:
                self.camera_label.configure(text="Test mode: camera unavailable", image="")
            else:
                self.camera_label.configure(text="Waiting for camera frames...", image="")
            return

        width = max(2, self.camera_label.winfo_width())
        height = max(2, self.camera_label.winfo_height())
        scale = min(width / image.width, height / image.height)
        out_w = max(1, int(image.width * scale))
        out_h = max(1, int(image.height * scale))
        frame = image.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
        self._camera_photo = ImageTk.PhotoImage(frame)
        self.camera_label.configure(image=self._camera_photo, text="")

    def _draw_target_animation(self, now_s: float):
        if self._target_anim_center_mm is None or self._target_anim_start_s is None:
            return
        age_s = now_s - self._target_anim_start_s
        if age_s < 0:
            return
        if age_s > self.TARGET_ANIM_DURATION_S:
            self._target_anim_center_mm = None
            self._target_anim_start_s = None
            return

        p = age_s / self.TARGET_ANIM_DURATION_S
        radius = self.TARGET_ANIM_START_RADIUS_PX - (
            self.TARGET_ANIM_START_RADIUS_PX - self.TARGET_DOT_RADIUS_PX
        ) * p
        cx, cy = self._world_to_canvas(*self._target_anim_center_mm)
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#ff8c1a",
            width=1,
        )

    def _start_target_animation(self, target_mm):
        if target_mm is None:
            return
        self._target_anim_center_mm = (target_mm[0], target_mm[1])
        self._target_anim_start_s = time.perf_counter()

    def _get_active_target_mm(self):
        if self.mode == self.MODE_MAP_WANDER and self.map_wander is not None:
            goal = self.map_wander.current_goal_board
            if goal is not None:
                return self.map_wander.board_to_state(*goal)
        if self.mode == self.MODE_ROUTINE and self.routine is not None:
            return self.routine.goal_xy_mm
        if self.teleop_target_mm is not None:
            return self.teleop_target_mm
        return None

    @staticmethod
    def _target_equal(a, b):
        if a is None or b is None:
            return a is b
        return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

    def _refresh_status(self):
        self.status_mode.set(f"mode={self.mode}")
        self.status_pose.set(f"x={self.state.x_mm:.1f} mm, y={self.state.y_mm:.1f} mm, theta={self.state.theta_rad:.2f} rad")
        self.status_wheels.set(
            f"L={self.state.left_wheel_mmps:.1f} mm/s, R={self.state.right_wheel_mmps:.1f} mm/s"
        )
        if self.mode == self.MODE_ROUTINE and self.routine is not None:
            gx, gy = self.routine.goal_xy_mm
            self.status_target.set(f"target=({gx:.1f}, {gy:.1f}) mm")
            elapsed_ms = (time.perf_counter() - self.routine.start_time_s) * 1000.0
            if self.routine.going_back:
                self.status_routine.set("routine=returning")
            else:
                left_s = max(0.0, (RoutineController.TIME_BEFORE_GOING_BACK_MS - elapsed_ms) / 1000.0)
                self.status_routine.set(f"routine time left={left_s:.1f}s")
        elif self.teleop_target_mm is not None:
            tx, ty = self.teleop_target_mm
            self.status_target.set(f"target=({tx:.1f}, {ty:.1f}) mm")
            self.status_routine.set("routine=inactive")
        else:
            self.status_target.set("target=none")
            self.status_routine.set("routine=inactive")

        if self.mode == self.MODE_CUBE_SEARCH and self.cube_search is not None:
            self.status_cube_search.set(self.cube_search.status_text)
        elif not self.status_cube_search.get().startswith("cube search=error"):
            self.status_cube_search.set("cube search=inactive")

        if self.mode == self.MODE_MAP_WANDER and self.map_wander is not None:
            self.status_map_wander.set(self.map_wander.status_text)
        else:
            self.status_map_wander.set("map wander=inactive")

        audio_state = get_audio_status()
        enabled = "on" if audio_state["enabled"] else "off"
        muted = "muted" if audio_state["muted"] else "live"
        if audio_state["is_playing"]:
            self.status_audio.set(
                f"audio: {enabled}, {muted}, vol={audio_state['volume']}, playing {audio_state['remaining_s']:.1f}s"
            )
        else:
            self.status_audio.set(f"audio: {enabled}, {muted}, vol={audio_state['volume']}, idle")

    def _world_to_canvas(self, x_mm: float, y_mm: float):
        """State-frame mm → canvas pixel via board frame."""
        bx = self._board_start_bx_mm + x_mm
        by = self._board_start_by_mm - y_mm
        return self._board_to_canvas(bx, by)

    def _canvas_to_world(self, x_px: float, y_px: float):
        """Canvas pixel → state-frame mm via board frame."""
        map_size_px, ox, oy = self._get_map_geometry()
        if map_size_px <= 0:
            return 0.0, 0.0
        scale = min(map_size_px / BOARD_WIDTH_MM, map_size_px / BOARD_HEIGHT_MM)
        x_off = ox + (map_size_px - BOARD_WIDTH_MM * scale) / 2.0
        y_off = oy + (map_size_px - BOARD_HEIGHT_MM * scale) / 2.0
        bx = (x_px - x_off) / scale
        by = (y_px - y_off) / scale
        return bx - self._board_start_bx_mm, self._board_start_by_mm - by

    def _get_map_geometry(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        map_size = min(width, height)
        ox = (width - map_size) / 2.0
        oy = (height - map_size) / 2.0
        return map_size, ox, oy

    @staticmethod
    def _fade_color(alpha: float):
        alpha = max(0.0, min(1.0, alpha))
        # dark blue → light grey on white background
        r0, g0, b0 = (30, 80, 220)
        r1, g1, b1 = (200, 200, 200)
        r = int(r1 + (r0 - r1) * alpha)
        g = int(g1 + (g0 - g1) * alpha)
        b = int(b1 + (b0 - b1) * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_close(self):
        self.cozmo.stop()
        self.root.destroy()

    def _configure_window_geometry(self):
        screen_w = self.root.winfo_screenwidth()
        total_w = int(screen_w * self.WINDOW_WIDTH_RATIO)
        self.map_panel_width_px = int(total_w * self.MAP_WIDTH_RATIO)
        self.camera_panel_width_px = int(total_w * self.CAMERA_WIDTH_RATIO)
        self.sidebar_width_px = max(240, total_w - self.map_panel_width_px - self.camera_panel_width_px)
        self.map_panel_size_px = self.map_panel_width_px
        self.main_panel_height_px = max(self.map_panel_size_px, int(self.camera_panel_width_px * 0.75))

        outer_w = self.map_panel_width_px + self.camera_panel_width_px + self.sidebar_width_px + 40
        outer_h = self.main_panel_height_px + self.WINDOW_HEIGHT_EXTRA_PX
        self.root.geometry(f"{outer_w}x{outer_h}")
        self.root.resizable(False, False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use toy client instead of robot")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with CozmoInterface(test_mode=args.test) as cozmo:
            root = tk.Tk()
            app = CozmoGui(root=root, cozmo=cozmo)
            del app
            root.mainloop()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
