"""
Map-based random wander controller.

Physical space:  30 × 20 inches exactly (BOARD_WIDTH_MM × BOARD_HEIGHT_MM).
Map file:        map.svg
  white pixels  → walkable ground
  black pixels  → obstacle
  red dot       → robot starting position (R≥235, G≤20, B≤20, pure red ±20)

The SVG is rasterized and stretched to fill the 30×20 inch space exactly
(MAP_RASTER_W × MAP_RASTER_H pixels, 1 px ≈ 1 mm).

Coordinate frames
-----------------
Board frame  origin = board top-left   x → right   y ↓ down   (image axes)
State frame  origin = robot start pos  x → right   y ↑ up     (odometry / math)

board_x = start_board_x + state_x
board_y = start_board_y - state_y
"""

import io
import math
import random
from typing import Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PilImage
except ImportError:
    PilImage = None

from helpers import RobotState, calculate_wheel_speeds_for_point

# ── Physical constants ────────────────────────────────────────────────────────
MM_PER_INCH     = 25.4
BOARD_WIDTH_MM  = 30.0 * MM_PER_INCH   # 762.0 mm
BOARD_HEIGHT_MM = 20.0 * MM_PER_INCH   # 508.0 mm

# ── Rasterisation (1 px ≈ 1 mm, stretched to board dims) ─────────────────────
MAP_RASTER_W = 762
MAP_RASTER_H = 508

# ── Red-dot start-position detection ─────────────────────────────────────────
RED_R_MIN = 235   # pure red: high R
RED_G_MAX = 20    #           low G
RED_B_MAX = 20    #           low B

# ── Obstacle safety margin ────────────────────────────────────────────────────
OBSTACLE_INFLATE_PX = 4

# ── Goal selection ────────────────────────────────────────────────────────────
MIN_GOAL_DIST_MM  = 150.0  # new goal must be at least this far from current pos
MAX_GOAL_ATTEMPTS = 150    # random samples before giving up and spinning

# ── Stuck / recovery ──────────────────────────────────────────────────────────
STUCK_TIMEOUT_S   = 6.0
STUCK_PROGRESS_MM = 15.0

# ── Motion ────────────────────────────────────────────────────────────────────
GOAL_REACHED_MM    = 60.0
V_MAX_MMPS         = 80.0
KP_TURN            = 3.0
ANGLE_DEADBAND_RAD = 0.05
OMEGA_MAX_SCALE    = 0.6
SPIN_MMPS          = 30.0


def _rasterize_svg(path: str, w: int, h: int):
    """Render an SVG to an RGB PIL Image stretched to (w × h) pixels."""
    if PilImage is None:
        return None
    # cairosvg — most reliable
    try:
        import cairosvg
        data = cairosvg.svg2png(url=str(path), output_width=w, output_height=h)
        return PilImage.open(io.BytesIO(data)).convert("RGB")
    except ImportError:
        pass
    # svglib fallback
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(path))
        if drawing is None:
            return None
        data = renderPM.drawToString(drawing, fmt="PNG")
        img = PilImage.open(io.BytesIO(data)).convert("RGB")
        return img.resize((w, h), PilImage.Resampling.LANCZOS)
    except ImportError:
        pass
    return None


class MapWanderController:
    """Random-goal wander over a rasterised SVG obstacle map."""

    def __init__(self, map_path: str, max_wheel_mmps: float, track_width_mm: float):
        self.max_wheel_mmps = max_wheel_mmps
        self.track_width_mm = track_width_mm
        self.omega_max = OMEGA_MAX_SCALE * max_wheel_mmps * 2.0 / track_width_mm
        self.status_text = "map wander=loading"

        self.map_w_px = MAP_RASTER_W
        self.map_h_px = MAP_RASTER_H
        self._safe = None       # np.ndarray bool: True = safe pixel
        self._raster_rgb = None  # np.ndarray (H,W,3) uint8 for GUI

        # Starting position in board frame (derived from red dot)
        self.start_bx_mm: float = 2.0 * MM_PER_INCH
        self.start_by_mm: float = 2.0 * MM_PER_INCH

        self._load_map(map_path)

        # Navigation state
        self._goal_state: Optional[Tuple[float, float]] = None  # state-frame goal
        self._last_progress_s: float = 0.0
        self._last_check_pos: Tuple[float, float] = (0.0, 0.0)
        self._spin_dir: float = 1.0
        self._spinning: bool = False

        # Marker detection (lazy import to avoid hard dependency)
        self._detector = None
        self.found_marker_id: Optional[int] = None
        try:
            from marker_search import Cv2MarkerDetector
            self._detector = Cv2MarkerDetector()
        except Exception:
            pass

    # ── map loading ───────────────────────────────────────────────────────────

    def _load_map(self, path: str) -> None:
        if np is None:
            self.status_text = "map wander=missing numpy"
            return

        img = _rasterize_svg(path, MAP_RASTER_W, MAP_RASTER_H)
        if img is None:
            self.status_text = "map wander=SVG load failed (install cairosvg or svglib)"
            return

        arr = np.array(img, dtype=np.uint8)   # (H, W, 3) RGB
        self._raster_rgb = arr

        # Detect start position: centroid of pure-red pixels
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        red_mask = (r >= RED_R_MIN) & (g <= RED_G_MAX) & (b <= RED_B_MAX)
        if red_mask.any():
            ys, xs = np.where(red_mask)
            self.start_bx_mm = float(xs.mean()) / MAP_RASTER_W * BOARD_WIDTH_MM
            self.start_by_mm = float(ys.mean()) / MAP_RASTER_H * BOARD_HEIGHT_MM

        # Free map: white (all channels bright) or red start marker
        free = arr.max(axis=2) > 200

        # Inflate obstacles for clearance
        try:
            import cv2
            k = OBSTACLE_INFLATE_PX * 2 + 1
            kernel = np.ones((k, k), np.uint8)
            self._safe = cv2.dilate((~free).astype(np.uint8), kernel) == 0
        except ImportError:
            self._safe = free.copy()

        self.status_text = (
            f"map wander=ready  start=({self.start_bx_mm:.0f}, {self.start_by_mm:.0f}) mm"
        )

    # ── coordinate helpers ────────────────────────────────────────────────────

    def state_to_board(self, sx: float, sy: float) -> Tuple[float, float]:
        return self.start_bx_mm + sx, self.start_by_mm - sy

    def board_to_state(self, bx: float, by: float) -> Tuple[float, float]:
        return bx - self.start_bx_mm, self.start_by_mm - by

    def board_to_px(self, bx_mm: float, by_mm: float) -> Tuple[int, int]:
        col = int(bx_mm / BOARD_WIDTH_MM * self.map_w_px)
        row = int(by_mm / BOARD_HEIGHT_MM * self.map_h_px)
        return (max(0, min(self.map_w_px - 1, col)),
                max(0, min(self.map_h_px - 1, row)))

    def is_safe_board(self, bx_mm: float, by_mm: float) -> bool:
        if self._safe is None:
            return True
        if not (0.0 <= bx_mm < BOARD_WIDTH_MM and 0.0 <= by_mm < BOARD_HEIGHT_MM):
            return False
        col, row = self.board_to_px(bx_mm, by_mm)
        return bool(self._safe[row, col])

    # ── goal picking ──────────────────────────────────────────────────────────

    def _pick_goal(self, sx: float, sy: float) -> Optional[Tuple[float, float]]:
        """Return a random safe state-frame goal ≥ MIN_GOAL_DIST_MM away."""
        bx0, by0 = self.state_to_board(sx, sy)
        for _ in range(MAX_GOAL_ATTEMPTS):
            bx = random.uniform(0.0, BOARD_WIDTH_MM)
            by = random.uniform(0.0, BOARD_HEIGHT_MM)
            if not self.is_safe_board(bx, by):
                continue
            if math.hypot(bx - bx0, by - by0) < MIN_GOAL_DIST_MM:
                continue
            return self.board_to_state(bx, by)
        return None

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        now_s: float,
        robot_state: RobotState,
        image=None,
    ) -> Tuple[float, float]:
        sx, sy = robot_state.x_mm, robot_state.y_mm

        # Check for AprilTags while wandering
        if (image is not None and self._detector is not None
                and self.found_marker_id is None):
            try:
                from marker_search import HOME_MARKER_ID
                dets, _, _ = self._detector.detect(image, target_marker_id=None)
                for det in dets:
                    if det.marker_id != HOME_MARKER_ID:
                        self.found_marker_id = det.marker_id
                        self.status_text = f"map wander=marker {det.marker_id} spotted!"
                        return (0.0, 0.0)
            except Exception:
                pass

        # Pick new goal if needed
        if self._goal_state is None and not self._spinning:
            g = self._pick_goal(sx, sy)
            if g is not None:
                self._goal_state = g
                self._last_progress_s = now_s
                self._last_check_pos = (sx, sy)
                bx, by = self.state_to_board(*g)
                self.status_text = f"map wander=new goal ({bx:.0f}, {by:.0f}) mm"
            else:
                self._spinning = True
                self._spin_dir = random.choice([-1.0, 1.0])

        if self._spinning:
            # Try to find a goal each tick while spinning
            g = self._pick_goal(sx, sy)
            if g is not None:
                self._goal_state = g
                self._spinning = False
                self._last_progress_s = now_s
                self._last_check_pos = (sx, sy)
            else:
                self.status_text = "map wander=searching (spin)"
                t = SPIN_MMPS * self._spin_dir
                return (t, -t)

        gx, gy = self._goal_state
        dist = math.hypot(gx - sx, gy - sy)

        if dist <= GOAL_REACHED_MM:
            self._goal_state = None
            self.status_text = "map wander=reached, picking next goal"
            return (0.0, 0.0)

        # Stuck detection
        if (now_s - self._last_progress_s) >= STUCK_TIMEOUT_S:
            progress = math.hypot(sx - self._last_check_pos[0], sy - self._last_check_pos[1])
            self._last_progress_s = now_s
            self._last_check_pos = (sx, sy)
            if progress < STUCK_PROGRESS_MM:
                self._goal_state = None
                self._spin_dir = random.choice([-1.0, 1.0])
                self._spinning = True
                self.status_text = "map wander=stuck, new goal"
                return (SPIN_MMPS * self._spin_dir, -SPIN_MMPS * self._spin_dir)

        bx_g, by_g = self.state_to_board(gx, gy)
        self.status_text = f"map wander=→({bx_g:.0f}, {by_g:.0f}) mm  d={dist:.0f}"

        return calculate_wheel_speeds_for_point(
            x_mm=sx, y_mm=sy, theta_rad=robot_state.theta_rad,
            target_x_mm=gx, target_y_mm=gy,
            max_wheel_mmps=self.max_wheel_mmps,
            track_width_mm=self.track_width_mm,
            v_max_mmps=V_MAX_MMPS,
            kp_turn=KP_TURN,
            angle_deadband_rad=ANGLE_DEADBAND_RAD,
            omega_max_radps=self.omega_max,
        )

    @property
    def current_goal_board(self) -> Optional[Tuple[float, float]]:
        """Current goal in board-frame mm, or None."""
        if self._goal_state is None:
            return None
        return self.state_to_board(*self._goal_state)

    @property
    def raster_image_pil(self):
        """The rasterised map as a PIL Image (for GUI display), or None."""
        if self._raster_rgb is None or PilImage is None:
            return None
        return PilImage.fromarray(self._raster_rgb)
