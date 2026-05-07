"""
Map-based coverage wander for Cozmo.

Physical space:  30 × 20 inches (BOARD_WIDTH_MM × BOARD_HEIGHT_MM)
Map PNG:         white pixels (grey > 128) = traversable; black = obstacle
                 top-left pixel = board top-left corner
Starting pose:   2 × 2 inches from top-left, facing east (theta = 0)

Coordinate frames
-----------------
Board frame  origin = board top-left   x → right   y ↓ down   (image axes)
State frame  origin = robot start pos  x → right   y ↑ up     (odometry / math)

board_x = START_X_MM + state_x
board_y = START_Y_MM - state_y
"""

import heapq
import math
import random
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PilImage
except ImportError:
    PilImage = None

from helpers import RobotState, calculate_wheel_speeds_for_point

# ── Physical constants ───────────────────────────────────────────────────────
MM_PER_INCH     = 25.4
BOARD_WIDTH_MM  = 30.0 * MM_PER_INCH   # 762.0 mm
BOARD_HEIGHT_MM = 20.0 * MM_PER_INCH   # 508.0 mm
START_X_MM      =  2.0 * MM_PER_INCH   # 50.8 mm from left edge
START_Y_MM      =  2.0 * MM_PER_INCH   # 50.8 mm from top edge

# ── Map processing ───────────────────────────────────────────────────────────
MAP_FREE_THRESHOLD  = 128  # grey value above which a pixel is considered free
OBSTACLE_INFLATE_PX = 4    # safety margin around obstacles (pixels)

# ── Coverage ─────────────────────────────────────────────────────────────────
COVERAGE_CELL_MM  = 60.0  # one coverage grid cell side length
VISIT_RADIUS_MM   = 50.0  # robot marks cells within this radius as visited
GOAL_REACHED_MM   = 55.0  # switch to next path waypoint when this close
PATH_STEP_PX      = 6     # keep every N-th A* pixel as a waypoint
MAX_PLAN_ATTEMPTS = 12    # goal candidates to try before giving up

# ── Stuck detection ───────────────────────────────────────────────────────────
STUCK_TIMEOUT_S   = 6.0   # seconds of low travel → declare stuck
STUCK_PROGRESS_MM = 15.0  # travel below this in STUCK_TIMEOUT_S → stuck

# ── Motion ───────────────────────────────────────────────────────────────────
V_MAX_MMPS         = 80.0
KP_TURN            = 3.0
ANGLE_DEADBAND_RAD = 0.05
OMEGA_MAX_SCALE    = 0.6
SPIN_MMPS          = 30.0  # turn-in-place speed while searching for a new goal


class MapWanderController:
    """Coverage wander driven by a PNG obstacle map and dead-reckoning odometry."""

    def __init__(
        self,
        map_path: str,
        max_wheel_mmps: float,
        track_width_mm: float,
    ):
        self.max_wheel_mmps = max_wheel_mmps
        self.track_width_mm = track_width_mm
        self.omega_max = OMEGA_MAX_SCALE * max_wheel_mmps * 2.0 / track_width_mm
        self.status_text = "map wander=starting"

        self.map_w_px = 0
        self.map_h_px = 0
        self._free: Optional["np.ndarray"] = None   # bool, True = traversable pixel
        self._safe: Optional["np.ndarray"] = None   # bool, True = safe after inflation
        self._load_map(map_path)

        # Coverage grid over board frame
        self._n_cols = max(1, int(math.ceil(BOARD_WIDTH_MM  / COVERAGE_CELL_MM)))
        self._n_rows = max(1, int(math.ceil(BOARD_HEIGHT_MM / COVERAGE_CELL_MM)))
        self._visited: Optional["np.ndarray"] = (
            np.zeros((self._n_rows, self._n_cols), dtype=bool) if np is not None else None
        )

        # Path following
        self._path: List[Tuple[float, float]] = []   # state-frame (x, y) waypoints
        self._path_idx: int = 0

        # Stuck / spin recovery
        self._last_progress_s: float = 0.0
        self._last_check_pos: Tuple[float, float] = (0.0, 0.0)
        self._spin_dir: float = 1.0
        self._spinning: bool = False

    # ── map loading ──────────────────────────────────────────────────────────

    def _load_map(self, path: str) -> None:
        if np is None or PilImage is None:
            self.status_text = "map wander=missing numpy/Pillow"
            return
        try:
            img = PilImage.open(path).convert("L")
        except Exception as exc:
            self.status_text = f"map wander=load error: {exc}"
            return
        self.map_w_px = img.width
        self.map_h_px = img.height
        arr = np.array(img, dtype=np.uint8)
        self._free = arr > MAP_FREE_THRESHOLD
        try:
            import cv2
            k = OBSTACLE_INFLATE_PX * 2 + 1
            kernel = np.ones((k, k), np.uint8)
            self._safe = cv2.dilate((~self._free).astype(np.uint8), kernel) == 0
        except ImportError:
            self._safe = self._free.copy()

    # ── coordinate conversion ────────────────────────────────────────────────

    @staticmethod
    def state_to_board(sx: float, sy: float) -> Tuple[float, float]:
        return START_X_MM + sx, START_Y_MM - sy

    @staticmethod
    def board_to_state(bx: float, by: float) -> Tuple[float, float]:
        return bx - START_X_MM, START_Y_MM - by

    def board_to_px(self, bx_mm: float, by_mm: float) -> Tuple[int, int]:
        col = int(bx_mm / BOARD_WIDTH_MM  * self.map_w_px)
        row = int(by_mm / BOARD_HEIGHT_MM * self.map_h_px)
        return (max(0, min(self.map_w_px  - 1, col)),
                max(0, min(self.map_h_px  - 1, row)))

    def px_to_board(self, col: int, row: int) -> Tuple[float, float]:
        return ((col + 0.5) / self.map_w_px  * BOARD_WIDTH_MM,
                (row + 0.5) / self.map_h_px * BOARD_HEIGHT_MM)

    def is_safe_board(self, bx_mm: float, by_mm: float) -> bool:
        if self._safe is None:
            return True
        col, row = self.board_to_px(bx_mm, by_mm)
        return bool(self._safe[row, col])

    # ── coverage tracking ────────────────────────────────────────────────────

    def _mark_visited(self, bx_mm: float, by_mm: float) -> None:
        if self._visited is None:
            return
        r = max(1, int(math.ceil(VISIT_RADIUS_MM / COVERAGE_CELL_MM)))
        cc = int(bx_mm / COVERAGE_CELL_MM)
        cr = int(by_mm / COVERAGE_CELL_MM)
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self._n_rows and 0 <= nc < self._n_cols:
                    self._visited[nr, nc] = True

    def _unvisited_safe_goals(self) -> List[Tuple[float, float]]:
        if self._visited is None:
            return []
        goals = []
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                if not self._visited[r, c]:
                    bx = (c + 0.5) * COVERAGE_CELL_MM
                    by = (r + 0.5) * COVERAGE_CELL_MM
                    if (0.0 < bx < BOARD_WIDTH_MM and 0.0 < by < BOARD_HEIGHT_MM
                            and self.is_safe_board(bx, by)):
                        goals.append((bx, by))
        return goals

    # ── A* pathfinding ───────────────────────────────────────────────────────

    def _astar(self, sc: int, sr: int, gc: int, gr: int) -> Optional[List[Tuple[int, int]]]:
        if self._safe is None:
            return None
        H, W = self._safe.shape
        if not (0 <= gr < H and 0 <= gc < W and self._safe[gr, gc]):
            return None

        start, goal = (sr, sc), (gr, gc)
        heap: list = [(0.0, start)]
        came: dict = {}
        g: dict = {start: 0.0}

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path: List[Tuple[int, int]] = []
                while cur in came:
                    path.append((cur[1], cur[0]))   # emit (col, row)
                    cur = came[cur]
                path.reverse()
                return path
            cr, cc = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W) or not self._safe[nr, nc]:
                        continue
                    ng = g[cur] + (1.414 if dr != 0 and dc != 0 else 1.0)
                    nb = (nr, nc)
                    if ng < g.get(nb, float("inf")):
                        g[nb] = ng
                        came[nb] = cur
                        heapq.heappush(heap, (ng + abs(nr - gr) + abs(nc - gc), nb))
        return None

    def _plan_path(self, sx: float, sy: float, goal_board: Tuple[float, float]) -> bool:
        bx, by = self.state_to_board(sx, sy)
        sc, sr = self.board_to_px(bx, by)
        gc, gr = self.board_to_px(goal_board[0], goal_board[1])

        if self._safe is None:
            gsx, gsy = self.board_to_state(goal_board[0], goal_board[1])
            self._path = [(gsx, gsy)]
            self._path_idx = 0
            return True

        raw = self._astar(sc, sr, gc, gr)
        if not raw:
            return False

        waypoints: List[Tuple[float, float]] = []
        for i, (col, row) in enumerate(raw):
            if i % PATH_STEP_PX == 0 or i == len(raw) - 1:
                gbx, gby = self.px_to_board(col, row)
                waypoints.append(self.board_to_state(gbx, gby))

        if not waypoints:
            waypoints = [self.board_to_state(goal_board[0], goal_board[1])]

        self._path = waypoints
        self._path_idx = 0
        return True

    # ── goal selection ───────────────────────────────────────────────────────

    def _pick_and_plan(self, sx: float, sy: float, now_s: float) -> bool:
        goals = self._unvisited_safe_goals()
        if not goals:
            if self._visited is not None:
                self._visited[:] = False
            goals = self._unvisited_safe_goals()
        if not goals:
            return False

        bx, by = self.state_to_board(sx, sy)
        goals.sort(key=lambda g: math.hypot(g[0] - bx, g[1] - by))
        # Prefer the farther half for better spatial coverage
        half = max(1, len(goals) // 2)
        far = goals[half:]
        random.shuffle(far)
        near = goals[:half]
        random.shuffle(near)

        for goal_board in (far + near)[:MAX_PLAN_ATTEMPTS]:
            if self._plan_path(sx, sy, goal_board):
                self._last_progress_s = now_s
                self._last_check_pos = (sx, sy)
                return True
        return False

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, now_s: float, robot_state: RobotState) -> Tuple[float, float]:
        sx, sy = robot_state.x_mm, robot_state.y_mm

        bx, by = self.state_to_board(sx, sy)
        self._mark_visited(bx, by)

        if not self._path or self._path_idx >= len(self._path):
            if self._pick_and_plan(sx, sy, now_s):
                self._spinning = False
            else:
                self._spinning = True
                self._spin_dir = random.choice([-1.0, 1.0])

        if self._spinning:
            self.status_text = "map wander=searching (spin)"
            t = SPIN_MMPS * self._spin_dir
            return (t, -t)

        wp_sx, wp_sy = self._path[self._path_idx]
        dist = math.hypot(wp_sx - sx, wp_sy - sy)

        if dist <= GOAL_REACHED_MM:
            self._path_idx += 1
            if self._path_idx >= len(self._path):
                self.status_text = "map wander=reaching next goal"
                self._path = []
            return (0.0, 0.0)

        # Stuck detection
        if (now_s - self._last_progress_s) >= STUCK_TIMEOUT_S:
            progress = math.hypot(sx - self._last_check_pos[0], sy - self._last_check_pos[1])
            self._last_progress_s = now_s
            self._last_check_pos = (sx, sy)
            if progress < STUCK_PROGRESS_MM:
                self._path = []
                self._spin_dir = random.choice([-1.0, 1.0])
                self._spinning = True
                self.status_text = "map wander=stuck, replanning"
                return (SPIN_MMPS * self._spin_dir, -SPIN_MMPS * self._spin_dir)

        final_board = self.state_to_board(*self._path[-1])
        self.status_text = (
            f"map wander=wp {self._path_idx + 1}/{len(self._path)} "
            f"→ ({final_board[0]:.0f}, {final_board[1]:.0f}) mm  d={dist:.0f}"
        )

        return calculate_wheel_speeds_for_point(
            x_mm=sx,
            y_mm=sy,
            theta_rad=robot_state.theta_rad,
            target_x_mm=wp_sx,
            target_y_mm=wp_sy,
            max_wheel_mmps=self.max_wheel_mmps,
            track_width_mm=self.track_width_mm,
            v_max_mmps=V_MAX_MMPS,
            kp_turn=KP_TURN,
            angle_deadband_rad=ANGLE_DEADBAND_RAD,
            omega_max_radps=self.omega_max,
        )

    @property
    def current_waypoint_state(self) -> Optional[Tuple[float, float]]:
        """Current target waypoint in state frame, or None."""
        if not self._path or self._path_idx >= len(self._path):
            return None
        return self._path[self._path_idx]

    @property
    def path_waypoints_state(self) -> List[Tuple[float, float]]:
        """Remaining path waypoints in state frame (from current index onward)."""
        if not self._path:
            return []
        return self._path[self._path_idx:]
