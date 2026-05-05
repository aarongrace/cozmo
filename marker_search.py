import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image

from helpers import calculate_wheel_speeds_for_point

try:
    import numpy as np
except Exception as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


CAMERA_HFOV_DEG = 58.0
MARKER_REAL_SIZE_MM = 20.0  # 2 cm AprilTag squares


@dataclass
class MarkerDetection:
    marker_id: int
    center_x: float
    center_y: float
    width_px: float
    height_px: float
    frame_width: int
    frame_height: int

    @property
    def offset_ratio(self) -> float:
        half = max(1.0, self.frame_width / 2.0)
        return (self.center_x - half) / half

    @property
    def height_ratio(self) -> float:
        return self.height_px / max(1.0, float(self.frame_height))

    def estimated_distance_mm(self) -> float:
        if self.height_px <= 0.0:
            return float("inf")
        f_px = (self.frame_width / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
        return (MARKER_REAL_SIZE_MM * f_px) / self.height_px


class Cv2MarkerDetector:
    WHITE_THRESHOLD = 175
    GROUND_ALPHA = 0.35
    GROUND_COLOR_BGR = (200, 60, 10)  # rich blue
    BORDER_COLOR_BGR = (0, 255, 255)  # yellow
    N_SECTORS = 5
    # Top half of the frame covers farther ground — use it for clearance
    CLEARANCE_TOP_RATIO = 0.5

    def __init__(self, dictionary_id=None):
        if np is None:
            raise RuntimeError(f"NumPy import failed: {NUMPY_IMPORT_ERROR}")
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {CV2_IMPORT_ERROR}")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV was installed without aruco support; install opencv-contrib-python"
            )
        if dictionary_id is None:
            dictionary_id = cv2.aruco.DICT_APRILTAG_36H11
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect(self, image: Image.Image, target_marker_id: Optional[int] = None):
        rgb = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        fh, fw = frame_bgr.shape[:2]

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame_bgr, self.dictionary, parameters=self.parameters
            )

        detections = []
        if ids is not None:
            for marker_corners, marker_id_arr in zip(corners, ids):
                marker_id = int(marker_id_arr[0])
                if target_marker_id is not None and marker_id != target_marker_id:
                    continue
                pts = marker_corners[0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                w_px = float(
                    max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
                )
                h_px = float(
                    max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
                )
                detections.append(
                    MarkerDetection(
                        marker_id=marker_id,
                        center_x=cx,
                        center_y=cy,
                        width_px=w_px,
                        height_px=h_px,
                        frame_width=fw,
                        frame_height=fh,
                    )
                )

        ground_mask = self._detect_ground(frame_bgr)
        annotated = self._annotate(frame_bgr, corners, ids, detections, ground_mask)
        return detections, annotated, ground_mask

    def _detect_ground(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sx, sy = w // 2, h - 1
        if int(gray[sy, sx]) < self.WHITE_THRESHOLD:
            return None
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(gray, flood_mask, (sx, sy), 255, loDiff=35, upDiff=35, flags=flags)
        return flood_mask[1:-1, 1:-1]

    def _annotate(self, frame_bgr, corners, ids, detections, ground_mask):
        h, w = frame_bgr.shape[:2]
        out = frame_bgr.copy()

        if ground_mask is not None and ground_mask.any():
            gm = ground_mask > 0
            overlay = out.copy()
            overlay[gm] = self.GROUND_COLOR_BGR
            cv2.addWeighted(out, 1.0 - self.GROUND_ALPHA, overlay, self.GROUND_ALPHA, 0, out)
            kernel = np.ones((3, 3), np.uint8)
            border = cv2.subtract(cv2.dilate(ground_mask, kernel), ground_mask)
            out[border > 0] = self.BORDER_COLOR_BGR

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)

        cx = w // 2
        deadband_px = int(w * MarkerSearchController.CENTER_DEADBAND_RATIO / 2.0)
        cv2.line(out, (cx, 0), (cx, h), (80, 200, 255), 1)
        cv2.line(out, (cx - deadband_px, 0), (cx - deadband_px, h), (70, 70, 70), 1)
        cv2.line(out, (cx + deadband_px, 0), (cx + deadband_px, h), (70, 70, 70), 1)

        for det in detections:
            dx, dy = int(det.center_x), int(det.center_y)
            d_mm = det.estimated_distance_mm()
            cv2.circle(out, (dx, dy), 5, (0, 255, 0), 2)
            cv2.putText(
                out,
                f"id={det.marker_id} {d_mm:.0f}mm",
                (max(0, dx - 50), max(16, dy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def ground_clearance(self, ground_mask) -> list:
        """Per-sector [0..1] free ratio in the far portion of the frame, left to right."""
        if ground_mask is None:
            return [0.0] * self.N_SECTORS
        h, w = ground_mask.shape
        top = ground_mask[: int(h * self.CLEARANCE_TOP_RATIO), :]
        th = top.shape[0]
        sw = max(1, w // self.N_SECTORS)
        result = []
        for i in range(self.N_SECTORS):
            x0 = i * sw
            x1 = (i + 1) * sw if i < self.N_SECTORS - 1 else w
            count = float(np.count_nonzero(top[:, x0:x1]))
            denom = float(th * (x1 - x0))
            result.append(count / denom if denom > 0.0 else 0.0)
        return result


class MarkerSearchController:
    DETECTION_INTERVAL_S = 0.12

    # Wandering
    WANDER_FORWARD_MMPS = 35.0
    WANDER_FORWARD_MAX_S = 1.5
    WANDER_TURN_MMPS = 22.0
    WANDER_TURN_S = 0.7
    WANDER_BACK_MMPS = 25.0
    WANDER_BACK_S = 0.5
    GROUND_OBSTACLE_THRESH = 0.20

    # Centering — turn speed reduced 70% vs wandering
    CENTER_DEADBAND_RATIO = 0.16
    CENTER_TURN_MMPS = 6.6  # WANDER_TURN_MMPS * 0.30

    # Approach
    APPROACH_MMPS = 30.0
    APPROACH_TURN_GAIN = 10.0
    APPROACH_OVERSHOOT_MM = 35.0

    # Lift timing
    LIFT_LOWER_WAIT_S = 1.2
    LIFT_RAISE_WAIT_S = 1.0

    # Return home
    RETURN_STOP_MM = 28.0
    RETURN_V_MAX_MMPS = 70.0
    RETURN_KP_TURN = 4.0
    RETURN_ANGLE_DEADBAND_RAD = 0.05

    def __init__(
        self,
        max_wheel_mmps: float,
        track_width_mm: float,
        target_marker_id: Optional[int] = None,
        base_xy_mm=(0.0, 0.0),
        on_lift_up: Optional[Callable] = None,
        on_lift_down: Optional[Callable] = None,
        get_cliff_detected: Optional[Callable] = None,
    ):
        self.max_wheel_mmps = max_wheel_mmps
        self.track_width_mm = track_width_mm
        self.target_marker_id = target_marker_id
        self.base_xy_mm = base_xy_mm
        self._on_lift_up = on_lift_up or (lambda: None)
        self._on_lift_down = on_lift_down or (lambda: None)
        self._get_cliff_detected = get_cliff_detected or (lambda: False)

        self.detector = Cv2MarkerDetector()
        self.state = "lowering_lift"
        self._state_start_s: Optional[float] = None
        self._action_done = False  # one-shot flag for per-state entry actions

        self.last_detection: Optional[MarkerDetection] = None
        self.last_detection_time_s = 0.0
        self._last_ground_mask = None
        self.annotated_image = None
        self.status_text = "cube search=starting"

        self._wander_sub: str = "forward"
        self._wander_sub_start_s: Optional[float] = None
        self._wander_turn_dir: float = 1.0

        self._approach_start_xy: Optional[tuple] = None
        self._approach_target_mm: float = 0.0

    # ------------------------------------------------------------------ public

    def update(self, now_s: float, image: Optional[Image.Image], robot_state) -> tuple:
        if self._state_start_s is None:
            self._state_start_s = now_s

        if image is not None and now_s - self.last_detection_time_s >= self.DETECTION_INTERVAL_S:
            self.last_detection_time_s = now_s
            dets, self.annotated_image, mask = self.detector.detect(image, self.target_marker_id)
            self.last_detection = self._pick_best(dets)
            self._last_ground_mask = mask

        handlers = {
            "lowering_lift": self._do_lower_lift,
            "wandering":     self._do_wander,
            "centering":     self._do_center,
            "approaching":   self._do_approach,
            "raising_lift":  self._do_raise_lift,
            "returning":     self._do_return,
            "finished":      self._do_finished,
        }
        return handlers.get(self.state, lambda *_: (0.0, 0.0))(now_s, robot_state)

    # ----------------------------------------------------------------- states

    def _do_lower_lift(self, now_s, _rs):
        if not self._action_done:
            self._on_lift_down()
            self._action_done = True
        elapsed = now_s - self._state_start_s
        self.status_text = f"cube search=lowering lift {elapsed:.1f}s"
        if elapsed >= self.LIFT_LOWER_WAIT_S:
            self._transition("wandering", now_s)
            self._wander_sub = "forward"
            self._wander_sub_start_s = now_s
        return (0.0, 0.0)

    def _do_wander(self, now_s, _rs):
        if self.last_detection is not None:
            self._transition("centering", now_s)
            self.status_text = f"cube search=found id={self.last_detection.marker_id}"
            return (0.0, 0.0)

        if self._get_cliff_detected():
            self._wander_sub = "backing"
            self._wander_sub_start_s = now_s
            self.status_text = "cube search=cliff, backing"
            return (-self.WANDER_BACK_MMPS, -self.WANDER_BACK_MMPS)

        if self._wander_sub_start_s is None:
            self._wander_sub_start_s = now_s
        elapsed = now_s - self._wander_sub_start_s

        if self._wander_sub == "backing":
            if elapsed >= self.WANDER_BACK_S:
                self._wander_sub = "turning"
                self._wander_sub_start_s = now_s
                self._wander_turn_dir = random.choice([-1.0, 1.0])
            return (-self.WANDER_BACK_MMPS, -self.WANDER_BACK_MMPS)

        if self._wander_sub == "turning":
            if elapsed >= self.WANDER_TURN_S:
                self._wander_sub = "forward"
                self._wander_sub_start_s = now_s
            else:
                t = self.WANDER_TURN_MMPS * self._wander_turn_dir
                self.status_text = f"cube search=wander turn {'L' if self._wander_turn_dir > 0 else 'R'}"
                return (t, -t)

        # forward sub-state — check ground ahead
        clearance = self.detector.ground_clearance(self._last_ground_mask)
        center = len(clearance) // 2

        if clearance[center] < self.GROUND_OBSTACLE_THRESH:
            best = max(range(len(clearance)), key=lambda i: clearance[i])
            if clearance[best] < self.GROUND_OBSTACLE_THRESH:
                self._wander_sub = "backing"
                self._wander_sub_start_s = now_s
                self.status_text = "cube search=wander all blocked"
                return (-self.WANDER_BACK_MMPS, -self.WANDER_BACK_MMPS)
            self._wander_turn_dir = 1.0 if best < center else -1.0
            self._wander_sub = "turning"
            self._wander_sub_start_s = now_s
            t = self.WANDER_TURN_MMPS * self._wander_turn_dir
            self.status_text = "cube search=wander obstacle turn"
            return (t, -t)

        if elapsed >= self.WANDER_FORWARD_MAX_S:
            self._wander_turn_dir = random.choice([-1.0, 1.0])
            self._wander_sub = "turning"
            self._wander_sub_start_s = now_s
            t = self.WANDER_TURN_MMPS * self._wander_turn_dir
            self.status_text = "cube search=wander random turn"
            return (t, -t)

        self.status_text = f"cube search=wandering fwd c={clearance[center]:.2f}"
        return (self.WANDER_FORWARD_MMPS, self.WANDER_FORWARD_MMPS)

    def _do_center(self, now_s, _rs):
        if self._get_cliff_detected():
            self._transition("wandering", now_s)
            self._wander_sub = "backing"
            self._wander_sub_start_s = now_s
            return (-self.WANDER_BACK_MMPS, -self.WANDER_BACK_MMPS)

        det = self.last_detection
        if det is None:
            self._transition("wandering", now_s)
            self._wander_sub = "forward"
            self._wander_sub_start_s = now_s
            self.status_text = "cube search=lost marker"
            return (0.0, 0.0)

        offset = det.offset_ratio
        if abs(offset) > self.CENTER_DEADBAND_RATIO:
            direction = 1.0 if offset > 0 else -1.0
            t = self.CENTER_TURN_MMPS * direction
            self.status_text = f"cube search=centering off={offset:.2f}"
            return (t, -t)

        dist_mm = det.estimated_distance_mm()
        self._approach_target_mm = dist_mm + self.APPROACH_OVERSHOOT_MM
        self._approach_start_xy = None
        self._transition("approaching", now_s)
        self.status_text = f"cube search=centered d={dist_mm:.0f}mm"
        return (0.0, 0.0)

    def _do_approach(self, now_s, robot_state):
        if self._get_cliff_detected():
            self._transition("wandering", now_s)
            self._wander_sub = "backing"
            self._wander_sub_start_s = now_s
            return (-self.WANDER_BACK_MMPS, -self.WANDER_BACK_MMPS)

        if self._approach_start_xy is None:
            self._approach_start_xy = (robot_state.x_mm, robot_state.y_mm)

        driven = _dist2d(self._approach_start_xy, (robot_state.x_mm, robot_state.y_mm))

        if driven >= self._approach_target_mm:
            self._transition("raising_lift", now_s)
            self.status_text = "cube search=at target, raising lift"
            return (0.0, 0.0)

        det = self.last_detection
        remaining = self._approach_target_mm - driven

        if det is not None:
            offset = det.offset_ratio
            correction = self.APPROACH_TURN_GAIN * offset
            left = max(-self.max_wheel_mmps, min(self.max_wheel_mmps, self.APPROACH_MMPS + correction))
            right = max(-self.max_wheel_mmps, min(self.max_wheel_mmps, self.APPROACH_MMPS - correction))
        else:
            left = right = self.APPROACH_MMPS

        self.status_text = f"cube search=approaching rem={remaining:.0f}mm"
        return (left, right)

    def _do_raise_lift(self, now_s, _rs):
        if not self._action_done:
            self._on_lift_up()
            self._action_done = True
        elapsed = now_s - self._state_start_s
        self.status_text = "cube search=raising lift"
        if elapsed >= self.LIFT_RAISE_WAIT_S:
            self._transition("returning", now_s)
            self.status_text = "cube search=returning"
        return (0.0, 0.0)

    def _do_return(self, now_s, robot_state):
        gx, gy = self.base_xy_mm
        dist = _dist2d((robot_state.x_mm, robot_state.y_mm), (gx, gy))
        if dist <= self.RETURN_STOP_MM:
            self._transition("finished", now_s)
            return (0.0, 0.0)
        omega_max = 0.6 * self.max_wheel_mmps * 2.0 / self.track_width_mm
        self.status_text = f"cube search=returning d={dist:.0f}mm"
        return calculate_wheel_speeds_for_point(
            x_mm=robot_state.x_mm,
            y_mm=robot_state.y_mm,
            theta_rad=robot_state.theta_rad,
            target_x_mm=gx,
            target_y_mm=gy,
            max_wheel_mmps=self.max_wheel_mmps,
            track_width_mm=self.track_width_mm,
            v_max_mmps=self.RETURN_V_MAX_MMPS,
            kp_turn=self.RETURN_KP_TURN,
            angle_deadband_rad=self.RETURN_ANGLE_DEADBAND_RAD,
            omega_max_radps=omega_max,
        )

    def _do_finished(self, _now_s, _rs):
        self.status_text = "cube search=finished"
        return (0.0, 0.0)

    # --------------------------------------------------------------- helpers

    def _transition(self, new_state: str, now_s: float):
        self.state = new_state
        self._state_start_s = now_s
        self._action_done = False

    def _pick_best(self, detections) -> Optional[MarkerDetection]:
        if not detections:
            return None
        return max(detections, key=lambda d: d.width_px * d.height_px)


def _dist2d(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
