from dataclasses import dataclass
from typing import Optional

from PIL import Image

from helpers import calculate_wheel_speeds_for_point

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - depends on local Python environment
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - depends on local OpenCV install
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


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
        half_width = max(1.0, self.frame_width / 2.0)
        return (self.center_x - half_width) / half_width

    @property
    def height_ratio(self) -> float:
        return self.height_px / max(1.0, float(self.frame_height))


class Cv2MarkerDetector:
    """Detects AprilTag 36h11 markers through OpenCV's aruco module."""

    def __init__(self, dictionary_id=None):
        if np is None:
            raise RuntimeError(f"NumPy import failed: {NUMPY_IMPORT_ERROR}")
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {CV2_IMPORT_ERROR}")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was installed without aruco support; install opencv-contrib-python")

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
        frame_height, frame_width = frame_bgr.shape[:2]

        if self.detector is not None:
            corners, ids, _rejected = self.detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                frame_bgr,
                self.dictionary,
                parameters=self.parameters,
            )

        detections = []
        if ids is not None:
            for marker_corners, marker_id_arr in zip(corners, ids):
                marker_id = int(marker_id_arr[0])
                if target_marker_id is not None and marker_id != target_marker_id:
                    continue

                pts = marker_corners[0]
                center_x = float(np.mean(pts[:, 0]))
                center_y = float(np.mean(pts[:, 1]))
                width_px = float(
                    max(
                        np.linalg.norm(pts[1] - pts[0]),
                        np.linalg.norm(pts[2] - pts[3]),
                    )
                )
                height_px = float(
                    max(
                        np.linalg.norm(pts[3] - pts[0]),
                        np.linalg.norm(pts[2] - pts[1]),
                    )
                )
                detections.append(
                    MarkerDetection(
                        marker_id=marker_id,
                        center_x=center_x,
                        center_y=center_y,
                        width_px=width_px,
                        height_px=height_px,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )

        annotated = self._annotate(frame_bgr, corners, ids, detections)
        return detections, annotated

    def _annotate(self, frame_bgr, corners, ids, detections):
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        deadband_px = int(w * MarkerSearchController.CENTER_DEADBAND_RATIO / 2.0)
        cv2.line(frame_bgr, (center_x, 0), (center_x, h), (80, 200, 255), 1)
        cv2.line(frame_bgr, (center_x - deadband_px, 0), (center_x - deadband_px, h), (70, 70, 70), 1)
        cv2.line(frame_bgr, (center_x + deadband_px, 0), (center_x + deadband_px, h), (70, 70, 70), 1)

        for detection in detections:
            cx = int(detection.center_x)
            cy = int(detection.center_y)
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                f"id {detection.marker_id} h={detection.height_ratio:.2f}",
                (max(0, cx - 45), max(16, cy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


class MarkerSearchController:
    DETECTION_INTERVAL_S = 0.15
    SEARCH_SPIN_MMPS = 20.0
    CENTER_TURN_MMPS = 14.0
    APPROACH_MMPS = 28.0
    APPROACH_TURN_GAIN = 12.0
    CENTER_DEADBAND_RATIO = 0.16
    CLOSE_HEIGHT_RATIO = 0.35
    RETURN_DELAY_S = 5.0
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
    ):
        self.max_wheel_mmps = max_wheel_mmps
        self.track_width_mm = track_width_mm
        self.target_marker_id = target_marker_id
        self.base_xy_mm = base_xy_mm
        self.detector = Cv2MarkerDetector()
        self.state = "searching"
        self.last_detection: Optional[MarkerDetection] = None
        self.last_detection_time_s = 0.0
        self.aligned_since_s: Optional[float] = None
        self.last_error = None
        self.annotated_image = None
        self.status_text = "cube search=ready"

    def update(self, now_s: float, image: Optional[Image.Image], robot_state):
        self.last_error = None

        if self.state == "returning":
            return self._return_home(robot_state)
        if self.state == "finished":
            self.status_text = "cube search=finished"
            return (0.0, 0.0)
        if self.state == "aligned":
            wait_left_s = self._handoff_wait_left(now_s)
            if wait_left_s <= 0.0:
                self.state = "returning"
                self.status_text = "cube search=returning"
                return self._return_home(robot_state)
            self.status_text = f"cube search=aligned; returning in {wait_left_s:.1f}s"
            return (0.0, 0.0)

        if image is None:
            self.state = "no_camera"
            self.status_text = "cube search=waiting for camera"
            return (0.0, 0.0)

        if now_s - self.last_detection_time_s >= self.DETECTION_INTERVAL_S:
            self.last_detection_time_s = now_s
            detections, self.annotated_image = self.detector.detect(image, self.target_marker_id)
            self.last_detection = self._select_detection(detections)

        detection = self.last_detection
        if detection is None:
            self.state = "searching"
            self.status_text = "cube search=searching"
            return (self.SEARCH_SPIN_MMPS, -self.SEARCH_SPIN_MMPS)

        offset = detection.offset_ratio
        if abs(offset) > self.CENTER_DEADBAND_RATIO:
            self.state = "centering"
            direction = 1.0 if offset > 0 else -1.0
            turn = self.CENTER_TURN_MMPS * direction
            self.status_text = f"cube search=centering id={detection.marker_id} offset={offset:.2f}"
            return (turn, -turn)

        if detection.height_ratio >= self.CLOSE_HEIGHT_RATIO:
            self.state = "aligned"
            self.aligned_since_s = now_s
            self.status_text = f"cube search=aligned id={detection.marker_id}"
            return (0.0, 0.0)

        self.state = "approaching"
        correction = self.APPROACH_TURN_GAIN * offset
        left = self.APPROACH_MMPS + correction
        right = self.APPROACH_MMPS - correction
        left = max(-self.max_wheel_mmps, min(self.max_wheel_mmps, left))
        right = max(-self.max_wheel_mmps, min(self.max_wheel_mmps, right))
        self.status_text = (
            f"cube search=approaching id={detection.marker_id} "
            f"offset={offset:.2f} size={detection.height_ratio:.2f}"
        )
        return (left, right)

    def _select_detection(self, detections):
        if not detections:
            return None
        return max(detections, key=lambda d: d.width_px * d.height_px)

    def _handoff_wait_left(self, now_s: float) -> float:
        if self.aligned_since_s is None:
            self.aligned_since_s = now_s
        return max(0.0, self.RETURN_DELAY_S - (now_s - self.aligned_since_s))

    def _return_home(self, robot_state):
        goal_x, goal_y = self.base_xy_mm
        dist_mm = ((goal_x - robot_state.x_mm) ** 2 + (goal_y - robot_state.y_mm) ** 2) ** 0.5
        if dist_mm <= self.RETURN_STOP_MM:
            self.state = "finished"
            self.status_text = "cube search=finished at base"
            return (0.0, 0.0)

        omega_max = 0.6 * self.max_wheel_mmps * 2.0 / self.track_width_mm
        self.status_text = f"cube search=returning dist={dist_mm:.1f}mm"
        return calculate_wheel_speeds_for_point(
            x_mm=robot_state.x_mm,
            y_mm=robot_state.y_mm,
            theta_rad=robot_state.theta_rad,
            target_x_mm=goal_x,
            target_y_mm=goal_y,
            max_wheel_mmps=self.max_wheel_mmps,
            track_width_mm=self.track_width_mm,
            v_max_mmps=self.RETURN_V_MAX_MMPS,
            kp_turn=self.RETURN_KP_TURN,
            angle_deadband_rad=self.RETURN_ANGLE_DEADBAND_RAD,
            omega_max_radps=omega_max,
        )
