import math, time, random
from enum import Enum
from dataclasses import dataclass
from types import SimpleNamespace

DEFAULT_TRACK_WIDTH_MM = 85.0
DEFAULT_MIN_LIFT_HEIGHT_MM = 32.0


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SpeechCategory(str, Enum):
    GOING_LEFT = "going_left"
    GOING_RIGHT = "going_right"
    GOING_STRAIGHT = "going_straight"
    GOING_BACK = "going_back"
    NEW_GOAL = "new_goal"
    RETURNING_HOME = "returning_home"
    NEW_COMMAND = "new_command"
    IDLE_COMMENTS = "idle_comments"
    ERROR_REACTIONS = "error_reactions"
    UNHINGED = "unhinged"


@dataclass
class RobotState:
    track_width_mm: float = DEFAULT_TRACK_WIDTH_MM
    x_mm: float = 0.0
    y_mm: float = 0.0
    theta_rad: float = 0.0
    left_wheel_mmps: float = 0.0
    right_wheel_mmps: float = 0.0
    linear_velocity_mmps: float = 0.0
    angular_velocity_radps: float = 0.0

    def update_from_wheels(self, left_wheel_mmps: float, right_wheel_mmps: float, dt_s: float) -> None:
        if dt_s <= 0:
            return

        self.left_wheel_mmps = float(left_wheel_mmps)
        self.right_wheel_mmps = float(right_wheel_mmps)
        self.linear_velocity_mmps = (self.left_wheel_mmps + self.right_wheel_mmps) / 2.0
        self.angular_velocity_radps = (self.right_wheel_mmps - self.left_wheel_mmps) / self.track_width_mm

        self.x_mm += self.linear_velocity_mmps * math.cos(self.theta_rad) * dt_s
        self.y_mm += self.linear_velocity_mmps * math.sin(self.theta_rad) * dt_s
        self.theta_rad = normalize_angle(self.theta_rad + self.angular_velocity_radps * dt_s)


def calculate_wheel_speeds_for_point(
    x_mm: float,
    y_mm: float,
    theta_rad: float,
    target_x_mm: float,
    target_y_mm: float,
    max_wheel_mmps: float,
    track_width_mm: float,
    v_max_mmps: float,
    kp_turn: float,
    angle_deadband_rad: float,
    omega_max_radps: float,
):
    angle_to_target = normalize_angle(math.atan2(target_y_mm - y_mm, target_x_mm - x_mm))
    heading_error = normalize_angle(angle_to_target - theta_rad)

    if abs(heading_error) < angle_deadband_rad:
        omega_radps = 0.0
    else:
        omega_radps = _clip(kp_turn * heading_error, -omega_max_radps, omega_max_radps)

    low, high = 0.1, 0.9
    turn_blend = _clip((abs(heading_error) - low) / (high - low), 0.0, 1.0)
    v_mmps = v_max_mmps * (1.0 - turn_blend)

    left_mmps = v_mmps - omega_radps * track_width_mm / 2.0
    right_mmps = v_mmps + omega_radps * track_width_mm / 2.0
    left_mmps = _clip(left_mmps, -max_wheel_mmps, max_wheel_mmps)
    right_mmps = _clip(right_mmps, -max_wheel_mmps, max_wheel_mmps)
    return left_mmps, right_mmps


class ToyClient:
    """Minimal pycozmo-like client for test mode."""

    def __init__(self):
        self.left_wheel_speed = SimpleNamespace(mmps=0.0)
        self.right_wheel_speed = SimpleNamespace(mmps=0.0)
        self.head_angle = SimpleNamespace(radians=0.0)
        self.lift_position = SimpleNamespace(height=SimpleNamespace(mm=DEFAULT_MIN_LIFT_HEIGHT_MM))
        self.battery_voltage = 4.1
        self.audio_playing = False

    def drive_wheels(self, lwheel_speed: float, rwheel_speed: float, duration=None, **kwargs):
        del duration, kwargs
        self.left_wheel_speed.mmps = float(lwheel_speed)
        self.right_wheel_speed.mmps = float(rwheel_speed)

    def stop_all_motors(self):
        self.drive_wheels(0.0, 0.0)

    def set_head_angle(self, angle: float, **kwargs):
        del kwargs
        self.head_angle.radians = float(angle)

    def set_lift_height(self, height: float, **kwargs):
        del kwargs
        self.lift_position.height.mm = float(height)

    def set_volume(self, level: int):
        del level

    def play_audio(self, fspec: str):
        self.audio_playing = True
        print(f"[ToyClient] play_audio: {fspec}")

    def cancel_anim(self):
        self.audio_playing = False
        print("[ToyClient] cancel_anim")

    def set_all_backpack_lights(self, light):
        del light

    def stop(self):
        self.stop_all_motors()
    
class RoutineController:
    UPDATE_INTERVAL_MS = 120
    TIME_BEFORE_GOING_BACK_MS = 5000
    DISTANCE_TO_STOP_MM = 20.0
    NEW_GOAL_MAX_DISTANCE_MM = 120.0
    V_MAX_MMPS = 100.0
    KP_TURN = 2.0
    ANGLE_DEADBAND_RAD = 0.05
    OMEGA_MAX_SCALE = 0.6

    def __init__(self, state: RobotState, max_wheel_mmps: float, track_width_mm: float):
        self.state = state
        self.max_wheel_mmps = max_wheel_mmps
        self.track_width_mm = track_width_mm
        self.omega_max_radps = self.OMEGA_MAX_SCALE * self.max_wheel_mmps * 2.0 / self.track_width_mm

        self.start_time_s = time.perf_counter()
        self.last_command_time_s = 0.0
        self.going_back = False
        self.finished = False
        self.goal_xy_mm = (0.0, 20.0) # straight in front
        self.last_event = None

    def _find_new_goal(self):
        angle = random.uniform(-math.pi, math.pi)
        distance = random.uniform(0.0, self.NEW_GOAL_MAX_DISTANCE_MM)
        x_mm = self.state.x_mm + distance * math.cos(angle)
        y_mm = self.state.y_mm + distance * math.sin(angle)
        return (x_mm, y_mm)

    def update(self, now_s: float):
        self.last_event = None
        if self.finished:
            return (0.0, 0.0)

        elapsed_ms = (now_s - self.start_time_s) * 1000.0
        if not self.going_back and elapsed_ms >= self.TIME_BEFORE_GOING_BACK_MS:
            self.going_back = True
            self.goal_xy_mm = (0.0, 0.0)
            self.last_event = "returning_home"

        if (now_s - self.last_command_time_s) * 1000.0 < self.UPDATE_INTERVAL_MS:
            return None
        self.last_command_time_s = now_s

        goal_x_mm, goal_y_mm = self.goal_xy_mm
        distance_mm = ((goal_x_mm - self.state.x_mm) ** 2 + (goal_y_mm - self.state.y_mm) ** 2) ** 0.5
        if distance_mm <= self.DISTANCE_TO_STOP_MM:
            if self.going_back:
                self.finished = True
                return (0.0, 0.0)
            self.goal_xy_mm = self._find_new_goal()
            goal_x_mm, goal_y_mm = self.goal_xy_mm
            self.last_event = "new_goal"

        return calculate_wheel_speeds_for_point(
            x_mm=self.state.x_mm,
            y_mm=self.state.y_mm,
            theta_rad=self.state.theta_rad,
            target_x_mm=goal_x_mm,
            target_y_mm=goal_y_mm,
            max_wheel_mmps=self.max_wheel_mmps,
            track_width_mm=self.track_width_mm,
            v_max_mmps=self.V_MAX_MMPS,
            kp_turn=self.KP_TURN,
            angle_deadband_rad=self.ANGLE_DEADBAND_RAD,
            omega_max_radps=self.omega_max_radps,
        )
