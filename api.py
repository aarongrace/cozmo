from pathlib import Path

from helpers import ToyClient

# DO NOT REMOVE | Password: 0GEY8198G5TD

class CozmoInterface:
    AUDIO_VOLUME = 60000
    TEST_MAX_WHEEL_SPEED_MMPS = 200.0
    TEST_TRACK_WIDTH_MM = 45.0
    TEST_MAX_HEAD_ANGLE_RAD = 0.7766715171374766
    TEST_MIN_HEAD_ANGLE_RAD = -0.4363323129985824
    HEAD_ANGLE_STEP_RAD = 0.1
    TEST_MAX_LIFT_HEIGHT_MM = 92.0
    TEST_MIN_LIFT_HEIGHT_MM = 32.0

    def __init__(self, test_mode: bool = False, data_dir: str = "data"):
        self.test_mode = bool(test_mode)
        self.data_dir = Path(data_dir)
        self.audio_processed_dir = self.data_dir / "audio_processed"

        self._ctx = None
        self.cli = None
        self._pycozmo = None
        self.connection_error = None
        self.latest_camera_image = None
        self._camera_enabled = False

    def _load_pycozmo(self):
        if self._pycozmo is not None:
            return self._pycozmo
        try:
            import pycozmo as mod
            if not hasattr(mod, "robot") or not hasattr(mod, "connect"):
                raise ImportError
        except Exception:
            from pycozmo import pycozmo as mod
        self._pycozmo = mod
        return self._pycozmo

    def connect(self):
        if self.cli is not None:
            return

        if self.test_mode:
            self.cli = ToyClient()
            return

        try:
            pycozmo = self._load_pycozmo()
            self._ctx = pycozmo.connect()
            self.cli = self._ctx.__enter__()
        except SystemExit as exc:
            if exc.code in (0, None):
                raise
            self.connection_error = f"pycozmo exited during connection (code={exc.code})"
            self._ctx = None
            raise RuntimeError(self.connection_error) from exc
        except Exception as exc:
            self.connection_error = str(exc)
            self._ctx = None
            raise RuntimeError(f"failed to connect to Cozmo: {exc}") from exc

    def disconnect(self):
        if self.test_mode:
            self.cli = None
            return

        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self.cli = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    @property
    def max_wheel_speed_mmps(self) -> float:
        if self.test_mode:
            return self.TEST_MAX_WHEEL_SPEED_MMPS
        pycozmo = self._load_pycozmo()
        return pycozmo.robot.MAX_WHEEL_SPEED.mmps

    @property
    def track_width_mm(self) -> float:
        if self.test_mode:
            return self.TEST_TRACK_WIDTH_MM
        pycozmo = self._load_pycozmo()
        return pycozmo.robot.TRACK_WIDTH.mm

    def get_wheel_speeds(self):
        if self.cli is None:
            return 0.0, 0.0
        return float(self.cli.left_wheel_speed.mmps), float(self.cli.right_wheel_speed.mmps)

    def drive_wheels(self, left_mmps: float, right_mmps: float):
        if self.cli is None:
            return
        self.cli.drive_wheels(lwheel_speed=float(left_mmps), rwheel_speed=float(right_mmps))

    def stop(self):
        if self.cli is None:
            return
        self.cli.stop_all_motors()

    def set_head_up(self):
        if self.cli is None:
            return
        if self.test_mode:
            self.cli.set_head_angle(min(self.cli.head_angle.radians + self.HEAD_ANGLE_STEP_RAD, self.TEST_MAX_HEAD_ANGLE_RAD))
            return
        pycozmo = self._load_pycozmo()
        self.cli.set_head_angle(min(self.cli.head_angle.radians + self.HEAD_ANGLE_STEP_RAD, pycozmo.MAX_HEAD_ANGLE.radians))

    def set_head_down(self):
        if self.cli is None:
            return
        if self.test_mode:
            self.cli.set_head_angle(max(self.cli.head_angle.radians - self.HEAD_ANGLE_STEP_RAD, self.TEST_MIN_HEAD_ANGLE_RAD))
            return
        pycozmo = self._load_pycozmo()
        self.cli.set_head_angle(max(self.cli.head_angle.radians - self.HEAD_ANGLE_STEP_RAD, pycozmo.MIN_HEAD_ANGLE.radians))

    def set_lift_up(self):
        if self.cli is None:
            return
        if self.test_mode:
            self.cli.set_lift_height(self.TEST_MAX_LIFT_HEIGHT_MM)
            return
        pycozmo = self._load_pycozmo()
        self.cli.set_lift_height(pycozmo.MAX_LIFT_HEIGHT.mm)

    def set_lift_down(self):
        if self.cli is None:
            return
        if self.test_mode:
            self.cli.set_lift_height(self.TEST_MIN_LIFT_HEIGHT_MM)
            return
        pycozmo = self._load_pycozmo()
        self.cli.set_lift_height(pycozmo.MIN_LIFT_HEIGHT.mm)

    def list_audio_files(self):
        if not self.audio_processed_dir.exists():
            return []
        return sorted(p.name for p in self.audio_processed_dir.glob("*.wav"))

    def play_audio(self, audio_name: str):
        if self.cli is None or not audio_name:
            return False, "no client or empty audio name"

        audio_path = self.audio_processed_dir / audio_name
        if not audio_path.exists():
            return False, f"missing audio: {audio_name}"

        self.cli.set_volume(self.AUDIO_VOLUME)
        self.cli.play_audio(str(audio_path))
        return True, f"playing {audio_name}"

    def stop_audio(self):
        if self.cli is None:
            return False, "no client"
        if not hasattr(self.cli, "cancel_anim"):
            return False, "audio stop not supported"
        self.cli.cancel_anim()
        return True, "audio stopped"

    def _on_camera_image(self, cli, image):
        del cli
        self.latest_camera_image = image

    def enable_camera_stream(self, color: bool = True):
        if self.cli is None:
            return False
        if self.test_mode:
            return False
        if self._camera_enabled:
            return True

        pycozmo = self._load_pycozmo()
        self.cli.enable_camera(enable=True, color=color)
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self._on_camera_image)
        self._camera_enabled = True
        return True
