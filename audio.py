import audioop
import hashlib
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from helpers import SpeechCategory


class _SpeechAudioController:
    TARGET_RATE = 22050
    TARGET_WIDTH = 2
    TARGET_CHANNELS = 1
    DEFAULT_VOLUME = 30000
    MAX_QUEUE_SIZE = 16
    MIN_ENQUEUE_INTERVAL_S = 0.2

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.cli = None
        self.data_dir = None
        self.lines = {}
        self.cache_dir = None

        self.enabled = False
        self.muted = False
        self.volume = self.DEFAULT_VOLUME

        self.playing_until_s = 0.0
        self.last_audio_end_s = 0.0

        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._worker = None
        self._stop_event = threading.Event()
        self._last_enqueue_s = 0.0

    def initialize(self, cli, data_dir: Path):
        self.cli = cli
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "audio_processed" / "speech_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        speech_path = self.data_dir / "speech_lines.json"
        self.lines = json.loads(speech_path.read_text()) if speech_path.exists() else {}

        if self.last_audio_end_s <= 0:
            # Start with a short quiet period at boot.
            self.last_audio_end_s = time.monotonic() + 3.0

        self._ensure_worker()

    def _ensure_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="SpeechAudioWorker", daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                req = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_request(req)
            except Exception:
                # Keep worker alive if one request fails.
                pass
            finally:
                self._queue.task_done()

    def _refresh_audio_state(self):
        now = time.monotonic()
        with self._lock:
            if self.playing_until_s > 0 and now >= self.playing_until_s:
                self.last_audio_end_s = self.playing_until_s
                self.playing_until_s = 0.0

    def is_playing(self):
        self._refresh_audio_state()
        with self._lock:
            return time.monotonic() < self.playing_until_s

    def stop_audio(self):
        if self.cli is None:
            return False, "audio not initialized"
        if hasattr(self.cli, "cancel_anim"):
            self.cli.cancel_anim()
        with self._lock:
            self.playing_until_s = 0.0
            self.last_audio_end_s = time.monotonic()
        return True, "audio stopped"

    def toggle_mute(self):
        if self.cli is None:
            return False, "audio not initialized"
        self.muted = not self.muted
        self.cli.set_volume(0 if self.muted else self.volume)
        return True, ("muted" if self.muted else "unmuted")

    def set_volume(self, volume: int):
        if self.cli is None:
            return False, "audio not initialized"
        self.volume = max(0, min(65535, int(volume)))
        if not self.muted:
            self.cli.set_volume(self.volume)
        return True, f"volume={self.volume}"

    def toggle_enabled(self):
        self.enabled = not self.enabled
        return True, ("audio enabled" if self.enabled else "audio disabled")

    def get_status(self):
        self._refresh_audio_state()
        now = time.monotonic()
        with self._lock:
            is_playing = now < self.playing_until_s
            remaining = max(0.0, self.playing_until_s - now)
        return {
            "enabled": self.enabled,
            "muted": self.muted,
            "volume": self.volume,
            "is_playing": is_playing,
            "remaining_s": remaining,
            "queued": self._queue.qsize(),
        }

    def _time_based_chance(self):
        self._refresh_audio_state()
        elapsed = max(0.0, time.monotonic() - self.last_audio_end_s)
        return min(0.3, 0.01 + 0.01 * elapsed)

    @staticmethod
    def _sanitize(text: str):
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", text)[:80]

    def _pick_line(self, category: SpeechCategory, line_override=None):
        if line_override and not str(line_override).lower().endswith(".wav"):
            return str(line_override)
        options = self.lines.get(category.value, [])
        if not options:
            return None
        return str(random.choice(options))

    def _resolve_override_wav(self, line_override):
        if not line_override:
            return None
        name = str(line_override)
        if not name.lower().endswith(".wav"):
            return None

        p = Path(name)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self.data_dir / "audio_processed" / p)
            candidates.append(self.data_dir / p)
            candidates.append(Path.cwd() / p)

        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    @staticmethod
    def _wav_duration_seconds(path: Path):
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return (frames / float(rate)) if rate > 0 else 0.0

    def _convert_wav_to_cozmo(self, src: Path, dst: Path):
        with wave.open(str(src), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        if n_channels == 2:
            frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
            n_channels = 1
        elif n_channels != 1:
            raise ValueError(f"unsupported channel count: {n_channels}")

        if sampwidth != self.TARGET_WIDTH:
            frames = audioop.lin2lin(frames, sampwidth, self.TARGET_WIDTH)
            sampwidth = self.TARGET_WIDTH

        if framerate != self.TARGET_RATE:
            frames, _ = audioop.ratecv(frames, sampwidth, n_channels, framerate, self.TARGET_RATE, None)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(self.TARGET_CHANNELS)
            wf.setsampwidth(self.TARGET_WIDTH)
            wf.setframerate(self.TARGET_RATE)
            wf.writeframes(frames)

    def _synthesize_line(self, text: str):
        safe = self._sanitize(text)
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        raw_wav = self.cache_dir / f"tts_raw_{safe}_{digest}.wav"
        cozmo_wav = self.cache_dir / f"tts_{safe}_{digest}.wav"

        if not cozmo_wav.exists():
            cmd = [
                sys.executable,
                "-c",
                (
                    "import pyttsx3,sys;"
                    "e=pyttsx3.init();"
                    "e.save_to_file(sys.argv[1],sys.argv[2]);"
                    "e.runAndWait()"
                ),
                text,
                str(raw_wav),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except Exception as exc:
                return None, f"tts subprocess failed: {exc}"
            self._convert_wav_to_cozmo(raw_wav, cozmo_wav)

        return cozmo_wav, "ok"

    def _process_request(self, req):
        if self.cli is None:
            return

        category = req["category"]
        always_play = req["always_play"]
        line_override = req.get("line_override")

        if always_play and hasattr(self.cli, "cancel_anim"):
            self.cli.cancel_anim()

        override_wav = self._resolve_override_wav(line_override)
        if override_wav is not None:
            wav_path = override_wav
        else:
            line = self._pick_line(category, line_override)
            if not line:
                return
            wav_path, _msg = self._synthesize_line(line)
            if wav_path is None:
                return

        self.cli.set_volume(0 if self.muted else self.volume)
        self.cli.play_audio(str(wav_path))

        duration_s = self._wav_duration_seconds(wav_path)
        now = time.monotonic()
        with self._lock:
            self.playing_until_s = now + max(0.05, duration_s)

    def play_audio(self, category: SpeechCategory, always_play: bool = False, line_override=None):
        if self.cli is None:
            return False, "audio not initialized"
        if not self.enabled:
            return False, "audio disabled"

        if not isinstance(category, SpeechCategory):
            category = SpeechCategory(category)

        if self.is_playing() and not always_play:
            return False, "audio already playing"
        if self._queue.qsize() > 0 and not always_play:
            return False, "audio queued"

        if not always_play:
            # Unhinged has a global 10% chance regardless of context.
            forced_unhinged = random.random() < 0.10
            if forced_unhinged:
                category = SpeechCategory.UNHINGED
            else:
                if category == SpeechCategory.UNHINGED:
                    if random.random() >= 0.10:
                        return False, "unhinged chance miss"
                elif random.random() >= self._time_based_chance():
                    return False, "chance miss"

        now = time.monotonic()
        if not always_play and (now - self._last_enqueue_s) < self.MIN_ENQUEUE_INTERVAL_S:
            return False, "audio throttle"
        self._last_enqueue_s = now

        req = {
            "category": category,
            "always_play": bool(always_play),
            "line_override": line_override,
        }
        try:
            self._queue.put_nowait(req)
        except queue.Full:
            return False, "audio queue full"
        return True, f"queued {category.value}"


def init_audio_controller(cli, data_dir="data"):
    _SpeechAudioController.instance().initialize(cli=cli, data_dir=Path(data_dir))


def play_audio(category: SpeechCategory, always_play: bool = False, line_override=None):
    return _SpeechAudioController.instance().play_audio(
        category=category, always_play=always_play, line_override=line_override
    )


def stop_audio():
    return _SpeechAudioController.instance().stop_audio()


def toggle_mute():
    return _SpeechAudioController.instance().toggle_mute()


def is_audio_playing():
    return _SpeechAudioController.instance().is_playing()


def set_volume(volume: int):
    return _SpeechAudioController.instance().set_volume(volume)


def toggle_audio_enabled():
    return _SpeechAudioController.instance().toggle_enabled()


def get_audio_status():
    return _SpeechAudioController.instance().get_status()
