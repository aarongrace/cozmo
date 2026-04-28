#!/usr/bin/env python3
import argparse
import warnings
import wave
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop

TARGET_RATE = 22050
TARGET_WIDTH = 2
TARGET_CHANNELS = 1


def process_wav(in_path: Path, out_path: Path):
    with wave.open(str(in_path), "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        n_frames = src.getnframes()
        frames = src.readframes(n_frames)

    if n_channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        n_channels = 1
    elif n_channels != 1:
        raise ValueError(f"unsupported channel count: {n_channels}")

    if sampwidth != TARGET_WIDTH:
        frames = audioop.lin2lin(frames, sampwidth, TARGET_WIDTH)
        sampwidth = TARGET_WIDTH

    if framerate != TARGET_RATE:
        frames, _ = audioop.ratecv(frames, sampwidth, n_channels, framerate, TARGET_RATE, None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as dst:
        dst.setnchannels(TARGET_CHANNELS)
        dst.setsampwidth(TARGET_WIDTH)
        dst.setframerate(TARGET_RATE)
        dst.writeframes(frames)


def process_all(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"input folder not found: {input_dir}")

    converted = 0
    skipped = 0
    for in_path in sorted(input_dir.rglob("*")):
        if not in_path.is_file():
            continue
        if in_path.suffix.lower() != ".wav":
            skipped += 1
            continue

        rel = in_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".wav")
        try:
            process_wav(in_path, out_path)
            print(f"converted: {in_path} -> {out_path}")
            converted += 1
        except Exception as exc:
            print(f"failed: {in_path} ({exc})")

    print(f"done. converted={converted}, skipped_non_wav={skipped}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert audio to Cozmo format (22050 Hz, 16-bit, mono WAV)")
    parser.add_argument("--input", default="data/audio_original", help="Input folder")
    parser.add_argument("--output", default="data/audio_processed", help="Output folder")
    return parser.parse_args()


def main():
    args = parse_args()
    process_all(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
