"""
English TTS via Kokoro. Writes a WAV file.

    python tts.py "hello, this is a test of the admissions advisor voice"
    python tts.py "..." --out reply.wav
"""

from __future__ import annotations

import argparse

import soundfile as sf
from kokoro import KPipeline

# 'a' = American English, 'b' = British.
LANG_CODE = "a"
VOICE = "af_heart"
SAMPLE_RATE = 24000

# "cuda" or "cpu" to force, None to auto-detect.
TTS_DEVICE: str | None = None


def load_pipeline() -> KPipeline:
    """Create the Kokoro pipeline once, so it isn't rebuilt per call."""
    if TTS_DEVICE is not None:
        device = TTS_DEVICE
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] device: {device}")
    return KPipeline(lang_code=LANG_CODE, device=device)


def stream_pcm(text: str, pipeline: KPipeline | None = None):
    """Yield little-endian int16 PCM bytes per Kokoro segment (mono,
    SAMPLE_RATE), so the client can play before synthesis finishes."""
    import numpy as np
    if pipeline is None:
        pipeline = load_pipeline()
    for _, _, audio in pipeline(text, voice=VOICE):
        a = np.asarray(audio, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        pcm16 = (a * 32767.0).astype("<i2")
        yield pcm16.tobytes()


def speak(text: str, out_path: str = "answer.wav",
          pipeline: KPipeline | None = None) -> str:
    """Synthesize `text` to a WAV file at out_path. Reuses a passed-in
    pipeline if given, else loads one. Returns the output path."""
    if pipeline is None:
        pipeline = load_pipeline()

    # Kokoro splits on sentences/length, so concatenate its segments.
    import numpy as np
    chunks = []
    for _, _, audio in pipeline(text, voice=VOICE):
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the given text.")

    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    # Kokoro gives float32 in [-1, 1]; browsers can't decode float WAV.
    full = np.asarray(full, dtype=np.float32)
    full = np.clip(full, -1.0, 1.0)
    pcm16 = (full * 32767.0).astype(np.int16)
    sf.write(out_path, pcm16, SAMPLE_RATE, subtype="PCM_16")
    print(f"wrote {out_path}  (audio length: {len(full) / SAMPLE_RATE:.1f}s)")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kokoro English TTS -> WAV")
    p.add_argument("text", help="Text to speak")
    p.add_argument("--out", default="answer.wav", help="Output WAV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    speak(args.text, args.out)


if __name__ == "__main__":
    main()