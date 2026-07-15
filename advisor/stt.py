"""
Standalone English STT via faster-whisper. Transcribes an audio file.

Isolated stage, like tts.py. main.py will chain this on the front:
audio -> transcribe -> retrieve -> generate -> speak.

    python stt.py recording.wav
    python stt.py query.mp3 --model large-v3-turbo

Models download on first run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load project-root .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from faster_whisper import WhisperModel

# whisper model and default language
MODEL_SIZE = "large-v3"
LANGUAGE = "en" 


def load_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """Load the model on GPU (float16). CTranslate2 backend."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[stt] model={model_size} device={device} compute={compute_type}")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(audio_path: str, model: WhisperModel | None = None,
               language: str = LANGUAGE) -> str:
    """Transcribe an audio file to text. Reuses a passed-in model if given,
    else loads one"""
    if model is None:
        model = load_model()
    segments, info = model.transcribe(audio_path, language=language)
    # segments is a generator; join the pieces into one transcript.
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="faster-whisper STT (audio file -> text)")
    p.add_argument("audio", help="Path to an audio file (wav/mp3/m4a/...)")
    p.add_argument("--model", default=MODEL_SIZE,
                   help="model size (large-v3, large-v3-turbo, medium, small)")
    p.add_argument("--language", default=LANGUAGE, help="forced language code")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    text = transcribe(args.audio, model=model, language=args.language)
    print("\n" + "=" * 80)
    print("TRANSCRIPT")
    print("=" * 80)
    print(text)


if __name__ == "__main__":
    main()