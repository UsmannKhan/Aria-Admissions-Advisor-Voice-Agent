"""
End-to-end voice pipeline orchestrator.

Loads every model once at startup, then runs:
    audio file -> STT (faster-whisper) -> retrieve -> generate (Gemini)
    -> TTS (Kokoro) -> spoken WAV

Only the answer prose is spoken; citations/grounding are printed.

    python main.py path/to/query.wav
    python main.py query.wav --k 6 --mode dense --out reply.wav
    python main.py --text "compare eligibility at LUMS and Habib"   # skip STT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_IN = PROJECT_ROOT / "data" / "audio" / "input"
AUDIO_OUT = PROJECT_ROOT / "data" / "audio" / "output"


sys.path.insert(0, str(PROJECT_ROOT / "advisor"))

# Let CTranslate2 (faster-whisper) find torch's bundled cuDNN 9 DLLs on Windows.
if sys.platform == "win32":
    try:
        import torch
        _torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if _torch_lib.is_dir():
            import os
            os.add_dll_directory(str(_torch_lib))
    except Exception:
        pass

load_dotenv(PROJECT_ROOT / ".env")

import stt
import tts
from generate import (answer_query, get_client, print_grounding, print_sources)


def warm_up(stt_model_size: str):
    """Load all models once. Returns the handles the pipeline reuses."""
    print("=" * 80)
    print("WARMING UP MODELS (one-time load)")
    print("=" * 80)

    t0 = time.time()
    whisper_model = stt.load_model(stt_model_size)
    print(f"  [stt]    loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    kokoro = tts.load_pipeline()
    print(f"  [tts]    loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    gemini = get_client()
    print(f"  [gemini] client ready in {time.time() - t0:.1f}s")

    print("warm-up complete.\n")
    return whisper_model, kokoro, gemini


def run_pipeline(query_text: str, k: int, mode: str, out_path: str,
                 kokoro, gemini, speak: bool = True) -> None:
    """retrieve -> generate -> (optional) speak, reusing warmed models."""
    t0 = time.time()
    results, parsed = answer_query(query_text, k, mode, gemini)
    gen_time = time.time() - t0
    if not results or parsed is None:
        print("\nNo sources retrieved; nothing to answer from.")
        return

    print("\n" + "=" * 80)
    print(f"ANSWER  (retrieve+generate: {gen_time:.1f}s)")
    print("=" * 80)
    print(parsed.answer)

    print_grounding(results, parsed.claims)
    print_sources(results, parsed.sources_used)

    # Speak the answer only, never the grounding/citations.
    if speak:
        print()
        t0 = time.time()
        tts.speak(parsed.answer, out_path=out_path, pipeline=kokoro)
        print(f"[tts] synthesis wall-time: {time.time() - t0:.1f}s")
    if speak:
        print()
        tts.speak(parsed.answer, out_path=out_path, pipeline=kokoro)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice admissions-advisor pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("audio", nargs="?", help="Path to a spoken-query audio file")
    src.add_argument("--text", help="Skip STT; use this typed query instead")

    p.add_argument("--k", type=int, default=6, help="top-k per entity (default 6)")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="dense")
    p.add_argument("--out", default=str(AUDIO_OUT / "answer.wav"),
                   help="spoken-answer WAV path")
    p.add_argument("--no-speak", action="store_true", help="skip TTS output")
    p.add_argument("--stt-model", default=stt.MODEL_SIZE,
                   help="faster-whisper model size")
    p.add_argument("--language", default=stt.LANGUAGE, help="forced STT language")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    whisper_model, kokoro, gemini = warm_up(args.stt_model)

    if args.text:
        query_text = args.text
        print(f"QUERY (typed): {query_text}")
    else:
        # Bare filenames resolve against data/audio/input/.
        audio_path = Path(args.audio)
        if not audio_path.exists():
            candidate = AUDIO_IN / args.audio
            if candidate.exists():
                audio_path = candidate
            else:
                print(f"ERROR: audio not found at '{args.audio}' or '{candidate}'.")
                return

        t0 = time.time()
        query_text = stt.transcribe(str(audio_path), model=whisper_model,
                                    language=args.language)
        print("\n" + "=" * 80)
        print(f"TRANSCRIPT ({time.time() - t0:.1f}s)")
        print("=" * 80)
        print(query_text)

    run_pipeline(query_text, args.k, args.mode, args.out,
                 kokoro, gemini, speak=not args.no_speak)


if __name__ == "__main__":
    main()