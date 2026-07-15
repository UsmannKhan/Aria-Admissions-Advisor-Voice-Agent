"""
OmniVoice Urdu TTS microservice. Runs in the separate omni-venv (its deps
conflict with the main pipeline). The main server calls this over HTTP only
for Urdu answers.

Run (in omni-venv, from project root):
    omni-venv\\Scripts\\activate
    uvicorn omni_service:app --host 127.0.0.1 --port 8800

this service only holds the GPU for the duration of a /synthesize call and offloads
to CPU after. The main server must offload Whisper before calling.

Voice: clones data/audio/sample/urdu_sample.wav for a consistent Urdu voice.
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
REF_AUDIO = PROJECT_ROOT / "data" / "audio" / "sample" / "urdu_sample.wav"
# Transcript of REF_AUDIO. Passing it saves OmniVoice from loading its own
# Whisper to transcribe the reference.
REF_TEXT = "آرٹیفیشل انٹیلیجنس اور جدید ٹیکنالوجی نے ہمارے کام کرنے کے انداز کو مکمل طور پر بدل دیا ہے۔ اب ہم اس نئے سسٹم کے ذریعے، اپنی آواز کو مزید بہتر اور واضح بنا سکتے ہیں۔"
SAMPLE_RATE = 24000

STATE: dict = {}


def _load_to_cpu():
    """Read the weights into CPU RAM at startup, so an Urdu request pays only
    the CPU->GPU move, not the disk read + deserialize."""
    from omnivoice import OmniVoice
    print("[omni] loading OmniVoice weights to CPU (one-time startup cost)...")
    t0 = time.time()
    # Loading to GPU is the documented path; push to CPU straight after.
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
    )
    model.to("cpu")
    torch.cuda.empty_cache()
    STATE["model"] = model
    STATE["on_gpu"] = False
    print(f"[omni] weights ready in CPU RAM in {time.time() - t0:.1f}s "
          "(GPU free until first Urdu request)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_to_cpu()
    _warmup()
    yield
    STATE.clear()


def _warmup():
    """Throwaway synthesis to compile and cache the CUDA kernels / diffusion
    sampler. Without it the first real request pays about 25s."""
    if not REF_AUDIO.exists():
        print("[omni] skip warmup: reference audio missing")
        return
    try:
        to_gpu()
        t0 = time.time()
        _ = STATE["model"].generate(text="سلام", ref_audio=str(REF_AUDIO),
                                    ref_text=REF_TEXT)
        print(f"[omni] warmup synth in {time.time() - t0:.1f}s "
              "(first real request will be fast)")
    except Exception as e:
        print(f"[omni] warmup failed (non-fatal): {e}")
    finally:
        to_cpu()


def to_gpu():
    """Move cached weights CPU -> GPU (no disk read)."""
    m = STATE["model"]
    if not STATE.get("on_gpu", False):
        t0 = time.time()
        m.to("cuda:0")
        STATE["on_gpu"] = True
        print(f"[omni] CPU -> GPU in {time.time() - t0:.2f}s")


def to_cpu():
    if STATE.get("on_gpu", False):
        STATE["model"].to("cpu")
        torch.cuda.empty_cache()
        STATE["on_gpu"] = False
        print("[omni] GPU -> CPU (VRAM freed)")


app = FastAPI(title="OmniVoice Urdu TTS", lifespan=lifespan)


class SynthIn(BaseModel):
    text: str
    offload_after: bool = True  # free GPU when done so Whisper can reload


@app.get("/health")
def health():
    return {"status": "ok", "loaded": "model" in STATE,
            "on_gpu": STATE.get("on_gpu", False),
            "ref_audio_exists": REF_AUDIO.exists()}


@app.post("/load")
def load():
    """Force weights onto the GPU now (caller should have freed VRAM first)."""
    to_gpu()
    return {"on_gpu": STATE.get("on_gpu", False)}


@app.post("/offload")
def offload():
    """Push the model off the GPU to free VRAM for Whisper."""
    to_cpu()
    return {"on_gpu": STATE.get("on_gpu", False)}


@app.post("/synthesize")
def synthesize(body: SynthIn):
    """Voice-clone the Urdu sample to speak `text`. Returns a 16-bit PCM WAV.
    Generate-and-return: RTF is ~0.15, so streaming isn't needed."""
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    if not REF_AUDIO.exists():
        raise HTTPException(500, f"reference audio not found: {REF_AUDIO}")
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded yet")

    model = STATE["model"]
    to_gpu()
    t0 = time.time()
    try:
        audio = model.generate(text=body.text, ref_audio=str(REF_AUDIO),
                               ref_text=REF_TEXT)
    except Exception as e:
        raise HTTPException(500, f"synthesis failed: {e}")
    finally:
        if body.offload_after:
            to_cpu()

    # audio is a list of torch.Tensor shape (1, T) at 24kHz.
    arr = audio[0]
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr).squeeze()
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype("<i2")

    buf = io.BytesIO()
    sf.write(buf, pcm16, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    dur = len(arr) / SAMPLE_RATE
    print(f"[omni] synth {len(body.text)} chars -> {dur:.1f}s audio "
          f"in {time.time() - t0:.2f}s")
    return Response(content=buf.read(), media_type="audio/wav",
                    headers={"X-Sample-Rate": str(SAMPLE_RATE)})