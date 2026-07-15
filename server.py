"""
FastAPI backend for the voice admissions advisor.

Models are loaded once at startup (lifespan) and kept warm in app state. Run
single-worker so they aren't duplicated across processes (VRAM).

    uvicorn server:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health              -> warm check
    POST /query               -> {text} -> {query, answer, claims, sources}
    POST /transcribe          -> audio file -> {transcript}
    POST /speak               -> {text} -> audio/wav (English, whole answer)
    POST /speak_stream        -> {text} -> int16 PCM stream (English)
    POST /speak_urdu          -> {text} -> audio/wav (via OmniVoice service)
    POST /speak_urdu_stream   -> {text} -> length-prefixed WAV stream (Urdu)
"""

from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_OUT = PROJECT_ROOT / "data" / "audio" / "output"
AUDIO_IN = PROJECT_ROOT / "data" / "audio" / "input"
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
AUDIO_IN.mkdir(parents=True, exist_ok=True)

# advisor package modules import each other with bare names.
sys.path.insert(0, str(PROJECT_ROOT / "advisor"))

# Let CTranslate2 find torch's cuDNN 9 DLLs on Windows.
if sys.platform == "win32":
    try:
        import os
        import torch
        _lib = Path(torch.__file__).resolve().parent / "lib"
        if _lib.is_dir():
            os.add_dll_directory(str(_lib))
    except Exception:
        pass

load_dotenv(PROJECT_ROOT / ".env")

import stt
import tts
from generate import answer_query, get_client

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = PROJECT_ROOT / "static"

# Handles populated at startup, reused across requests.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] warming up models...")
    STATE["whisper"] = stt.load_model()
    STATE["kokoro"] = tts.load_pipeline()
    STATE["gemini"] = get_client()
    import retrieve
    retrieve.warm_embedder()   # load bge-m3 now, else the first query is slow
    print("[server] warm-up complete.")
    yield
    STATE.clear()


app = FastAPI(title="Admissions Advisor", lifespan=lifespan)


class QueryIn(BaseModel):
    text: str
    k: int = 6
    mode: str = "dense"
    language: str = "en"


class SpeakIn(BaseModel):
    text: str
    language: str = "en"


# OmniVoice Urdu TTS microservice (runs in omni-venv on a separate port).
OMNI_SERVICE_URL = "http://127.0.0.1:8800"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(STATE.keys())}


@app.post("/query")
def query(body: QueryIn):
    """Retrieve + generate. Returns the structured answer (no audio here)."""
    if not body.text.strip():
        raise HTTPException(400, "empty query")
    lang = body.language if body.language in ("en", "ur") else "en"
    results, parsed = answer_query(body.text, body.k, body.mode, STATE["gemini"],
                                   language=lang)
    if not results or parsed is None:
        return {"query": body.text, "answer": None,
                "claims": [], "sources": []}

    # Map sources_used -> URLs grouped by institution.
    sources: dict[str, list[str]] = {}
    for n in parsed.sources_used:
        if 1 <= n <= len(results):
            meta = results[n - 1].get("meta", {})
            ent = meta.get("entity_id", "?")
            url = meta.get("source_url", "")
            if url and url not in sources.setdefault(ent, []):
                sources[ent].append(url)

    def resolve_claim_sources(nums):
        out = []
        for n in nums:
            if 1 <= n <= len(results):
                meta = results[n - 1].get("meta", {})
                out.append({
                    "n": n,
                    "entity": meta.get("entity_id", "?"),
                    "heading": meta.get("headings", ""),
                    "url": meta.get("source_url", ""),
                })
        return out

    return {
        "query": body.text,
        "answer": parsed.answer,
        "language": lang,
        "claims": [{"text": c.text, "sources": resolve_claim_sources(c.sources)}
                   for c in parsed.claims],
        "sources": sources,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    """Save a recorded audio file (e.g. WebM from the browser mic) and
    transcribe it with the warm Whisper model. The language comes from the
    frontend toggle rather than auto-detection, which tends to read Urdu as
    Hindi."""
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    lang = language if language in ("en", "ur") else "en"
    ext = Path(audio.filename or "rec.webm").suffix or ".webm"
    in_path = AUDIO_IN / f"rec_{uuid.uuid4().hex[:8]}{ext}"
    in_path.write_bytes(raw)
    try:
        text = stt.transcribe(str(in_path), model=STATE["whisper"], language=lang)
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")
    return {"transcript": text, "language": lang}


@app.post("/speak")
def speak(body: SpeakIn):
    """Synthesize text to a WAV and return it. (Non-streaming fallback.)"""
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    out_path = AUDIO_OUT / f"answer_{uuid.uuid4().hex[:8]}.wav"
    tts.speak(body.text, out_path=str(out_path), pipeline=STATE["kokoro"])
    return FileResponse(str(out_path), media_type="audio/wav",
                        filename=out_path.name)


def _split_sentences_ur(text: str) -> list[str]:
    """Split Urdu/English text into sentence-ish chunks for streamed TTS, on
    the Urdu full stop, question marks and periods. Short fragments are
    merged so chunks aren't tiny."""
    import re
    parts = re.split(r'(?<=[۔؟\?\.])\s+', text.strip())
    chunks, buf = [], ""
    for p in parts:
        if not p.strip():
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= 40:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks or [text]


@app.post("/speak_urdu_stream")
def speak_urdu_stream(body: SpeakIn):
    """Stream Urdu TTS sentence by sentence, so the browser can start playing
    the first sentence while the rest is still synthesizing.

    Wire format: for each chunk, a 4-byte big-endian length prefix followed by
    that chunk's WAV bytes. The browser reads [len][wav][len][wav]..."""
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    import httpx, struct

    sentences = _split_sentences_ur(body.text)

    def gen():
        with httpx.Client(timeout=120.0) as client:
            for i, sent in enumerate(sentences):
                try:
                    r = client.post(f"{OMNI_SERVICE_URL}/synthesize",
                                    json={"text": sent,
                                          # keep model on GPU between chunks;
                                          # only offload after the last one
                                          "offload_after": i == len(sentences) - 1})
                    if r.status_code != 200:
                        print(f"[urdu_stream] chunk {i} failed: {r.status_code}")
                        continue
                    wav = r.content
                    yield struct.pack(">I", len(wav)) + wav
                except Exception as e:
                    print(f"[urdu_stream] chunk {i} error: {e}")
                    continue

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": "24000", "Cache-Control": "no-cache"},
    )


@app.post("/speak_urdu")
def speak_urdu(body: SpeakIn):
    """Route Urdu TTS to the OmniVoice microservice and return its WAV.
    OmniVoice manages its own GPU load/offload."""
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    import httpx
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{OMNI_SERVICE_URL}/synthesize",
                            json={"text": body.text})
        if r.status_code != 200:
            raise HTTPException(502, f"omni service error: {r.status_code} {r.text}")
    except httpx.ConnectError:
        raise HTTPException(503,
            "OmniVoice service not reachable. Start it in omni-venv: "
            "uvicorn omni_service:app --port 8800")
    return Response(content=r.content, media_type="audio/wav",
                    headers={"X-Sample-Rate": r.headers.get("X-Sample-Rate", "24000")})


@app.post("/speak_stream")
def speak_stream(body: SpeakIn):
    """Stream raw int16 PCM (mono, 24kHz) per Kokoro segment, so the client
    can begin playback before synthesis finishes."""
    if not body.text.strip():
        raise HTTPException(400, "empty text")

    def gen():
        for pcm_bytes in tts.stream_pcm(body.text, pipeline=STATE["kokoro"]):
            yield pcm_bytes

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(tts.SAMPLE_RATE),
            "Cache-Control": "no-cache",
        },
    )