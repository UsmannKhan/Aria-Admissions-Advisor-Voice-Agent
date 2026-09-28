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

import sqlite3
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

from advisor import stt, tts
from advisor.baseline.pipeline import answer_query
from advisor.multi_agent import graph
from advisor.multi_agent import verify as V
from advisor.core import llm

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
    llm.warm()   # fail on a missing API key now, not on the first question
    from advisor.core import retrieval
    retrieval.warm_embedder()   # load bge-m3 now, else the first query is slow
    retrieval.get_corpus(retrieval.get_collection())   # BM25 index, once
    graph.warm_graph()          # compile into the global ask() uses
    print("[server] warm-up complete.")
    yield
    STATE.clear()


app = FastAPI(title="Admissions Advisor", lifespan=lifespan)


class QueryIn(BaseModel):
    text: str
    k: int = 6
    mode: str = "dense"
    language: str = "en"
    # one conversation per browser session; history and profile are restored
    # from the checkpointer under this key
    session: str = "web"
    # "full" = multi-agent graph, "baseline" = single-agent arm (both kept for
    # side-by-side demos)
    arm: str = "full"


class SpeakIn(BaseModel):
    text: str
    language: str = "en"
    # spoken forms for the figures in `text` (from /query). Swapped in just
    # before TTS since the synthesiser can't read bare digits; the displayed
    # answer keeps the digits.
    readings: list[dict] = []


# OmniVoice Urdu TTS microservice (runs in omni-venv on a separate port).
OMNI_SERVICE_URL = "http://127.0.0.1:8800"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(STATE.keys())}


def _resolve(results: list[dict], nums) -> list[dict]:
    """Turn 1-based source numbers into what the grounding panel shows."""
    out = []
    for n in nums:
        if 1 <= n <= len(results):
            meta = results[n - 1].get("meta", {})
            out.append({"n": n,
                        "entity": meta.get("entity_id", "?"),
                        "heading": meta.get("headings", ""),
                        "url": meta.get("source_url", "")})
    return out


def _sources_by_entity(results: list[dict], nums) -> dict[str, list[str]]:
    by_entity: dict[str, list[str]] = {}
    for s in _resolve(results, nums):
        if s["url"] and s["url"] not in by_entity.setdefault(s["entity"], []):
            by_entity[s["entity"]].append(s["url"])
    return by_entity


@app.post("/query")
def query(body: QueryIn):
    """Answer one turn. No audio here; the client asks for that separately."""
    if not body.text.strip():
        raise HTTPException(400, "empty query")
    lang = body.language if body.language in ("en", "ur") else "en"

    if body.arm == "baseline":
        results, parsed = answer_query(body.text, body.k, body.mode,
                                       language=lang)
        if not results or parsed is None:
            return {"query": body.text, "answer": None, "claims": [],
                    "sources": {}, "arm": "baseline"}
        return {
            "query": body.text,
            "answer": parsed.answer,
            "language": lang,
            "arm": "baseline",
            "claims": [{"text": c.text, "sources": _resolve(results, c.sources)}
                       for c in parsed.claims],
            "sources": _sources_by_entity(results, parsed.sources_used),
        }

    out = graph.ask(body.text, k=body.k, session=body.session, language=lang)
    results = out.get("sources") or []
    claims = out.get("claims") or []

    # Claims already carry indices into the merged source list, renumbered by
    # assemble() when partials were combined.
    used = [n for c in claims for n in c.get("sources", [])]
    return {
        "query": body.text,
        "answer": out.get("answer"),
        "language": lang,
        "arm": "full",
        "claims": [{"text": c["text"], "sources": _resolve(results, c.get("sources", []))}
                   for c in claims],
        "sources": _sources_by_entity(results, used or range(1, len(results) + 1)),
        # what the verifier changed, for the panel and for debugging
        "violations": out.get("violations") or [],
        # spoken forms for the figures; the client sends them back with /speak
        "readings": out.get("readings") or [],
        "profile": out.get("profile") or {},
    }


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    """Delete a conversation's checkpoints (history, profile, retrieved
    chunks). A few dozen test conversations ran to megabytes."""
    if not session_id.strip():
        raise HTTPException(400, "empty session id")
    try:
        conn = sqlite3.connect(str(graph.SESSIONS_DB))
        with conn:
            for table in ("writes", "checkpoints"):
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
        conn.close()
    except sqlite3.Error as exc:
        raise HTTPException(500, f"could not clear session: {exc}")
    return {"cleared": session_id}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    """Transcribe an uploaded recording (e.g. browser WebM) with the warm
    Whisper. Language from the UI toggle; auto-detect reads Urdu as Hindi."""
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
    """Non-streaming fallback."""
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

    # substitute before splitting, or a reading can be cut across two chunks
    sentences = _split_sentences_ur(V.speakable(body.text, body.readings))

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
                            json={"text": V.speakable(body.text, body.readings)})
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