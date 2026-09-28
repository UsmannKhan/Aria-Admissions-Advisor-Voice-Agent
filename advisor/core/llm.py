"""Generation client and model selection.

generate() dispatches on the model id: unprefixed ids (gemini-3.7-flash,
gemma-4-31b-it) use google-genai, openrouter:<id> goes through OpenRouter's
OpenAI-compatible API, local:<name> hits LOCAL_OPENAI_BASE_URL. The sweep
switches models through the ADVISOR_*_MODEL env vars, not code.

Gemini gets server-side constrained decoding (response_schema). The OpenAI
path asks for the schema, validates with pydantic, repairs once and counts
failures per model.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

ANSWER_MODEL = os.environ.get("ADVISOR_ANSWER_MODEL", "gemini-3.7-flash")
PLANNER_MODEL = os.environ.get("ADVISOR_PLANNER_MODEL", "gemini-3.5-flash-lite")

# not ANSWER_MODEL, so the verifier never judges its own model; fixed across
# the sweep so every generator gets the same judge
VERIFIER_MODEL = os.environ.get("ADVISOR_VERIFIER_MODEL", "gemini-3.5-flash-lite")

# default 'high' cost 16s per call for no gain
THINKING_LEVEL = "low"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# one OpenRouter id can be served by several providers at different
# quantisations; pin one for reproducible cells
OPENROUTER_PROVIDER = os.environ.get("ADVISOR_OPENROUTER_PROVIDER", "")
LOCAL_BASE_URL = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")

_RETRIES = 2          # transient API failures only; schema repair is separate
_RETRY_WAIT = 2.0

_client: "genai.Client | None" = None
_oai_clients: dict[str, Any] = {}
_lock = threading.Lock()


def get_client(api_key: str | None = None) -> "genai.Client":
    """One client per process; agents run on separate threads."""
    global _client
    if _client is not None and api_key is None:
        return _client

    key = (api_key
           or os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GOOGLE_API_KEY"))
    if not key:
        print(f"ERROR: no GEMINI_API_KEY in the environment or "
              f"{PROJECT_ROOT / '.env'}", file=sys.stderr)
        sys.exit(1)

    if api_key is not None:
        return genai.Client(api_key=key)

    with _lock:
        if _client is None:
            # hard timeout: under rate limiting a wedged connection hung
            # forever and even Ctrl+C was deferred
            _client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=120_000))   # ms
    return _client


def _split_backend(model: str) -> tuple[str, str]:
    """('openrouter', 'qwen/qwen3-32b') from 'openrouter:qwen/qwen3-32b'.
    Anything unprefixed is Gemini, so the existing model ids keep working."""
    backend, sep, rest = model.partition(":")
    if sep and backend in ("openrouter", "local"):
        return backend, rest
    return "gemini", model


def _openai_client(backend: str):
    """One client per backend, same reasoning as get_client()."""
    if backend in _oai_clients:
        return _oai_clients[backend]
    from openai import OpenAI

    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(f"no OPENROUTER_API_KEY in the environment or "
                               f"{PROJECT_ROOT / '.env'}")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    else:
        # Ollama and vLLM accept any non-empty key.
        client = OpenAI(base_url=LOCAL_BASE_URL,
                        api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "local"))

    with _lock:
        _oai_clients.setdefault(backend, client)
    return _oai_clients[backend]


@dataclass
class GenResult:
    text: str
    parsed: Any = None            # pydantic instance when a schema was given
    model: str = ""
    backend: str = ""
    usage: dict = field(default_factory=dict)


# per-model usage for the cost table. schema_repairs: needed the repair call;
# schema_failures: unusable even after it.
_usage: dict[str, dict] = {}


def _account(model: str, usage: dict, repaired: bool = False,
             failed: bool = False, provider: str = "") -> None:
    with _lock:
        row = _usage.setdefault(model, {"calls": 0, "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "thinking_tokens": 0,
                                        "schema_repairs": 0,
                                        "schema_failures": 0,
                                        "providers": {}})
        row["calls"] += 1
        if provider:
            row["providers"][provider] = row["providers"].get(provider, 0) + 1
        row["prompt_tokens"] += usage.get("prompt_tokens", 0)
        row["completion_tokens"] += usage.get("completion_tokens", 0)
        # billed but reported separately; without it reasoning models are
        # under-counted
        row["thinking_tokens"] += usage.get("thinking_tokens", 0)
        row["schema_repairs"] += int(repaired)
        row["schema_failures"] += int(failed)


def usage_summary() -> dict[str, dict]:
    with _lock:
        return {m: dict(row) for m, row in _usage.items()}


def reset_usage() -> None:
    with _lock:
        _usage.clear()


def _gen_gemini(model: str, contents: str, system: str | None,
                schema: type[BaseModel] | None, thinking: str) -> GenResult:
    # Gemma on the Google API only accepts thinking "minimal" (off) or "high"
    # (on). Our "low" maps to off, the cheap setting on both families.
    if model.startswith("gemma"):
        thinking = "minimal" if thinking in ("low", "minimal") else "high"
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
        **({"response_mime_type": "application/json",
            "response_schema": schema} if schema else {}),
    )
    # own retries: Gemma's free tier is 16K tokens/min and one RAG prompt is a
    # third of that, so a 429 means wait for the next minute
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = get_client().models.generate_content(
                model=model, contents=contents, config=cfg)
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
                wait = 20.0 * (attempt + 1)          # 20/40/60s: TPM windows
            elif attempt < 3:
                wait = _RETRY_WAIT * (attempt + 1)
            else:
                raise
            if attempt == 3:
                raise
            print(f"[llm] {model}: rate limited, waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/3)")
            time.sleep(wait)
    else:
        raise last_exc

    meta = getattr(resp, "usage_metadata", None)
    usage = {"prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
             "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
             "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0}
    _account(model, usage)
    return GenResult(text=resp.text or "",
                     parsed=resp.parsed if schema else None,
                     model=model, backend="gemini", usage=usage)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _extract_json(text: str) -> str:
    """JSON object from whatever the model wrapped it in (code fence, a
    sentence first): first '{' to last '}'."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _schema_clause(schema: type[BaseModel]) -> str:
    return ("\n\nReply with ONE JSON object matching this JSON Schema exactly. "
            "No prose before or after it, no markdown fences.\nSchema:\n"
            + json.dumps(schema.model_json_schema()))


def _chat(client, model: str, messages: list[dict], backend: str,
          thinking: str, response_format: dict | None) -> Any:
    """One chat call. Retries transient failures and drops optional params
    the provider rejects."""
    kwargs: dict = {"model": model, "messages": messages}
    if response_format:
        kwargs["response_format"] = response_format
    if backend == "openrouter":
        extra: dict = {}
        if thinking:
            # OpenRouter's normalised reasoning control; unsupported models
            # ignore it. exclude keeps reasoning tokens out of the content.
            extra["reasoning"] = {"effort": thinking, "exclude": True}
        if OPENROUTER_PROVIDER:
            # preferred provider, with fallback if it's down; the provider that
            # served each call is logged in usage
            extra["provider"] = {"order": [p.strip() for p in OPENROUTER_PROVIDER.split(",")],
                                 "allow_fallbacks": True}
        if extra:
            kwargs["extra_body"] = extra

    last_exc: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            # a 4xx about a param will fail the same way again, so retry
            # without the param
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            if "reasoning" in msg and "extra_body" in kwargs:
                kwargs["extra_body"].pop("reasoning", None)
                if not kwargs["extra_body"]:
                    kwargs.pop("extra_body")
                continue
            if attempt < _RETRIES:
                if "429" in msg:
                    time.sleep(20.0 * (attempt + 1))   # rate-limit window
                else:
                    time.sleep(_RETRY_WAIT * (attempt + 1))
    raise last_exc


def _reasoning_tokens(usage) -> int:
    """OpenAI-compatible responses put reasoning tokens under
    completion_tokens_details; absent on models that do not reason."""
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


def _gen_openai(backend: str, model: str, contents: str, system: str | None,
                schema: type[BaseModel] | None, thinking: str) -> GenResult:
    client = _openai_client(backend)

    system = system or ""
    response_format = None
    if schema:
        # schema in the prompt (some providers ignore response_format) and in
        # response_format. strict is off: it rejects defaulted fields, which
        # Answer and _Judgement have.
        system += _schema_clause(schema)
        response_format = {"type": "json_schema",
                           "json_schema": {"name": schema.__name__.strip("_"),
                                           "schema": schema.model_json_schema()}}

    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": contents}]

    resp = _chat(client, model, messages, backend, thinking, response_format)
    text = (resp.choices[0].message.content or "").strip()
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
             "thinking_tokens": _reasoning_tokens(u)}
    full = f"{backend}:{model}"
    provider = getattr(resp, "provider", "") or ""   # OpenRouter names the upstream

    if not schema:
        _account(full, usage, provider=provider)
        return GenResult(text=text, model=model, backend=backend, usage=usage)

    repaired = False
    for round_ in range(2):
        try:
            parsed = schema.model_validate(json.loads(_extract_json(text)))
            _account(full, usage, repaired=repaired, provider=provider)
            return GenResult(text=text, parsed=parsed, model=model,
                             backend=backend, usage=usage)
        except Exception as exc:
            if round_ == 1:
                break
            repaired = True
            # one repair attempt with the output and the error; more rounds
            # cost too much latency in a voice turn
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                 f"That was not valid against the schema ({exc}). "
                 "Reply again with ONLY the corrected JSON object."}]
            resp = _chat(client, model, messages, backend, thinking,
                         response_format)
            text = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            usage["thinking_tokens"] += _reasoning_tokens(u)

    _account(full, usage, repaired=repaired, failed=True, provider=provider)
    raise ValueError(f"{full} returned no valid {schema.__name__} "
                     f"after repair: {text[:200]!r}")


def generate(model: str, contents: str, system: str | None = None,
             schema: type[BaseModel] | None = None,
             thinking: str = THINKING_LEVEL) -> GenResult:
    """One generation on any backend. With a schema, .parsed is validated or
    the call raises (callers treat that like an outage)."""
    backend, name = _split_backend(model)
    if backend == "gemini":
        return _gen_gemini(name, contents, system, schema, thinking)
    return _gen_openai(backend, name, contents, system, schema, thinking)


def warm() -> None:
    """Build clients for the configured models at startup, so a missing key
    fails the boot, not the first question."""
    for m in {ANSWER_MODEL, PLANNER_MODEL, VERIFIER_MODEL}:
        backend, _ = _split_backend(m)
        if backend == "gemini":
            get_client()
        else:
            _openai_client(backend)
