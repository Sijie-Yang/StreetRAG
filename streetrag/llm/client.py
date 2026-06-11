"""OpenAI client: structured outputs, embeddings, function calling."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T")


def _registry_dir(registry_path: str) -> Path:
    return Path(registry_path).parent


def load_rag_settings(registry_path: str) -> dict:
    """Merge RAG settings: workspace-global first, then per-city overrides
    (multi-city layout puts registries in data/cities/<name>/)."""
    d = _registry_dir(registry_path)
    dirs = []
    if d.parent.name == "cities":
        dirs.append(d.parent.parent)
    dirs.append(d)
    settings: dict = {}
    for folder in dirs:
        for fname in ("RAG_setting.json", "RAG_setting.local.json"):
            p = folder / fname
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    settings.update(json.load(f))
    return settings


def resolve_api_key(settings: dict) -> str:
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    return (settings.get("openai_api_key") or "").strip()


def require_api_key(settings: dict) -> str:
    key = resolve_api_key(settings)
    if not key or key.startswith("your_"):
        raise SystemExit(
            "OpenAI API key not configured.\n"
            "  Set environment variable: export OPENAI_API_KEY=sk-...\n"
            "  Or create data/RAG_setting.local.json with {\"openai_api_key\": \"sk-...\"}"
        )
    return key


def llm_logs_dir(registry_path: str) -> Path:
    d = _registry_dir(registry_path) / "llm_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def llm_cache_dir(registry_path: str) -> Path:
    d = _registry_dir(registry_path) / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def embeddings_paths(registry_path: str) -> tuple[Path, Path]:
    d = _registry_dir(registry_path)
    return d / "embeddings.npz", d / "embeddings.meta.json"


def _cache_key(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def _load_cached(registry_path: str, key: str) -> Optional[dict]:
    p = llm_cache_dir(registry_path) / f"{key}.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cached(registry_path: str, key: str, payload: dict) -> None:
    p = llm_cache_dir(registry_path) / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _log_call(
    registry_path: str,
    *,
    prompt: str,
    system: str,
    response_text: str,
    model: str,
    seed: Optional[int],
    temperature: float,
    schema_name: Optional[str],
) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "schema": schema_name,
        "system": system,
        "prompt": prompt,
        "response": response_text,
    }
    fname = time.strftime("%Y%m%d-%H%M%S") + "-" + _cache_key(rec)[:8] + ".json"
    with open(llm_logs_dir(registry_path) / fname, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


def _build_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Package 'openai' not installed. Run: pip install -e ."
        ) from e
    return OpenAI(api_key=api_key)


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    if "chat" in m:  # gpt-5.x-chat-latest behave like classic chat models
        return False
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _sampling_kwargs(model: str, temperature: float, seed: Optional[int]) -> dict:
    """Reasoning models (gpt-5*, o-series) reject custom temperature/seed."""
    if _is_reasoning_model(model):
        return {}
    kw: dict = {"temperature": temperature}
    if seed is not None:
        kw["seed"] = seed
    return kw


def chat_text(
    *,
    settings: dict,
    registry_path: str,
    system: str,
    prompt: str,
    use_cache: bool = True,
) -> str:
    api_key = require_api_key(settings)
    model = settings.get("llm_model", "gpt-4o-mini")
    temperature = float(settings.get("llm_temperature", 0.0))
    seed = settings.get("llm_seed", 42)
    max_retries = int(settings.get("llm_max_retries", 3))
    cache_payload = {
        "kind": "chat_text",
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "system": system,
        "prompt": prompt,
    }
    key = _cache_key(cache_payload)
    if use_cache:
        cached = _load_cached(registry_path, key)
        if cached and "response" in cached:
            return cached["response"]
    client = _build_client(api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **_sampling_kwargs(model, temperature, seed),
            )
            text = resp.choices[0].message.content or ""
            _log_call(
                registry_path,
                prompt=prompt,
                system=system,
                response_text=text,
                model=model,
                seed=seed,
                temperature=temperature,
                schema_name=None,
            )
            _save_cached(registry_path, key, {"response": text, **cache_payload, "ts": time.time()})
            return text
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"LLM chat_text failed after {max_retries} retries: {last_err}")


def chat_structured(
    *,
    settings: dict,
    registry_path: str,
    system: str,
    prompt: str,
    schema_model: Type[T],
    use_cache: bool = True,
) -> T:
    if not issubclass(schema_model, BaseModel):
        raise RuntimeError("chat_structured requires a pydantic BaseModel subclass")
    api_key = require_api_key(settings)
    model = settings.get("llm_model", "gpt-4o-mini")
    temperature = float(settings.get("llm_temperature", 0.0))
    seed = settings.get("llm_seed", 42)
    max_retries = int(settings.get("llm_max_retries", 3))
    cache_payload = {
        "kind": "chat_structured",
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "system": system,
        "prompt": prompt,
        "schema_name": schema_model.__name__,
    }
    key = _cache_key(cache_payload)
    if use_cache:
        cached = _load_cached(registry_path, key)
        if cached and "response_json" in cached:
            try:
                return schema_model(**cached["response_json"])
            except Exception:
                pass
    client = _build_client(api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            parsed = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format=schema_model,
                **_sampling_kwargs(model, temperature, seed),
            )
            obj: T = parsed.choices[0].message.parsed
            obj_json = obj.model_dump()
            _log_call(
                registry_path,
                prompt=prompt,
                system=system,
                response_text=json.dumps(obj_json, ensure_ascii=False),
                model=model,
                seed=seed,
                temperature=temperature,
                schema_name=schema_model.__name__,
            )
            _save_cached(
                registry_path, key,
                {"response_json": obj_json, **cache_payload, "ts": time.time()},
            )
            return obj
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"LLM chat_structured failed after {max_retries} retries: {last_err}")


def chat_tools(
    *,
    settings: dict,
    registry_path: str,
    system: str,
    prompt: str,
    tools: List[dict],
    use_cache: bool = True,
) -> dict:
    """Function-calling completion. Returns {tool_name, arguments}."""
    api_key = require_api_key(settings)
    model = settings.get("llm_model", "gpt-4o-mini")
    temperature = float(settings.get("llm_temperature", 0.0))
    seed = settings.get("llm_seed", 42)
    max_retries = int(settings.get("llm_max_retries", 3))
    cache_payload = {
        "kind": "chat_tools",
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "system": system,
        "prompt": prompt,
        "tools": tools,
    }
    key = _cache_key(cache_payload)
    if use_cache:
        cached = _load_cached(registry_path, key)
        if cached and "tool_call" in cached:
            return cached["tool_call"]
    client = _build_client(api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice="required",
                **_sampling_kwargs(model, temperature, seed),
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                raise RuntimeError("No tool call returned")
            tc = msg.tool_calls[0]
            result = {
                "tool_name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            _log_call(
                registry_path,
                prompt=prompt,
                system=system,
                response_text=json.dumps(result, ensure_ascii=False),
                model=model,
                seed=seed,
                temperature=temperature,
                schema_name=tc.function.name,
            )
            _save_cached(
                registry_path, key,
                {"tool_call": result, **cache_payload, "ts": time.time()},
            )
            return result
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"LLM chat_tools failed after {max_retries} retries: {last_err}")


def embed_texts(
    *,
    settings: dict,
    registry_path: str,
    texts: List[str],
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    api_key = require_api_key(settings)
    model = settings.get("embedding_model", "text-embedding-3-small")
    client = _build_client(api_key)
    batch_size = 256
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t if (t and t.strip()) else " " for t in batch]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    if mat.size == 0 or query_vec.size == 0:
        return []
    q = query_vec.astype(np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = np.linalg.norm(mat, axis=1, keepdims=True)
    mn = np.where(mn < 1e-12, 1.0, mn)
    normed = mat / mn
    sims = normed @ qn
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        order = idx[np.argsort(-sims[idx])]
    return [int(i) for i in order]
