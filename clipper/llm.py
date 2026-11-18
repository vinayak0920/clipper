"""AI editing: judges the moment, tightens start/end, writes the hook, turns Hindi into Hinglish.

Runs on whichever provider is available:
  your own key (any of the four below) - used first, unlimited, never touches Clipper Cloud
  cloud      - Clipper Cloud, when signed in; counts against monthly hook credits
  ollama     - free, runs on your own machine (https://ollama.com), no key
  anthropic  - ANTHROPIC_API_KEY
  openai     - OPENAI_API_KEY (also any OpenAI-compatible server via OPENAI_BASE_URL)
  groq       - GROQ_API_KEY
  gemini     - GEMINI_API_KEY
AI_PROVIDER=auto picks: own key, then Clipper Cloud, then Ollama. AI_PROVIDER=off disables it,
and clips then get a template title.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx

from . import cloud, config  # noqa: F401  (config loads .env)

log = logging.getLogger("llm")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# A clip never waits longer than this for the AI. Past it, the clip renders with a plain trim and title.
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "12"))
DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
}
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_cache = {"at": 0.0, "provider": None}


def _ollama_up() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        want = model_for("ollama")
        return any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
    except Exception:
        return False


def provider() -> str | None:
    """Resolved provider name, or None when AI editing is off. Cached for 30s."""
    if time.time() - _cache["at"] < 30:
        return _cache["provider"]
    choice = os.getenv("AI_PROVIDER", "auto").strip().lower()
    found = None
    if choice == "off":
        found = None
    elif choice == "cloud":
        found = "cloud" if cloud.signed_in() else None
    elif choice != "auto":
        found = choice if (choice == "ollama" or os.getenv(KEY_ENV.get(choice, ""), "")) else None
    else:
        found = next((p for p, env in KEY_ENV.items() if os.getenv(env)), None)
        if not found and cloud.signed_in():
            found = "cloud"
        if not found and _ollama_up():
            found = "ollama"
    _cache.update(at=time.time(), provider=found)
    return found


def reset():
    _cache["at"] = 0.0


def model_for(p: str) -> str:
    return os.getenv(f"{p.upper()}_MODEL") or DEFAULT_MODELS.get(p, "")


def enabled() -> bool:
    return provider() is not None


def describe() -> str:
    p = provider()
    if p == "cloud":
        return "Clipper Cloud"
    return f"{p} ({model_for(p)})" if p else ""


def _ask(prompt: str, max_tokens: int = 800) -> str:
    p = provider()
    if not p:
        raise RuntimeError("no AI provider")
    model = model_for(p)
    if p == "ollama":
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0.4, "num_predict": max_tokens},
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=AI_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"]
    if p == "anthropic":
        r = httpx.post("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        }, json={"model": model, "max_tokens": max_tokens,
                 "messages": [{"role": "user", "content": prompt}]}, timeout=AI_TIMEOUT)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    if p in ("openai", "groq"):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if p == "openai" \
            else "https://api.groq.com/openai/v1"
        r = httpx.post(f"{base}/chat/completions", headers={
            "Authorization": f"Bearer {os.environ[KEY_ENV[p]]}",
        }, json={"model": model, "max_tokens": max_tokens, "temperature": 0.4,
                 "response_format": {"type": "json_object"},
                 "messages": [{"role": "user", "content": prompt}]}, timeout=AI_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if p == "gemini":
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4,
                                       "responseMimeType": "application/json"}},
            timeout=AI_TIMEOUT)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(x.get("text", "") for x in parts)
    raise RuntimeError(f"unknown AI_PROVIDER {p}")


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def judge(words: list, chat: list[str], keywords: list[str], channel: str,
          duration: float, reaction_at: float) -> dict | None:
    """words: list of captions.Word relative to clip start. Returns keep/start/end/title/reason."""
    if not enabled():
        return None
    if provider() == "cloud":
        try:
            return cloud.hook(words, chat, keywords, channel, duration, reaction_at)
        except cloud.OutOfCredits as e:
            log.info("cloud hooks: %s, falling back", e)
            if not _ollama_up():
                return None
            _cache.update(at=time.time(), provider="ollama")
        except Exception as e:
            log.warning("cloud hook failed: %s", e)
            return None
    lines = " ".join(f"[{w.start:.1f}] {w.text.strip()}" for w in words) or "(no speech)"
    prompt = f"""You edit short-form clips from live streams. Streamer: {channel}.
The raw clip is {duration:.1f}s long. Chat exploded at about {reaction_at:.1f}s (chat lags the video by 2-6s).
Chat messages during the spike: {json.dumps(chat[:15], ensure_ascii=False)}
Hype tokens: {keywords}
Transcript with start times in seconds:
{lines}

Decide:
1. keep: true if this works as a standalone short (a funny line, a big play, a reaction, drama). false for ads, AFK, chat spamming for no visible reason.
2. start and end in seconds: begin just before the setup, end right after the payoff/reaction. 12 to 59 seconds long. Never cut mid-sentence.
3. title: a hook for TikTok/Shorts, max 8 words, in the language/style the streamer speaks (Hinglish in Latin letters for Hindi streams). Make the viewer need to see what happens. No hashtags, no quotes, no emojis.
Reply with JSON only: {{"keep": true, "start": 0, "end": 30, "title": "...", "reason": "..."}}"""
    try:
        out = _json(_ask(prompt))
        if not isinstance(out, dict):
            return None
        s = max(0.0, float(out.get("start", 0)))
        e = min(duration, float(out.get("end", duration)))
        if e - s < 8:
            s, e = 0.0, duration
        keep = out.get("keep", True)
        if isinstance(keep, str):
            keep = keep.strip().lower() not in ("false", "no", "0")
        out.update(start=s, end=e, keep=bool(keep), title=str(out.get("title", "")).strip().strip('"'))
        return out
    except Exception as ex:
        log.warning("AI judge failed (%s): %s", provider(), ex)
        return None


def transliterate(words: list) -> list | None:
    """Devanagari -> Latin Hinglish, one output word per input word so timings stay intact."""
    if not enabled() or not words:
        return None
    if provider() == "cloud":
        try:
            return cloud.transliterate(words)
        except Exception as e:
            log.info("cloud transliterate skipped: %s", e)
            return None
    src = [w.text.strip() for w in words]
    prompt = (
        "Transliterate each item from Devanagari Hindi into casual Hinglish in Latin letters, the way Indians type in chat "
        "(e.g. 'भाई क्या कर रहा है' -> 'bhai kya kar raha hai'). Keep English words as English. "
        f"There are {len(src)} items. Reply with JSON only: {{\"words\": [one string per item, same order, same count]}}\n"
        + json.dumps(src, ensure_ascii=False)
    )
    try:
        out = _json(_ask(prompt, max_tokens=2000))
        arr = out.get("words") if isinstance(out, dict) else None
        if isinstance(arr, list) and len(arr) == len(src):
            return [str(x) for x in arr]
        log.warning("transliterate returned %s items for %s", len(arr or []), len(src))
    except Exception as ex:
        log.warning("transliterate failed: %s", ex)
    return None
