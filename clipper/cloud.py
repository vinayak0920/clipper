"""Client for Clipper Cloud (the hosted service).

Signed-in users get, without any keys of their own:
  * Twitch streamer discovery by region
  * AI hooks and Hinglish captions, counted against their monthly hook credits
Anything a user configures themselves (their own Twitch app, their own AI key, local Ollama)
is used directly and never touches the cloud.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from .config import write_env

log = logging.getLogger("cloud")
DEFAULT_URL = "https://apavin.com/api/clipper"   # override with CLIPPER_CLOUD_URL to run your own
_me_cache = {"at": 0.0, "data": None}


class OutOfCredits(Exception):
    pass


def url() -> str:
    return (os.getenv("CLIPPER_CLOUD_URL") or DEFAULT_URL).rstrip("/")


def token() -> str:
    return os.getenv("CLIPPER_TOKEN", "")


def signed_in() -> bool:
    return bool(token())


def _req(method: str, path: str, auth: bool = True, timeout: float = 30, **kw):
    headers = {"Authorization": f"Bearer {token()}"} if auth else {}
    r = httpx.request(method, url() + path, headers=headers, timeout=timeout, **kw)
    if r.status_code == 402:
        raise OutOfCredits(r.json().get("detail", "No hook credits left"))
    if r.status_code == 401 and auth:
        _me_cache.update(at=0.0, data=None)
    if r.status_code >= 400:
        try:
            msg = r.json().get("detail")
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(msg or f"Clipper Cloud answered {r.status_code}")
    return r.json()


# ------------------------------------------------------------ account
def login_url(return_url: str) -> str:
    from urllib.parse import quote
    return f"{url()}/v1/auth/google?return={quote(return_url, safe='')}"


def save_token(token: str) -> dict | None:
    """Keeps the token Google sign-in handed back, only after Clipper Cloud accepts it."""
    try:
        r = httpx.get(url() + "/v1/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    write_env({"CLIPPER_TOKEN": token})
    _me_cache.update(at=time.time(), data=r.json())
    return r.json()


def start_sign_in(email: str):
    return _req("POST", "/v1/auth/start", auth=False, json={"email": email})


def verify(email: str, code: str) -> dict:
    out = _req("POST", "/v1/auth/verify", auth=False, json={"email": email, "code": code})
    write_env({"CLIPPER_TOKEN": out["token"]})
    _me_cache.update(at=time.time(), data=out["me"])
    return out["me"]


def sign_out():
    write_env({"CLIPPER_TOKEN": ""})
    _me_cache.update(at=0.0, data=None)


def me(fresh: bool = False) -> dict | None:
    if not signed_in():
        return None
    if not fresh and _me_cache["data"] and time.time() - _me_cache["at"] < 30:
        return _me_cache["data"]
    try:
        data = _req("GET", "/v1/me", timeout=8)
    except Exception as e:
        log.debug("me failed: %s", e)
        return _me_cache["data"]
    _me_cache.update(at=time.time(), data=data)
    return data


def hooks_left() -> int:
    m = me()
    if m and m.get("unlimited"):
        return 10**9
    return int(m["hooks_left"]) if m else 0


def upgrade() -> dict:
    return _req("POST", "/v1/billing/subscribe", json={})


def cancel() -> dict:
    return _req("POST", "/v1/billing/cancel")


# ------------------------------------------------------------ services
def live(region: str, platforms: list[str], min_viewers: int) -> dict:
    return _req("GET", "/v1/live", params={"region": region, "platforms": ",".join(platforms),
                                            "min_viewers": min_viewers}, timeout=40)


def hook(words: list, chat: list[str], keywords: list[str], channel: str, duration: float, reaction_at: float) -> dict:
    out = _req("POST", "/v1/hook", timeout=120, json={
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
        "chat": chat, "keywords": keywords, "channel": channel,
        "duration": duration, "reaction_at": reaction_at,
    })
    _me_cache["at"] = 0.0
    return out


def transliterate(words: list) -> list[str]:
    out = _req("POST", "/v1/transliterate", timeout=120, json={
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words]})
    return out["words"]
