"""Find live streamers on Twitch and Kick, filtered by country / region."""
from __future__ import annotations

import logging
import time

import httpx

from . import cloud
from .config import settings
from .regions import REGIONS, matches_keywords

log = logging.getLogger("discovery")

try:
    from curl_cffi import requests as cffi_requests  # gets past Cloudflare on kick.com
except ImportError:
    cffi_requests = None


class _Token:
    def __init__(self):
        self.value = ""
        self.expires = 0.0

    def valid(self) -> bool:
        return bool(self.value) and time.time() < self.expires - 60


_twitch_token = _Token()
_kick_token = _Token()


# ---------------------------------------------------------------- Twitch
def twitch_token() -> str:
    if _twitch_token.valid():
        return _twitch_token.value
    if not (settings.twitch_client_id and settings.twitch_client_secret):
        raise RuntimeError("Sign in to Clipper, or add your own Twitch app in Settings")
    r = httpx.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": settings.twitch_client_id,
            "client_secret": settings.twitch_client_secret,
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    _twitch_token.value = j["access_token"]
    _twitch_token.expires = time.time() + j.get("expires_in", 3600)
    return _twitch_token.value


def _twitch_get(path: str, params) -> dict:
    r = httpx.get(
        f"https://api.twitch.tv/helix/{path}",
        params=params,
        headers={"Client-Id": settings.twitch_client_id, "Authorization": f"Bearer {twitch_token()}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def twitch_streams(language: str, pages: int = 1) -> list[dict]:
    out, cursor = [], None
    for _ in range(pages):
        params = {"language": language, "first": 100, "type": "live"}
        if cursor:
            params["after"] = cursor
        j = _twitch_get("streams", params)
        for s in j.get("data", []):
            thumb = (s.get("thumbnail_url") or "").replace("{width}", "440").replace("{height}", "248")
            out.append({
                "platform": "twitch",
                "channel": s["user_login"],
                "name": s.get("user_name") or s["user_login"],
                "title": s.get("title", ""),
                "category": s.get("game_name", ""),
                "viewers": s.get("viewer_count", 0),
                "language": s.get("language", language),
                "tags": s.get("tags") or [],
                "thumbnail": thumb,
                "started_at": s.get("started_at"),
            })
        cursor = (j.get("pagination") or {}).get("cursor")
        if not cursor:
            break
    return out


def twitch_lookup(login: str) -> dict | None:
    j = _twitch_get("streams", {"user_login": login})
    return j["data"][0] if j.get("data") else None


# ---------------------------------------------------------------- Kick
def kick_token() -> str:
    if _kick_token.valid():
        return _kick_token.value
    r = httpx.post(
        "https://id.kick.com/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": settings.kick_client_id,
            "client_secret": settings.kick_client_secret,
        },
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    _kick_token.value = j["access_token"]
    _kick_token.expires = time.time() + j.get("expires_in", 3600)
    return _kick_token.value


def _kick_row(s: dict, language: str) -> dict:
    cat = s.get("category") or {}
    if isinstance(s.get("categories"), list) and s["categories"]:
        cat = s["categories"][0]
    # kick.com's web feed has the livestream slug in "slug" and the channel slug in channel.slug
    channel = (s.get("channel") or {}).get("slug") or s.get("broadcaster_slug") or s.get("slug") or ""
    return {
        "platform": "kick",
        "channel": channel,
        "name": (s.get("channel") or {}).get("user", {}).get("username") or channel,
        "title": s.get("stream_title") or s.get("session_title") or "",
        "category": cat.get("name", ""),
        "viewers": s.get("viewer_count") or s.get("viewers") or 0,
        "language": s.get("language", language),
        "tags": s.get("custom_tags") or s.get("tags") or [],
        "thumbnail": _kick_thumb(s.get("thumbnail")),
        "started_at": s.get("started_at") or s.get("start_time"),
    }


def _kick_thumb(t) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        if t.get("src") or t.get("url"):
            return t.get("src") or t.get("url")
        srcset = t.get("srcset") or ""
        parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
        return parts[-1] if parts else ""
    return ""


KICK_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German",
    "it": "Italian", "tr": "Turkish", "ru": "Russian", "pl": "Polish", "ar": "Arabic", "ja": "Japanese",
    "ko": "Korean", "id": "Indonesian", "tl": "Filipino", "ur": "Urdu", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "mr": "Marathi", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "gu": "Gujarati",
    "vi": "Vietnamese", "th": "Thai",
}
_kick_web_cache = {"at": 0.0, "rows": []}


def _kick_web_all(pages: int = 6) -> list[dict]:
    """Page through Kick's public browse feed, then filter its mixed-language rows locally.

    Kick's web feed returns a small page even when `limit=100` is requested.
    Stopping when a page has fewer than 100 rows therefore only searched page 1.
    """
    if time.time() - _kick_web_cache["at"] < 60:
        return _kick_web_cache["rows"]
    if cffi_requests is None:
        raise RuntimeError("Install curl_cffi to query Kick's public browse feed")

    rows = []
    for page in range(1, pages + 1):
        r = cffi_requests.get(
            "https://kick.com/stream/livestreams/en",
            params={"page": page, "limit": 100, "sort": "desc"},
            impersonate="chrome", timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        if isinstance(data, dict):
            data = data.get("livestreams", [])
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        last_page = payload.get("last_page") or payload.get("lastPage")
        if last_page and page >= int(last_page):
            break

    _kick_web_cache.update(at=time.time(), rows=rows)
    return rows


def kick_streams(language: str) -> list[dict]:
    # Official public API (needs a free app from kick.com/settings/developer)
    if settings.kick_client_id and settings.kick_client_secret:
        r = httpx.get(
            "https://api.kick.com/public/v1/livestreams",
            params={"language": language, "limit": 100, "sort": "viewer_count"},
            headers={"Authorization": f"Bearer {kick_token()}"},
            timeout=15,
        )
        r.raise_for_status()
        return [_kick_row(s, language) for s in r.json().get("data", [])]
    # Fallback without API keys: scan Kick's mixed-language browse pages and
    # filter the returned rows by the stream's language locally.
    if cffi_requests is None:
        raise RuntimeError("Add KICK_CLIENT_ID / KICK_CLIENT_SECRET to .env or pip install curl_cffi")
    name = KICK_LANG_NAMES.get(language, language).lower()
    rows = []
    for s in _kick_web_all():
        stream_language = str(s.get("language") or s.get("language_code") or "").lower()
        # Some browse rows omit language; retain those rather than dropping a live channel.
        if stream_language and stream_language not in (name, language.lower()):
            continue
        rows.append(_kick_row(s, language))
    return rows


def kick_channel_info(slug: str) -> dict:
    """Returns channel id (used for chat), chatroom id and live status from kick.com's web API."""
    if cffi_requests is None:
        raise RuntimeError("pip install curl_cffi")
    r = cffi_requests.get(f"https://kick.com/api/v2/channels/{slug}", impersonate="chrome", timeout=15)
    r.raise_for_status()
    j = r.json()
    return {
        "channel_id": j.get("id"),
        "chatroom_id": (j.get("chatroom") or {}).get("id"),
        "is_live": bool(j.get("livestream")),
        "title": (j.get("livestream") or {}).get("session_title", ""),
    }


# ---------------------------------------------------------------- Combined
def live_by_region(region_key: str, platforms: list[str], min_viewers: int = 0) -> dict:
    region = REGIONS[region_key]
    rows, errors = [], []

    # No Twitch app of your own? Signed-in users get Twitch listings from Clipper Cloud.
    own_twitch = bool(settings.twitch_client_id and settings.twitch_client_secret)
    if "twitch" in platforms and not own_twitch and cloud.signed_in():
        if region.get("all_languages"):
            errors.append("Twitch all-language discovery needs your own Twitch app; choose a region for Clipper Cloud listings.")
        else:
            try:
                rows.extend(cloud.live(region_key, ["twitch"], 0)["streams"])
            except Exception as e:
                errors.append(f"twitch (Clipper Cloud): {e}")
        platforms = [p for p in platforms if p != "twitch"]

    def collect(fn, lang, platform, keyword_only=False, **kw):
        try:
            for s in fn(lang, **kw):
                if keyword_only and not matches_keywords(region, s["title"], s["tags"]):
                    continue
                rows.append(s)
        except Exception as e:  # keep going if one language fails
            errors.append(f"{platform} {lang}: {e}")

    for p in platforms:
        if region.get("all_languages") and p == "kick":
            try:
                for raw in _kick_web_all():
                    lang = str(raw.get("language") or raw.get("language_code") or "")
                    rows.append(_kick_row(raw, lang))
            except Exception as e:
                errors.append(f"kick all languages: {e}")
            continue

        fn = twitch_streams if p == "twitch" else kick_streams
        if region.get("all_languages") and p == "twitch":
            collect(fn, None, p, pages=4)
            continue
        for lang in region["languages"]:
            collect(fn, lang, p)
        for lang in region.get("keyword_languages", []):
            # English streams from this country: scan more pages, keep tag/title matches
            kw = {"pages": 4} if p == "twitch" else {}
            collect(fn, lang, p, keyword_only=True, **kw)

    seen, unique = set(), []
    for s in sorted(rows, key=lambda x: -int(x["viewers"] or 0)):
        key = (s["platform"], s["channel"].lower())
        if key in seen or int(s["viewers"] or 0) < min_viewers or not s["channel"]:
            continue
        seen.add(key)
        unique.append(s)
    return {"streams": unique, "errors": errors}


def reset_tokens():
    """Forget cached app tokens after keys change in settings."""
    for t in (_twitch_token, _kick_token):
        t.value, t.expires = "", 0.0
