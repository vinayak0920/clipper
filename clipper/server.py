from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import cloud, discovery, llm, pipeline
from .config import ROOT, settings, write_env
from .monitor import manager
from .regions import REGIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="Clipper")
app.mount("/clips", StaticFiles(directory=str(settings.clips_dir)), name="clips")
app.mount("/ui", StaticFiles(directory=str(ROOT / "assets")), name="ui")
STARTED = time.time()

# ------------------------------------------------------------------ settings schema
# kind: text | secret | select | number | toggle
SCHEMA = [
    {"group": "twitch", "title": "Your own Twitch app", "about": "Optional. Signed-in accounts list Twitch streamers through Clipper Cloud. Add an app from dev.twitch.tv only if you'd rather use your own.",
     "link": "https://dev.twitch.tv/console/apps", "fields": [
        {"key": "TWITCH_CLIENT_ID", "label": "Client ID", "kind": "text"},
        {"key": "TWITCH_CLIENT_SECRET", "label": "Client secret", "kind": "secret"},
    ]},
    {"group": "kick", "title": "Your own Kick app", "about": "Optional. Without keys, Kick streamers come from kick.com's public browse feed.",
     "link": "https://kick.com/settings/developer", "fields": [
        {"key": "KICK_CLIENT_ID", "label": "Client ID", "kind": "text"},
        {"key": "KICK_CLIENT_SECRET", "label": "Client secret", "kind": "secret"},
    ]},
    {"group": "ai", "title": "AI hooks", "about": "Writes the hook, trims to the payoff, skips weak moments and turns Hindi into Hinglish. Signed in, this runs on Clipper Cloud. Your own key, or Ollama on this computer, is unlimited and skips your credits.",
     "fields": [
        {"key": "AI_PROVIDER", "label": "Use", "kind": "select", "default": "auto",
         "options": [["auto", "Automatic: my key, then Clipper Cloud, then Ollama"], ["cloud", "Clipper Cloud (uses hook credits)"], ["ollama", "Ollama on this computer"],
                     ["anthropic", "Claude (Anthropic key)"], ["openai", "OpenAI key or compatible server"],
                     ["groq", "Groq key"], ["gemini", "Gemini key"], ["off", "Off, captions only"]]},
        {"key": "OLLAMA_MODEL", "label": "Ollama model", "kind": "text", "default": "qwen2.5:7b"},
        {"key": "ANTHROPIC_API_KEY", "label": "Anthropic key", "kind": "secret"},
        {"key": "OPENAI_API_KEY", "label": "OpenAI key", "kind": "secret"},
        {"key": "OPENAI_BASE_URL", "label": "OpenAI-compatible URL", "kind": "text", "placeholder": "https://api.openai.com/v1"},
        {"key": "GROQ_API_KEY", "label": "Groq key", "kind": "secret"},
        {"key": "GEMINI_API_KEY", "label": "Gemini key", "kind": "secret"},
    ]},
    {"group": "detection", "title": "Moment detection", "about": "Applies to streamers you start watching after saving.", "fields": [
        {"key": "SENSITIVITY", "label": "Chat must run this many times its normal speed", "kind": "number", "default": "2.6", "min": 1.5, "max": 5, "step": 0.1},
        {"key": "PRE_ROLL", "label": "Seconds kept before chat reacts", "kind": "number", "default": "22", "min": 5, "max": 45, "step": 1},
        {"key": "POST_ROLL", "label": "Seconds kept after it calms", "kind": "number", "default": "6", "min": 0, "max": 20, "step": 1},
        {"key": "COOLDOWN_SECONDS", "label": "Wait between clips of one streamer (s)", "kind": "number", "default": "75", "min": 15, "max": 600, "step": 5},
        {"key": "MAX_CLIP", "label": "Longest clip (s)", "kind": "number", "default": "59", "min": 15, "max": 180, "step": 1},
    ]},
    {"group": "output", "title": "Recording and captions", "fields": [
        {"key": "STREAM_QUALITY", "label": "Recording quality", "kind": "select", "default": "720p60,720p,best",
         "options": [["720p60,720p,best", "720p (recommended)"], ["1080p60,1080p,best", "1080p, more bandwidth"], ["480p,best", "480p, saves bandwidth"]]},
        {"key": "BUFFER_SECONDS", "label": "Rolling buffer per streamer (s)", "kind": "number", "default": "240", "min": 90, "max": 900, "step": 30},
        {"key": "WHISPER_MODEL", "label": "Transcription model", "kind": "select", "default": "small",
         "options": [["base", "Base, fastest"], ["small", "Small, balanced"], ["medium", "Medium, better Hindi"], ["large-v3", "Large v3, best, slow on CPU"]]},
        {"key": "HINDI_SCRIPT", "label": "Hindi captions", "kind": "select", "default": "latin",
         "options": [["latin", "Hinglish (bhai kya hua)"], ["devanagari", "Devanagari (भाई क्या हुआ)"]]},
        {"key": "CAPTION_LANGUAGE", "label": "Force caption language", "kind": "text", "placeholder": "auto, or hi, en, es..."},
        {"key": "SHOW_TITLE", "label": "Hook bar on clips", "kind": "toggle", "default": "1"},
    ]},
    {"group": "advanced", "title": "Advanced", "about": "Only change this if you run your own Clipper Cloud server.", "fields": [
        {"key": "CLIPPER_CLOUD_URL", "label": "Clipper Cloud URL", "kind": "text", "placeholder": "https://apavin.com/api/clipper"},
    ]},
]
FIELDS = {f["key"]: f for g in SCHEMA for f in g["fields"]}


@app.middleware("http")
async def local_only(request: Request, call_next):
    # The dashboard can read and write API keys, so refuse anything not addressed to this machine
    host = (request.headers.get("host") or "").split(":")[0]
    if request.url.path.startswith("/api") and host not in ("localhost", "127.0.0.1", "[::1]", "::1"):
        return JSONResponse({"detail": "Open the dashboard at http://localhost"}, status_code=403)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/config")
def config():
    acct = cloud.me()
    return {
        "account": acct,
        "signed_in": cloud.signed_in(),
        "regions": [{"key": k, "label": v["label"], "languages": v["languages"]} for k, v in REGIONS.items()],
        "twitch_ready": bool(settings.twitch_client_id and settings.twitch_client_secret) or bool(acct),
        "twitch_own": bool(settings.twitch_client_id and settings.twitch_client_secret),
        "kick_official": bool(settings.kick_client_id and settings.kick_client_secret),
        "ai": llm.enabled(),
        "ai_name": llm.describe(),
        "sensitivity": settings.sensitivity,
        "tools": {t: bool(shutil.which(t)) for t in ("ffmpeg", "ffprobe", "streamlink")},
    }


# ------------------------------------------------------------------ settings
def _mask(v: str) -> str:
    return ("•" * 6 + v[-4:]) if len(v) > 8 else ("•" * 6 if v else "")


@app.get("/api/settings")
def get_settings():
    groups = []
    for g in SCHEMA:
        fields = []
        for f in g["fields"]:
            val = os.getenv(f["key"], "")
            item = {k: v for k, v in f.items()}
            if f["kind"] == "secret":
                item.update(value="", set=bool(val), hint=_mask(val))
            else:
                item["value"] = val if val != "" else f.get("default", "")
            fields.append(item)
        groups.append({**{k: v for k, v in g.items() if k != "fields"}, "fields": fields})
    return {"groups": groups, "env_file": str(ROOT / ".env")}


class SaveReq(BaseModel):
    values: dict[str, str]


@app.post("/api/settings")
def save_settings(req: SaveReq):
    updates = {}
    for k, v in req.values.items():
        if k not in FIELDS:
            raise HTTPException(400, f"unknown setting {k}")
        f = FIELDS[k]
        v = str(v).strip()
        if f["kind"] == "number":
            try:
                num = float(v)
            except ValueError:
                raise HTTPException(400, f"{f['label']} must be a number")
            v = str(min(max(num, f.get("min", num)), f.get("max", num)))
        if k == "CLIPPER_CLOUD_URL" and v and (not v.startswith(("https://", "http://")) or " " in v):
            raise HTTPException(400, "Clipper Cloud URL must start with https://, or leave it empty")
        updates[k] = v
    write_env(updates)
    settings.reload()
    discovery.reset_tokens()
    llm.reset()
    if "WHISPER_MODEL" in updates:
        pipeline.reset_model()
    return {"ok": True, "saved": sorted(updates)}


@app.post("/api/test/{service}")
async def test_service(service: str):
    t0 = time.time()
    try:
        if service == "twitch":
            rows = await asyncio.to_thread(discovery.twitch_streams, "en")
            msg = f"Connected. {len(rows)} English streams live right now."
        elif service == "kick":
            rows = await asyncio.to_thread(discovery.kick_streams, "en")
            src = "official API" if settings.kick_client_id else "public feed"
            msg = f"Connected via {src}. {len(rows)} English streams found."
        elif service == "ai":
            llm.reset()
            if not llm.enabled():
                return {"ok": False, "message": "No AI available. Start Ollama with a downloaded model, or add a key and pick it under Use."}
            out = await asyncio.to_thread(llm._ask, 'Reply with JSON only: {"ok": true}', 30)
            if '"ok"' not in out:
                raise RuntimeError(f"unexpected reply: {out[:80]}")
            msg = f"{llm.describe()} replied in {time.time() - t0:.1f}s."
        else:
            raise HTTPException(404)
        return {"ok": True, "message": msg}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        hint = "Key or secret is wrong." if code in (400, 401, 403) else f"Service answered {code}."
        return {"ok": False, "message": hint}
    except Exception as e:
        return {"ok": False, "message": str(e)[:200]}


# ------------------------------------------------------------------ account (Clipper Cloud)
class EmailReq(BaseModel):
    email: str


class CodeReq(BaseModel):
    email: str
    code: str


def _cloud_call(fn, *a):
    try:
        return fn(*a)
    except cloud.OutOfCredits as e:
        raise HTTPException(402, str(e))
    except httpx.HTTPError:
        raise HTTPException(502, "Can't reach Clipper Cloud. Check your internet connection.")
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/account")
async def account():
    return {"signed_in": cloud.signed_in(), "me": await asyncio.to_thread(cloud.me, True), "cloud_url": cloud.url()}


@app.post("/api/account/start")
async def account_start(req: EmailReq):
    await asyncio.to_thread(_cloud_call, cloud.start_sign_in, req.email)
    return {"ok": True}


@app.post("/api/account/verify")
async def account_verify(req: CodeReq):
    me = await asyncio.to_thread(_cloud_call, cloud.verify, req.email, req.code)
    settings.reload()
    llm.reset()
    return {"ok": True, "me": me}


@app.get("/api/account/login")
def account_login(request: Request):
    """Where "Continue with Google" goes: apavin.com, then back to /auth/callback on this computer."""
    host = request.headers.get("host") or "localhost:8000"
    return {"url": cloud.login_url(f"http://{host}/auth/callback")}


class TokenReq(BaseModel):
    token: str


@app.post("/api/account/token")
async def account_token(req: TokenReq):
    me = await asyncio.to_thread(cloud.save_token, req.token.strip())
    if not me:
        raise HTTPException(400, "That sign-in didn't work. Try again.")
    settings.reload()
    llm.reset()
    return {"ok": True, "me": me}


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback():
    # The token is in the URL fragment, which only this page's script can read.
    return """<!doctype html><meta charset=utf-8><title>Signing in</title>
<body style="font:16px system-ui;background:#fffcf5;color:#111;display:grid;place-items:center;height:100vh;margin:0">
<p id=m>Signing you in...</p>
<script>
const h = new URLSearchParams(location.hash.slice(1));
const t = h.get("clipper_token"), err = h.get("clipper_error");
history.replaceState(null, "", "/auth/callback");
if (!t) { document.getElementById("m").textContent = err || "Sign-in didn't finish."; setTimeout(() => location.href = "/#settings", 2500); }
else fetch("/api/account/token", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({token: t})})
  .then(r => { location.href = r.ok ? "/#settings" : "/#settings"; });
</script>"""


@app.post("/api/account/signout")
def account_signout():
    cloud.sign_out()
    llm.reset()
    return {"ok": True}


@app.post("/api/account/upgrade")
async def account_upgrade():
    return await asyncio.to_thread(_cloud_call, cloud.upgrade)


@app.post("/api/account/cancel")
async def account_cancel():
    return await asyncio.to_thread(_cloud_call, cloud.cancel)


# ------------------------------------------------------------------ ollama
PULL = {"model": "", "status": "", "done": 0, "total": 0, "error": "", "running": False}


@app.get("/api/ollama")
async def ollama_status():
    info = {"installed": bool(shutil.which("ollama")), "running": False, "models": [], "pull": PULL}
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(f"{llm.OLLAMA_URL}/api/tags")
            info["running"] = r.status_code == 200
            info["models"] = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return info


class PullReq(BaseModel):
    model: str


@app.post("/api/ollama/pull")
def ollama_pull(req: PullReq):
    if PULL["running"]:
        return PULL

    def run():
        PULL.update(model=req.model, status="starting", done=0, total=0, error="", running=True)
        try:
            with httpx.stream("POST", f"{llm.OLLAMA_URL}/api/pull", json={"name": req.model}, timeout=None) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    j = json.loads(line)
                    if j.get("error"):
                        PULL["error"] = j["error"]
                    PULL["status"] = j.get("status", PULL["status"])
                    if j.get("total"):
                        PULL.update(total=j["total"], done=j.get("completed", 0))
        except Exception as e:
            PULL["error"] = str(e)[:200]
        finally:
            PULL["running"] = False
            llm.reset()

    threading.Thread(target=run, daemon=True).start()
    return PULL


# ------------------------------------------------------------------ live
@app.get("/api/live")
async def live(region: str, platforms: str = "twitch,kick", min_viewers: int = 0):
    if region not in REGIONS:
        raise HTTPException(404, "unknown region")
    plats = [p for p in platforms.split(",") if p in ("twitch", "kick")]
    return await asyncio.to_thread(discovery.live_by_region, region, plats, min_viewers)


class AddReq(BaseModel):
    platform: str
    channel: str
    name: str = ""
    sensitivity: Optional[float] = None


@app.post("/api/monitors")
async def add_monitor(req: AddReq):
    if req.platform not in ("twitch", "kick"):
        raise HTTPException(400, "platform must be twitch or kick")
    m = await manager.add(req.platform, req.channel, req.name, req.sensitivity)
    return m.status()


@app.get("/api/monitors")
def monitors():
    return [m.status() for m in manager.monitors.values()]


@app.delete("/api/monitors/{key}")
async def remove_monitor(key: str):
    await manager.remove(key)
    return {"ok": True}


class SensReq(BaseModel):
    sensitivity: float


@app.post("/api/monitors/{key}/sensitivity")
def set_sensitivity(key: str, req: SensReq):
    m = manager.monitors.get(key)
    if not m:
        raise HTTPException(404)
    m.detector.sensitivity = max(1.3, min(6.0, req.sensitivity))
    return m.status()


@app.post("/api/monitors/{key}/clip")
def clip_now(key: str, seconds: float = 30):
    m = manager.monitors.get(key)
    if not m:
        raise HTTPException(404)
    m.manual_clip(seconds)
    return {"ok": True}


@app.get("/api/clips")
def clips():
    return pipeline.list_clips()


@app.delete("/api/clips/{cid}")
def delete_clip(cid: str):
    d = (settings.clips_dir / cid).resolve()
    if d.parent != settings.clips_dir.resolve():
        raise HTTPException(400)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.get("/api/stats")
def stats():
    clips = pipeline.list_clips()
    day = time.time() - 86400
    jobs = [j for m in manager.monitors.values() for j in m.jobs]
    size = sum(f.stat().st_size for f in settings.data_dir.rglob("*") if f.is_file())
    return {
        "watching": len(manager.monitors),
        "recording": sum(1 for m in manager.monitors.values() if m.recorder.is_recording()),
        "clips_total": len(clips),
        "clips_today": sum(1 for c in clips if c.get("created", 0) > day),
        "in_progress": sum(1 for j in jobs if j["status"] not in ("done", "dropped") and not j["status"].startswith(("error", "dropped"))),
        "disk_mb": round(size / 1e6),
        "uptime": int(time.time() - STARTED),
    }


@app.on_event("startup")
async def startup():
    pipeline.warm_up()


@app.on_event("shutdown")
async def shutdown():
    await manager.stop_all()
