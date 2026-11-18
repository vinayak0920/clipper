from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # dotenv is optional
    dotenv_values = load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


class Settings:
    """Values come from .env / environment. reload() re-reads them after the dashboard saves."""

    def __init__(self):
        self.reload(override=False)

    def reload(self, override: bool = True):
        if load_dotenv and ENV_FILE.exists():
            if override:
                # blank lines in .env never wipe values set elsewhere; cleared keys are handled by write_env
                for k, v in dotenv_values(ENV_FILE).items():
                    if v:
                        os.environ[k] = v
            else:
                load_dotenv(ENV_FILE)
        g = os.getenv
        self.twitch_client_id = g("TWITCH_CLIENT_ID", "")
        self.twitch_client_secret = g("TWITCH_CLIENT_SECRET", "")
        self.kick_client_id = g("KICK_CLIENT_ID", "")
        self.kick_client_secret = g("KICK_CLIENT_SECRET", "")
        self.stream_quality = g("STREAM_QUALITY", "720p60,720p,best")
        self.buffer_seconds = _f("BUFFER_SECONDS", 240)
        self.segment_seconds = _f("SEGMENT_SECONDS", 4)
        self.sensitivity = _f("SENSITIVITY", 2.6)
        self.warmup_seconds = _f("WARMUP_SECONDS", 45)
        self.cooldown_seconds = _f("COOLDOWN_SECONDS", 45)
        self.pre_roll = _f("PRE_ROLL", 22)
        self.post_roll = _f("POST_ROLL", 6)
        self.max_clip = _f("MAX_CLIP", 59)
        self.min_clip = _f("MIN_CLIP", 12)
        self.whisper_model = g("WHISPER_MODEL", "small")
        self.whisper_device = g("WHISPER_DEVICE", "auto")
        self.caption_language = g("CAPTION_LANGUAGE", "")
        self.data_dir = Path(g("DATA_DIR", str(ROOT / "data")))
        self.buffers_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    @property
    def buffers_dir(self) -> Path:
        return self.data_dir / "buffers"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def fonts_dir(self) -> Path:
        return ROOT / "assets" / "fonts"


def write_env(updates: dict[str, str]):
    """Update KEY=value lines in .env in place, keep comments and order, append new keys."""
    if not ENV_FILE.exists():
        example = ROOT / ".env.example"
        ENV_FILE.write_text(example.read_text() if example.exists() else "", encoding="utf-8")
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    left = dict(updates)
    for i, line in enumerate(lines):
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in left:
            key = m.group(1)
            lines[i] = f"{key}={_quote(left.pop(key))}"
    for key, val in left.items():
        lines.append(f"{key}={_quote(val)}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        ENV_FILE.chmod(0o600)
    except OSError:
        pass
    for key, val in updates.items():
        os.environ[key] = str(val)


def _quote(v) -> str:
    v = str(v).replace("\n", " ").strip()
    return f'"{v}"' if (" " in v or "#" in v) else v


settings = Settings()
