"""Detects streamer speech while the game image is paused or nearly still.

This is a lightweight candidate trigger for high-chat channels. It deliberately
uses speech recognition plus low gameplay motion; it cannot prove where the
streamer is looking or identify a game's exact pause/alive state.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _frames(video: Path, offset: float, duration: float) -> list[np.ndarray]:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{offset:.2f}", "-i", str(video),
           "-t", f"{duration:.2f}", "-an", "-vf", "fps=1,scale=160:90:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=12)
    if p.returncode:
        return []
    size = 160 * 90
    return [np.frombuffer(p.stdout[i:i + size], np.uint8).reshape(90, 160)
            for i in range(0, len(p.stdout) - size + 1, size)]


def _game_is_still(frames: list[np.ndarray]) -> bool:
    if len(frames) < 5:
        return False
    # Ignore screen edges where webcam overlays, chat and alerts commonly sit.
    motion = [float(np.mean(np.abs(a[8:82, 16:144].astype(np.int16) -
                              b[8:82, 16:144].astype(np.int16))))
              for a, b in zip(frames, frames[1:])]
    quiet = sum(v < 2.2 for v in motion)
    return quiet >= max(4, int(len(motion) * 0.70))


def _audio(video: Path, offset: float, duration: float, wav: Path) -> bool:
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{offset:.2f}", "-i", str(video),
           "-t", f"{duration:.2f}", "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    p = subprocess.run(cmd, capture_output=True, timeout=12)
    return p.returncode == 0 and wav.exists() and wav.stat().st_size > 1000


def detect(video: Path, offset: float, duration: float, wav: Path, budget: float = 5.0) -> tuple[bool, str]:
    """Return (candidate, transcript) for speech over a still game screen."""
    if not _game_is_still(_frames(video, offset, duration)):
        return False, ""
    if not _audio(video, offset, duration, wav):
        return False, ""
    try:
        from .pipeline import transcribe
        words, _ = transcribe(wav, budget=budget)
    except Exception:
        return False, ""
    spoken = [w for w in words if w.end - w.start >= 0.08]
    transcript = " ".join(w.text for w in spoken).strip()
    spoken_time = sum(max(0.0, w.end - w.start) for w in spoken)
    return len(spoken) >= 4 and spoken_time >= 1.4, transcript
