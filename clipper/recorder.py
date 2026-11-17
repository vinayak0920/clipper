"""Records a live stream into a rolling buffer of short .ts segments.

streamlink pulls the HLS stream (Twitch or Kick), ffmpeg splits it into ~4s
segments named by wall-clock start time, and anything older than
BUFFER_SECONDS is deleted. When a moment is found we stitch the segments that
cover it. Nothing is re-encoded while recording, so CPU stays near zero.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .config import settings

log = logging.getLogger("recorder")


def stream_url(platform: str, channel: str) -> str:
    return f"https://www.twitch.tv/{channel}" if platform == "twitch" else f"https://kick.com/{channel}"


class Recorder:
    def __init__(self, platform: str, channel: str):
        self.platform = platform
        self.channel = channel
        self.dir = settings.buffers_dir / f"{platform}_{channel.lower()}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sl = None
        self.ff = None
        self.stopped = False
        self.error = ""
        self._thread = None

    # ------------------------------------------------------------ process control
    def start(self):
        self.stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _spawn(self):
        sl_cmd = [
            "streamlink", "--stdout", "--loglevel", "warning",
            "--hls-live-edge", "2", "--stream-segment-threads", "2",
            stream_url(self.platform, self.channel), settings.stream_quality,
        ]
        if self.platform == "twitch":
            sl_cmd[1:1] = ["--twitch-disable-ads", "--twitch-low-latency"]
        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
            "-f", "segment", "-segment_time", str(settings.segment_seconds),
            "-strftime", "1",
            str(self.dir / "seg_%s.ts"),
        ]
        self.sl = subprocess.Popen(sl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.ff = subprocess.Popen(ff_cmd, stdin=self.sl.stdout, stderr=subprocess.PIPE)
        self.sl.stdout.close()

    def _run(self):
        backoff = 5
        while not self.stopped:
            try:
                self._spawn()
                log.info("recording %s/%s", self.platform, self.channel)
                while not self.stopped and self.ff.poll() is None:
                    self._prune()
                    time.sleep(2)
                if self.stopped:
                    break
                err = (self.sl.stderr.read() or b"").decode(errors="ignore")[-400:]
                self.error = err.strip() or "stream ended"
                log.warning("recorder %s ended: %s", self.channel, self.error)
            except FileNotFoundError as e:
                self.error = f"missing tool: {e.filename} (install streamlink and ffmpeg)"
                log.error(self.error)
                return
            finally:
                self._kill()
            time.sleep(backoff)            # streamer offline or network blip: retry
            backoff = min(backoff * 2, 60)

    def _kill(self):
        for p in (self.sl, self.ff):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    def stop(self):
        self.stopped = True
        self._kill()

    # ------------------------------------------------------------ buffer access
    def segments(self) -> list[tuple[float, Path]]:
        segs = []
        for p in self.dir.glob("seg_*.ts"):
            try:
                segs.append((float(p.stem.split("_")[1]), p))
            except ValueError:
                pass
        return sorted(segs)

    def _prune(self):
        cutoff = time.time() - settings.buffer_seconds
        for ts, p in self.segments():
            if ts < cutoff:
                p.unlink(missing_ok=True)

    def buffered_seconds(self) -> float:
        s = self.segments()
        return (time.time() - s[0][0]) if s else 0.0

    def is_recording(self) -> bool:
        s = self.segments()
        return bool(s) and time.time() - s[-1][0] < settings.segment_seconds * 4

    def extract(self, start: float, end: float, out: Path) -> float:
        """Concatenate the segments covering [start, end]. Returns the offset (s) of `start` inside `out`."""
        segs = self.segments()
        # the newest segment is still being written; only use finished ones
        chosen = []
        for i, (ts, p) in enumerate(segs[:-1]):
            nxt = segs[i + 1][0]
            if nxt > start and ts < end:
                chosen.append((ts, p))
        if not chosen:
            raise RuntimeError("no buffered video for this moment (buffer too short or stream lagging)")
        with open(out, "wb") as w:
            for _, p in chosen:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
        return max(0.0, start - chosen[0][0])
