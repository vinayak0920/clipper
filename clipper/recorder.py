"""Records a live stream into a rolling buffer of short .ts segments.

streamlink pulls the HLS stream (Twitch or Kick), ffmpeg splits it into ~4s
numbered segments and lists each one's stream time in a CSV. Stream time is
mapped to wall-clock time (to line up with chat), and anything older than
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
        self._gen = 0                 # bumps on every reconnect: new files, new stream clock
        self._list = None             # this run's segment CSV
        self._read = 0                # bytes of it already parsed
        self._offset = None           # wall clock minus stream time, for this run
        self._index: list[tuple[float, float, Path]] = []   # (wall start, wall end, file), oldest first
        self._lock = threading.Lock()

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
        # Numbered files, never named by the clock: the stream arrives in bursts, so two
        # segments can start in the same second, and a clock name would overwrite one of them.
        self._gen += 1
        self._list = self.dir / f"list_{self._gen}.csv"
        self._read = 0
        self._offset = None
        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0",
            "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
            "-f", "segment", "-segment_time", str(settings.segment_seconds),
            "-segment_list", str(self._list), "-segment_list_type", "csv", "-segment_list_flags", "+live",
            str(self.dir / f"seg_{self._gen}_%06d.ts"),
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
                    self._scan()
                    self._prune()
                    time.sleep(1)
                self._scan()
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
    def _scan(self):
        """Reads newly finished segments from ffmpeg's CSV and gives each a wall-clock span."""
        lst = self._list
        if not lst or not lst.exists():
            return
        with open(lst, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(self._read)
            chunk = f.read()
        if not chunk:
            return
        done, _, rest = chunk.rpartition("\n")
        if not done:
            return
        self._read += len(done) + 1
        now = time.time()
        with self._lock:
            for line in done.splitlines():
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    name, st, en = parts[0], float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                # A segment is listed the moment it is complete, never before its content
                # was live, so the smallest (seen - stream end) is the closest mapping.
                off = now - en
                self._offset = off if self._offset is None else min(self._offset, off)
                path = self.dir / name
                if path.exists():
                    self._index.append((st + self._offset, en + self._offset, path))
            self._index.sort(key=lambda x: x[0])

    def entries(self) -> list[tuple[float, float, Path]]:
        with self._lock:
            return list(self._index)

    def segments(self) -> list[tuple[float, Path]]:
        return [(s, p) for s, _, p in self.entries()]

    def _prune(self):
        cutoff = time.time() - settings.buffer_seconds
        with self._lock:
            keep = []
            for s, e, p in self._index:
                if e < cutoff:
                    p.unlink(missing_ok=True)
                else:
                    keep.append((s, e, p))
            self._index = keep
        for old in self.dir.glob("list_*.csv"):
            if old != self._list and not any(p.name.startswith(f"seg_{old.stem.split('_')[1]}_") for _, _, p in self.entries()):
                old.unlink(missing_ok=True)

    def buffered_seconds(self) -> float:
        e = self.entries()
        return (e[-1][1] - e[0][0]) if e else 0.0

    def is_recording(self) -> bool:
        e = self.entries()
        return bool(e) and time.time() - e[-1][1] < max(20.0, settings.segment_seconds * 5)

    def latest_end(self) -> float:
        e = self.entries()
        return e[-1][1] if e else 0.0

    def extract(self, start: float, end: float, out: Path) -> float:
        """Concatenates the finished segments covering [start, end]. Returns the offset (s) of `start` inside `out`."""
        chosen = [(s, p) for s, e, p in self.entries() if e > start and s < end]
        if not chosen:
            raise RuntimeError("no buffered video for this moment (buffer too short or stream lagging)")
        with open(out, "wb") as w:
            for _, p in chosen:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
        return max(0.0, start - chosen[0][0])
