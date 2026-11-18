"""One Monitor per streamer: recorder + chat listener + detector. A shared worker renders clips."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import discovery, pipeline
from .chat import kick_chat, twitch_chat
from .config import settings
from .detector import Detector, Moment
from .recorder import Recorder
from .talk_detector import detect as detect_talk_pause

log = logging.getLogger("monitor")
# whisper + ffmpeg are heavy: a few clips at a time (pipeline.RENDERERS), queue the rest
WORKER = ThreadPoolExecutor(max_workers=pipeline.RENDERERS, thread_name_prefix="render")
# a clip that waited this long in the queue is skipped: by then the moment is old news
# and rendering it would only push newer clips back
STALE_AFTER = 150


class Monitor:
    def __init__(self, platform: str, channel: str, name: str = "", sensitivity: float | None = None,
                 category: str = ""):
        self.platform = platform
        self.channel = channel.strip().lower().lstrip("@")
        self.name = name or channel
        self.category = category.strip()
        self.key = f"{platform}:{self.channel}"
        self.detector = Detector(
            sensitivity=sensitivity or settings.sensitivity,
            warmup=settings.warmup_seconds,
            cooldown=settings.cooldown_seconds,
        )
        self.recorder = Recorder(platform, self.channel)
        self.stop_evt = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.started = time.time()
        self.msg_count = 0
        self.moments: list[dict] = []
        self.jobs: list[dict] = []
        self.error = ""
        self._talk_last_scan = 0.0
        self._talk_last_clip = 0.0

    def on_message(self, ts: float, user: str, text: str):
        self.msg_count += 1
        self.detector.add(ts, text)

    async def start(self):
        self.recorder.start()
        if self.platform == "twitch":
            self.tasks.append(asyncio.create_task(twitch_chat(self.channel, self.on_message, self.stop_evt)))
        else:
            try:
                info = await asyncio.to_thread(discovery.kick_channel_info, self.channel)
                if not info.get("channel_id"):
                    raise RuntimeError("channel not found")
                self.tasks.append(asyncio.create_task(kick_chat(info["channel_id"], self.on_message, self.stop_evt)))
            except Exception as e:
                self.error = f"Kick chat unavailable: {e}"
                log.warning("%s: %s", self.key, self.error)
        self.tasks.append(asyncio.create_task(self._loop()))

    async def _loop(self):
        while not self.stop_evt.is_set():
            await asyncio.sleep(1)
            m = self.detector.tick()
            if m:
                if not self.recorder.is_recording():
                    log.info("%s: moment found but stream isn't recording, skipped", self.key)
                else:
                    log.info("%s: moment score=%.1f rate=%.1f/s base=%.2f %s", self.key, m.score, m.rate, m.baseline, m.keywords)
                    self.submit(m)

            # For high-volume channels, also look for a spoken moment over a
            # paused/still game. This path does not require a chat spike.
            now = time.monotonic()
            if (self.detector.baseline is not None and self.detector.baseline >= 3.0
                    and self.detector.active is None and now - self._talk_last_scan >= 15
                    and m is None and now - self._talk_last_clip >= 60 and self.recorder.is_recording()):
                self._talk_last_scan = now
                result = await asyncio.to_thread(self._talk_pause_candidate)
                if result:
                    moment, transcript = result
                    self._talk_last_clip = time.monotonic()
                    log.info("%s: paused-game speech candidate: %s", self.key, transcript[:120])
                    self.submit(moment, trigger="talk_pause")
    def _talk_pause_candidate(self):
        """Probe a recent completed buffer window without blocking the event loop."""
        end = self.recorder.latest_end()
        duration = 10.0
        start = end - duration
        entries = self.recorder.entries()
        if not entries or start < entries[0][0] or end <= start:
            return None
        with tempfile.TemporaryDirectory(prefix="clipper-talk-") as tmp:
            raw, wav = Path(tmp) / "probe.ts", Path(tmp) / "speech.wav"
            try:
                offset = self.recorder.extract(start, end, raw)
                found, transcript = detect_talk_pause(raw, offset, duration, wav)
            except Exception as e:
                log.debug("%s: paused-game probe failed: %s", self.key, e)
                return None
        if not found:
            return None
        moment = Moment(start=start, peak=end - 3, end=end, score=0, rate=0,
                        baseline=round(self.detector.baseline or 0, 2),
                        top_messages=[transcript] if transcript else [],
                        keywords=["talk_pause"])
        return moment, transcript

    def submit(self, m: Moment, manual: bool = False, trigger: str = "chat"):
        job = {"at": m.peak, "score": m.score, "status": "queued", "clip": None,
               "keywords": m.keywords, "manual": manual, "trigger": trigger}
        self.jobs.append(job)
        self.jobs = self.jobs[-20:]
        meta = {"platform": self.platform, "channel": self.channel, "name": self.name,
                "category": self.category, "trigger": trigger}

        loop = asyncio.get_running_loop()

        async def prepare():
            # wait for the reaction to finish recording outside the render queue,
            # so one clip's wait never holds up another clip's render
            need = m.end + settings.post_roll
            job["status"] = "recording the reaction"
            limit = time.time() + settings.post_roll + 25
            while self.recorder.latest_end() < need and time.time() < limit:
                await asyncio.sleep(1)
            job["status"] = "cutting"
            try:
                piece = await asyncio.to_thread(pipeline.cut, m, self.recorder)
            except Exception as e:
                log.warning("cut failed: %s", e)
                job["status"] = f"error: {e}"[:200]
                return
            job["status"] = "queued"
            job["queued_at"] = time.time()
            loop.run_in_executor(WORKER, lambda: work(piece))

        def work(piece):
            if time.time() - job["queued_at"] > STALE_AFTER and not manual:
                job["status"] = "skipped: too many clips at once"
                pipeline.discard(piece)
                return
            t0 = time.time()
            try:
                job["clip"] = pipeline.make_clip(m, self.recorder, meta, on_status=lambda s: job.update(status=s),
                                                 force_keep=manual or trigger == "talk_pause", piece=piece)
                if job["clip"] is None and not job["status"].startswith("dropped"):
                    job["status"] = "dropped"
            except Exception as e:
                log.exception("clip failed")
                job["status"] = f"error: {e}"[:200]
                pipeline.discard(piece)
            job["took"] = round(time.time() - t0)
            log.info("%s: clip %s in %ss", self.key, job["status"], job["took"])

        loop.create_task(prepare())

    def manual_clip(self, seconds: float = 30):
        """'Clip that' button: last `seconds` of video plus a few seconds after the click."""
        now = time.time()
        start = now - max(5.0, seconds - settings.pre_roll)
        recent = [t for _, t, _ in list(self.detector.msgs)[-40:]]
        m = Moment(start=start, peak=now, end=now, score=0, rate=0,
                   baseline=round(self.detector.baseline or 0, 2), top_messages=recent[-10:], keywords=["manual"])
        self.submit(m, manual=True)

    async def stop(self):
        self.stop_evt.set()
        self.recorder.stop()
        for t in self.tasks:
            t.cancel()

    def status(self) -> dict:
        d = self.detector
        hist = list(d.history)[-90:]
        return {
            "key": self.key,
            "platform": self.platform,
            "channel": self.channel,
            "name": self.name,
            "category": self.category,
            "uptime": int(time.time() - self.started),
            "recording": self.recorder.is_recording(),
            "buffered": int(self.recorder.buffered_seconds()),
            "recorder_error": self.recorder.error if not self.recorder.is_recording() else "",
            "chat_error": self.error,
            "messages": self.msg_count,
            "rate": hist[-1][1] if hist else 0,
            "baseline": round(d.baseline or 0, 2),
            "warm": time.time() - d.started >= d.warmup,
            "sensitivity": d.sensitivity,
            "spike_active": d.active is not None,
            "history": [[h[1], h[2]] for h in hist],
            "jobs": self.jobs[-8:],
        }


class Manager:
    def __init__(self):
        self.monitors: dict[str, Monitor] = {}

    async def add(self, platform: str, channel: str, name: str = "", sensitivity: float | None = None,
                  category: str = "") -> Monitor:
        key = f"{platform}:{channel.strip().lower().lstrip('@')}"
        if key in self.monitors:
            return self.monitors[key]
        m = Monitor(platform, channel, name, sensitivity, category)
        self.monitors[key] = m
        await m.start()
        return m

    async def remove(self, key: str):
        m = self.monitors.pop(key, None)
        if m:
            await m.stop()

    async def stop_all(self):
        for k in list(self.monitors):
            await self.remove(k)


manager = Manager()
