"""Moment -> finished vertical clip.

1. stitch buffered segments around the moment
2. transcribe with faster-whisper (word timestamps)
3. optional Claude pass: keep/drop, tighter start/end, hook title
4. pick a 9:16 layout from face detection
5. render with animated captions burned in
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm, reframe
from .captions import DEVANAGARI, Word, build_ass, build_title_ass
from .config import settings
from .detector import Moment

log = logging.getLogger("pipeline")

# How many clips are made at the same time. Two on an 8-core Mac keeps each one fast.
RENDERERS = int(os.getenv("RENDERERS", "0")) or (3 if (os.cpu_count() or 4) >= 12 else 2 if (os.cpu_count() or 4) >= 6 else 1)
TRANSCRIBE_BUDGET = float(os.getenv("TRANSCRIBE_SECONDS", "18"))   # captions stop here, the clip still renders
_whisper = None


def _model():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        device = settings.whisper_device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        _whisper = WhisperModel(settings.whisper_model, device=device,
                                compute_type="float16" if device == "cuda" else "int8",
                                cpu_threads=max(2, (os.cpu_count() or 4) // RENDERERS),
                                num_workers=RENDERERS)
    return _whisper


def warm_up():
    """Loads Whisper in the background at startup, so the first clip doesn't pay for it."""
    def go():
        try:
            with _model_lock:
                _model()
            log.info("whisper %s loaded", settings.whisper_model)
        except Exception as e:
            log.warning("whisper preload failed: %s", e)
    threading.Thread(target=go, daemon=True, name="whisper-preload").start()


_model_lock = threading.Lock()
_side = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clip-side")


def _analyze_layout(video: Path, start: float, duration: float, category: str):
    """Call the category-aware framer, while remaining compatible with older copies."""
    try:
        return reframe.analyze(video, start=start, duration=duration, samples=14, category=category)
    except TypeError as e:
        if "category" not in str(e) and "keyword" not in str(e):
            raise
        log.warning("reframe.py is outdated; using its category-blind layout until it is replaced")
        return reframe.analyze(video, start=start, duration=duration, samples=14)


_encoder = None
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
_RATE = ["-b:v", "9M", "-maxrate", "12M", "-bufsize", "18M"]
# Hardware H.264 encoders, several times faster than x264. Tried in order, first that works wins.
HW_ENCODERS = {
    "darwin": [["-c:v", "h264_videotoolbox", *_RATE, "-profile:v", "high", "-allow_sw", "1"]],
    "other": [
        ["-c:v", "h264_nvenc", "-preset", "p4", *_RATE, "-profile:v", "high"],   # NVIDIA
        ["-c:v", "h264_qsv", "-preset", "veryfast", *_RATE],                     # Intel
        ["-c:v", "h264_amf", "-quality", "speed", "-rc", "vbr_peak", *_RATE],    # AMD
    ],
}


def _encoder_works(enc: list[str]) -> bool:
    """FFmpeg lists GPU encoders even with no matching GPU, so encode a second of test video to be sure."""
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                            "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-t", "1",
                            *enc, "-pix_fmt", "yuv420p", "-f", "null", "-"],
                           capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
        return p.returncode == 0
    except Exception:
        return False


def video_encoder() -> list[str]:
    """Hardware H.264 (Apple, NVIDIA, Intel or AMD) when this machine has it, x264 otherwise."""
    global _encoder
    if _encoder is None:
        _encoder = X264
        if os.getenv("HW_ENCODE", "1") == "1":
            try:
                listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                                        stdin=subprocess.DEVNULL).stdout
                for enc in HW_ENCODERS["darwin" if sys.platform == "darwin" else "other"]:
                    if enc[1] in listed and _encoder_works(enc):
                        _encoder = enc
                        log.info("video encoder: %s", enc[1])
                        break
            except Exception:
                pass
    return _encoder


def reset_model():
    global _whisper
    _whisper = None


def _has_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 90):
    if cmd[0] == "ffmpeg" and "-nostdin" not in cmd:
        cmd = [cmd[0], "-nostdin", *cmd[1:]]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-800:]}")
    return p


def run_render(cmd: list[str], cwd: Path, duration: float, status, timeout: float | None = None):
    """Run a render while forwarding FFmpeg progress to the clip job status."""
    if cmd[0] == "ffmpeg" and "-nostdin" not in cmd:
        cmd = [cmd[0], "-nostdin", *cmd[1:]]
    timeout = timeout or float(os.getenv("RENDER_TIMEOUT_SECONDS", "300"))
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1, errors="replace")
    progress_lines: queue.Queue[str] = queue.Queue()
    errors: deque[str] = deque(maxlen=40)

    def collect(stream, sink):
        try:
            for line in iter(stream.readline, ""):
                sink(line.rstrip())
        finally:
            stream.close()

    readers = [
        threading.Thread(target=collect, args=(p.stdout, progress_lines.put), daemon=True),
        threading.Thread(target=collect, args=(p.stderr, errors.append), daemon=True),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    last_percent = -1
    status("rendering 0%")
    while True:
        if time.monotonic() >= deadline:
            p.kill()
            p.wait()
            for reader in readers:
                reader.join(timeout=1)
            detail = "\n".join(errors)[-800:]
            raise TimeoutError(f"ffmpeg render exceeded {timeout:.0f}s" + (f": {detail}" if detail else ""))
        try:
            line = progress_lines.get(timeout=0.25)
        except queue.Empty:
            if p.poll() is not None:
                break
            continue
        if line.startswith("out_time_us="):
            try:
                out_time = int(line.split("=", 1)[1]) / 1_000_000
                percent = min(99, int(out_time / max(duration, 0.1) * 100))
                if percent >= last_percent + 5:
                    last_percent = percent
                    status(f"rendering {percent}%")
            except ValueError:
                pass
        elif line == "progress=end":
            break

    returncode = p.wait()
    for reader in readers:
        reader.join(timeout=1)
    if returncode:
        detail = "\n".join(errors)[-800:]
        raise RuntimeError(f"ffmpeg render failed (exit {returncode})" + (f": {detail}" if detail else ""))
    status("rendering 100%")


def probe_duration(path: Path) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def transcribe(audio: Path, budget: float = TRANSCRIBE_BUDGET) -> tuple[list[Word], str]:
    """Word-timed transcript, cut off at `budget` seconds so one clip can't stall the queue."""
    deadline = time.time() + budget
    if not _model_lock.acquire(timeout=min(15.0, budget)):
        raise RuntimeError("speech model is still loading (first run downloads it)")
    try:
        model = _model()
    finally:
        _model_lock.release()
    segments, info = model.transcribe(
        str(audio), word_timestamps=True, vad_filter=True,
        language=settings.caption_language or None, beam_size=1,
        condition_on_previous_text=False,
    )
    words = []
    for seg in segments:              # lazy: each step decodes the next piece of audio
        for w in seg.words or []:
            if w.word.strip():
                words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        if time.time() > deadline:
            log.warning("transcription hit its %.0fs budget, using what it has", budget)
            break
    return words, info.language


def tighten(words: list[Word], duration: float) -> tuple[float, float]:
    """No-LLM fallback: start on a word boundary near the beginning, end just after the last word."""
    start, end = 0.0, duration
    if words:
        early = [w for w in words if w.start <= 6.0]
        if early:
            start = max(0.0, early[0].start - 0.35)
        end = min(duration, max(words[-1].end + 0.8, duration - 2.0))
    if end - start < settings.min_clip:
        start, end = 0.0, duration
    return start, min(end, start + settings.max_clip)


def cut(moment: Moment, recorder) -> dict:
    """Copies the buffered video around a moment into its own folder. Cheap; done before
    the clip waits in the render queue, so the rolling buffer can't drop it meanwhile."""
    cid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = settings.clips_dir / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    win_start = moment.start - settings.pre_roll
    win_end = moment.end + settings.post_roll
    if win_end - win_start > settings.max_clip + 10:
        win_start = win_end - (settings.max_clip + 10)
    raw = out_dir / "raw.ts"
    try:
        offset = recorder.extract(win_start, win_end, raw)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    duration = win_end - win_start
    avail = probe_duration(raw) - offset
    if avail > 0:
        duration = min(duration, avail)
    return {"cid": cid, "out_dir": out_dir, "raw": raw, "offset": offset, "duration": duration, "win_start": win_start}


def discard(piece: dict):
    shutil.rmtree(piece["out_dir"], ignore_errors=True)


def ready_at(moment: Moment) -> float:
    """Wall-clock time when the reaction after a moment has finished recording."""
    return moment.end + settings.post_roll + settings.segment_seconds * 1.5


def make_clip(moment: Moment, recorder, meta: dict, on_status=None, force_keep: bool = False, piece: dict | None = None) -> dict | None:
    status = on_status or (lambda s: None)
    if piece is None:
        wait = ready_at(moment) - time.time()
        if wait > 0:
            time.sleep(wait)
        status("cutting")
        piece = cut(moment, recorder)
    cid, out_dir, raw = piece["cid"], piece["out_dir"], piece["raw"]
    offset, duration, win_start = piece["offset"], piece["duration"], piece["win_start"]

    # face framing runs alongside transcription; it only needs the video
    layout_job = _side.submit(_analyze_layout, raw, offset, min(duration, 40), meta.get("category", ""))
    t0 = time.time()

    # 2. transcribe the window
    status("transcribing")
    wav = out_dir / "audio.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.2f}", "-i", str(raw), "-t", f"{duration:.2f}",
         "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    try:
        words, lang = transcribe(wav)
    except Exception as e:
        log.warning("transcription failed: %s", e)
        words, lang = [], ""

    # 3. judge + trim
    reaction_at = moment.start - win_start
    verdict = llm.judge(words, moment.top_messages, moment.keywords, meta["channel"], duration, reaction_at)
    if verdict and verdict.get("keep") is False and not force_keep:
        status("dropped by AI: " + str(verdict.get("reason", ""))[:80])
        _cleanup(out_dir, keep_meta=False)
        return None
    if verdict:
        start, end = verdict["start"], verdict["end"]
        title = verdict.get("title") or ""
    else:
        start, end = tighten(words, duration)
        title = ""
    if not title:
        hype = moment.keywords[0].upper() if moment.keywords and not force_keep else "moment"
        title = f"{meta.get('name') or meta['channel']} {hype}"

    clip_words = [Word(w.text, w.start - start, w.end - start) for w in words if w.start >= start - 0.05 and w.end <= end + 0.3]
    if lang == "hi" and os.getenv("HINDI_SCRIPT", "latin") == "latin" and any(DEVANAGARI.search(w.text) for w in clip_words):
        latin = llm.transliterate(clip_words)
        if latin:
            for w, t in zip(clip_words, latin):
                w.text = t

    # 4. layout
    status("framing")
    try:
        layout = layout_job.result(timeout=30)
    except Exception as e:
        log.warning("framing failed, using the plain layout: %s", e)
        layout = reframe.choose_layout(1280, 720, [], 0)

    # 5. captions + render
    status("rendering")
    ass = build_ass(clip_words, y=layout.caption_y)
    if os.getenv("SHOW_TITLE", "1") == "1":
        # Keep the title in the upper panel regardless of which panel holds gameplay.
        game_top = getattr(layout, "game_top", True)
        hook_y = 240 if game_top else getattr(reframe, "CAM_TOP_H", 700) + 120
        ass += build_title_ass(title, end - start, y=hook_y)
    (out_dir / "captions.ass").write_text(ass, encoding="utf-8")
    fonts = str(settings.fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    # reset timestamps to 0 so caption timing lines up with the cut (stream PTS never start at 0)
    graph = ("[0:v]setpts=PTS-STARTPTS[s0];" + layout.filter_graph().replace("[0:v]", "[s0]")
             + f";[v]subtitles=filename=captions.ass:fontsdir={fonts}[vout]")
    def render(enc):
        run_render([
            "ffmpeg", "-y", "-loglevel", "error",
            "-progress", "pipe:1", "-stats_period", "1", "-nostats",
            "-ss", f"{offset + start:.2f}", "-i", "raw.ts", "-t", f"{end - start:.2f}",
            "-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?",
            *enc, "-r", "30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-af", "asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.5:LRA=11",
            "-movflags", "+faststart", "clip.mp4",
        ], cwd=out_dir, duration=end - start, status=status)
    try:
        render(video_encoder())
    except Exception as e:
        if video_encoder()[1] == "libx264":
            raise
        log.warning("hardware encoder failed, using x264: %s", str(e)[-200:])
        global _encoder
        _encoder = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
        render(_encoder)
    status("saving thumbnail")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", "clip.mp4", "-frames:v", "1",
         "-vf", "scale=360:-2", "thumb.jpg"], cwd=out_dir)

    info = {
        "id": cid,
        "platform": meta["platform"],
        "channel": meta["channel"],
        "name": meta.get("name") or meta["channel"],
        "category": meta.get("category", ""),
        "trigger": meta.get("trigger", "chat"),
        "title": title,
        "created": time.time(),
        "moment_at": moment.peak,
        "duration": round(end - start, 1),
        "score": moment.score,
        "chat_rate": moment.rate,
        "baseline": moment.baseline,
        "keywords": moment.keywords,
        "chat": moment.top_messages,
        "language": lang,
        "layout": layout.mode,
        "game_top": getattr(layout, "game_top", True),
        "layout_label": ("Gameplay top · facecam bottom" if getattr(layout, "game_top", True) else "Facecam top · gameplay bottom")
                        if layout.mode == "split" else {"face": "Face crop", "fill": "Blurred fill"}.get(layout.mode, layout.mode),
        "transcript": " ".join(w.text for w in clip_words),
        "ai": bool(verdict),
        "ai_reason": (verdict or {}).get("reason", ""),
        "render_seconds": round(time.time() - t0, 1),
        "video": f"/clips/{cid}/clip.mp4",
        "thumb": f"/clips/{cid}/thumb.jpg",
    }
    (out_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _cleanup(out_dir)
    status("done")
    return info


def _cleanup(out_dir: Path, keep_meta: bool = True):
    for name in ("raw.ts", "audio.wav"):
        if os.getenv("KEEP_RAW") != "1":
            (out_dir / name).unlink(missing_ok=True)
    if not keep_meta:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


def list_clips() -> list[dict]:
    out = []
    for m in settings.clips_dir.glob("*/meta.json"):
        try:
            out.append(json.loads(m.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(out, key=lambda c: -c.get("created", 0))
