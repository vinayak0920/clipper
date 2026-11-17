"""Turns a 16:9 stream into a 9:16 vertical layout.

Three layouts, picked automatically from face detection on sampled frames:
  split : gaming stream with a facecam overlay -> facecam on top, gameplay below
  face  : IRL / just chatting (the face is big) -> 9:16 crop that follows the face
  fill  : no face found -> sharp 4:3 center on a blurred, darkened copy of the frame
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import ROOT

log = logging.getLogger("reframe")
OUT_W, OUT_H = 1080, 1920
TOP_H = 700                     # facecam panel height in split layout
YUNET = ROOT / "assets" / "models" / "yunet.onnx"


@dataclass
class Layout:
    mode: str
    src_w: int
    src_h: int
    cam: tuple[int, int, int, int] | None = None   # x, y, w, h in source pixels
    face_cx: int | None = None
    caption_y: int = 1360

    def filter_graph(self) -> str:
        """ffmpeg filter_complex producing [v] at 1080x1920."""
        W, H = self.src_w, self.src_h
        if self.mode == "split" and self.cam:
            x, y, w, h = self.cam
            game_w = _even(min(W, H * OUT_W / (OUT_H - TOP_H)))
            game_x = _even((W - game_w) / 2)
            return (
                f"[0:v]split=2[a][b];"
                f"[a]crop={w}:{h}:{x}:{y},scale={OUT_W}:{TOP_H}:flags=lanczos,setsar=1[top];"
                f"[b]crop={game_w}:{H}:{game_x}:0,scale={OUT_W}:{OUT_H - TOP_H}:flags=lanczos,setsar=1[bot];"
                f"[top][bot]vstack=inputs=2[v]"
            )
        if self.mode == "face" and self.face_cx is not None:
            cw = _even(min(W, H * 9 / 16))
            cx = int(min(max(self.face_cx - cw / 2, 0), W - cw))
            return f"[0:v]crop={cw}:{H}:{_even(cx)}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v]"
        fg_w = _even(min(W, H * 4 / 3))
        fg_x = _even((W - fg_w) / 2)
        return (
            f"[0:v]split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"boxblur=8:2,eq=brightness=-0.18:saturation=1.2,scale={OUT_W}:{OUT_H},setsar=1[bg];"
            f"[b]crop={fg_w}:{H}:{fg_x}:0,scale={OUT_W}:-2:flags=lanczos,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-40[v]"
        )


def _even(v: float) -> int:
    v = int(v)
    return v - (v % 2)


def _detector(w: int, h: int):
    if YUNET.exists():
        try:
            d = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.75, 0.3, 50)
            return ("yunet", d)
        except Exception as e:  # older OpenCV
            log.debug("yunet unavailable: %s", e)
    return ("haar", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))


def _sample_frames(video: Path, start: float, duration: float, samples: int, sw: int, sh: int):
    """Small BGR frames spread over the window, decoded by ffmpeg with a fast seek
    instead of reading every frame from the start of the file."""
    import subprocess
    import numpy as np
    rate = samples / max(duration, 1.0)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{duration:.2f}", "-an",
           "-vf", f"fps={rate:.4f},scale={sw}:{sh}", "-frames:v", str(samples),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    size = sw * sh * 3
    return [np.frombuffer(out[i:i + size], np.uint8).reshape(sh, sw, 3) for i in range(0, len(out) - size + 1, size)]


def analyze(video: Path, start: float = 0.0, duration: float = 30.0, samples: int = 14) -> Layout:
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()
    scale = 640 / W
    sw, sh = 640, int(H * scale) // 2 * 2
    kind, det = _detector(sw, sh)
    try:
        frames = _sample_frames(video, start, duration, samples, sw, sh)
    except Exception as e:
        log.warning("frame sampling failed: %s", e)
        frames = []
    faces = []
    for small in frames:
        found = []
        if kind == "yunet":
            _, res = det.detect(small)
            if res is not None:
                found = [tuple(r[:4]) for r in res]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found = [tuple(r) for r in det.detectMultiScale(gray, 1.1, 6, minSize=(24, 24))]
        if found:
            x, y, w, h = max(found, key=lambda r: r[2] * r[3])
            faces.append((x / scale, y / scale, w / scale, h / scale))
    return choose_layout(W, H, faces, len(frames))


def choose_layout(W: int, H: int, faces: list[tuple], sampled: int) -> Layout:
    if sampled == 0 or len(faces) < max(2, sampled * 0.35):
        return Layout("fill", W, H, caption_y=1500)

    cxs = [x + w / 2 for x, y, w, h in faces]
    cys = [y + h / 2 for x, y, w, h in faces]
    fh = statistics.median(h for *_, h in faces)
    cx, cy = statistics.median(cxs), statistics.median(cys)

    if fh / H > 0.17:                                   # face is the main subject
        return Layout("face", W, H, face_cx=int(cx), caption_y=1380)

    spread = (statistics.pstdev(cxs) + statistics.pstdev(cys)) / W
    if spread > 0.06:                                   # faces jump around: probably in-game characters
        return Layout("fill", W, H, caption_y=1500)

    # facecam overlay: build a box around the face with the top panel's aspect ratio
    aspect = OUT_W / TOP_H
    ch = min(H, fh * 3.4)
    cw = min(W, ch * aspect)
    ch = cw / aspect
    x = min(max(cx - cw / 2, 0), W - cw)
    y = min(max(cy - ch * 0.45, 0), H - ch)
    return Layout("split", W, H, cam=(_even(x), _even(y), _even(cw), _even(ch)), caption_y=TOP_H + 170)
