"""Turns a 16:9 stream into a 9:16 vertical layout.

Three layouts, picked from the stream category and persistent facecam detections:
  split : gameplay and a close, centered facecam share the vertical frame
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
TOP_H = 1180                    # larger gameplay panel when the game is on top
CAM_TOP_H = 700                 # smaller, centered facecam panel for GTA/story games
YUNET = ROOT / "assets" / "models" / "yunet.onnx"


@dataclass
class Layout:
    mode: str
    src_w: int
    src_h: int
    cam: tuple[int, int, int, int] | None = None   # x, y, w, h in source pixels
    face_cx: int | None = None
    caption_y: int = 1360
    game_top: bool = True

    def filter_graph(self) -> str:
        """ffmpeg filter_complex producing [v] at 1080x1920."""
        W, H = self.src_w, self.src_h
        if self.mode == "split" and self.cam:
            x, y, w, h = self.cam
            game_h = TOP_H if self.game_top else OUT_H - CAM_TOP_H
            cam_h = OUT_H - game_h
            game_w = _even(min(W, H * OUT_W / game_h))
            game_x = _even((W - game_w) / 2)
            cam = f"[b]crop={w}:{h}:{x}:{y},scale={OUT_W}:{cam_h}:flags=lanczos,setsar=1[cam]"
            game = f"[a]crop={game_w}:{H}:{game_x}:0,scale={OUT_W}:{game_h}:flags=lanczos,setsar=1[game]"
            stack = "[game][cam]" if self.game_top else "[cam][game]"
            return f"[0:v]split=2[a][b];{game};{cam};{stack}vstack=inputs=2[v]"
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
            d = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.55, 0.3, 100)
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


def analyze(video: Path, start: float = 0.0, duration: float = 30.0, samples: int = 14,
            category: str = "") -> Layout:
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()
    scale = 960 / W
    sw, sh = 960, int(H * scale) // 2 * 2
    kind, det = _detector(sw, sh)
    try:
        frames = _sample_frames(video, start, duration, samples, sw, sh)
    except Exception as e:
        log.warning("frame sampling failed: %s", e)
        frames = []
    faces = []
    faces_by_frame = []
    for small in frames:
        found = []
        if kind == "yunet":
            _, res = det.detect(small)
            if res is not None:
                found = [tuple(r[:4]) for r in res]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found = [tuple(r) for r in det.detectMultiScale(gray, 1.05, 4, minSize=(18, 18))]
        full = [(x / scale, y / scale, w / scale, h / scale) for x, y, w, h in found]
        faces.extend(full)
        faces_by_frame.append(full)
    return choose_layout(W, H, faces, len(frames), category=category, faces_by_frame=faces_by_frame)


def _game_category(category: str) -> str:
    return " ".join(category.lower().replace("&", "and").split())


def _gta_style(category: str) -> bool:
    c = _game_category(category)
    return any(k in c for k in ("grand theft auto", "gta", "red dead redemption", "cyberpunk 2077"))


def _is_game(category: str) -> bool:
    c = _game_category(category)
    return bool(c) and c not in ("just chatting", "talk shows and podcasts", "music", "irl")


def _stable_facecam(faces_by_frame: list[list[tuple]], W: int, H: int) -> tuple | None:
    """Find a small, persistent face in one screen location (usually the webcam overlay)."""
    clusters = []
    for frame_i, detections in enumerate(faces_by_frame):
        claimed = set()
        for face in detections:
            x, y, w, h = face
            if not 0.035 <= h / H <= 0.24:
                continue
            cx, cy = x + w / 2, y + h / 2
            # Webcam overlays are normally near an edge. A game character in the
            # middle of the gameplay should not be mistaken for the streamer.
            near_edge = cx < W * 0.38 or cx > W * 0.62 or cy < H * 0.22 or cy > H * 0.78
            if not near_edge:
                continue
            best, best_dist = None, 1.0
            for ci, c in enumerate(clusters):
                if ci in claimed:
                    continue
                mx, my = c["cx"], c["cy"]
                dist = max(abs(cx - mx) / W, abs(cy - my) / H)
                size_delta = abs(h - c["h"]) / H
                if dist < 0.08 and size_delta < 0.10 and dist < best_dist:
                    best, best_dist = ci, dist
            if best is None:
                clusters.append({"cx": cx, "cy": cy, "h": h, "items": [(frame_i, face)]})
                claimed.add(len(clusters) - 1)
            else:
                c = clusters[best]
                c["items"].append((frame_i, face))
                n = len(c["items"])
                c["cx"] += (cx - c["cx"]) / n
                c["cy"] += (cy - c["cy"]) / n
                c["h"] += (h - c["h"]) / n
                claimed.add(best)
    min_hits = max(2, int(len(faces_by_frame) * 0.20 + 0.99))
    viable = [c for c in clusters if len({i for i, _ in c["items"]}) >= min_hits]
    if not viable:
        return None
    best = max(viable, key=lambda c: len(c["items"]) * (c["h"] / H) ** 0.25)
    boxes = [box for _, box in best["items"]]
    return tuple(statistics.median(box[i] for box in boxes) for i in range(4))


def choose_layout(W: int, H: int, faces: list[tuple], sampled: int, category: str = "",
                  faces_by_frame: list[list[tuple]] | None = None) -> Layout:
    facecam = _stable_facecam(faces_by_frame or [], W, H)
    if facecam:
        x, y, w, h = facecam
        game_top = not _gta_style(category)
        game_h = TOP_H if game_top else OUT_H - CAM_TOP_H
        cam_h = OUT_H - game_h
        aspect = OUT_W / cam_h
        ch = min(H, h * 3.4)
        cw = min(W, ch * aspect)
        ch = cw / aspect
        cx, cy = x + w / 2, y + h / 2
        crop_x = min(max(cx - cw / 2, 0), W - cw)
        crop_y = min(max(cy - ch * 0.45, 0), H - ch)
        cap_y = TOP_H - 185 if game_top else OUT_H - 210
        return Layout("split", W, H, cam=(_even(crop_x), _even(crop_y), _even(cw), _even(ch)),
                      caption_y=cap_y, game_top=game_top)

    # Game categories should never turn a large in-game portrait into a face-only crop.
    if _is_game(category) or sampled == 0 or len(faces) < max(2, sampled * 0.35):
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

    # For non-game streams, follow a dominant face with a full-height portrait crop.
    return Layout("face", W, H, face_cx=int(cx), caption_y=1380)
