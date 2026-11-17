"""Animated word-by-word captions as an .ass subtitle file (burned in by ffmpeg/libass).

Look: big heavy font, thick black outline, 1-3 words on screen at a time.
Each group pops in (scale 70% -> 100%), the word being spoken turns colored
and bumps up to ~118% scale, emphasis words get a second color.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

W, H = 1080, 1920
HIGHLIGHTS = ["&H0000E5FF&", "&H006EFF3D&"]   # ASS is BGR: yellow #FFE500, green #3DFF6E
EMPHASIS = "&H004D4DFF&"                         # red #FF4D4D for shouts / swears / numbers
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SHOUT = re.compile(r"(!|\?{2,}|^\d+$)")


@dataclass
class Word:
    text: str
    start: float
    end: float


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[Word], max_words=3, max_chars=16, gap=0.45) -> list[list[Word]]:
    groups, cur = [], []
    for w in words:
        if not w.text.strip():
            continue
        chars = sum(len(x.text) + 1 for x in cur) + len(w.text)
        pause = cur and (w.start - cur[-1].end) > gap
        end_sentence = cur and cur[-1].text.rstrip().endswith((".", "?", "!", ","))
        if cur and (len(cur) >= max_words or chars > max_chars or pause or end_sentence):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[Word], y: int = 1360, font_size: int = 112) -> str:
    uses_devanagari = any(DEVANAGARI.search(w.text) for w in words)
    font = "Noto Sans Devanagari" if uses_devanagari else "Montserrat Black"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,10,5,5,50,50,0,1
Style: Hook,{font},84,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,16,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    groups = group_words(words)
    for gi, g in enumerate(groups):
        hl = HIGHLIGHTS[gi % len(HIGHLIGHTS)]
        g_end = g[-1].end + 0.12
        if gi + 1 < len(groups):
            g_end = min(g_end + 0.25, groups[gi + 1][0].start)   # hold briefly, never overlap the next group
        for wi, w in enumerate(g):
            seg_start = w.start if wi else g[0].start
            seg_end = g[wi + 1].start if wi + 1 < len(g) else g_end
            if seg_end <= seg_start:
                seg_end = seg_start + 0.08
            parts = []
            for j, x in enumerate(g):
                t = _esc(x.text.strip())
                t = t if uses_devanagari else t.upper()
                if j == wi:
                    color = EMPHASIS if SHOUT.search(x.text.strip()) else hl
                    parts.append(
                        f"{{\\c{color}\\fscx100\\fscy100\\t(0,70,\\fscx120\\fscy120)\\t(70,150,\\fscx114\\fscy114)}}"
                        f"{t}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}"
                    )
                else:
                    parts.append(t)
            entry = "\\fscx72\\fscy72\\t(0,110,\\fscx100\\fscy100)" if wi == 0 else ""
            text = f"{{\\an5\\pos({W // 2},{y}){entry}}}" + " ".join(parts)
            lines.append(f"Dialogue: 0,{_ts(seg_start)},{_ts(seg_end)},Cap,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 18) -> str:
    """Balanced lines (no lonely last word), max 3."""
    words = text.split()
    n = min(3, max(1, -(-len(text) // width)))
    target = len(text) / n
    lines, cur = [], ""
    for i, word in enumerate(words):
        left = len(words) - i
        if cur and len(lines) < n - 1 and (len(cur) + 1 + len(word) > target + 3) and left >= 1:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return "\\N".join(lines)


def build_title_ass(title: str, duration: float, y: int = 230, seconds: float = 3.0) -> str:
    """Hook bar: black text on a white box, pops in with a slight tilt, fades out after `seconds`."""
    t = _esc(title).strip()
    if not t:
        return ""
    if not DEVANAGARI.search(t):
        t = t.upper()
    end = min(duration, seconds)
    return (
        f"Dialogue: 1,{_ts(0)},{_ts(end)},Hook,,0,0,0,,"
        f"{{\\an5\\pos({W // 2},{y})\\frz2\\fscx60\\fscy60\\t(0,140,\\fscx106\\fscy106)"
        f"\\t(140,220,\\fscx100\\fscy100)\\fad(0,200)}}{_wrap(t)}\n"
    )
