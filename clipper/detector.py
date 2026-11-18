"""Finds highlight moments from chat behaviour.

Each channel gets its own baseline, so a 200-viewer stream and a 50k-viewer stream
are judged against their own normal chat speed, not a global number.

Signals:
  * chat speed over the last 5s vs the channel's slow-moving baseline
  * hype share: messages with laugh / shock / clip tokens (KEKW, LUL, 😂, 💀, clip it...)
  * spam share: many people sending the same emote or word at once
"""
from __future__ import annotations

import collections
import math
import re
import time
from dataclasses import dataclass, field

HYPE = {
    # laughs
    "kekw", "lul", "lmao", "lmfao", "lol", "omegalul", "icant", "kekl", "pepelaugh", "xd", "haha", "hahaha",
    "😂", "🤣", "💀", "😭",
    # shock / hype
    "pog", "pogchamp", "poggers", "pogu", "wtf", "omg", "holy", "wow", "insane", "clean", "nuts", "monkas",
    "w", "ww", "www", "l", "ll", "gg", "ez", "noway", "🔥", "😱", "😳", "‼️", "?????",
    # clip requests
    "clip", "clipit", "clipped", "clipthat",
    # Hindi / Hinglish chat
    "bhai", "op", "bawaal", "gazab", "mast", "jhakaas", "khatarnak", "hahahaha",
}
TOKEN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


@dataclass
class Moment:
    start: float        # when chat started reacting (wall clock)
    peak: float
    end: float
    score: float
    rate: float
    baseline: float
    top_messages: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


class Detector:
    def __init__(self, sensitivity=2.6, warmup=45.0, cooldown=75.0, window=5.0):
        self.sensitivity = sensitivity
        self.warmup = warmup
        self.cooldown = cooldown
        self.window = window
        self.msgs: collections.deque = collections.deque()   # (ts, text, tokens)
        self.started = time.time()
        self.baseline = None
        self.base_var = 0.0
        self.base_hype = 0.0        # how 'hype' this chat normally is (some chats spam LUL all day)
        self.active = None          # dict while a spike is in progress
        self.last_moment_end = 0.0
        self.history: collections.deque = collections.deque(maxlen=180)  # (ts, rate, ratio) for the UI

    def add(self, ts: float, text: str):
        self.msgs.append((ts, text, _tokens(text)))

    def _window(self, now: float):
        while self.msgs and self.msgs[0][0] < now - 120:
            self.msgs.popleft()
        return [m for m in self.msgs if m[0] >= now - self.window]

    def tick(self, now: float | None = None) -> Moment | None:
        """Call once per second. Returns a Moment when a spike has finished."""
        now = now or time.time()
        win = self._window(now)
        rate = len(win) / self.window

        hype = sum(1 for _, _, tk in win if any(t in HYPE for t in tk))
        hype_share = hype / len(win) if win else 0.0
        firsts = collections.Counter(tk[0] for _, _, tk in win if tk)
        spam_share = (firsts.most_common(1)[0][1] / len(win)) if win and len(win) >= 4 else 0.0

        if self.baseline is None:
            self.baseline = rate
        base = max(self.baseline, 0.15)
        ratio = rate / base
        extra_hype = max(0.0, hype_share - self.base_hype)
        boosted = ratio * (1 + 1.2 * extra_hype + 0.4 * spam_share)
        self.history.append((now, round(rate, 2), round(boosted, 2)))

        warm = now - self.started >= self.warmup
        # chat arrives in random bursts; on small channels that noise alone can triple the rate,
        # so the count must also clear a Poisson-style bar for this channel's normal speed
        expected = base * self.window
        sd = math.sqrt(max(expected, self.base_var * self.window ** 2))
        significant = len(win) >= expected + 4 * sd + 3
        # Big chats (3+ messages a second) rarely triple their speed, but they have far more
        # moments worth clipping: lower the bar a little and allow clips closer together.
        big = base >= 3.0
        sens = self.sensitivity * (0.8 if big else 1.0)
        cooldown = min(self.cooldown, 20.0) if big else self.cooldown
        firing = (
            warm
            and significant
            and boosted >= sens
            and ratio >= max(1.6 if big else 1.8, sens * 0.7)
        )

        result = None
        if self.active is None:
            if firing and now - self.last_moment_end >= cooldown:
                self.active = {"start": now - self.window, "peak": now, "peak_score": boosted,
                               "peak_rate": rate, "msgs": list(win)}
        else:
            a = self.active
            last_ts = a["msgs"][-1][0] if a["msgs"] else 0.0
            a["msgs"].extend(m for m in win if m[0] > last_ts)
            if boosted > a["peak_score"]:
                a.update(peak=now, peak_score=boosted, peak_rate=rate)
            cooled = ratio < 1.5 or boosted < sens * 0.55
            too_long = now - a["start"] > 30
            if cooled or too_long:
                result = self._finish(now)

        # Baseline only learns from calm periods, so a long hype stretch doesn't become the new normal
        if self.active is None:
            alpha = 0.02 if warm else 0.15
            diff = rate - self.baseline
            self.baseline += alpha * diff
            self.base_var = (1 - alpha) * (self.base_var + alpha * diff * diff)
            if win:
                self.base_hype += alpha * (hype_share - self.base_hype)
        return result

    def _finish(self, now: float) -> Moment:
        a = self.active
        self.active = None
        self.last_moment_end = now
        texts = [t for _, t, _ in a["msgs"]]
        counts = collections.Counter(t for _, _, tk in a["msgs"] for t in set(tk) if t in HYPE)
        common = collections.Counter(texts).most_common(8)
        return Moment(
            start=a["start"], peak=a["peak"], end=now,
            score=round(a["peak_score"], 2), rate=round(a["peak_rate"], 2),
            baseline=round(self.baseline or 0, 2),
            top_messages=[t for t, _ in common],
            keywords=[k for k, _ in counts.most_common(6)],
        )
