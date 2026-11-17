"""Live chat listeners. Both are read-only and need no login.

Twitch: anonymous IRC over WebSocket (justinfan nick).
Kick:   public chat history endpoint, polled every second.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import random
import re
import time
from typing import Callable

import websockets


log = logging.getLogger("chat")
OnMessage = Callable[[float, str, str], None]  # (timestamp, user, text)

KICK_EMOTE = re.compile(r"\[emote:\d+:([^\]]+)\]")


async def twitch_chat(channel: str, on_message: OnMessage, stop: asyncio.Event):
    nick = f"justinfan{random.randint(10000, 99999)}"
    while not stop.is_set():
        try:
            async with websockets.connect("wss://irc-ws.chat.twitch.tv:443", ping_interval=60) as ws:
                await ws.send("CAP REQ :twitch.tv/tags")
                await ws.send("PASS SCHMOOPIIE")
                await ws.send(f"NICK {nick}")
                await ws.send(f"JOIN #{channel.lower()}")
                log.info("twitch chat connected: %s", channel)
                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        continue
                    for line in str(raw).split("\r\n"):
                        if not line:
                            continue
                        if line.startswith("PING"):
                            await ws.send("PONG :tmi.twitch.tv")
                            continue
                        if " PRIVMSG " in line:
                            user, text = _parse_privmsg(line)
                            on_message(time.time(), user, text)
        except Exception as e:
            if stop.is_set():
                break
            log.warning("twitch chat %s dropped (%s), reconnecting", channel, e)
            await asyncio.sleep(3)


def _parse_privmsg(line: str) -> tuple[str, str]:
    rest = line
    user = ""
    if rest.startswith("@"):
        tags, rest = rest[1:].split(" ", 1)
        for kv in tags.split(";"):
            if kv.startswith("display-name="):
                user = kv.split("=", 1)[1]
    prefix_end = rest.find(" PRIVMSG ")
    if not user and rest.startswith(":"):
        user = rest[1:rest.find("!")]
    msg_start = rest.find(" :", prefix_end + 1)
    return user, rest[msg_start + 2:] if msg_start >= 0 else ""


async def kick_chat(channel_id: int, on_message: OnMessage, stop: asyncio.Event, interval: float = 1.0):
    """Kick moved live chat to an authenticated socket, so we poll the public history
    endpoint (last 25 messages) once a second and keep only messages we haven't seen.
    Handles up to ~25 msgs/s exactly; above that the count saturates, which still reads as a spike."""
    from curl_cffi import requests as cffi_requests

    url = f"https://web.kick.com/api/v1/chat/{channel_id}/history"
    seen: collections.OrderedDict = collections.OrderedDict()
    session = cffi_requests.Session(impersonate="chrome")
    first = True
    fails = 0
    log.info("kick chat polling: channel %s", channel_id)
    while not stop.is_set():
        t0 = time.time()
        try:
            r = await asyncio.to_thread(session.get, url, timeout=10)
            r.raise_for_status()
            msgs = ((r.json().get("data") or {}).get("messages")) or []
            fresh = [m for m in reversed(msgs) if m.get("id") not in seen]
            for m in fresh:
                seen[m.get("id")] = 1
                if first:
                    continue          # don't count the backlog as live activity
                text = KICK_EMOTE.sub(lambda x: x.group(1), m.get("content", "") or "")
                user = (m.get("sender") or {}).get("username", "")
                on_message(time.time(), user, text)
            while len(seen) > 500:
                seen.popitem(last=False)
            first = False
            fails = 0
        except Exception as e:
            fails += 1
            if fails in (1, 10) or fails % 60 == 0:
                log.warning("kick chat %s poll failed (%s)", channel_id, e)
        await asyncio.sleep(max(0.2, interval - (time.time() - t0)))
