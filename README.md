# Clipper

Open-source live clipper for **Twitch** and **Kick**. Choose streamers by country, and Clipper records them live, spots the moments chat loses it, and gives you vertical 9:16 clips with animated captions and a hook on top, ready for TikTok, Shorts and Reels.

Recording, transcription and rendering all run on your own machine. Sign in with Google for Twitch listings and AI hooks with no keys at all, or bring your own key or Ollama and use it unlimited and offline. No watermark, ever.

## What it does

- **Choose streamers by region.** India, Brazil, Spain/LatAm, Turkey, Arabic, Japan, Korea and more. Mix Twitch and Kick, sorted by live viewers, or add any channel by name.
- **Records live.** Keeps a rolling buffer of the last 4 minutes per streamer. The stream is copied as-is, not re-encoded, so CPU use stays low.
- **Finds the moment from chat.** Each streamer's normal chat speed is learned first. A clip fires when chat clears a statistical bar for its size and either jumps well above that normal speed or fills with laugh, shock and "clip it" words (KEKW, LUL, 💀, W, bhai, op...). Hype words only count above what that chat usually spams.
- **Cuts with context.** Each clip starts about 20 seconds before chat reacted, so the setup is in, and ends just after the payoff.
- **Picks the vertical layout itself.** Stream category and persistent facecam detection choose one of three:
  - **split:** gaming streams use two panels with a centered close-up facecam. Most games put larger gameplay above the facecam; GTA, Red Dead Redemption and Cyberpunk put the facecam above larger gameplay.
  - **face:** crop that follows the streamer, for IRL streams
  - **blurred fill:** centered gameplay over a blurred copy of itself when a webcam cannot be located
- Gameplay framing uses the center of the source feed, which generally keeps the crosshair and active play visible. It does not yet read a game's alive/dead state or track the in-game player automatically.
- **Animated captions.** Word by word: 1-3 words on screen, the spoken word pops and turns yellow or green, shouts go red. Hindi can come out as Hinglish ("bhai kya hua") or in Devanagari.
- **AI hooks and trimming (optional).** An AI writes a short hook in the streamer's language, trims the clip to the setup and payoff, and skips moments that won't work as a short. The hook shows as a bold white bar for the first 3 seconds.
- **Dashboard.** Three views:
  - **Live:** a monitor wall with a recording light, chat graph and level meter for each streamer, plus a sensitivity slider and a **Clip last 30s** button.
  - **Clips:** a gallery to filter, play, download, copy the hook or delete clips.
  - **Settings:** paste keys, test each connection and tune detection, with no file editing.

## Quick start (macOS)

```bash
git clone https://github.com/vinayak0920/clipper.git
cd clipper
./run.sh
```

`run.sh` installs ffmpeg (through Homebrew) and the Python packages, creates `.env` from `.env.example`, and opens http://localhost:8000.

Linux works too: install `ffmpeg` and Python 3.10+, then run the same script. Windows: use WSL.

## Setup

Open **Settings** in the dashboard and sign in with Google. That's all most people need: Twitch listings and 50 AI hooks a month are free, and Pro (₹300 a month) gives 3,000 hooks.

Prefer your own keys? Paste them, press **Save changes**, then **Test connection**. Each service gets a status light, and Ollama models can be downloaded from the same page. Everything is saved to `.env` on your computer, and the dashboard only answers requests addressed to localhost. None of the keys are required to clip a channel you add by name.

| What | Why | How |
|---|---|---|
| Twitch app (optional when signed in) | List Twitch streamers by region | [dev.twitch.tv/console/apps](https://dev.twitch.tv/console/apps): turn on two-factor auth on your Twitch account, register an app with redirect `https://localhost`, then copy `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET` |
| Kick app (optional) | Kick's official listing API | [kick.com/settings/developer](https://kick.com/settings/developer). Without it, Clipper reads Kick's public browse feed |
| AI hooks (optional when signed in) | Hooks, trimming, Hinglish | Free and local: `brew install ollama && ollama pull qwen2.5:7b`. Or set one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` |

`AI_PROVIDER=auto` uses your own key first, then Clipper Cloud if you're signed in, then Ollama. Without either, clips still get captions, plus a simple title instead of an AI hook. Any OpenAI-compatible server (LM Studio, vLLM, OpenRouter) works through `OPENAI_BASE_URL`.

## How detection works

Every second, for each streamer:

1. Count chat messages in the last 5 seconds.
2. Compare that count to the streamer's own slow-moving baseline, which only learns from calm periods.
3. Check the count clears `expected + 4 * sd + 3`. Small chats are noisy, and this stops random bursts from firing.
4. Weight the result by extra hype words and spam.
5. When the spike cools down, cut the clip and queue it for rendering.

For high-chat channels (baseline at least 3 messages/second), Clipper also scans every 15 seconds for a 10-second stretch with a nearly still game image and recognizable speech. That can catch a streamer talking to chat while a game is paused even when chat never produces a new spike. It waits at least 60 seconds between these candidates. This is an approximation: it cannot verify gaze direction or know the game's alive/dead state, and loud game audio can affect speech recognition.

Turn the sensitivity up (3-4) for fewer, bigger moments, or down (2-2.3) to catch more.

## Project layout

```
clipper/
  discovery.py   live streamers by region (Twitch Helix, Kick public API / browse feed)
  regions.py     region -> languages + country keywords
  recorder.py    streamlink -> ffmpeg rolling segment buffer
  chat.py        Twitch IRC (anonymous), Kick chat feed
  detector.py    per-channel chat spike detection
  talk_detector.py  paused-game speech candidate detection for high-chat channels
  pipeline.py    cut -> transcribe -> AI judge -> layout -> render
  reframe.py     face detection and 9:16 layouts
  captions.py    animated .ass captions and hook bar
  llm.py         own key / Clipper Cloud / Ollama
  cloud.py       Clipper Cloud client (sign-in, listings, hooks)
  server.py      FastAPI backend
web/index.html   dashboard
```

## Settings

All settings live in `.env`. See `.env.example` for the full list.

| Setting | Default | |
|---|---|---|
| `SENSITIVITY` | 2.6 | how far above normal chat must go |
| `PRE_ROLL` / `POST_ROLL` | 22 / 6 | seconds kept before chat reacted / after it calmed |
| `BUFFER_SECONDS` | 240 | rolling video buffer per streamer |
| `STREAM_QUALITY` | `720p60,720p,best` | streamlink quality order |
| `WHISPER_MODEL` | small | `base` is faster, `medium` / `large-v3` are more accurate for Hindi |
| `HINDI_SCRIPT` | latin | `latin` for Hinglish, `devanagari` for Hindi script |
| `SHOW_TITLE` | 1 | hook bar on or off |

## Limits

- Kick's public chat feed returns the last 25 messages. Polling once a second is accurate up to about 25 messages/second. Above that it caps out, which still registers as a spike.
- Twitch ad breaks are skipped while recording, which leaves a short gap in the buffer.
- One clip renders at a time. On an M-series Mac, expect about 15-40 seconds per clip.

## Clip responsibly

Clipper records public streams for highlights. Before posting someone else's content, get the streamer's permission or join their clipping programme. Follow each platform's terms of service.

## Contributing

Issues and pull requests are welcome. Good first areas are new regions and hype words for more languages, caption styles, and more layouts.

## Made by Apavin

Clipper is a free tool from [Apavin](https://apavin.com/stream-clipper). Pro accounts are at [apavin.com/stream-clipper](https://apavin.com/stream-clipper).

## License

[MIT](LICENSE) © 2026 Vinayak Kaushal. Bundled fonts and the face model keep their own licences, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
