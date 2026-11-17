#!/usr/bin/env bash
# One command setup + start. Opens the dashboard at http://localhost:8000
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg (one time)..."; brew install ffmpeg
  else
    echo "ffmpeg is missing. Install Homebrew from https://brew.sh then run: brew install ffmpeg"; exit 1
  fi
fi

PY=python3
for v in python3.12 python3.11 python3.13 python3.10; do command -v $v >/dev/null 2>&1 && PY=$v && break; done

if [ ! -d .venv ]; then
  echo "Creating Python environment with $PY..."
  $PY -m venv .venv
fi
source .venv/bin/activate
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch .venv/.installed
fi
[ -f .env ] || cp .env.example .env

if command -v ollama >/dev/null 2>&1; then
  curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &)
else
  echo "Tip: for free local AI hooks, install Ollama (brew install ollama) and run: ollama pull qwen2.5:7b"
fi

PORT="${PORT:-8000}"
( sleep 2; open "http://localhost:$PORT" 2>/dev/null || true ) &
exec python -m uvicorn clipper.server:app --host 127.0.0.1 --port "$PORT"
