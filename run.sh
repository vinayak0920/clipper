#!/usr/bin/env bash
# One command setup + start. Opens the dashboard at http://localhost:8000
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing FFmpeg with subtitle support (one time)..."
    brew install ffmpeg-full
    export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"
  else
    echo "ffmpeg is missing. Install FFmpeg with libass/subtitles support, then run this script again."; exit 1
  fi
fi

if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -Eq '[[:space:]]subtitles[[:space:]]'; then
  if command -v brew >/dev/null 2>&1; then
    echo "Your FFmpeg build has no subtitles filter. Installing the full Homebrew build..."
    brew install ffmpeg-full
    export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"
  fi
fi
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -Eq '[[:space:]]subtitles[[:space:]]'; then
  echo "FFmpeg needs the subtitles filter (libass). Install a build with libass, then run this script again."; exit 1
fi

PYTHON_BIN=""
for v in python3.12 python3.11 python3.13 python3.10; do
  if command -v "$v" >/dev/null 2>&1; then PYTHON_BIN="$v"; break; fi
done
if [ -z "$PYTHON_BIN" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing supported Python 3.12 (one time)..."
    brew install python@3.12
    PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"
  else
    echo "Python 3.10-3.13 is required. Install Python 3.12, then run this script again."; exit 1
  fi
fi

if [ -d .venv ]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  case "$VENV_VERSION" in
    3.10|3.11|3.12|3.13) ;;
    *)
      BACKUP_SUFFIX="$(date +%Y%m%d%H%M%S)"
      echo "Moving unsupported or broken .venv (Python ${VENV_VERSION:-unknown}) to .venv-backup-$BACKUP_SUFFIX"
      mv .venv ".venv-backup-$BACKUP_SUFFIX"
      ;;
  esac
fi

if [ ! -d .venv ]; then
  echo "Creating Python environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
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
