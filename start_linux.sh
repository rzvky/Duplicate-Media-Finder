#!/bin/bash
# Duplicate Media Finder: run with: bash start_linux.sh (first run sets everything up).
cd "$(dirname "$0")" || exit 1
VENV=".venv"

setup() {
  echo ""
  echo " Duplicate Media Finder: first-time setup"
  echo " This needs internet once and takes about 2 to 5 minutes."
  echo ""
  PY=""
  for c in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo " Python 3 is not installed on this computer."
    echo " 1. Install Python 3.10 or newer with venv support."
    echo " 2. Then run start_linux.sh again."
    echo " (Ubuntu/Debian: sudo apt install python3 python3-venv python3-tk)"
    read -r -p " Press Enter to close. " _
    exit 1
  fi
  echo " Using Python: $PY ($("$PY" --version 2>&1))"
  "$PY" -m venv "$VENV" || fail
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q --upgrade pip
  "$VENV/bin/python" -m pip install --disable-pip-version-check "Pillow>=10" "numpy>=1.22" "pillow-heif>=0.16" || fail
  echo ""; echo " Installing camera RAW support..."
  "$VENV/bin/python" -m pip install --disable-pip-version-check "rawpy>=0.19" \
    || echo " Note: camera RAW support could not be installed. Everything else still works."
  echo ""; echo " Installing the AI engine..."
  "$VENV/bin/python" -m pip install --disable-pip-version-check "onnxruntime>=1.16" \
    || echo " Note: AI matching could not be installed. Exact and resized copies are still found."
  echo ""; echo " Installing video support..."
  "$VENV/bin/python" -m pip install --disable-pip-version-check "av>=14" \
    || echo " Note: video support could not be installed. Photos are still checked."
  echo ""
  "$VENV/bin/python" -m dupefinder check
  echo ""; echo " Setup finished."; echo ""
}

fail() {
  echo ""
  echo " Setup did not finish. Check your internet connection, then run this file again."
  rm -rf "$VENV"
  read -r -p " Press Enter to close. " _
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  setup
fi
# Older installs: add video support once (skipped if it could not be installed before)
if ! "$VENV/bin/python" -c "import av" >/dev/null 2>&1 && [ ! -f "$VENV/no_video_support.txt" ]; then
  echo " Adding video support, one time only..."
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q "av>=14" \
    || echo "Video support could not be installed. Delete the .venv folder to try again." > "$VENV/no_video_support.txt"
fi
"$VENV/bin/python" -m dupefinder "$@"
echo ""
read -r -p " Press Enter to close this window. " _
