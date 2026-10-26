#!/bin/sh
# Build the test games' histories on first start, then run the requested command.
# GH_GAME picks the game the dashboard hunts in (crypt or ironworks); GH_REPO points at any
# other game repository with a glitchhunter.json instead.
set -e

if [ -z "$GH_REPO" ]; then
  if [ ! -d "$GH_WORK_DIR/fixture/.git" ]; then
    python -m glitchhunter fixture --game crypt
  fi
  if [ ! -d "$GH_WORK_DIR/fixture-ironworks/.git" ]; then
    python -m glitchhunter fixture --game ironworks
  fi
fi

case "$1" in
  serve)
    exec python -m glitchhunter serve --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  *)
    exec python -m glitchhunter "$@"
    ;;
esac
