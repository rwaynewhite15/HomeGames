#!/usr/bin/env bash
#
# Pull new code and restart HomeGames — but only when something actually
# changed, and preferably not on top of a game in progress.
#
# Run by homegames-deploy.timer every couple of minutes. Safe to run by hand.
#
# The whole reason this is careful: the server keeps CLIENTS and ROOMS in
# memory only. stats.json, ai_profiles.json and chess_brains/ are written to
# disk as games finish, so results and AI learning survive a restart — but
# every game still in progress is lost. A deploy that restarts blindly will
# eventually land in the middle of somebody's Euchre hand.
set -euo pipefail

REPO="${HOMEGAMES_REPO:-/home/pi/HomeGames}"
BRANCH="${HOMEGAMES_BRANCH:-main}"
SERVICE="${HOMEGAMES_SERVICE:-homegames.service}"
PORT="${HOMEGAMES_PORT:-4001}"
VENV="$REPO/.venv"

# How many times in a row we will step aside for a game in progress before
# deploying anyway. At the timer's 2-minute cadence, 15 is about half an hour —
# long enough to let a game finish, short enough that one abandoned room cannot
# hold a deploy back forever.
MAX_DEFERRALS="${HOMEGAMES_MAX_DEFERRALS:-15}"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/homegames-deploy"
DEFER_FILE="$STATE_DIR/deferrals"

log() { echo "[deploy $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$STATE_DIR"
cd "$REPO"

# --- is there anything new? ------------------------------------------------
# Fetch failures are normal on a home connection: the Pi may be up before the
# router is. Not an error, just nothing to do this tick.
if ! git fetch --quiet origin "$BRANCH" 2>/dev/null; then
    log "could not reach origin; will try again next tick"
    exit 0
fi

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0                      # nothing new — the overwhelmingly common case
fi

log "new commit on $BRANCH: ${LOCAL:0:8} -> ${REMOTE:0:8}"

# --- refuse to clobber local edits -----------------------------------------
# Someone may have been debugging directly on the Pi. Losing that to an
# automatic pull would be worse than skipping the deploy.
if ! git diff --quiet || ! git diff --cached --quiet; then
    log "ERROR: working tree has uncommitted changes; refusing to pull."
    log "       commit, stash or discard them on the Pi, then re-run."
    exit 1
fi

# --- wait for a game to finish, within reason ------------------------------
PLAYING=""
if command -v curl >/dev/null 2>&1; then
    PLAYING=$(curl -fsS --max-time 3 "http://localhost:$PORT/api/health" 2>/dev/null \
              | python3 -c 'import json,sys; print(json.load(sys.stdin).get("playing", 0))' \
              2>/dev/null || echo "")
fi

if [ -n "$PLAYING" ] && [ "$PLAYING" != "0" ]; then
    COUNT=$(cat "$DEFER_FILE" 2>/dev/null || echo 0)
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -lt "$MAX_DEFERRALS" ]; then
        echo "$COUNT" > "$DEFER_FILE"
        log "$PLAYING game(s) in progress; deferring ($COUNT/$MAX_DEFERRALS)"
        exit 0
    fi
    log "$PLAYING game(s) in progress but deferred $COUNT times; deploying anyway"
fi
rm -f "$DEFER_FILE"

# --- pull ------------------------------------------------------------------
# Fast-forward only. A merge commit created unattended on the Pi is a mess
# nobody will expect to find there; if it will not fast-forward, something
# needs a human.
if ! git merge --ff-only "origin/$BRANCH"; then
    log "ERROR: cannot fast-forward — the Pi has diverged from $BRANCH."
    log "       resolve it on the Pi, then re-run."
    exit 1
fi

# --- dependencies ----------------------------------------------------------
# Only when requirements.txt actually changed, so the normal deploy stays fast.
# This is what installs python-chess the first time you deploy the chess work;
# without it the server comes back up with chess quietly missing.
if ! git diff --quiet "$LOCAL" "$REMOTE" -- requirements.txt; then
    log "requirements.txt changed; updating the virtualenv"
    "$VENV/bin/pip" install --quiet --upgrade -r requirements.txt
fi

# --- restart ---------------------------------------------------------------
log "restarting $SERVICE"
sudo systemctl restart "$SERVICE"

sleep 3
if systemctl is-active --quiet "$SERVICE"; then
    log "deployed ${REMOTE:0:8} — service is up"
else
    log "ERROR: $SERVICE did not come back. Recent log:"
    journalctl -u "$SERVICE" -n 20 --no-pager || true
    exit 1
fi
