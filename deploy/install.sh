#!/usr/bin/env bash
#
# One-shot installer for the Pi. Run it from the repo, as the user that should
# own the server (not root):
#
#     cd ~/HomeGames && ./deploy/install.sh
#
# Sets up:
#   - a virtualenv with python-chess in it
#   - homegames.service         — the server itself, started at boot
#   - homegames-deploy.timer    — checks main every 2 minutes, deploys changes
#   - a one-command sudoers rule so the deploy can restart the service
#
# Re-running it is safe; it overwrites the units and reloads systemd.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
BRANCH="${HOMEGAMES_BRANCH:-main}"
UNIT_DIR=/etc/systemd/system
SYSTEMCTL="$(command -v systemctl)"

if [ "$USER_NAME" = "root" ]; then
    echo "Run this as the user that should own the server, not as root." >&2
    echo "The deploy runs git as this user; root-owned .git objects would" >&2
    echo "break every later git command you run on the Pi." >&2
    exit 1
fi

echo "Repo:    $REPO"
echo "User:    $USER_NAME:$GROUP_NAME"
echo "Branch:  $BRANCH"
echo

# --- virtualenv ------------------------------------------------------------
# Raspberry Pi OS Bookworm and later mark the system interpreter
# externally-managed (PEP 668), so `pip install chess` against it fails with
# "externally-managed-environment". A venv sidesteps that without
# --break-system-packages and without touching the system python.
if [ ! -d "$REPO/.venv" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$REPO/.venv"
fi
echo "Installing dependencies..."
"$REPO/.venv/bin/pip" install --quiet --upgrade pip
"$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements.txt"
"$REPO/.venv/bin/python" -c 'import chess; print("  python-chess", chess.__version__)'

chmod +x "$REPO/deploy/deploy.sh"

# --- units -----------------------------------------------------------------
render() {
    sed -e "s|__REPO__|$REPO|g" \
        -e "s|__USER__|$USER_NAME|g" \
        -e "s|__GROUP__|$GROUP_NAME|g" \
        -e "s|__BRANCH__|$BRANCH|g" "$1"
}

echo "Installing systemd units..."
render "$REPO/deploy/homegames.service"        | sudo tee "$UNIT_DIR/homegames.service"        >/dev/null
render "$REPO/deploy/homegames-deploy.service" | sudo tee "$UNIT_DIR/homegames-deploy.service" >/dev/null
sudo cp "$REPO/deploy/homegames-deploy.timer" "$UNIT_DIR/homegames-deploy.timer"

# --- sudoers ---------------------------------------------------------------
# Exactly one command, no password. Validated before install: a malformed
# sudoers file can lock you out of sudo entirely, so it is written to a temp
# file, checked with visudo -c, and only then moved into place.
echo "Granting $USER_NAME permission to restart the service..."
SUDO_TMP="$(mktemp)"
echo "$USER_NAME ALL=(root) NOPASSWD: $SYSTEMCTL restart homegames.service" > "$SUDO_TMP"
if sudo visudo -c -f "$SUDO_TMP" >/dev/null; then
    sudo install -m 0440 -o root -g root "$SUDO_TMP" /etc/sudoers.d/homegames-deploy
    rm -f "$SUDO_TMP"
else
    rm -f "$SUDO_TMP"
    echo "Refusing to install a sudoers file that does not validate." >&2
    exit 1
fi

# --- go --------------------------------------------------------------------
sudo "$SYSTEMCTL" daemon-reload
sudo "$SYSTEMCTL" enable --now homegames.service
sudo "$SYSTEMCTL" enable --now homegames-deploy.timer

echo
echo "Done."
echo "  Server:     sudo systemctl status homegames"
echo "  Logs:       journalctl -u homegames -f"
echo "  Deploys:    journalctl -u homegames-deploy -f"
echo "  Next check: $(systemctl show homegames-deploy.timer -p NextElapseUSecRealtime --value 2>/dev/null || echo 'within 2 min')"
echo "  Deploy now: ./deploy/deploy.sh"
