#!/usr/bin/env bash
# One-time, permanent Dobot serial setup. Run with:  sudo bash tools/setup_dobot_permanent.sh
# Does NOT move the arm. Idempotent.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

USER_NAME="${SUDO_USER:-sreehitha}"
RULE=/etc/udev/rules.d/99-dobot.rules

# 1. Serial access: add the user to dialout (takes effect after re-login).
usermod -aG dialout "$USER_NAME"

# 2. udev rule: Dobot Magician = Silicon Labs CP210x (10c4:ea60).
#    Gives group dialout rw access and a stable /dev/dobot symlink.
cat > "$RULE" <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0660", GROUP="dialout", SYMLINK+="dobot"
EOF

# 3. brltty (braille daemon) can grab USB-serial adapters; this arm doesn't need it.
if dpkg -s brltty >/dev/null 2>&1; then
  systemctl disable --now brltty-udev.service brltty.service 2>/dev/null || true
  systemctl mask brltty-udev.service brltty.service 2>/dev/null || true
fi

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty

echo "Done. Log out and back in (or reboot the VM) so the dialout group applies."
echo "Then verify:  id | grep dialout ; ls -l /dev/dobot"
