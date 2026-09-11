#!/bin/bash
# Disable gnome-remote-desktop (RDP on port 3389) and turn off SSH password auth.
# Run from a real terminal: bash sh/harden_network.sh

set -e

echo "=== Disabling gnome-remote-desktop ==="
sudo systemctl disable --now gnome-remote-desktop
systemctl --user disable --now gnome-remote-desktop 2>/dev/null || true

echo "=== Disabling SSH password authentication ==="
sudo sed -i.bak 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
grep -q '^PermitRootLogin' /etc/ssh/sshd_config \
    || echo 'PermitRootLogin no' | sudo tee -a /etc/ssh/sshd_config
sudo systemctl reload ssh

echo "=== Verifying (3389 should be gone, 22 should remain) ==="
ss -tlnp | grep -E ':22|:3389' || echo "(no matching ports — 3389 is closed)"
echo "Done."
