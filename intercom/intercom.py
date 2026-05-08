#!/usr/bin/env python3
"""
intercom.py — File-based communication between Claude Code instances.

Each instance gets a random ID and communicates via a shared channel file.

Usage:
    python intercom/intercom.py init              # Join the channel, get an ID
    python intercom/intercom.py send "message"    # Send a message
    python intercom/intercom.py read              # Read new messages from others
    python intercom/intercom.py peek              # Check for new messages without advancing
    python intercom/intercom.py history           # Show full channel history
"""

import os
import sys
import random
import time
from datetime import datetime

INTERCOM_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNEL_FILE = os.path.join(INTERCOM_DIR, "channel.txt")
ID_FILE_PATTERN = os.path.join(INTERCOM_DIR, "instance_{pid}.id")


def get_id_file():
    """Get the ID file path for this process's parent (the Claude session)."""
    # Use parent PID since this script is invoked by Claude, and the
    # parent shell session is the stable identifier.
    ppid = os.getppid()
    return os.path.join(INTERCOM_DIR, f"instance_{ppid}.id")


def load_my_id():
    """Load this instance's ID, or return None if not initialized."""
    id_file = get_id_file()
    if os.path.exists(id_file):
        with open(id_file) as f:
            return f.read().strip()
    # Also check all instance files — the PPID may change between calls,
    # so fall back to any .id file modified recently by this session
    for fn in os.listdir(INTERCOM_DIR):
        if fn.startswith("instance_") and fn.endswith(".id"):
            path = os.path.join(INTERCOM_DIR, fn)
            with open(path) as f:
                return f.read().strip()
    return None


def save_my_id(instance_id):
    """Save this instance's ID."""
    id_file = get_id_file()
    with open(id_file, 'w') as f:
        f.write(instance_id)


def get_cursor_file(instance_id):
    """Get the read-cursor file for an instance."""
    return os.path.join(INTERCOM_DIR, f"cursor_{instance_id}.pos")


def load_cursor(instance_id):
    """Load the read cursor (byte position in channel file)."""
    cursor_file = get_cursor_file(instance_id)
    if os.path.exists(cursor_file):
        with open(cursor_file) as f:
            return int(f.read().strip())
    return 0


def save_cursor(instance_id, pos):
    """Save the read cursor."""
    cursor_file = get_cursor_file(instance_id)
    with open(cursor_file, 'w') as f:
        f.write(str(pos))


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cmd_init():
    """Initialize this instance with a random ID."""
    existing = load_my_id()
    if existing:
        print(f"Already initialized as claude_{existing}")
        return

    instance_id = str(random.randint(1000000, 9999999))
    save_my_id(instance_id)

    # Announce on channel
    msg = f"[{timestamp()}] claude_{instance_id}: ** joined the channel **\n"
    with open(CHANNEL_FILE, 'a') as f:
        f.write(msg)

    # Set cursor to end of file (don't show old messages)
    if os.path.exists(CHANNEL_FILE):
        save_cursor(instance_id, os.path.getsize(CHANNEL_FILE))

    print(f"Initialized as claude_{instance_id}")
    print(f"Use 'python intercom/intercom.py send \"message\"' to send.")
    print(f"Use 'python intercom/intercom.py read' to check for messages.")


def cmd_send(message):
    """Send a message to the channel."""
    instance_id = load_my_id()
    if not instance_id:
        print("Error: not initialized. Run 'python intercom/intercom.py init' first.")
        sys.exit(1)

    msg = f"[{timestamp()}] claude_{instance_id}: {message}\n"
    with open(CHANNEL_FILE, 'a') as f:
        f.write(msg)
    print(f"Sent as claude_{instance_id}")


def cmd_read(peek_only=False):
    """Read new messages from others."""
    instance_id = load_my_id()
    if not instance_id:
        print("Error: not initialized. Run 'python intercom/intercom.py init' first.")
        sys.exit(1)

    if not os.path.exists(CHANNEL_FILE):
        print("No messages yet.")
        return

    cursor = load_cursor(instance_id)
    file_size = os.path.getsize(CHANNEL_FILE)

    if cursor >= file_size:
        print("No new messages.")
        return

    with open(CHANNEL_FILE) as f:
        f.seek(cursor)
        new_content = f.read()
        new_pos = f.tell()

    # Filter out own messages
    my_tag = f"claude_{instance_id}:"
    lines = new_content.strip().split('\n')
    other_messages = [l for l in lines if my_tag not in l]

    if other_messages:
        print("--- New messages ---")
        for line in other_messages:
            print(line)
        print("---")
    else:
        print("No new messages from others.")

    # Advance cursor (unless peek)
    if not peek_only:
        save_cursor(instance_id, new_pos)


def cmd_history():
    """Show full channel history."""
    if not os.path.exists(CHANNEL_FILE):
        print("No messages yet.")
        return
    with open(CHANNEL_FILE) as f:
        print(f.read())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "init":
        cmd_init()
    elif cmd == "send":
        if len(sys.argv) < 3:
            print("Usage: python intercom/intercom.py send \"message\"")
            sys.exit(1)
        cmd_send(" ".join(sys.argv[2:]))
    elif cmd == "read":
        cmd_read()
    elif cmd == "peek":
        cmd_read(peek_only=True)
    elif cmd == "history":
        cmd_history()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
