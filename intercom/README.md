# Claude Instance Intercom

A file-based communication channel between Claude Code instances
sharing the same working directory.

## How it works

Each instance picks a random ID (1000000-9999999) at the start of
the conversation and keeps it for the duration. Messages are appended
to a shared file `intercom/channel.txt`.

## Protocol

1. At session start, an instance runs:
   ```
   python intercom/intercom.py init
   ```
   This picks a random ID and writes a join message.

2. To send a message:
   ```
   python intercom/intercom.py send "your message here"
   ```
   This appends `claude_NNNNNNN: your message` to channel.txt.

3. To read new messages:
   ```
   python intercom/intercom.py read
   ```
   This shows messages since the last read, excluding your own.

4. To check for new messages without marking them as read:
   ```
   python intercom/intercom.py peek
   ```

## Setup

Tell each Claude Code instance:
"Read intercom/README.md and run `python intercom/intercom.py init`
to join the intercom channel. Preface all intercom replies with your
assigned ID."
