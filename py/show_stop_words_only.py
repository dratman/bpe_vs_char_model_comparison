import re
import sys

# Load stop words
with open("stop_words.txt") as f:
    stop_words = set(line.strip() for line in f)

text = sys.stdin.read()

# Replace each word: keep if it's a stop word, otherwise replace with _
result = re.sub(r"[a-zA-Z]+", lambda m: m.group().lower() if m.group().lower() in stop_words else "_", text)

print(result)
