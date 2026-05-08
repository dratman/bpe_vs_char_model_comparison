import re
from collections import Counter
import sys

how_many = 1000000  # Limit for most common words to display

def unique_words_and_counts(chunk_size=1024*1024):  # 1MB chunks
    word_counts = Counter()
    total_words = 0

    while chunk := sys.stdin.read(chunk_size):
        # Convert chunk to lowercase
        chunk = chunk.lower()

        # Find words containing only alphabetic characters
        words = re.findall(r'\b[a-z]+\b', chunk)

        # Update word counts and total word count
        word_counts.update(words)
        total_words += len(words)

    return word_counts, total_words

def print_word_counts(word_counts, total_words, limit=None):
    print("Word counts (sorted by frequency):")
    cumulative_percent = 0.0
    print(f"{'Word':<20} {'Count':<10} {'% of Total':<15} {'Cumulative %':<15}")
    print("-" * 60)

    for word, count in word_counts.most_common(limit):
        percent = (count / total_words) # * 100 (let spreadsheet multiply by 100 for percent)
        cumulative_percent += percent
        print(f"{word:<20} {count:<10} {percent:<15.6f} {cumulative_percent:<15.6f}")

# Get word counts and total word count from the input
word_counts, total_words = unique_words_and_counts()

# Print the word counts with percentages
print_word_counts(word_counts, total_words, limit=how_many)
