#!/bin/zsh
#
# sample_imitator_L10_compare.sh - Compare and rollout for the layer-10 imitator
#

LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/sample_imitator_L10_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"

echo "Output will be saved to: $LOG_FILE"
echo "You can watch with: tail -f $LOG_FILE"
echo ""

python3 -u -c "
import sys; sys.path.insert(0, 'py')
import torch

# Run compare mode
print('=== COMPARE MODE ===')
print()

from sample_imitator import *
import numpy as np

device = 'mps'
torch.manual_seed(42)
np.random.seed(42)

small_model, small_args = load_small_model('pt/bpe_16L16H_iter170000.pt', device=device, dtype=torch.bfloat16)
imitator, imit_config, split_layer = load_imitator('pt/imitator_L10_full.pt', device)

model_base = 'pt/bpe_16L16H_iter170000'.replace('.pt','')
if '_iter' in model_base:
    model_base = model_base.rsplit('_iter', 1)[0]
meta_path = model_base + '_meta.pkl'
tokenizer = load_tokenizer(meta_path)
print(f'Tokenizer: {tokenizer.tokenizer_type}, vocab={tokenizer.vocab_size}')

all_tokens = load_corpus_tokens('txt_local/corpus_books_shuffled_2026_04_18.txt', tokenizer, 'txt_local/corpus_tokens_32k.pt')
n_val = int(len(all_tokens) * 0.1)
val_tokens = all_tokens[-n_val:]
print(f'Using {len(val_tokens):,} validation tokens')

block_size = imit_config.block_size

compare_predictions(imitator, small_model, tokenizer, val_tokens,
                    split_layer, block_size, device, num_samples=3)

print()
print()
print('=== ROLLOUT MODE ===')
print()

rollout(imitator, small_model, tokenizer, val_tokens,
        split_layer, block_size, device,
        prefix_tokens=64, rollout_tokens=200, num_samples=3)
" 2>&1 > "$LOG_FILE"

echo "Done. Results in: $LOG_FILE"
