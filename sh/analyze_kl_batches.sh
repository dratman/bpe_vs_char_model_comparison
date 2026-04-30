#!/bin/zsh
#
# analyze_kl_batches.sh - Run the KL imitator on many batches and
# separate into high-match and low-match groups. Show the actual text.
#

LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/analyze_kl_batches_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"

echo "Output will be saved to: $LOG_FILE"
echo ""

python3 -u -c "
import sys; sys.path.insert(0, 'py')
import torch
import torch.nn.functional as F
import numpy as np
from imitator_model import Imitator, ImitatorConfig
from small_model_split import load_small_model, forward_to_layer, forward_from_layer
from tokenizer import load_tokenizer

device = 'mps'
torch.manual_seed(123)

# Load everything
small_model, small_args = load_small_model('pt/bpe_16L16H_iter170000.pt', device=device, dtype=torch.float32)

print('Loading imitator...')
ckpt = torch.load('pt/imitator_L8_kl.pt', map_location=device, weights_only=False)
config = ImitatorConfig(**ckpt['imitator_config'])
imitator = Imitator(config).to(device)
imitator.load_state_dict(ckpt['imitator'])
imitator.eval()
split_layer = ckpt['split_layer']
print(f'Imitator loaded: split_layer={split_layer}, iter={ckpt.get(\"iter_num\", \"?\")}')

tokenizer = load_tokenizer('pt/bpe_16L16H_meta.pkl')

# Load tokens
all_tokens = torch.load('txt_local/corpus_tokens_32k.pt', map_location='cpu', weights_only=True)
n_val = int(len(all_tokens) * 0.1)
val_tokens = all_tokens[-n_val:]
print(f'Val tokens: {len(val_tokens):,}')

block_size = config.block_size
imitator_dtype = next(imitator.parameters()).dtype

# Run many single-window evaluations
num_windows = 100
results = []

print(f'Evaluating {num_windows} windows...')
with torch.no_grad():
    for i in range(num_windows):
        start = torch.randint(len(val_tokens) - block_size, (1,)).item()
        idx = val_tokens[start:start + block_size].unsqueeze(0).to(device)

        # Get real activations
        acts = forward_to_layer(small_model, idx, split_layer)
        inp = acts[:, :-1, :].to(imitator_dtype)
        tgt = acts[:, 1:, :]

        # Imitator prediction
        pred = imitator(inp)

        # Decode both through back half
        real_logits = forward_from_layer(small_model, tgt, split_layer)
        pred_logits = forward_from_layer(small_model, pred, split_layer)

        real_top1 = real_logits.argmax(dim=-1)[0]
        pred_top1 = pred_logits.argmax(dim=-1)[0]

        match_rate = (real_top1 == pred_top1).float().mean().item()

        # Cosine similarity
        cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1).mean().item()

        # Decode the text
        text = tokenizer.decode(idx[0].tolist())

        results.append({
            'match': match_rate,
            'cos': cos,
            'text': text[:300],
            'idx': i,
        })

        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{num_windows}')

# Sort by match rate
results.sort(key=lambda r: r['match'])

# Distribution
matches = [r['match'] for r in results]
print()
print('='*70)
print(f'Match rate distribution over {num_windows} windows:')
print(f'  Min:    {min(matches):.3f}')
print(f'  25th:   {np.percentile(matches, 25):.3f}')
print(f'  Median: {np.median(matches):.3f}')
print(f'  75th:   {np.percentile(matches, 75):.3f}')
print(f'  Max:    {max(matches):.3f}')
print(f'  Mean:   {np.mean(matches):.3f}')
print()

# Count by range
low = sum(1 for m in matches if m < 0.35)
mid = sum(1 for m in matches if 0.35 <= m < 0.65)
high = sum(1 for m in matches if m >= 0.65)
print(f'  Low (<35%):   {low} windows')
print(f'  Mid (35-65%): {mid} windows')
print(f'  High (>65%):  {high} windows')

# Show worst 10
print()
print('='*70)
print('WORST 10 windows (lowest match rate):')
print('='*70)
for r in results[:10]:
    print(f'  match={r[\"match\"]:.3f}  cos={r[\"cos\"]:.3f}')
    # Show first 200 chars, replace newlines
    display = r['text'][:200].replace(chr(10), ' ')
    print(f'  text: {display}')
    print()

# Show best 10
print('='*70)
print('BEST 10 windows (highest match rate):')
print('='*70)
for r in results[-10:]:
    print(f'  match={r[\"match\"]:.3f}  cos={r[\"cos\"]:.3f}')
    display = r['text'][:200].replace(chr(10), ' ')
    print(f'  text: {display}')
    print()
" 2>&1 > "$LOG_FILE"

echo "Done. Results in: $LOG_FILE"
