#!/usr/bin/env python
"""
word_mode_grid_eval.py  (coupler-queue 0007; extends word_mode_real_fraction.py)

For each arm pt/yawl_word_L<L>H4_e<N>_untied_cuda: load checkpoints at 5K/10K/15K/20K,
generate words at temperatures 0.8 and 0.5, and report real-word fraction overall and
by generated-length bucket, novel-real share (val split vs train split), empty and
over-length counts, val bpc and params.  Results -> JSON (one record per arm/iter/temp).

Usage: python py/word_mode_grid_eval.py --out doc/0007_grid_results.json
"""
import argparse, json, math, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from word_mode_real_fraction import load_model, generate_words
from tokenizer import load_tokenizer

ARMS = [(32, 2), (32, 4)] + [(e, l) for e in (64, 128, 256) for l in (1, 2, 4)]
ITERS = [5000, 10000, 15000, 20000]
TEMPS = [0.8, 0.5]
BUCKETS = [(1, 4), (5, 7), (8, 10), (11, 14)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wordlist', default='txt_local/yawl_word_list_width_26.txt')
    ap.add_argument('--out', default='doc/0007_grid_results.json')
    ap.add_argument('--n_words', type=int, default=1000)
    ap.add_argument('--max_letters', type=int, default=14)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yawl = [w.strip() for w in open(args.wordlist) if w.strip()]
    yawl_set = set(yawl)
    trained = [w for w in yawl if len(w) <= args.max_letters]
    mean_train_len = sum(map(len, trained)) / len(trained)

    records = []
    for e, l in ARMS:
        base = f'pt/yawl_word_L{l}H4_e{e}_untied_cuda'
        val_path = base + '_val_words.txt'
        if not os.path.exists(base + '_final.pt') or not os.path.exists(val_path):
            print('MISSING arm', base); continue
        val_set = set(w.strip() for w in open(val_path) if w.strip())
        tok = load_tokenizer(base + '_meta.pkl')
        for it in ITERS:
            path = f'{base}_final.pt' if it == 20000 else f'{base}_iter{it}.pt'
            model, ck = load_model(path, device)
            n_params = sum(p.numel() for p in model.parameters())
            bv = ck.get("best_val_loss"); bv = float(bv) if bv is not None else None
            for T in TEMPS:
                words = generate_words(model, tok, device, args.n_words, 18, T, 40, 42)
                # generate_words uses max_new_tokens = max_letters+2 -> 20 for max_letters=18
                n = len(words)
                empty = sum(w == '' for w in words)
                over = sum(len(w) > args.max_letters for w in words)
                real = [w for w in words if w in yawl_set and 0 < len(w) <= args.max_letters]
                bk = {}
                for lo, hi in BUCKETS:
                    inb = [w for w in words if lo <= len(w) <= hi]
                    bk[f'{lo}-{hi}'] = {'n': len(inb), 'real': sum(w in yawl_set for w in inb)}
                lens = [len(w) for w in words if w]
                rec = dict(arm=f'e{e}_L{l}', n_embd=e, n_layer=l, iter=it, temp=T,
                           n_params=n_params, best_val_bpc=(bv / math.log(2)) if bv else None,
                           ckpt_iter=ck.get('iter_num'),
                           n=n, real=len(real), empty=empty, overlong=over,
                           real_val=sum(w in val_set for w in real),
                           real_train=sum(w not in val_set for w in real),
                           buckets=bk, mean_gen_len=sum(lens) / max(1, len(lens)),
                           mean_train_len=mean_train_len, samples=words[:20])
                records.append(rec)
                print(f"{rec['arm']} it{it} T{T}: real {100*len(real)/n:.1f}% "
                      f"novel(val) {rec['real_val']} train {rec['real_train']} empty {empty} long {over}", flush=True)
            del model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, 'w'), indent=1)
    print('wrote', args.out)

if __name__ == '__main__':
    main()
