"""
Probe-batch instrumentation for the char-vs-BPE training-dynamics study (coupler-queue 0012).

ProbeRunner tokenizes the frozen probe batch (doc/0012_probe_batch.json: 64 passages x 1024 chars)
with the model's own tokenizer and, on request, runs it through a model under torch.no_grad()
(explicit fp32 attention via CausalSelfAttention.attn_probe) and returns:

  * per layer / head / passage kind: mean attention entropy and mean effective number of attended
    positions (mean of e^H per query row; rows 1..n-2 of the input, row 0 is trivially 0);
  * loss per passage kind in bits per character. v2 probe batch (coupler-queue 0013): both tokenizers
    are scored on the IDENTICAL character span of each passage, given the identical preceding text
    (spans from the sidecar <probe>_spans.json: the characters after BPE token 0 up to the last BPE
    token predicted inside the BPE window and inside the char window). bits/char = bits over exactly
    those characters / span length. A batch without a sidecar (v1) falls back to the 0012 behaviour;
  * induction score per layer/head on repetition passages: mean attention from every second-copy
    input position j (token j equals token j-k) to key position j-k+1;
  * repetition improvement: bits/char of the second copy vs the first copy (each over 512 chars).

No random numbers are drawn, so the training RNG state is untouched.
"""
import json
import math
import os

import torch
import torch.nn.functional as F

KINDS = ('prose', 'morphology', 'short_sentence', 'repetition')


class ProbeRunner:
    def __init__(self, probe_path, tokenizer, device, max_input=None, chunk=8):
        with open(probe_path, encoding='utf-8') as f:
            data = json.load(f)
        self.passages = data['passages']
        self.device = device
        self.chunk = chunk
        self.kinds = [p['kind'] for p in self.passages]
        self.n_chars = []    # characters in the scored span
        self.n_truncated = 0
        self.ids = []        # list of 1-D long tensors (input + target tokens, length n_in + 1)
        self.lo = []         # first input position whose target is scored
        self.split = []      # input position that predicts the first element of the second copy
        self.rep_k = []      # token index where the second copy starts (repetition passages) else None
        span_path = probe_path[:-len('.json')] + '_spans.json' if probe_path.endswith('.json') else None
        spans = None
        if span_path and os.path.exists(span_path):
            with open(span_path) as f:
                sp = json.load(f)
            spans = sp['spans']
            ctxs = sp['meta']
            want = ctxs['bpe_ctx'] if tokenizer.tokenizer_type == 'bpe' else ctxs['char_ctx']
            assert max_input is None or max_input == want, \
                f"spans were built for context {want}, run uses block_size {max_input}"
            assert len(spans) == len(self.passages)
        self.spans = spans
        for i, p in enumerate(self.passages):
            text = p['text']
            if tokenizer.tokenizer_type == 'bpe':
                enc = tokenizer.tokenizer.encode(text)
                ids, offs = enc.ids, enc.offsets
                k = None
                if p['kind'] == 'repetition':
                    ro = p['repeat_offset']
                    k = next(j for j, (a, b) in enumerate(offs) if a >= ro)
                    assert len(ids) == 2 * k and ids[:k] == ids[k:], "repetition passage not token-aligned"
            else:
                ids = tokenizer.encode(text)
                k = p['repeat_offset'] if p['kind'] == 'repetition' else None
            ro = p.get('repeat_offset')
            if spans is not None:
                s0, e0 = spans[i]['start'], spans[i]['end']
                if tokenizer.tokenizer_type == 'bpe':
                    m = spans[i]['bpe_pred_tokens']
                    assert offs[1][0] == s0 and offs[m][1] == e0, "span does not match BPE token offsets"
                    ids = ids[:m + 1]
                    lo, split = 0, (k - 1 if k is not None else None)
                else:
                    ids = ids[:e0]               # inputs chars 0..e0-2 predict chars 1..e0-1
                    lo, split = s0 - 1, (ro - 1 if ro is not None else None)
                covered = e0 - s0
            else:                                # v1 behaviour
                covered = len(text)
                if max_input is not None and len(ids) > max_input + 1:
                    ids = ids[:max_input + 1]
                    covered = offs[max_input][1]
                    self.n_truncated += 1
                lo, split = 0, (k - 1 if k is not None else None)
            self.n_chars.append(covered)
            self.ids.append(torch.tensor(ids, dtype=torch.long))
            self.lo.append(lo)
            self.split.append(split)
            self.rep_k.append(k)
        self.n_layer = None
        if self.n_truncated:
            print(f"ProbeRunner: {self.n_truncated} passages truncated to {max_input} input tokens")
        if spans is not None:
            print(f"ProbeRunner: shared scored spans from {span_path} "
                  f"(span chars min/mean/max {min(self.n_chars)}/{sum(self.n_chars)/len(self.n_chars):.1f}/{max(self.n_chars)})")

    @torch.no_grad()
    def run(self, model, ctx):
        was_training = model.training
        model.eval()
        layers = list(model.transformer.h)
        L = len(layers)
        H = model.config.n_head
        kinds = KINDS
        ent_sum = {k: torch.zeros(L, H, dtype=torch.float64) for k in kinds}
        eff_sum = {k: torch.zeros(L, H, dtype=torch.float64) for k in kinds}
        row_cnt = {k: 0 for k in kinds}
        ind_sum = torch.zeros(L, H, dtype=torch.float64)
        ind_cnt = 0
        bits = {k: 0.0 for k in kinds}
        chars = {k: 0 for k in kinds}
        rep_bits_first = rep_bits_second = 0.0
        rep_chars_first = rep_chars_second = 0

        N = len(self.passages)
        for c0 in range(0, N, self.chunk):
            idxs = list(range(c0, min(N, c0 + self.chunk)))
            lens = [len(self.ids[i]) - 1 for i in idxs]            # input lengths
            T = max(lens)
            x = torch.zeros(len(idxs), T, dtype=torch.long)
            y = torch.full((len(idxs), T), -1, dtype=torch.long)
            for b, i in enumerate(idxs):
                n = lens[b]
                x[b, :n] = self.ids[i][:-1]
                y[b, :n] = self.ids[i][1:]
            x, y = x.to(self.device), y.to(self.device)
            valid = (y >= 0)

            # row validity for entropy: input rows 1..n-2 (row 0 trivial; last row n-1 is fine too,
            # but we keep rows 1..n-1 inclusive of last predicting row)
            row_ok = torch.zeros(len(idxs), T, dtype=torch.bool, device=self.device)
            for b in range(len(idxs)):
                row_ok[b, max(1, self.lo[idxs[b]]):lens[b]] = True
            kind_of = [self.kinds[i] for i in idxs]
            kind_masks = {k: torch.tensor([kk == k for kk in kind_of], device=self.device) for k in kinds}

            # induction targets
            rep_b = [b for b, i in enumerate(idxs) if self.kinds[i] == 'repetition']
            ind_rows = {}   # b -> (j tensor, key tensor)
            for b in rep_b:
                k = self.rep_k[idxs[b]]
                n = lens[b]
                j = torch.arange(k, n, device=self.device)         # second-copy input positions
                ind_rows[b] = (j, j - k + 1)

            def make_hook(layer_i):
                def hook(att):                                      # (B, H, T, T) fp32
                    logp = torch.log(att.clamp_min(1e-30))
                    ent = -(att * logp).sum(-1)                     # (B, H, T)
                    eff = torch.exp(ent)
                    for kname in kinds:
                        m = kind_masks[kname]
                        if not m.any():
                            continue
                        r = row_ok[m]                               # (b, T)
                        e = ent[m]                                  # (b, H, T)
                        f = eff[m]
                        ent_sum[kname][layer_i] += (e * r[:, None, :]).sum((0, 2)).double().cpu()
                        eff_sum[kname][layer_i] += (f * r[:, None, :]).sum((0, 2)).double().cpu()
                    for b, (j, key) in ind_rows.items():
                        ind_sum[layer_i] += att[b][:, j, key].sum(-1).double().cpu()
                return hook

            for li, blk in enumerate(layers):
                blk.attn.attn_probe = make_hook(li)
            try:
                with ctx:
                    logits, _ = model(x, y)
            finally:
                for blk in layers:
                    blk.attn.attn_probe = None
            for kname in kinds:
                m = kind_masks[kname]
                if m.any():
                    row_cnt[kname] += int((row_ok[m]).sum().item())
            ind_cnt += sum(len(ind_rows[b][0]) for b in ind_rows)

            nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                  ignore_index=-1, reduction='none').view(len(idxs), T)
            nll_bits = nll / math.log(2.0)
            for b, i in enumerate(idxs):
                kname = self.kinds[i]
                lo = self.lo[i]
                bits[kname] += float(nll_bits[b, lo:lens[b]].sum())
                chars[kname] += self.n_chars[i]
                if kname == 'repetition':
                    sp = self.split[i]
                    ro = self.passages[i]['repeat_offset']
                    # target m (m>=1) is predicted at input m-1; first copy = inputs lo..sp-1
                    rep_bits_first += float(nll_bits[b, lo:sp].sum())
                    rep_bits_second += float(nll_bits[b, sp:lens[b]].sum())
                    if self.spans is not None:
                        rep_chars_first += ro - self.spans[i]['start']
                        rep_chars_second += self.spans[i]['end'] - ro
                    else:
                        rep_chars_first += ro
                        rep_chars_second += self.n_chars[i] - ro

        out = dict(
            attn_entropy={k: (ent_sum[k] / max(row_cnt[k], 1)).tolist() for k in kinds},
            attn_eff_n={k: (eff_sum[k] / max(row_cnt[k], 1)).tolist() for k in kinds},
            bpc={k: bits[k] / chars[k] for k in kinds},
            induction=(ind_sum / max(ind_cnt, 1)).tolist(),
            rep_first_bpc=rep_bits_first / rep_chars_first,
            rep_second_bpc=rep_bits_second / rep_chars_second,
        )
        out['rep_improvement_bpc'] = out['rep_first_bpc'] - out['rep_second_bpc']
        if was_training:
            model.train()
        return out
