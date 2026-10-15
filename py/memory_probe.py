"""A6000 memory/throughput probe (coupler-queue 0008).

One config per process:  python py/memory_probe.py one --size S125 --mode bf16 --mb 8 [--vocab 8192]
Full sweep (each config in a fresh subprocess):  python py/memory_probe.py sweep
Attention backend check:  python py/memory_probe.py attn

Mirrors train.py: GPT/GPTConfig from model.py (bias=True, tied weights, dropout 0),
configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9,0.95), allow_fused=False)
(= --no_fused), autocast(cuda, bf16) for bf16 modes (no GradScaler: train.py only
uses one for fp16), plain fp32 (no TF32 enabled, as in train.py) for fp32. Loss comes
from model.forward. Copied from train.py (inline there): the autocast ctx setup.
"""
import argparse, contextlib, json, subprocess, sys, time, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

SIZES = {'S125': (12, 12, 768), 'M350': (24, 16, 1024), 'L760': (36, 20, 1280)}
MODES = ['fp32', 'bf16', 'bf16+fp32attn']
MBS = [4, 8, 16, 32]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'doc', '0008_memory_probe.jsonl')
GiB = 2 ** 30


def run_one(size, mode, mb, vocab, block=1024):
    nl, nh, ne = SIZES[size]
    rec = dict(size=size, mode=mode, mb=mb, vocab=vocab, block=block)
    try:
        cfg = GPTConfig(n_layer=nl, n_head=nh, n_embd=ne, block_size=block, vocab_size=vocab,
                        bias=True, dropout=0.0, tie_weights=True, fp32_attention=(mode == 'bf16+fp32attn'))
        model = GPT(cfg).to('cuda')
        opt = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95),
                                         device_type='cuda', allow_fused=False)
        ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if mode != 'fp32' else contextlib.nullcontext())
        nparams = sum(p.numel() for p in model.parameters())
        rec['params'] = nparams
        rec['params_nonembed'] = nparams - model.transformer.wpe.weight.numel() - model.transformer.wte.weight.numel()
        X = torch.randint(0, vocab, (mb, block), device='cuda')
        Y = torch.randint(0, vocab, (mb, block), device='cuda')

        def step():
            with ctx:
                _, loss = model(X, Y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(10):
            step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        tps = 10 * mb * block / dt
        rec.update(status='ok', peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GiB, 2),
                   peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2),
                   tok_per_s=round(tps), tflops=round(6 * nparams * tps / 1e12, 1))
    except torch.cuda.OutOfMemoryError:
        rec.update(status='OOM')
    return rec


def emit(rec):
    line = json.dumps(rec)
    print(line, flush=True)
    with open(OUT, 'a') as f:
        f.write(line + '\n')


def attn_check():
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))
    emit(dict(kind='versions', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    B, H, T, D = 8, 16, 1024, 64
    for dt in (torch.float32, torch.bfloat16):
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            rec = dict(kind='attn', dtype=str(dt), backend=be.name)
            try:
                q, k, v = [torch.randn(B, H, T, D, device='cuda', dtype=dt, requires_grad=True) for _ in range(3)]
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                with sdpa_kernel(be):
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    y.sum().backward()
                torch.cuda.synchronize()
                rec.update(runs=True, peak_extra_mib=round((torch.cuda.max_memory_allocated() - base) / 2**20, 1))
            except Exception as e:
                rec.update(runs=False, error=str(e).strip().splitlines()[0][:160])
            emit(rec)
    # unrestricted dispatch: which kernels does PyTorch actually launch?
    for dt in (torch.float32, torch.bfloat16):
        q, k, v = [torch.randn(B, H, T, D, device='cuda', dtype=dt, requires_grad=True) for _ in range(3)]
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
            torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages()
                        if any(s in e.key.lower() for s in ('attn', 'flash', 'efficient', 'softmax'))})
        emit(dict(kind='attn_default', dtype=str(dt),
                  peak_extra_mib=round((torch.cuda.max_memory_allocated() - base) / 2**20, 1),
                  kernels=[n[:90] for n in names][:8]))


def sweep():
    py = sys.executable

    def go(size, mode, mb, vocab):
        r = subprocess.run([py, os.path.abspath(__file__), 'one', '--size', size, '--mode', mode,
                            '--mb', str(mb), '--vocab', str(vocab)], capture_output=True, text=True)
        if r.returncode != 0:
            emit(dict(size=size, mode=mode, mb=mb, vocab=vocab, status='ERROR', stderr=r.stderr[-300:]))
            return True
        return '"status": "OOM"' in r.stdout
    for size in SIZES:
        for mode in MODES:
            for mb in MBS:
                if go(size, mode, mb, 8192):
                    break
    for mb in MBS:
        if go('M350', 'bf16', mb, 32768):
            break
    subprocess.run([py, os.path.abspath(__file__), 'attn'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['one', 'sweep', 'attn'])
    ap.add_argument('--size'); ap.add_argument('--mode'); ap.add_argument('--mb', type=int)
    ap.add_argument('--vocab', type=int, default=8192)
    a = ap.parse_args()
    if a.cmd == 'one':
        emit(run_one(a.size, a.mode, a.mb, a.vocab))
    elif a.cmd == 'attn':
        attn_check()
    else:
        sweep()
