"""Memory/throughput probe (coupler-queue 0008 CUDA, 0009 MPS).

One config per process:  python py/memory_probe.py one --size S125 --mode bf16 --mb 8 [--vocab 8192] [--device cuda|mps]
Full sweep (each config in a fresh subprocess):  python py/memory_probe.py sweep [--device cuda|mps]
Attention backend check (CUDA only):  python py/memory_probe.py attn

Mirrors train.py: GPT/GPTConfig from model.py (bias=True, tied weights, dropout 0),
configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9,0.95), allow_fused=False)
(= --no_fused), autocast(<device>, bf16) for bf16 modes (no GradScaler: train.py only
uses one for fp16), plain fp32 (no TF32 enabled, as in train.py) for fp32. Loss comes
from model.forward. Copied from train.py (inline there): the autocast ctx setup.

--device cuda is the original 0008 path and is byte-for-byte unchanged in what it
measures and prints (writes to doc/0008_memory_probe.jsonl).

--device mps (0009) mirrors the SAME train.py autocast call on Apple-Silicon MPS and
writes to doc/0009_mps_probe.jsonl. It adds, per run: whether autocast actually engaged
(torch 2.5.1 has an MPS-autocast bug — see 0009.result.md), whether the loss stayed
finite on every step, and MPS memory sampled with current/driver_allocated_memory
(MPS has no peak-stats API, so these are post-step samples, not transient peaks).
Mode 'bf16native' is an MPS-only DIAGNOSTIC — model+inputs cast to bf16, no autocast —
to measure the bf16 compute ceiling; it is NOT what train.py does.
"""
import argparse, contextlib, json, subprocess, sys, time, os, platform
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

SIZES = {'S125': (12, 12, 768), 'M350': (24, 16, 1024), 'L760': (36, 20, 1280)}
MODES = ['fp32', 'bf16', 'bf16+fp32attn']
MPS_MODES = ['fp32', 'bf16', 'bf16+fp32attn', 'bf16native']
MPS_GRID = {'S125': [4, 8, 16, 32], 'M350': [4, 8, 16, 32], 'L760': [4, 8]}
MBS = [4, 8, 16, 32]
DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'doc')
OUT_CUDA = os.path.join(DOC, '0008_memory_probe.jsonl')
OUT_MPS = os.path.join(DOC, '0009_mps_probe.jsonl')
OUT = OUT_CUDA  # reset in __main__ from --device
GiB = 2 ** 30


def run_one(size, mode, mb, vocab, block=1024, device='cuda'):
    nl, nh, ne = SIZES[size]
    rec = dict(size=size, mode=mode, mb=mb, vocab=vocab, block=block)
    native_bf16 = (mode == 'bf16native')             # MPS diagnostic only; NOT what train.py does
    autocast_mode = mode in ('bf16', 'bf16+fp32attn')  # the modes where train.py enters autocast
    try:
        cfg = GPTConfig(n_layer=nl, n_head=nh, n_embd=ne, block_size=block, vocab_size=vocab,
                        bias=True, dropout=0.0, tie_weights=True, fp32_attention=(mode == 'bf16+fp32attn'))
        model = GPT(cfg).to(device)
        if native_bf16:
            model = model.to(torch.bfloat16)         # real bf16 compute (no autocast) — the ceiling
        opt = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95),
                                         device_type=device, allow_fused=False)
        ctx = (torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
               if autocast_mode else contextlib.nullcontext())
        nparams = sum(p.numel() for p in model.parameters())
        rec['params'] = nparams
        rec['params_nonembed'] = nparams - model.transformer.wpe.weight.numel() - model.transformer.wte.weight.numel()
        X = torch.randint(0, vocab, (mb, block), device=device)
        Y = torch.randint(0, vocab, (mb, block), device=device)

        def step():
            with ctx:
                _, loss = model(X, Y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss

        if device == 'cuda':
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
        else:  # mps
            rec['device'] = 'mps'
            for _ in range(2):
                step()
            torch.mps.synchronize()
            with ctx:  # did autocast actually engage? (torch 2.5.1 MPS-autocast bug)
                rec['autocast_requested'] = autocast_mode
                rec['autocast_enabled'] = bool(torch.is_autocast_enabled('mps')) if autocast_mode else False
            max_cur = max_drv = 0
            finite = []
            t0 = time.time()
            for _ in range(10):
                loss = step()
                finite.append(torch.isfinite(loss.detach()))   # read after final sync; no per-step host sync
                c = torch.mps.current_allocated_memory()        # allocator reads; neither syncs the GPU
                d = torch.mps.driver_allocated_memory()
                max_cur = c if c > max_cur else max_cur
                max_drv = d if d > max_drv else max_drv
            torch.mps.synchronize()
            dt = time.time() - t0
            tps = 10 * mb * block / dt
            # MPS has no peak-memory API; peak_alloc/peak_reserved are post-step SAMPLE maxima.
            rec.update(status='ok', peak_alloc_gib=round(max_cur / GiB, 2),
                       peak_reserved_gib=round(max_drv / GiB, 2),
                       tok_per_s=round(tps), tflops=round(6 * nparams * tps / 1e12, 1),
                       loss_finite=bool(torch.stack(finite).all().item()))
    except torch.cuda.OutOfMemoryError:
        rec.update(status='OOM')
    except RuntimeError as e:
        if device == 'cuda':
            raise                                    # keep CUDA path byte-for-byte identical
        msg = str(e)
        if 'out of memory' in msg.lower() or "can't allocate" in msg.lower():
            rec.update(status='OOM')
        else:
            rec.update(status='ERROR', error=msg.strip().splitlines()[0][:200])
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


def sweep(device='cuda'):
    py = sys.executable
    if device == 'cuda':
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
    else:  # mps
        emit(dict(kind='versions', device='mps', torch=torch.__version__,
                  macos=platform.mac_ver()[0], mps_available=torch.backends.mps.is_available()))

        def go(size, mode, mb):
            try:
                r = subprocess.run([py, os.path.abspath(__file__), 'one', '--size', size, '--mode', mode,
                                    '--mb', str(mb), '--vocab', '8192', '--device', 'mps'],
                                   capture_output=True, text=True, timeout=15 * 60)
            except subprocess.TimeoutExpired:
                emit(dict(size=size, mode=mode, mb=mb, vocab=8192, device='mps', status='timeout',
                          note='exceeded 15 min; skipping larger micro-batches of this size/mode'))
                return True
            if r.returncode != 0:
                emit(dict(size=size, mode=mode, mb=mb, vocab=8192, device='mps', status='ERROR',
                          stderr=r.stderr[-300:]))
                return True
            return '"status": "OOM"' in r.stdout
        for size, mbs in MPS_GRID.items():
            for mode in MPS_MODES:
                for mb in mbs:
                    if go(size, mode, mb):
                        break


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['one', 'sweep', 'attn'])
    ap.add_argument('--size'); ap.add_argument('--mode'); ap.add_argument('--mb', type=int)
    ap.add_argument('--vocab', type=int, default=8192)
    ap.add_argument('--device', choices=['cuda', 'mps'], default='cuda')
    a = ap.parse_args()
    OUT = OUT_MPS if a.device == 'mps' else OUT_CUDA
    if a.cmd == 'one':
        emit(run_one(a.size, a.mode, a.mb, a.vocab, device=a.device))
    elif a.cmd == 'attn':
        attn_check()
    else:
        sweep(device=a.device)
