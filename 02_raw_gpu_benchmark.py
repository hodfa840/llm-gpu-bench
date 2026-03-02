#!/usr/bin/env python3
"""
Raw GPU Benchmark — Compute throughput & memory bandwidth.
Measures:
  • FP32 / FP16 / BF16 / TF32 matrix multiply TFLOPS
  • INT8 TOPS
  • Memory bandwidth (GB/s)
  • VRAM usage (allocated / peak / total)
  • System RAM usage
  • CPU utilisation during benchmark
"""

import torch
import time
import json
import os
import threading
import psutil
import sys
from pathlib import Path

R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"; BOLD="\033[1m"; RST="\033[0m"

device = torch.device("cuda:0")
proc   = psutil.Process(os.getpid())

def hr(w=72): print(f"{DIM}{'─'*w}{RST}")


# ─── System snapshot ──────────────────────────────────────────────────────────

def sys_snapshot(label=""):
    ram    = psutil.virtual_memory()
    swap   = psutil.swap_memory()
    cpu    = psutil.cpu_percent(interval=0.1)
    vram_a = torch.cuda.memory_allocated(device)  / 1e9
    vram_r = torch.cuda.memory_reserved(device)   / 1e9
    vram_p = torch.cuda.max_memory_allocated(device) / 1e9
    return {
        "label"           : label,
        "cpu_pct"         : cpu,
        "ram_used_gb"     : round(ram.used   / 1e9, 2),
        "ram_avail_gb"    : round(ram.available / 1e9, 2),
        "ram_total_gb"    : round(ram.total  / 1e9, 2),
        "ram_pct"         : ram.percent,
        "swap_used_gb"    : round(swap.used  / 1e9, 2),
        "vram_alloc_gb"   : round(vram_a, 2),
        "vram_reserved_gb": round(vram_r, 2),
        "vram_peak_gb"    : round(vram_p, 2),
        "vram_total_gb"   : round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }

def print_snapshot(snap):
    vt = snap["vram_total_gb"]
    va = snap["vram_alloc_gb"]
    vp = snap["vram_peak_gb"]
    rt = snap["ram_total_gb"]
    ru = snap["ram_used_gb"]

    vram_bar_len = int((va / vt) * 30) if vt else 0
    ram_bar_len  = int((ru / rt) * 30) if rt else 0

    print(f"\n  {BOLD}{Y}System Resources{RST}")
    hr()
    print(f"  {W}CPU usage        {RST}: {C}{snap['cpu_pct']:>5.1f}%{RST}")
    print(f"  {W}RAM used/total   {RST}: {C}{ru:.2f} GB / {rt:.2f} GB  ({snap['ram_pct']:.1f}%){RST}  "
          f"{G}{'▓' * ram_bar_len}{'░' * (30 - ram_bar_len)}{RST}")
    print(f"  {W}Swap used        {RST}: {C}{snap['swap_used_gb']:.2f} GB{RST}")
    print(f"  {W}VRAM allocated   {RST}: {C}{va:.2f} GB / {vt:.2f} GB{RST}  "
          f"{G if va/vt < 0.8 else Y}{'▓' * vram_bar_len}{'░' * (30 - vram_bar_len)}{RST}")
    print(f"  {W}VRAM peak        {RST}: {C}{vp:.2f} GB{RST}")
    print(f"  {W}VRAM reserved    {RST}: {C}{snap['vram_reserved_gb']:.2f} GB{RST}")


# ─── CPU usage sampler (background thread) ────────────────────────────────────

class CPUSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples  = []
        self._stop    = threading.Event()
        self._t       = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    @property
    def avg(self): return sum(self.samples) / len(self.samples) if self.samples else 0
    @property
    def peak(self): return max(self.samples) if self.samples else 0


# ─── Benchmarks ───────────────────────────────────────────────────────────────

def warmup_and_sync(fn, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()

def timed(fn, iters=20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_matmul(dtype, N=8192):
    a = torch.randn(N, N, dtype=dtype, device=device)
    b = torch.randn(N, N, dtype=dtype, device=device)
    fn = lambda: torch.matmul(a, b)
    warmup_and_sync(fn)

    torch.cuda.reset_peak_memory_stats(device)
    sampler = CPUSampler().start()
    t = timed(fn, iters=30)
    sampler.stop()

    flops  = 2 * N**3
    tflops = flops / t / 1e12
    return tflops, t * 1e3, sampler.avg, sampler.peak


def bench_int8(N=8192):
    a = torch.randint(-128, 127, (N, N), dtype=torch.int8, device=device)
    b = torch.randint(-128, 127, (N, N), dtype=torch.int8, device=device)
    fn = lambda: torch._int_mm(a, b)
    warmup_and_sync(fn)
    sampler = CPUSampler().start()
    t = timed(fn, iters=30)
    sampler.stop()
    tops = 2 * N**3 / t / 1e12
    return tops, sampler.avg


def bench_bandwidth(size_gb=2.0):
    n_elements = int(size_gb * 1e9 / 4)
    x = torch.zeros(n_elements, dtype=torch.float32, device=device)
    y = torch.zeros_like(x)
    fn = lambda: y.copy_(x)
    warmup_and_sync(fn)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = CPUSampler().start()
    t = timed(fn, iters=50)
    sampler.stop()
    gb_per_s = size_gb * 2 / t
    return gb_per_s, sampler.avg


def bench_training_step(batch=64, seq=512, hidden=4096):
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden),
    ).to(device).to(torch.float16)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, seq, hidden, dtype=torch.float16, device=device)

    from torch.amp import autocast
    def step():
        optim.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            out  = model(x)
            loss = out.mean()
        loss.backward()
        optim.step()

    warmup_and_sync(step, warmup=2)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = CPUSampler().start()
    t = timed(step, iters=10)
    sampler.stop()

    vram_peak = torch.cuda.max_memory_allocated(device) / 1e9
    tokens_per_s = (batch * seq) / t
    return tokens_per_s, t * 1e3, vram_peak, sampler.avg


def main():
    results = {}

    print(f"\n{BOLD}{M}{'═'*72}")
    print(f"  ⚡  RAW GPU COMPUTE BENCHMARK")
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'═'*72}{RST}")

    # ── Baseline system snapshot ────────────────────────────────────────────────
    snap_baseline = sys_snapshot("baseline")
    print_snapshot(snap_baseline)

    # ── FLOPS ──────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{Y}  [1] Matrix Multiply Throughput (N=8192){RST}")
    hr()
    print(f"  {DIM}Tip: this is what drives LLM attention & MLP layers{RST}\n")

    for dtype, label in [
        (torch.float32, "FP32  "),
        (torch.float16, "FP16  "),
        (torch.bfloat16,"BF16  "),
    ]:
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        try:
            tf, ms, cpu_avg, cpu_pk = bench_matmul(dtype)
            bar = f"{G}{'█' * min(int(tf/3), 50)}{RST}"
            snap = sys_snapshot()
            print(f"  {W}{label}{RST}  {C}{tf:6.1f} TFLOPS{RST}  {bar}")
            print(f"         CPU avg {cpu_avg:.1f}%  peak {cpu_pk:.1f}%  │  "
                  f"VRAM peak {snap['vram_peak_gb']:.2f} GB  │  "
                  f"RAM {snap['ram_used_gb']:.2f}/{snap['ram_total_gb']:.2f} GB")
            results[label.strip()] = {"tflops": round(tf, 2), "cpu_avg_pct": round(cpu_avg, 1)}
        except Exception as e:
            print(f"  {R}{label}: FAILED — {e}{RST}")

    # INT8
    torch.cuda.reset_peak_memory_stats(device)
    try:
        tops, cpu_avg = bench_int8()
        snap = sys_snapshot()
        bar = f"{G}{'█' * min(int(tops/3), 50)}{RST}"
        print(f"  {W}INT8  {RST}  {C}{tops:6.1f} TOPS  {RST}  {bar}")
        print(f"         CPU avg {cpu_avg:.1f}%  │  VRAM peak {snap['vram_peak_gb']:.2f} GB  │  RAM {snap['ram_used_gb']:.2f} GB")
        results["INT8"] = {"tops": round(tops, 2), "cpu_avg_pct": round(cpu_avg, 1)}
    except Exception as e:
        print(f"  {Y}INT8: skipped — {e}{RST}")

    # ── Memory Bandwidth ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{Y}  [2] GPU Memory Bandwidth{RST}")
    hr()
    print(f"  {DIM}Tip: bandwidth is the bottleneck for LLM decode (memory-bound){RST}\n")

    bw, cpu_avg = bench_bandwidth(size_gb=2.0)
    snap = sys_snapshot()
    bar = f"{G}{'█' * min(int(bw/20), 50)}{RST}"
    print(f"  {W}Sequential copy (2 GB){RST}  {C}{bw:6.1f} GB/s{RST}  {bar}")
    print(f"  CPU avg {cpu_avg:.1f}%  │  VRAM peak {snap['vram_peak_gb']:.2f} GB  │  RAM {snap['ram_used_gb']:.2f} GB")
    results["Bandwidth_GBps"] = round(bw, 1)

    # ── Training Sim ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{Y}  [3] Simulated Training Step (fp16, FF layer, batch=64, seq=512, hidden=4096){RST}")
    hr()
    tok_s, ms, vram_peak, cpu_avg = bench_training_step()
    snap = sys_snapshot()
    print(f"  {W}Tokens/s (fwd+bwd+optim){RST}  {C}{tok_s:>10,.0f} tok/s{RST}  {DIM}({ms:.1f} ms/step){RST}")
    print(f"  VRAM peak         {vram_peak:.2f} GB")
    print(f"  CPU avg           {cpu_avg:.1f}%")
    print(f"  RAM used          {snap['ram_used_gb']:.2f} / {snap['ram_total_gb']:.2f} GB  ({snap['ram_pct']:.1f}%)")
    results["Training_toks_per_s"] = round(tok_s)

    # ── Final system snapshot ───────────────────────────────────────────────────
    snap_final = sys_snapshot("after benchmark")
    print_snapshot(snap_final)

    # ── Save ────────────────────────────────────────────────────────────────────
    out = Path("results_raw_benchmark.json")
    out.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "baseline": snap_baseline,
        "final": snap_final,
        "benchmarks": results,
    }, indent=2))
    print(f"\n  {DIM}Results saved → {out}{RST}")
    print(f"{BOLD}{M}{'═'*72}{RST}\n")


if __name__ == "__main__":
    main()
