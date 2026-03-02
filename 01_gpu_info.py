#!/usr/bin/env python3
"""
GPU Info & LLM Capability Summary
Shows GPU specs and what LLMs can fit in VRAM at various precisions.
"""

import torch
import subprocess
import json
from datetime import datetime

# ─── ANSI colors ───────────────────────────────────────────────────────────────
R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B  = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"

def hr(char="─", width=70, color=C):
    print(f"{color}{char * width}{RST}")

def header(title):
    hr("═")
    print(f"{BOLD}{C}  {title}{RST}")
    hr("═")

def section(title):
    print(f"\n{BOLD}{Y}  ▶  {title}{RST}")
    hr("─", 70, DIM)

# ─── VRAM requirements table ───────────────────────────────────────────────────
MODELS = [
    # (name,            params_B, context_K)
    ("TinyLlama-1.1B",       1.1,   2),
    ("Phi-2 2.7B",           2.7,   2),
    ("Mistral-7B",           7.0,   8),
    ("LLaMA-3-8B",           8.0,   8),
    ("Gemma-9B",             9.0,   8),
    ("Mistral-11B",         11.0,   8),
    ("LLaMA-3-13B",         13.0,   8),
    ("Mixtral-8x7B (MoE)",  46.7,   32),
    ("LLaMA-3.3-70B",        70.0,   8),
    ("Qwen2.5-72B",         72.0,   32),
    ("Mixtral-8x22B (MoE)", 141.0,  64),
    ("LLaMA-3.1-405B",     405.0,  128),
    ("DeepSeek-V3 671B",   671.0,  128),
]

BYTES_PER_PARAM = {
    "fp32"    : 4.0,
    "bf16/fp16": 2.0,
    "fp8"     : 1.0,
    "int8/Q8" : 1.0,
    "Q4 (4-bit)": 0.5,
    "Q2 (2-bit)": 0.25,
}

KV_BYTES_PER_TOKEN = 0.5  # rough estimate per billion params at 8K ctx

def vram_needed(params_b, bytes_per_param, ctx_k=8, overhead=1.15):
    """Estimate VRAM in GB (model weights + KV cache overhead)."""
    weights_gb = params_b * 1e9 * bytes_per_param / 1e9
    kv_gb      = params_b * KV_BYTES_PER_TOKEN * ctx_k / 8  # rough
    return (weights_gb + kv_gb) * overhead

def fits(vram_needed, total_vram):
    if vram_needed <= total_vram * 0.85:
        return f"{G}✓ fits{RST}"
    elif vram_needed <= total_vram:
        return f"{Y}⚠ tight{RST}"
    else:
        return f"{R}✗ OOM{RST}"


def main():
    print(f"\n{BOLD}{M}{'═'*70}")
    print(f"  🖥️  GPU LLM CAPABILITY REPORT  — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*70}{RST}\n")

    # ── GPU Hardware ────────────────────────────────────────────────────────────
    section("GPU Hardware")
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1e9

    specs = {
        "GPU Name"            : props.name,
        "VRAM"                : f"{total_vram:.1f} GB",
        "SM Count"            : props.multi_processor_count,
        "CUDA Capability"     : f"{props.major}.{props.minor}",
        "L2 Cache"            : f"{props.L2_cache_size / 1e6:.0f} MB",
        "Memory Bus Width"    : f"{props.memory_bus_width} bit",
    }

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version,memory.bandwidth",
             "--format=csv,noheader"], timeout=10, text=True
        ).strip().split(",")
        if len(smi) >= 2:
            specs["Driver Version"] = smi[1].strip()
    except Exception:
        pass

    try:
        smi2 = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.limit",
             "--format=csv,noheader,nounits"], timeout=10, text=True
        ).strip().split(",")
        if len(smi2) == 3:
            specs["SM Clock"]   = f"{smi2[0].strip()} MHz"
            specs["Mem Clock"]  = f"{smi2[1].strip()} MHz"
            specs["TDP"]        = f"{smi2[2].strip()} W"
    except Exception:
        pass

    for k, v in specs.items():
        print(f"  {BOLD}{W}{k:<22}{RST}: {C}{v}{RST}")

    # ── Theoretical Compute ─────────────────────────────────────────────────────
    section("Theoretical Compute (estimated)")

    # Blackwell GB202: 96 TFLOPs FP16 Tensor, memory bandwidth ~1.8 TB/s
    # RTX PRO 6000 Blackwell: 188 SMs × 2 × 128 FP16 cores = 48128 → ~96 TF
    cuda_cores = props.multi_processor_count * 128  # estimate for Blackwell
    print(f"  {W}Approx CUDA cores  {RST}: {C}{cuda_cores:,}{RST}")
    print(f"  {W}FP16 TFLOPs (est.) {RST}: {C}~{cuda_cores * 2 * 2.4 / 1e3:.0f} TFLOPs{RST}")
    print(f"  {W}INT8 TOPs  (est.)  {RST}: {C}~{cuda_cores * 2 * 2.4 * 2 / 1e3:.0f} TOPs{RST}")
    print(f"  {W}Architecture       {RST}: {C}Blackwell (SM {props.major}.{props.minor}){RST}")

    # ── VRAM Fit Table ──────────────────────────────────────────────────────────
    section(f"LLM VRAM Requirements vs Your {total_vram:.0f} GB GPU")

    precisions = ["bf16/fp16", "fp8", "Q4 (4-bit)", "Q2 (2-bit)"]
    col_w = 12

    # Header row
    print(f"  {'Model':<28}", end="")
    print(f"{'Params':>8}", end="")
    for p in precisions:
        print(f"  {p:>{col_w}}", end="")
    print()
    hr("─", 70, DIM)

    for name, params, ctx in MODELS:
        print(f"  {W}{name:<28}{RST}", end="")
        print(f"{DIM}{params:>6.1f}B{RST}", end="")
        for p in precisions:
            bpp  = BYTES_PER_PARAM[p]
            vram = vram_needed(params, bpp, ctx)
            fstr = fits(vram, total_vram)
            print(f"  {vram:>6.1f}GB {fstr}", end="")
        print()

    print(f"\n  {DIM}* Estimates include ~15% overhead for KV cache & activations. Actual usage varies.{RST}")

    # ── What can you actually run? ──────────────────────────────────────────────
    section(f"Models You Can Run (fp16/bf16 on {total_vram:.0f} GB VRAM)")
    can_run_fp16 = [(n, p) for n, p, _ in MODELS if p * 2 * 1.15 <= total_vram]
    can_run_4bit = [(n, p) for n, p, _ in MODELS if p * 0.5 * 1.15 <= total_vram]

    print(f"  {G}{BOLD}✓ fp16/bf16 (full precision):{RST}")
    for n, p in can_run_fp16:
        print(f"    {G}•{RST} {n} ({p}B) — ~{p*2:.0f} GB")

    print(f"\n  {Y}{BOLD}✓ 4-bit quantized (bitsandbytes / GGUF Q4):{RST}")
    for n, p in can_run_4bit:
        print(f"    {Y}•{RST} {n} ({p}B) — ~{p*0.5:.0f} GB")

    # ── Recommended benchmark repos ─────────────────────────────────────────────
    section("Recommended GPU / LLM Benchmark Repositories")
    repos = [
        ("vllm-project/vllm",          "benchmark_throughput.py + benchmark_latency.py — production-grade"),
        ("ggerganov/llama.cpp",         "llama-bench — CPU+GPU, GGUF, Q4/Q8/fp16 support"),
        ("huggingface/optimum-benchmark","HF-native, multi-backend (PyTorch, ONNX, TRT)"),
        ("NVIDIA/TensorRT-LLM",         "TensorRT-LLM bench — highest throughput on NVIDIA"),
        ("mlcommons/inference",          "MLPerf Inference — industry standard, LLM track"),
        ("ray-project/llmperf",          "LLM API throughput & latency benchmarking"),
        ("Mozilla-Ocho/llamafile",       "Single-file benchmark, GGUF, GPU-accelerated"),
        ("EleutherAI/lm-evaluation-harness","Quality benchmarks (MMLU, HellaSwag, etc.)"),
    ]
    for repo, desc in repos:
        print(f"  {C}github.com/{repo}{RST}")
        print(f"    {DIM}{desc}{RST}")

    print(f"\n{BOLD}{M}{'═'*70}{RST}\n")


if __name__ == "__main__":
    main()
