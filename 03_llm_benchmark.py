#!/usr/bin/env python3
"""
LLM Inference Benchmark — Multi-family Model Catalog
=====================================================
Families covered:
  • OPT        — Meta (fully open, no auth)
  • Qwen 2.5   — Alibaba (fully open): general + coder variants
  • Phi-4      — Microsoft (fully open, SOTA ~14B)
  • Mistral    — Mistral AI (open): 7B, Nemo-12B, Small-3-24B, Mixtral-8x7B
  • DeepSeek   — DeepSeek AI (R1 distills + V2-Lite, fully open)
  • GLM-4      — Tsinghua (fully open)
  • OLMo-2     — Allen AI (fully open: weights + data + code)
  • Kimi/Moonshot — Moonshot AI (open)
  • LLaMA-3    — Meta (requires HF token + access request)
  • Gemma-2    — Google (requires HF token + access request)

Each model is checked for HF access before downloading.
Runs bf16 AND 4-bit NF4 quantization for each.

Metrics reported per model & precision:
  • Decode throughput (tok/s)  ← most important
  • Prefill throughput (tok/s)
  • Time-to-first-token (ms)
  • VRAM model delta (weights)
  • VRAM peak (inference)
  • System RAM used / total
  • CPU % during prefill and decode
"""

import bench_config          # loads config.env & HF token before any HF calls
import torch, time, json, gc, os, threading, argparse, psutil, sys
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

R="\033[91m"; G="\033[92m"; Y="\033[93m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"; BOLD="\033[1m"; RST="\033[0m"

device = "cuda:0"

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CATALOG
#  Fields: (model_id, display_name, family, params_B, run_fp16, run_4bit, gated)
#  gated=True → requires HF token + accepted license
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CATALOG = [

    # ── OPT (Meta, fully open) ────────────────────────────────────────────────
    ("facebook/opt-125m",                "OPT-125M",             "OPT",      0.1,  True,  True,  False),
    ("facebook/opt-1.3b",                "OPT-1.3B",             "OPT",      1.3,  True,  True,  False),
    ("facebook/opt-6.7b",                "OPT-6.7B",             "OPT",      6.7,  True,  True,  False),
    ("facebook/opt-13b",                 "OPT-13B",              "OPT",     13.0,  True,  True,  False),
    ("facebook/opt-30b",                 "OPT-30B",              "OPT",     30.0,  True,  True,  False),
    ("facebook/opt-66b",                 "OPT-66B",              "OPT",     66.0,  False, True,  False),  # fp16 ~132GB

    # ── Qwen 2.5 — General (Alibaba, fully open, SOTA at 72B) ────────────────
    ("Qwen/Qwen2.5-7B-Instruct",          "Qwen2.5-7B",          "Qwen",     7.0,  True,  True,  False),
    ("Qwen/Qwen2.5-14B-Instruct",         "Qwen2.5-14B",         "Qwen",    14.0,  True,  True,  False),
    ("Qwen/Qwen2.5-32B-Instruct",         "Qwen2.5-32B",         "Qwen",    32.0,  True,  True,  False),
    ("Qwen/Qwen2.5-72B-Instruct",         "Qwen2.5-72B",         "Qwen",    72.0,  False, True,  False),  # fp16~144GB, 4-bit~36GB ✓

    # ── Qwen 2.5 — Coder (SOTA for coding tasks) ──────────────────────────────
    ("Qwen/Qwen2.5-Coder-7B-Instruct",    "Qwen2.5-Coder-7B",   "Qwen",     7.0,  True,  True,  False),
    ("Qwen/Qwen2.5-Coder-32B-Instruct",   "Qwen2.5-Coder-32B",  "Qwen",    32.0,  True,  True,  False),  # ★ SOTA coding

    # ── Phi-4 (Microsoft, fully open, ★ SOTA for ~14B class) ─────────────────
    ("microsoft/phi-4",                    "Phi-4-14B",           "Phi",     14.0,  True,  True,  False),

    # ── Mistral (open) ────────────────────────────────────────────────────────
    ("mistralai/Mistral-7B-Instruct-v0.3",        "Mistral-7B",          "Mistral",  7.0,  True,  True,  False),
    ("mistralai/Mistral-Nemo-Instruct-2407",       "Mistral-Nemo-12B",   "Mistral", 12.0,  True,  True,  False),
    ("mistralai/Mistral-Small-24B-Instruct-2501",  "Mistral-Small3-24B", "Mistral", 24.0,  True,  True,  False),  # ★ Jan 2026 SOTA
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",       "Mixtral-8x7B",       "Mistral", 46.7,  True,  True,  False),  # MoE, ~96GB active weights

    # ── DeepSeek (open — distills + V2-Lite) ──────────────────────────────────
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  "DS-R1-Qwen-7B",  "DeepSeek",  7.0, True,  True, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DS-R1-Llama-8B", "DeepSeek",  8.0, True,  True, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DS-R1-Qwen-14B", "DeepSeek", 14.0, True,  True, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DS-R1-Qwen-32B", "DeepSeek", 32.0, True,  True, False),
    ("deepseek-ai/DeepSeek-V2-Lite",             "DeepSeek-V2-Lite","DeepSeek", 15.7, True,  True, False),

    # ── GLM-4 (Tsinghua, open) ────────────────────────────────────────────────
    ("THUDM/glm-4-9b-chat",              "GLM-4-9B",             "GLM",      9.0,  True,  True,  False),

    # ── OLMo-2 (Allen AI, fully open) ────────────────────────────────────────
    ("allenai/OLMo-2-1124-7B",           "OLMo-2-7B",            "OLMo",     7.0,  True,  True,  False),
    ("allenai/OLMo-2-1124-13B",          "OLMo-2-13B",           "OLMo",    13.0,  True,  True,  False),

    # ── Kimi / Moonshot (open) ────────────────────────────────────────────────
    ("moonshotai/Kimi-VL-A3B-Instruct",  "Kimi-VL-3B",           "Kimi",     3.0,  True,  True,  False),

    # ── LLaMA-3 (gated — needs HF token + Meta access) ───────────────────────
    ("meta-llama/Llama-3.2-1B-Instruct",  "LLaMA-3.2-1B",   "LLaMA",  1.0,  True,  True,  True),
    ("meta-llama/Llama-3.2-3B-Instruct",  "LLaMA-3.2-3B",   "LLaMA",  3.0,  True,  True,  True),
    ("meta-llama/Llama-3.1-8B-Instruct",  "LLaMA-3.1-8B",   "LLaMA",  8.0,  True,  True,  True),
    ("meta-llama/Llama-3.3-70B-Instruct", "LLaMA-3.3-70B",  "LLaMA", 70.0,  True,  True,  True),  # newer than 3.1-70B; fp16 ~140GB → bf16 on 102GB is tight

    # ── Gemma-2 (gated — needs HF token + Google access) ─────────────────────
    ("google/gemma-2-9b-it",             "Gemma-2-9B",           "Gemma",    9.0,  True,  True,  True),
    ("google/gemma-2-27b-it",            "Gemma-2-27B",          "Gemma",   27.0,  True,  True,  True),
]

# Family → suggested selection for --quick-family
FAMILY_DEFAULTS = {
    "opt"      : ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-6.7b"],
    "qwen"     : ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct"],
    "phi"      : ["microsoft/phi-4"],
    "mistral"  : ["mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-Small-24B-Instruct-2501"],
    "deepseek" : ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
    "glm"      : ["THUDM/glm-4-9b-chat"],
    "olmo"     : ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-13B"],
    "llama"    : ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
    "gemma"    : ["google/gemma-2-9b-it"],
    "kimi"     : ["moonshotai/Kimi-VL-A3B-Instruct"],
}


def hr(w=72): print(f"{DIM}{'─'*w}{RST}")
def free_mem(): gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


# ── HF Access Checker ─────────────────────────────────────────────────────────

def check_hf_access(model_id, hf_api):
    """Returns ('ok'|'gated_denied'|'not_found'|'open'), info_str"""
    try:
        info = hf_api.model_info(model_id)
        if info.gated:
            # Try to fetch tokenizer config to confirm actual access
            try:
                hf_hub_download(model_id, "tokenizer_config.json")
                return "ok", f"gated — access GRANTED ✓"
            except GatedRepoError:
                return "gated_denied", "gated — access DENIED (need HF token + accepted license)"
            except Exception:
                return "ok", "gated — likely accessible"
        return "ok", "open"
    except RepositoryNotFoundError:
        return "not_found", "repository not found"
    except GatedRepoError:
        return "gated_denied", "gated — access DENIED"
    except HfHubHTTPError as e:
        return "ok", f"HTTP {e.response.status_code}"
    except Exception as e:
        return "unknown", str(e)


def pre_flight_check(models_to_run, hf_api, skip_check=False):
    """Check HF access for all models before starting benchmarks.
    If skip_check=True, skip network API calls and approve all open models directly.
    """
    print(f"\n{BOLD}{Y}  🔍 Pre-flight: {len(models_to_run)} model(s)  "
          f"{'(access check SKIPPED — using local cache)' if skip_check else '(checking HF access…)'}{RST}")
    hr()
    approved  = []
    skipped   = []
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or hf_api.token)
    print(f"  HF token: {'✓ found' if has_token else '✗ not set — gated models will be skipped'}")
    print()

    for entry in models_to_run:
        model_id, name, family, params, fp16, bit4, gated = entry
        if gated and not has_token:
            status = f"{R}SKIP — gated, no HF token{RST}"
            skipped.append(entry)
        elif skip_check:
            # Trust user: approve open models, gate check skipped
            if gated:
                status = f"{Y}ASSUME OK (gated, token present, check skipped){RST}"
            else:
                status = f"{G}OK  (open model, check skipped){RST}"
            approved.append(entry)
        else:
            access, info = check_hf_access(model_id, hf_api)
            if access in ("ok", "unknown"):
                status = f"{G}OK{RST}  {DIM}({info}){RST}"
                approved.append(entry)
            else:
                status = f"{R}SKIP — {info}{RST}"
                skipped.append(entry)

        tag = f"{DIM}[{family}]{RST}"
        print(f"  {W}{name:<24}{RST} {tag:<16}  {status}")

    print()
    if skipped:
        print(f"  {Y}Skipped {len(skipped)} model(s) (no access / no token).{RST}")
        print(f"  {DIM}To access gated models: huggingface-cli login{RST}")
    print(f"  {G}Will benchmark {len(approved)} model(s).{RST}")
    hr()
    return approved


# ── Resource monitoring ───────────────────────────────────────────────────────

class CPUSampler:
    def __init__(self, interval=0.5):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
    def start(self): self._t.start(); return self
    def stop(self):  self._stop.set(); self._t.join()
    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)
    @property
    def avg(self):  return round(sum(self.samples)/len(self.samples), 1) if self.samples else 0.0
    @property
    def peak(self): return round(max(self.samples), 1) if self.samples else 0.0


def sys_snap():
    vm = psutil.virtual_memory()
    return {
        "ram_used_gb"  : round(vm.used  / 1e9, 2),
        "ram_total_gb" : round(vm.total / 1e9, 2),
        "ram_pct"      : vm.percent,
        "swap_gb"      : round(psutil.swap_memory().used / 1e9, 2),
        "vram_alloc_gb": round(torch.cuda.memory_allocated(device)     / 1e9, 2),
        "vram_peak_gb" : round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_fp16(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()

def load_4bit(model_id):
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, device_map="auto",
        trust_remote_code=True).eval()


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, prompt, max_new_tokens):
    inputs   = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)

    # Prefill
    sampler1 = CPUSampler().start()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    sampler1.stop()

    # Decode
    past = out.past_key_values
    nxt  = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen  = [nxt.item()]

    sampler2 = CPUSampler().start()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if nxt.item() == tokenizer.eos_token_id:
                break
            out  = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt  = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(nxt.item())
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t1
    sampler2.stop()

    snap = sys_snap()
    n_gen = len(gen)

    return {
        "ttft_ms"         : round(ttft * 1e3, 1),
        "prefill_tok_s"   : round(n_prompt / ttft, 1),
        "decode_tok_s"    : round(n_gen / decode_time if decode_time > 0 else 0, 1),
        "total_tok_s"     : round((n_prompt + n_gen) / (ttft + decode_time), 1),
        "n_prompt"        : n_prompt,
        "n_generated"     : n_gen,
        "cpu_prefill_pct" : sampler1.avg,
        "cpu_decode_avg"  : sampler2.avg,
        "cpu_decode_peak" : sampler2.peak,
        "vram_peak_gb"    : snap["vram_peak_gb"],
        "ram_used_gb"     : snap["ram_used_gb"],
        "ram_total_gb"    : snap["ram_total_gb"],
        "ram_pct"         : snap["ram_pct"],
        "swap_gb"         : snap["swap_gb"],
    }


def print_result(label, res, vram_model_gb):
    tok_s = res["decode_tok_s"]
    col   = G if tok_s > 60 else (Y if tok_s > 20 else R)
    bar   = f"{col}{'█' * min(int(tok_s/4), 48)}{RST}"

    vt = res["vram_total_gb"] if "vram_total_gb" in res else torch.cuda.get_device_properties(0).total_memory/1e9
    vp = res["vram_peak_gb"]
    rt = res["ram_total_gb"]
    ru = res["ram_used_gb"]
    vbar = int((vp/vt)*28) if vt else 0
    rbar = int((ru/rt)*28) if rt else 0

    print(f"\n  {BOLD}{W}── {label} ──{RST}")
    print(f"  {'VRAM weights':<28}: {C}{vram_model_gb:.2f} GB{RST}")
    print(f"  {'VRAM peak (incl. KV cache)':<28}: {C}{vp:.2f} / {vt:.1f} GB{RST}  "
          f"{G if vp/vt<0.85 else Y}{'▓'*vbar}{'░'*(28-vbar)}{RST}")
    print(f"  {'RAM used':<28}: {C}{ru:.2f} / {rt:.2f} GB ({res['ram_pct']:.1f}%){RST}  "
          f"{'▓'*rbar}{'░'*(28-rbar)}")
    print(f"  {'Swap used':<28}: {C}{res['swap_gb']:.2f} GB{RST}")
    print(f"  {'CPU during prefill':<28}: {C}{res['cpu_prefill_pct']:.1f}%{RST}")
    print(f"  {'CPU during decode':<28}: {C}{res['cpu_decode_avg']:.1f}% avg  /  {res['cpu_decode_peak']:.1f}% peak{RST}")
    print(f"  {'Time-to-first-token':<28}: {C}{res['ttft_ms']:.0f} ms{RST}")
    print(f"  {'Prefill':<28}: {C}{res['prefill_tok_s']:,.0f} tok/s{RST}")
    print(f"  {BOLD}{'Decode ★':<28}{RST}: {col}{tok_s:,.1f} tok/s{RST}  {bar}")


def bench_one(entry, prompt, max_gen, results):
    model_id, name, family, params, do_fp16, do_4bit, _ = entry

    print(f"\n{BOLD}{M}{'═'*72}{RST}")
    print(f"{BOLD}{M}  [{family}] {name}  ({model_id})  ~{params}B params{RST}")
    print(f"{BOLD}{M}{'═'*72}{RST}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mres = {}
    for prec_key, do_run, loader, label in [
        ("bf16",     do_fp16, load_fp16, "bf16/fp16"),
        ("4bit_nf4", do_4bit, load_4bit, "4-bit NF4 (bitsandbytes)"),
    ]:
        if not do_run:
            continue
        print(f"\n  {Y}▶ Loading {label}…{RST}", flush=True)
        free_mem()
        v_before = torch.cuda.memory_allocated(device) / 1e9
        try:
            model   = loader(model_id)
            v_after = torch.cuda.memory_allocated(device) / 1e9
            v_delta = v_after - v_before

            res = run_inference(model, tokenizer, prompt, max_gen)
            print_result(label, res, v_delta)
            mres[prec_key] = {**res, "vram_model_gb": round(v_delta, 2)}

            del model; free_mem()
        except torch.cuda.OutOfMemoryError:
            print(f"  {R}  ✗ OOM — not enough VRAM for {label}{RST}")
            mres[prec_key] = {"error": "OOM"}
            free_mem()
        except Exception as e:
            print(f"  {R}  ✗ Error: {e}{RST}")
            mres[prec_key] = {"error": str(e)}
            free_mem()

    results[name] = {"family": family, "model_id": model_id, "params_B": params, **mres}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{Y}  ⚙  Config (config.env){RST}")
    bench_config.print_config_summary()

    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--models",  nargs="+", default=None,
                        help="Specific model IDs from the catalog")
    parser.add_argument("--family",  nargs="+", default=None,
                        help="Filter by family: opt, qwen, mistral, deepseek, glm, olmo, llama, gemma, kimi")
    parser.add_argument("--quick",   action="store_true",
                        help="Only opt-125m + opt-1.3b (fast sanity check)")
    parser.add_argument("--max-gen", type=int, default=bench_config.get_int("MAX_NEW_TOKENS", 200))
    parser.add_argument("--prompt",  type=str,
                        default="Explain in detail the theory of general relativity — covering space-time curvature, gravitational lensing, time dilation, and experimental evidence:")
    parser.add_argument("--no-access-check", action="store_true",
                        help="Skip HF API preflight check (faster, use when network is slow/blocked)")
    args = parser.parse_args()

    # Default families from config if not overridden on CLI
    if not args.family and not args.models and not args.quick:
        cfg_families = bench_config.get_list("DEFAULT_FAMILIES")
        if cfg_families:
            args.family = cfg_families

    hf_api = HfApi()

    print(f"\n{BOLD}{M}{'═'*72}")
    print(f"  🚀 LLM INFERENCE BENCHMARK — Multi-family")
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  RAM   : {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"  CPUs  : {psutil.cpu_count()} logical  ({psutil.cpu_count(logical=False)} physical)")
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"  HF    : {'✓ token loaded' if has_token else 'no token (gated models will be skipped)'}")
    print(f"  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*72}{RST}\n")

    # ── Build run list ─────────────────────────────────────────────────────────
    if args.quick:
        run_list = [e for e in MODEL_CATALOG if e[0] in ("facebook/opt-125m", "facebook/opt-1.3b")]
    elif args.models:
        id_set   = set(args.models)
        run_list = [e for e in MODEL_CATALOG if e[0] in id_set]
        # also support display names
        if not run_list:
            run_list = [e for e in MODEL_CATALOG if e[1] in id_set]
    elif args.family:
        fams = {f.lower() for f in args.family}
        run_list = [e for e in MODEL_CATALOG if e[2].lower() in fams]
    else:
        run_list = MODEL_CATALOG  # all

    if not run_list:
        print(f"  {R}No matching models found. Check --models / --family args.{RST}")
        sys.exit(1)

    # ── Pre-flight access check ────────────────────────────────────────────────
    approved = pre_flight_check(run_list, hf_api, skip_check=args.no_access_check)
    if not approved:
        print(f"\n  {R}No accessible models to benchmark. Exiting.{RST}")
        sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────────
    results = {}
    for entry in approved:
        bench_one(entry, args.prompt, args.max_gen, results)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{BOLD}{M}{'═'*72}")
    print(f"  📊 SUMMARY — Decode tok/s  |  VRAM weights  |  RAM used")
    print(f"{'═'*72}{RST}")
    print(f"{BOLD}{W}  {'Model':<24}{'Family':<12}{'bf16 tok/s':>10}  {'VRAM':>8}  {'4bit tok/s':>10}  {'VRAM':>8}  {'RAM':>7}{RST}")
    hr()

    by_family = {}
    for name, data in results.items():
        fam = data.get("family", "?")
        by_family.setdefault(fam, []).append((name, data))

    for fam, entries in sorted(by_family.items()):
        print(f"\n  {BOLD}{Y}[{fam}]{RST}")
        for name, data in entries:
            bf  = data.get("bf16",     {})
            nf4 = data.get("4bit_nf4", {})
            bt  = bf.get("decode_tok_s",  None)
            nt  = nf4.get("decode_tok_s", None)
            bv  = bf.get("vram_model_gb", None)
            nv  = nf4.get("vram_model_gb", None)
            ram = nf4.get("ram_used_gb") or bf.get("ram_used_gb")
            bs  = f"{bt:>8.1f}" if isinstance(bt, float) else f"{'OOM':>8}"
            ns  = f"{nt:>8.1f}" if isinstance(nt, float) else f"{'OOM':>8}"
            bvs = f"{bv:.1f}G"  if isinstance(bv, float) else "—"
            nvs = f"{nv:.1f}G"  if isinstance(nv, float) else "—"
            rs  = f"{ram:.1f}G" if isinstance(ram, float) else "—"
            col = G if isinstance(bt, float) or isinstance(nt, float) else R
            print(f"  {col}{name:<24}{data.get('family',''):<12}{RST}{C}{bs}{RST}  {DIM}{bvs:>8}{RST}  {C}{ns}{RST}  {DIM}{nvs:>8}{RST}  {DIM}{rs:>7}{RST}")

    output = {
        "gpu"            : torch.cuda.get_device_name(0),
        "vram_gb"        : round(torch.cuda.get_device_properties(0).total_memory/1e9, 1),
        "ram_total_gb"   : round(psutil.virtual_memory().total/1e9, 1),
        "cpu_count"      : psutil.cpu_count(),
        "date"           : datetime.now().isoformat(),
        "max_new_tokens" : args.max_gen,
        "results"        : results,
    }
    outfile = Path("results_llm_benchmark.json")
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  {DIM}Full results → {outfile}{RST}")
    print(f"{BOLD}{M}{'═'*72}{RST}\n")


if __name__ == "__main__":
    main()
