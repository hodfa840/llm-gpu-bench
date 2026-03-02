#!/usr/bin/env python3
"""
Real LLM Fine-tuning Benchmark
================================
Uses ACTUAL pretrained models + REAL HuggingFace datasets.

Default dataset: tatsu-lab/alpaca (instruction fine-tuning, 52K examples)
Other options:  dolly, oasst1, wikitext2, wikitext103

Fine-tuning methods benchmarked:
  1. Full Fine-tuning (bf16 + gradient checkpointing)
  2. LoRA   (bf16 base, low-rank adapters only)
  3. QLoRA  (4-bit NF4 base + LoRA)

Metrics per method:
  • Tokens/s            ← key training speed metric
  • Samples/s
  • ms/step
  • Training loss       ← real loss on real data
  • VRAM peak
  • System RAM
  • CPU %

NOTE on 671B (DeepSeek-V3/R1 full):
  671B × 0.5 bytes (4-bit) = ~335 GB weights alone.
  CANNOT run on a single 102 GB GPU under ANY quantization.
  Minimum hardware = 4-8× H100 80GB with tensor parallelism.
  Only the distilled versions (7B/8B/14B/32B) are single-GPU feasible.
"""

import bench_config
import torch, time, json, gc, os, random, threading, argparse, psutil
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

R="\033[91m"; G="\033[92m"; Y="\033[93m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"; BOLD="\033[1m"; RST="\033[0m"
device = "cuda:0"

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET CATALOG
# ══════════════════════════════════════════════════════════════════════════════
DATASETS = {
    # Instruction fine-tuning
    "alpaca"    : ("tatsu-lab/alpaca",                    None,                  "train"),
    "dolly"     : ("databricks/databricks-dolly-15k",     None,                  "train"),
    "oasst1"    : ("OpenAssistant/oasst1",                None,                  "train"),
    # Language modelling
    "wikitext2" : ("wikitext",  "wikitext-2-raw-v1",                             "train"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1",                           "train"),
}
DEFAULT_DATASET = "alpaca"


def format_alpaca(ex):
    parts = [f"### Instruction:\n{ex.get('instruction','')}"]
    if ex.get("input","").strip():
        parts.append(f"### Input:\n{ex['input']}")
    parts.append(f"### Response:\n{ex.get('output','')}")
    return {"text": "\n\n".join(parts)}


def format_dolly(ex):
    return {"text": f"### Instruction:\n{ex.get('instruction','')}\n\n### Response:\n{ex.get('response','')}"}


def load_real_dataset(dataset_name: str, tokenizer, seq_len: int, num_samples: int = 1024):
    """Load and tokenize a real dataset. Returns list of token-id tensors."""
    if dataset_name not in DATASETS:
        print(f"  {Y}Unknown dataset '{dataset_name}' → falling back to alpaca{RST}")
        dataset_name = "alpaca"

    ds_id, subset, split = DATASETS[dataset_name]
    print(f"  {DIM}Loading: {ds_id}  (subset={subset or 'default'}, split={split}){RST}", flush=True)

    try:
        kwargs = {"split": split, "trust_remote_code": True}
        if subset:
            kwargs["name"] = subset
        raw = load_dataset(ds_id, **kwargs)

        # Apply instruction formatting
        if "alpaca" in ds_id:
            raw = raw.map(format_alpaca)
        elif "dolly" in ds_id:
            raw = raw.map(format_dolly)

        # Get text column
        text_col = "text" if "text" in raw.column_names else raw.column_names[0]
        n = min(num_samples, len(raw))
        texts = [raw[i][text_col] for i in range(n) if raw[i].get(text_col, "").strip()]

        print(f"  {DIM}Tokenizing {len(texts)} samples…{RST}", flush=True)
        tokenized = tokenizer(texts, truncation=True, max_length=seq_len,
                              padding=False, return_attention_mask=False)
        pool = [torch.tensor(ids) for ids in tokenized["input_ids"] if len(ids) > 16]
        print(f"  {G}✓ Dataset ready: {len(pool)} sequences  (dataset: {dataset_name}){RST}")
        return pool, dataset_name

    except Exception as e:
        print(f"  {R}Dataset load failed: {e}{RST}")
        print(f"  {Y}→ Using synthetic fallback text{RST}")
        return None, "synthetic"


SYNTHETIC_FALLBACK = [
    "The transformer architecture uses self-attention to process sequences in parallel.",
    "Fine-tuning adapts a pretrained model to a specific domain using a smaller dataset.",
    "LoRA trains low-rank decomposition matrices instead of full weight matrices.",
    "Quantization reduces model weights from 16-bit to 4-bit to fit larger models in VRAM.",
    "Gradient checkpointing recomputes activations during backward pass to save memory.",
] * 50


def make_batch(tokenizer, batch_size, seq_len, token_pool=None):
    """Build a training batch from the token pool or synthetic fallback."""
    if token_pool and len(token_pool) >= batch_size:
        samples = random.sample(token_pool, batch_size)
        ids  = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, s in enumerate(samples):
            l = min(len(s), seq_len)
            ids[i, :l]  = s[:l]
            mask[i, :l] = 1
    else:
        enc  = tokenizer(SYNTHETIC_FALLBACK[:batch_size], return_tensors="pt",
                         padding="max_length", truncation=True, max_length=seq_len)
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]

    ids    = ids.to(device)
    mask   = mask.to(device)
    labels = ids.clone()
    labels[mask == 0] = -100
    return ids, mask, labels


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════
FINETUNE_MODELS = [
    # (model_id,                                  name,           params_B, full,  lora,  qlora)
    ("facebook/opt-1.3b",                         "OPT-1.3B",      1.3,    True,  True,  True),
    ("Qwen/Qwen2.5-7B-Instruct",                  "Qwen2.5-7B",    7.0,    True,  True,  True),
    ("Qwen/Qwen2.5-14B-Instruct",                 "Qwen2.5-14B",  14.0,    True,  True,  True),
    ("Qwen/Qwen2.5-32B-Instruct",                 "Qwen2.5-32B",  32.0,   False,  True,  True),  # full OOM (needs ~512GB)
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "DS-R1-7B",     7.0,    True,  True,  True),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "DS-R1-14B",   14.0,    True,  True,  True),
    ("microsoft/phi-4",                            "Phi-4-14B",   14.0,    True,  True,  True),
]

LORA_CFG = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)


def hr(w=72): print(f"{DIM}{'─'*w}{RST}")
def free_mem(): gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


class CPUSampler:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
    def start(self): self._t.start(); return self
    def stop(self):  self._stop.set(); self._t.join()
    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(0.5)
    @property
    def avg(self): return round(sum(self.samples)/len(self.samples), 1) if self.samples else 0.0


def sys_snap():
    vm = psutil.virtual_memory()
    return {
        "ram_used_gb"  : round(vm.used   / 1e9, 2),
        "ram_total_gb" : round(vm.total  / 1e9, 2),
        "ram_pct"      : vm.percent,
        "vram_peak_gb" : round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def run_bench(model, tokenizer, batch_size, seq_len, iters, use_grad_ckpt, token_pool):
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5)

    # Warmup step
    ids, mask, labels = make_batch(tokenizer, batch_size, seq_len, token_pool)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out  = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = out.loss
    loss.backward(); optim.step(); optim.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    sampler = CPUSampler().start()
    times   = []

    for _ in range(iters):
        ids, mask, labels = make_batch(tokenizer, batch_size, seq_len, token_pool)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
        loss.backward(); optim.step(); optim.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    sampler.stop()
    snap = sys_snap()
    avg_t = sum(times) / len(times)

    return {
        "tokens_per_s" : round((batch_size * seq_len) / avg_t),
        "samples_per_s": round(batch_size / avg_t, 2),
        "ms_per_step"  : round(avg_t * 1e3, 1),
        "final_loss"   : round(loss.item(), 4),
        "batch_size"   : batch_size,
        "seq_len"      : seq_len,
        "cpu_avg_pct"  : round(sampler.avg, 1),
        **snap,
    }


def print_result(label, res, pct):
    tok_s = res["tokens_per_s"]
    col   = G if tok_s > 2000 else (Y if tok_s > 500 else R)
    bar   = f"{col}{'█' * min(int(tok_s/200), 48)}{RST}"
    vt, vp = res["vram_total_gb"], res["vram_peak_gb"]
    rt, ru = res["ram_total_gb"],  res["ram_used_gb"]
    vbar   = int((vp/vt)*26) if vt else 0
    rbar   = int((ru/rt)*26) if rt else 0

    print(f"\n  {BOLD}{W}── {label} ──{RST}")
    print(f"  {'Trainable params':<28}: {C}{pct:.3f}% of all params{RST}")
    print(f"  {'Batch / Seq len':<28}: {C}bs={res['batch_size']}  seq={res['seq_len']}{RST}")
    print(f"  {BOLD}{'★ Tokens/s':<28}{RST}: {col}{tok_s:>10,}{RST}  {bar}")
    print(f"  {'Samples/s':<28}: {C}{res['samples_per_s']:>10.2f}{RST}")
    print(f"  {'ms / step':<28}: {C}{res['ms_per_step']:>10.1f} ms{RST}")
    print(f"  {'Training loss (real data)':<28}: {C}{res['final_loss']:>10.4f}{RST}")
    print(f"  {'VRAM peak':<28}: {C}{vp:.2f} / {vt:.1f} GB{RST}  "
          f"{G if vp/vt<0.85 else Y}{'▓'*vbar}{'░'*(26-vbar)}{RST}")
    print(f"  {'RAM used':<28}: {C}{ru:.2f} / {rt:.2f} GB ({res['ram_pct']:.1f}%){RST}  "
          f"{'▓'*rbar}{'░'*(26-rbar)}")
    print(f"  {'CPU avg':<28}: {C}{res['cpu_avg_pct']:.1f}%{RST}" if 'cpu_avg_pct' in res
          else f"  {'CPU avg':<28}: {DIM}n/a{RST}")


def bench_model(model_id, name, params_B, do_full, do_lora, do_qlora,
                batch_size, seq_len, iters, token_pool, dataset_name, results):
    print(f"\n{BOLD}{M}{'═'*72}{RST}")
    print(f"{BOLD}{M}  🏋️  {name}  ({model_id})  ~{params_B}B params{RST}")
    print(f"{BOLD}{M}  Dataset: {dataset_name}{RST}")
    print(f"{BOLD}{M}{'═'*72}{RST}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-tokenize for this model's tokenizer if pool was built with a different one
    # (pool is rebuilt per model for accuracy)
    if token_pool is None:
        pool = None
    else:
        # The pool was tokenized externally; use as-is (same seq_len)
        pool = token_pool

    model_results = {}

    for method, do_run, loader_fn, label, grad_ckpt in [
        ("full_ft",  do_full,  lambda: AutoModelForCausalLM.from_pretrained(
             model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True),
         f"Full FT (bf16, grad ckpt)",                True),
        ("lora",     do_lora,  lambda: get_peft_model(
             AutoModelForCausalLM.from_pretrained(
                 model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True),
             LORA_CFG),
         f"LoRA (bf16 base, r=16)",                   False),
        ("qlora",    do_qlora, lambda: get_peft_model(
             prepare_model_for_kbit_training(
                 AutoModelForCausalLM.from_pretrained(
                     model_id, device_map="auto", trust_remote_code=True,
                     quantization_config=BitsAndBytesConfig(
                         load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))),
             LORA_CFG),
         f"QLoRA (4-bit NF4, r=16)",                  False),
    ]:
        if not do_run:
            continue

        est = {"full_ft": params_B*16, "lora": params_B*2+1, "qlora": params_B*0.5+1}[method]
        print(f"\n  {Y}▶ {label}…{RST}  {DIM}(est. VRAM: ~{est:.0f} GB){RST}", flush=True)
        free_mem()

        try:
            model = loader_fn()
            model.train()
            total, trainable = count_trainable(model)
            pct = trainable / total * 100

            res = run_bench(model, tokenizer, batch_size, seq_len, iters, grad_ckpt, pool)
            res["trainable_pct"] = round(pct, 4)

            print_result(label, res, pct)
            model_results[method] = res
            del model; free_mem()

        except torch.cuda.OutOfMemoryError:
            needed = est
            print(f"  {R}  ✗ OOM — needs ~{needed:.0f} GB, GPU has "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB{RST}")
            model_results[method] = {"error": "OOM", "vram_needed_gb": round(needed, 1)}
            free_mem()
        except Exception as e:
            print(f"  {R}  ✗ Error: {e}{RST}")
            model_results[method] = {"error": str(e)}
            free_mem()

    results[name] = {"model_id": model_id, "params_B": params_B,
                     "dataset": dataset_name, **model_results}


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Benchmark")
    parser.add_argument("--models",      nargs="+", default=None, help="Model display names to run")
    parser.add_argument("--quick",       action="store_true", help="Only OPT-1.3B")
    parser.add_argument("--dataset",     default=DEFAULT_DATASET,
                        choices=list(DATASETS.keys()), help=f"Dataset to use (default: {DEFAULT_DATASET})")
    parser.add_argument("--num-samples", type=int, default=1024, help="Samples to load from dataset")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--seq-len",     type=int, default=512)
    parser.add_argument("--iters",       type=int, default=8)
    args = parser.parse_args()

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{BOLD}{M}{'═'*72}")
    print(f"  🏋️  LLM FINE-TUNING BENCHMARK")
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM    : {total_vram:.1f} GB")
    print(f"  RAM     : {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"  CPUs    : {psutil.cpu_count()} logical")
    print(f"  Dataset : {args.dataset}  ({args.num_samples} samples)")
    print(f"  Batch   : bs={args.batch_size}  seq={args.seq_len}  iters={args.iters}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*72}{RST}")

    print(f"\n  {BOLD}{Y}⚠  671B models (DeepSeek-V3/R1 full):{RST}")
    print(f"  {DIM}  671B × 0.5 bytes (4-bit) = 335 GB — impossible on {total_vram:.0f} GB single GPU.")
    print(f"  {DIM}  Needs 4-8× H100 80GB. Only distilled (7B-32B) are single-GPU feasible.{RST}\n")

    # Pre-load dataset with a temporary tokenizer (opt-125m, very fast)
    print(f"\n  {BOLD}{Y}📥 Loading dataset ({args.dataset})…{RST}")
    tmp_tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    token_pool, ds_name = load_real_dataset(args.dataset, tmp_tok, args.seq_len, args.num_samples)
    del tmp_tok

    if args.quick:
        run_list = [FINETUNE_MODELS[0]]
    elif args.models:
        names = set(args.models)
        run_list = [m for m in FINETUNE_MODELS if m[1] in names]
    else:
        run_list = FINETUNE_MODELS

    results = {}
    for model_id, name, params_B, do_full, do_lora, do_qlora in run_list:
        bench_model(model_id, name, params_B, do_full, do_lora, do_qlora,
                    args.batch_size, args.seq_len, args.iters,
                    token_pool, ds_name, results)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n\n{BOLD}{M}{'═'*72}")
    print(f"  📊 FINE-TUNING SUMMARY  (dataset: {ds_name})")
    print(f"{'═'*72}{RST}")
    print(f"{BOLD}{W}  {'Model':<22}  {'Method':<24}  {'Tok/s':>8}  {'ms/step':>8}  {'VRAM':>8}  {'Loss':>7}{RST}")
    hr()

    for name, data in results.items():
        for method, label in [("full_ft","Full FT"), ("lora","LoRA"), ("qlora","QLoRA")]:
            d = data.get(method)
            if not d:
                continue
            if "error" in d:
                needed = f"  (~{d['vram_needed_gb']:.0f}GB needed)" if "vram_needed_gb" in d else ""
                print(f"  {R}{name:<22}  {label:<24}  {'OOM':>8}{RST}{DIM}{needed}{RST}")
            else:
                col = G if d["tokens_per_s"] > 2000 else (Y if d["tokens_per_s"] > 500 else R)
                print(f"  {W}{name:<22}{RST}  {C}{label:<24}{RST}  "
                      f"{col}{d['tokens_per_s']:>8,}{RST}  "
                      f"{d['ms_per_step']:>8.1f}  "
                      f"{d['vram_peak_gb']:>7.1f}G  "
                      f"{DIM}{d['final_loss']:>7.4f}{RST}")

    outfile = Path("results_finetune_benchmark.json")
    outfile.write_text(json.dumps({
        "gpu"         : torch.cuda.get_device_name(0),
        "vram_gb"     : round(total_vram, 1),
        "ram_total_gb": round(psutil.virtual_memory().total/1e9, 1),
        "date"        : datetime.now().isoformat(),
        "dataset"     : ds_name,
        "batch_size"  : args.batch_size,
        "seq_len"     : args.seq_len,
        "results"     : results,
    }, indent=2, default=str))
    print(f"\n  {DIM}Results saved → {outfile}{RST}")
    print(f"{BOLD}{M}{'═'*72}{RST}\n")


if __name__ == "__main__":
    main()
