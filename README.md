# llm-gpu-bench

A straightforward benchmark suite for testing how well a GPU handles Large Language Models. It covers raw compute, inference speed, and fine-tuning throughput — the three things that actually matter when you want to know if a GPU is useful for AI work.

It was originally run on an NVIDIA RTX PRO 6000 Blackwell (96 GB), but it works on any CUDA-capable GPU. The goal is to give people a consistent way to compare hardware across different machines.

---

## What it measures

**1. Raw GPU hardware** (`01_gpu_info.py`, `02_raw_gpu_benchmark.py`)
- TFLOPS at FP32, FP16, BF16, INT8
- Memory bandwidth (GB/s)
- VRAM capacity and theoretical LLM model size limits

**2. LLM inference** (`03_llm_benchmark.py`)
- Real text generation speed (tokens/sec) across multiple model families
- Tests both full precision (BF16) and 4-bit quantized (NF4) loading
- Measures Time to First Token, prefill speed, decode speed, and VRAM usage
- Covers OPT, Qwen 2.5, Mistral, DeepSeek R1, Phi-4, GLM-4, OLMo-2, and more

**3. Fine-tuning** (`04_finetune_benchmark.py`)
- Training throughput in tokens/sec using the Alpaca dataset
- Compares Full Fine-Tuning, LoRA, and QLoRA
- Reports peak VRAM and CPU load during training
- Clearly marks OOM (out of memory) failures with estimated requirements

**4. System monitoring** (throughout)
- CPU %, RAM, and Swap tracked during every benchmark
- Low CPU % confirms the GPU is the bottleneck, not the data pipeline

**5. Email report** (`05_email_results.py`)
- Sends a formatted HTML summary when the full run completes

---

## Project structure

```
llm-gpu-bench/
├── 01_gpu_info.py             # Hardware info and VRAM capacity table
├── 02_raw_gpu_benchmark.py    # TFLOPS and bandwidth tests
├── 03_llm_benchmark.py        # Inference benchmark (multi-family)
├── 04_finetune_benchmark.py   # Fine-tuning benchmark (Full FT / LoRA / QLoRA)
├── 05_email_results.py        # HTML results email sender
├── bench_config.py            # Config loader (reads config.env)
├── run_all_benchmarks.sh      # Runs all steps in sequence
├── config.env.example         # Template for your settings
└── report/
    └── main.tex               # LaTeX report (upload to Overleaf)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/llm-gpu-bench.git
cd llm-gpu-bench
```

### 2. Create a conda environment

```bash
conda create -n ml python=3.10 -y
conda activate ml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft bitsandbytes accelerate psutil huggingface_hub
```

### 3. Configure settings

Copy the example config and fill in your details:

```bash
cp config.env.example config.env
```

Open `config.env` and set:
- `HF_TOKEN` — your Hugging Face token (required for gated models like LLaMA-3 and Gemma-2)
- `EMAIL_TO`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` — only needed if you want email reports
- `DEFAULT_FAMILIES` — which model families to benchmark (see options below)

Your `config.env` is in `.gitignore` and will never be committed.

Alternatively, put your Hugging Face token in a file called `hf_token` in the project root. That file is also ignored by git.

---

## Running

### Full benchmark (downloads models as needed)

```bash
./run_all_benchmarks.sh
```

### Quick sanity check (OPT-125M and OPT-1.3B only, no downloads)

```bash
./run_all_benchmarks.sh --quick
```

### Run individual scripts

```bash
# GPU info only
conda run -n ml python 01_gpu_info.py

# Raw compute
conda run -n ml python 02_raw_gpu_benchmark.py

# Inference — specific families only
conda run -n ml python 03_llm_benchmark.py --families opt mistral

# Skip HF access check (useful on restricted networks)
conda run -n ml python 03_llm_benchmark.py --no-access-check

# Fine-tuning — specific models, specific dataset
conda run -n ml python 04_finetune_benchmark.py --dataset alpaca --quick
```

---

## Model families

| Flag | Models included |
|------|----------------|
| `opt` | OPT-125M → 30B (Meta, fully open) |
| `qwen` | Qwen2.5 7B, 14B, 32B, 72B + Coder variants |
| `phi` | Phi-4 14B (Microsoft) |
| `mistral` | Mistral-7B, Nemo-12B, Small3-24B, Mixtral-8x7B |
| `deepseek` | DeepSeek-R1 Distill: 7B, 8B, 14B, 32B |
| `glm` | GLM-4-9B (Tsinghua) |
| `olmo` | OLMo-2 7B and 13B (Allen AI) |
| `kimi` | Kimi-VL-3B (Moonshot AI) |
| `llama` | LLaMA-3.1-8B, LLaMA-3.3-70B (gated — requires HF token) |
| `gemma` | Gemma-2-9B, Gemma-2-27B (gated — requires HF token) |

Set `DEFAULT_FAMILIES` in `config.env` to choose which ones run automatically. Leave it blank to run everything (requires several hundred GB of disk and many hours).

---

## Output files

After each run, results are saved locally as JSON:

| File | Contents |
|------|----------|
| `results_raw_benchmark.json` | TFLOPS, bandwidth, training sim |
| `results_llm_benchmark.json` | Inference speed per model and precision |
| `results_finetune_benchmark.json` | Training throughput, VRAM, CPU per method |

These files are ignored by git so you can keep them locally without uploading.

---

## Disk space

Downloading models takes significant disk space. Approximate sizes:

| Family | Size on disk |
|--------|-------------|
| OPT (all sizes up to 30B) | ~100 GB |
| Qwen2.5 (7B to 32B) | ~120 GB |
| Mistral (7B to Mixtral) | ~120 GB |
| DeepSeek R1 distills (7B–32B) | ~80 GB |
| LLaMA-3.3-70B | ~140 GB |

Models are cached in `~/.cache/huggingface/hub/` by default.

---

## Known limitations

- **14B+ Full Fine-Tuning** requires more than 200 GB of VRAM and will always OOM on a single GPU. This is expected. Use LoRA or QLoRA instead.
- **Mixtral-8x7B** in 4-bit sometimes fails depending on which version of bitsandbytes is installed. This is a known upstream issue.
- **DeepSeek-V2-Lite** requires a newer version of Transformers than what ships with some conda defaults.
- **Gated models** (LLaMA-3, Gemma-2) require accepting the license on Hugging Face and providing a valid token.
- **Slow internet** will make the first run take many hours due to model downloads. All subsequent runs use the local cache.

---

## Results from testing on RTX PRO 6000 Blackwell (96 GB)

A full benchmark was run on this hardware in February 2026. Some highlights:

| Model | Precision | Decode Speed | VRAM |
|-------|-----------|-------------|------|
| Qwen2.5-7B | BF16 | 55.7 tok/s | 15.2 GB |
| DeepSeek-R1-14B | BF16 | 35.1 tok/s | 29.5 GB |
| Qwen2.5-32B | BF16 | 19.1 tok/s | 65.6 GB |
| Qwen2.5-32B | 4-bit | 12.3 tok/s | 19.5 GB |

The full LaTeX report with all tables is in `report/main.tex` and can be compiled on Overleaf.

---

## Contributing

If you run this on different hardware, feel free to open a PR with your JSON results and a note about the GPU you used. The goal is to build up a collection of results across different cards.

---

## License

MIT
