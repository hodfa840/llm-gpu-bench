#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  GPU LLM Benchmark Suite — Full Run
#  Reads settings from config.env in the same directory.
#
#  Usage:
#    ./run_all_benchmarks.sh              # runs default families from config.env
#    ./run_all_benchmarks.sh --quick      # only OPT-125M / OPT-1.3B (fast test)
#    ./run_all_benchmarks.sh --no-email   # skip emailing results
# ══════════════════════════════════════════════════════════════════════════════

CONDA_ENV="ml"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/benchmark_run_$(date +%Y%m%d_%H%M).log"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
RED='\033[0;31m'; NC='\033[0m'; BOLD='\033[1m'

QUICK=false
SEND_EMAIL=true
for arg in "$@"; do
  case $arg in
    --quick)    QUICK=true ;;
    --no-email) SEND_EMAIL=false ;;
  esac
done

divider() { echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"; }
step()    { echo -e "\n${BOLD}${YELLOW}▶  $1${NC}\n"; }
ok()      { echo -e "${GREEN}  ✅  $1${NC}"; }
fail()    { echo -e "${RED}  ✗   $1${NC}"; }

echo -e "${BOLD}${GREEN}"
divider
echo "  🚀  GPU LLM BENCHMARK SUITE"
echo "  $(date)"
echo "  Log: $LOG"
divider
echo -e "${NC}"

cd "$DIR"

# ── 1. GPU Info & VRAM capability table ──────────────────────────────────────
step "Step 1/5 — GPU Info & LLM Capability Table"
conda run -n "$CONDA_ENV" python "$DIR/01_gpu_info.py" 2>&1 | tee -a "$LOG" \
  && ok "GPU info done" || fail "GPU info failed"

# ── 2. Raw compute & memory bandwidth ─────────────────────────────────────────
step "Step 2/5 — Raw GPU Compute, FLOPS & Memory Bandwidth"
conda run -n "$CONDA_ENV" python "$DIR/02_raw_gpu_benchmark.py" 2>&1 | tee -a "$LOG" \
  && ok "Raw benchmark done" || fail "Raw benchmark failed"

# ── 3. LLM Inference (multi-family, fp16 + 4-bit NF4) ─────────────────────────
step "Step 3/5 — LLM Inference Benchmark (fp16 + 4-bit)"
echo "  Note: models are downloaded from HuggingFace on first run."
echo "  Default families come from config.env (DEFAULT_FAMILIES)."
if $QUICK; then
  conda run -n "$CONDA_ENV" python "$DIR/03_llm_benchmark.py" --quick --no-access-check 2>&1 | tee -a "$LOG"
else
  conda run -n "$CONDA_ENV" python "$DIR/03_llm_benchmark.py" --no-access-check 2>&1 | tee -a "$LOG"
fi
ok "LLM inference benchmark done" || fail "LLM inference benchmark failed"

# ── 4. Fine-tuning benchmark (Full FT + LoRA + QLoRA on real data) ────────────
step "Step 4/5 — Fine-tuning Benchmark (Full FT + LoRA + QLoRA)"
echo "  Dataset: tatsu-lab/alpaca (52K instruction pairs)"
echo "  Methods: Full Fine-tuning, LoRA (r=16), QLoRA (4-bit NF4 + LoRA)"
if $QUICK; then
  conda run -n "$CONDA_ENV" python "$DIR/04_finetune_benchmark.py" --quick --dataset alpaca 2>&1 | tee -a "$LOG"
else
  conda run -n "$CONDA_ENV" python "$DIR/04_finetune_benchmark.py" --dataset alpaca 2>&1 | tee -a "$LOG"
fi
ok "Fine-tuning benchmark done" || fail "Fine-tuning benchmark failed"

# ── 5. Email results ───────────────────────────────────────────────────────────
if $SEND_EMAIL; then
  step "Step 5/5 — Email Results"
  echo "  Sending report…  (recipient from config.env → EMAIL_TO)"
  conda run -n "$CONDA_ENV" python "$DIR/05_email_results.py" 2>&1 | tee -a "$LOG" \
    && ok "Email sent" || fail "Email failed (check SMTP settings in config.env)"
else
  echo -e "\n${YELLOW}  ⚠  Email skipped (--no-email flag){NC}"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
divider
echo -e "${BOLD}${GREEN}  ✅  All benchmarks complete!${NC}"
echo -e "  Results saved:"
ls -1 "$DIR"/results_*.json 2>/dev/null | sed 's/^/    • /'
echo -e "  Full log: $LOG"
divider
