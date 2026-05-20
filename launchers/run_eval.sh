#!/bin/bash
# Evaluate ONE merged checkpoint on ONE dataset at the 38k regime, with an
# optional seed for repeated-sampling runs.
# Usage: bash run_eval.sh <method_name> <dataset> [tp] [seed]
#   method_name:  qwen3_4b_base_{opsd|eopd|pwopsd|reopold}
#               | dsr1_l8b_{opsd|eopd|pwopsd|reopold}
#               | olmo3_7b_think_{opsd|eopd|pwopsd|reopold}
#   dataset:     math500 | aime24 | aime25 | hmmt25
#   tp:          tensor parallel size (default 4)
#   seed:        optional integer seed for vLLM SamplingParams. When given,
#                output filename gets a _seed{N} suffix and --seed N is
#                passed through.
#
# The launcher uses flock + tmp+atomic-mv + JSON validation so a stale
# JSON from a different sampling regime cannot satisfy the skip check.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"
DATASET="${2:-}"
TP="${3:-4}"
SEED="${4:-}"          # optional; empty = no --seed (legacy)
HOSTLABEL="${HOSTLABEL:-worker}"

if [ -z "$NAME" ] || [ -z "$DATASET" ]; then
  echo "Usage: bash $0 <method_name> <dataset> [tp] [seed]" >&2; exit 2
fi
case "$NAME" in
  qwen3_4b_base_opsd|qwen3_4b_base_eopd|qwen3_4b_base_pwopsd|qwen3_4b_base_reopold) ;;
  dsr1_l8b_opsd|dsr1_l8b_eopd|dsr1_l8b_pwopsd|dsr1_l8b_reopold) ;;
  olmo3_7b_think_opsd|olmo3_7b_think_eopd|olmo3_7b_think_pwopsd|olmo3_7b_think_reopold) ;;
  *) echo "ERROR: unknown method '$NAME'" >&2; exit 2;;
esac
if [ -n "$SEED" ] && ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: seed must be a non-negative integer, got '$SEED'" >&2; exit 2
fi
case "$DATASET" in
  math500|aime24|aime25|hmmt25) ;;
  *) echo "ERROR: unknown dataset '$DATASET'" >&2; exit 2;;
esac

SENTINEL="$REPO_ROOT/outputs/.${NAME}_ready_for_eval"
if [ ! -f "$SENTINEL" ]; then
  echo "[$(date)] ERROR: sentinel $SENTINEL not present — aborting." >&2; exit 3
fi

MERGED="$REPO_ROOT/outputs/${NAME}_merged"
if [ ! -f "$MERGED/config.json" ]; then
  echo "[$(date)] ERROR: merged model not found at $MERGED" >&2; exit 3
fi
if ! find "$MERGED" -maxdepth 1 -name "*.safetensors" -size +0 | grep -q . ; then
  echo "[$(date)] ERROR: $MERGED has no non-empty .safetensors shards" >&2; exit 3
fi

cd "$REPO_ROOT/code"
eval "$(conda shell.bash hook)"
conda activate pwopsd
if [ "${WANDB_MODE:-}" != "offline" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then : "${WANDB_API_KEY:?Set WANDB_API_KEY env var (or use WANDB_MODE=offline)}"; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
export TRITON_CACHE_DIR="${HOME}/.cache/triton"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export TORCH_HOME="${HOME}/.cache/torch"
export XDG_CACHE_HOME="${HOME}/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$REPO_ROOT/eval_results/logs"
mkdir -p "$LOG_DIR"

SEED_SUFFIX=""
if [ -n "$SEED" ]; then
  SEED_SUFFIX="_seed${SEED}"
fi
if [ "$DATASET" = "math500" ]; then
  OUTFILE="$REPO_ROOT/eval_results/${NAME}_38k${SEED_SUFFIX}.json"
else
  OUTFILE="$REPO_ROOT/eval_results/${NAME}_${DATASET}_38k${SEED_SUFFIX}.json"
fi
LOCK="$OUTFILE.lock"
LOGFILE="$LOG_DIR/${NAME}_${DATASET}_38k${SEED_SUFFIX}.${HOSTLABEL}.log"

validate_eval_json() {
  # args: file, dataset, optional expected-seed (empty = don't check)
  # Pins the full eval regime (dataset, val_n, max_new_tokens, temperature,
  # top_p, top_k, num_problems, model dir) so a stale JSON from a different
  # regime cannot satisfy the skip check.
  python - "$1" "$2" "${3:-}" "$MERGED" <<'PY' >/dev/null 2>&1
import json, sys, math
f, ds, expected_seed, merged = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(f))
assert d.get("dataset") == ds, f"dataset mismatch: {d.get('dataset')} vs {ds}"
assert int(d.get("val_n", -1)) == 12
assert int(d.get("max_new_tokens", -1)) == 38912
assert math.isclose(float(d.get("temperature", -1.0)), 1.0, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(d.get("top_p", -1.0)), 0.95, rel_tol=0.0, abs_tol=1e-9)
assert int(d.get("top_k", -999)) == -1
# Pin remaining sampling-time knobs so a stale JSON from a different regime is rejected.
assert math.isclose(float(d.get("min_p", -1.0)), 0.0, rel_tol=0.0, abs_tol=1e-9), f"min_p mismatch: {d.get(chr(39)+chr(109)+chr(105)+chr(110)+chr(95)+chr(112)+chr(39))}"
assert math.isclose(float(d.get("presence_penalty", -1.0)), 0.0, rel_tol=0.0, abs_tol=1e-9), f"presence_penalty mismatch: {d.get(chr(39)+chr(112)+chr(114)+chr(101)+chr(115)+chr(101)+chr(110)+chr(99)+chr(101)+chr(95)+chr(112)+chr(101)+chr(110)+chr(97)+chr(108)+chr(116)+chr(121)+chr(39))}"
assert d.get("enable_thinking", False) is True, f"enable_thinking must be True for paper evals"
assert "average_at_n_pct" in d and "pass_at_n_pct" in d and "majority_vote_at_n_pct" in d
# num_problems sanity: each dataset has a fixed expected size.
expected_n = {"math500": 500, "aime24": 30, "aime25": 30, "hmmt25": 30}.get(ds)
got_n = int(d.get("num_problems", -1))
assert expected_n is None or got_n == expected_n, f"num_problems {got_n} vs expected {expected_n}"
# Model identity: stored base_model path must match the merged dir we'd run now.
got_model = d.get("base_model", "")
assert got_model == merged, f"base_model mismatch: {got_model} vs {merged}"
if expected_seed != "":
    got = d.get("seed", None)
    assert got is not None and int(got) == int(expected_seed), \
        f"seed mismatch: expected {expected_seed}, got {got}"
PY
}

cd "$REPO_ROOT/eval"
(
  if ! flock -n 9; then
    echo "[$(date)] SKIP ($HOSTLABEL): lock held for $NAME/$DATASET" | tee -a "$LOGFILE"
    exit 0
  fi
  if [ -f "$OUTFILE" ]; then
    if validate_eval_json "$OUTFILE" "$DATASET" "$SEED"; then
      echo "[$(date)] SKIP ($HOSTLABEL): $OUTFILE exists and validates" | tee -a "$LOGFILE"
      exit 0
    else
      echo "[$(date)] WARN ($HOSTLABEL): $OUTFILE exists but invalid; removing" | tee -a "$LOGFILE"
      rm -f "$OUTFILE"
    fi
  fi
  echo "" | tee -a "$LOGFILE"
  echo "============ EVAL 38k ($HOSTLABEL TP=$TP): $NAME on $DATASET  $(date) ============" | tee -a "$LOGFILE"
  TMP="${OUTFILE}.tmp.${HOSTLABEL}.$$"
  SEED_ARG=""
  if [ -n "$SEED" ]; then
    SEED_ARG="--seed $SEED"
  fi
  # NB: $SEED_ARG is intentionally UNQUOTED so empty expands to nothing.
  # Pass enable_thinking / min_p / presence_penalty explicitly so the JSON
  # validator (which pins these to True / 0.0 / 0.0) keeps matching even if
  # evaluate_math.py defaults drift in the future.
  NCCL_P2P_DISABLE=1 python evaluate_math.py \
    --base_model "$MERGED" --dataset "$DATASET" --val_n 12 --temperature 1.0 \
    --top_p 0.95 --top_k -1 \
    --min_p 0.0 --presence_penalty 0.0 --enable_thinking \
    --max_new_tokens 38912 --tensor_parallel_size $TP \
    $SEED_ARG \
    --output_file "$TMP" 2>&1 | tee -a "$LOGFILE"
  if [ ! -f "$TMP" ]; then
    echo "[$(date)] ERROR ($HOSTLABEL): $TMP not produced" | tee -a "$LOGFILE" >&2
    exit 1
  fi
  if ! validate_eval_json "$TMP" "$DATASET" "$SEED"; then
    echo "[$(date)] ERROR ($HOSTLABEL): post-eval validation failed" | tee -a "$LOGFILE" >&2
    mv "$TMP" "${TMP}.invalid"
    exit 1
  fi
  mv "$TMP" "$OUTFILE"
  echo "[$(date)] SAVED ($HOSTLABEL): $OUTFILE" | tee -a "$LOGFILE"
) 9>"$LOCK"

echo "[$(date)] DONE: $NAME on $DATASET ($HOSTLABEL)"
