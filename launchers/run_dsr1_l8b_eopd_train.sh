#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B + EOPD baseline (method=eopd): paper-faithful EOPD
# implementation (entropy-conditioned per-token reverse-KL + top-k forward-KL).
# Output: outputs/dsr1_l8b_eopd/ (LoRA adapter)
#         outputs/dsr1_l8b_eopd_merged/ (merged full model)
#
# EOPD hyperparameters baked into the trainer:
#   - tau = 0.8 (entropy threshold above which forward KL is added)
#   - top-k = 16 (forward-KL truncation support)
#   - PPO clip eps = 0.2 (PPO Eq 8 default)
# L^OPD matches paper Eq 7+8 (PPO-clipped per-sampled-token reverse-KL).
# L^FKL matches paper Eq 10 (top-k forward KL with teacher renormalized
# over its own top-k support and student using its full-vocab probability
# at those indices).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/code"
eval "$(conda shell.bash hook)"
conda activate pwopsd
if [ "${WANDB_MODE:-}" != "offline" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then : "${WANDB_API_KEY:?Set WANDB_API_KEY env var (or use WANDB_MODE=offline)}"; fi
export WANDB_PROJECT="${WANDB_PROJECT:-pwopsd}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
export TRITON_CACHE_DIR="${HOME}/.cache/triton"
export VLLM_CACHE_DIR="${HOME}/.cache/vllm"
export TORCH_HOME="${HOME}/.cache/torch"
export XDG_CACHE_HOME="${HOME}/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENFORCE_EAGER=0

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

validate_merged_dir() {
  local d="$1"
  [ -f "$d/config.json" ] || return 1
  local n_shards
  n_shards=$(find "$d" -maxdepth 1 -name "*.safetensors" -size +0 -printf 1 2>/dev/null | wc -c)
  [ "$n_shards" -ge 1 ] || return 1
  [ -f "$d/tokenizer.json" ] || [ -f "$d/tokenizer.model" ] || [ -f "$d/tokenizer_config.json" ] || return 1
  return 0
}

mark_ready() {
  local NAME="$1"
  local MERGED="$REPO_ROOT/outputs/${NAME}_merged"
  date > "$MERGED/.complete"
  date > "$REPO_ROOT/outputs/.${NAME}_ready_for_eval"
  echo "[$(date)] sentinel: $MERGED/.complete and outputs/.${NAME}_ready_for_eval"
}

train_and_merge() {
  local NAME="$1"; shift
  local PORT=$((29500 + RANDOM % 200))
  local MERGED="$REPO_ROOT/outputs/${NAME}_merged"

  if [ -d "$MERGED" ]; then
    if validate_merged_dir "$MERGED"; then
      echo "[$(date)] SKIP train+merge: ${NAME}_merged already exists and validates"
      mark_ready "$NAME"
      return 0
    else
      echo "[$(date)] WARN: ${NAME}_merged exists but is incomplete; will RE-MERGE"
      rm -rf "$MERGED"
    fi
  fi

  echo ""; echo "============ TRAINING: $NAME on $MODEL  $(date) ============"
  cd "$REPO_ROOT/code"
  accelerate launch --config_file "$REPO_ROOT/code/accelerate_4gpu.yaml" \
    --num_processes 4 --gradient_accumulation_steps 4 --main_process_port $PORT \
    ea_opsd_train.py --model_name_or_path "$MODEL" "$@" \
    --learning_rate 5e-6 --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --gradient_checkpointing --num_train_epochs 30 --max_steps 150 \
    --max_completion_length 1024 --save_steps 25 --logging_steps 2 \
    --attn_implementation sdpa --torch_dtype bfloat16 --max_length 20000 \
    --beta 0 --use_vllm --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.30 --vllm_tensor_parallel_size 1 \
    --use_peft --lora_r 64 --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 --top_p 0.95 --top_k 20 --lmbda 1 \
    --fixed_teacher --jsd_token_clip 0.05 \
    --wandb_project ${WANDB_PROJECT:-pwopsd} \
    --output_dir "$REPO_ROOT/outputs/$NAME" --run_config "$NAME"

  ADAPTER="$REPO_ROOT/outputs/$NAME/checkpoint-100"
  echo "[$(date)] ============ MERGE: $NAME -> $MERGED ============"
  cd "$REPO_ROOT/code" && python merge_lora.py "$MODEL" "$ADAPTER" "$MERGED"

  if ! validate_merged_dir "$MERGED"; then
    echo "[$(date)] ERROR: merge of $NAME did not produce a valid merged dir at $MERGED"
    return 2
  fi
  mark_ready "$NAME"
}

train_and_merge "dsr1_l8b_eopd" \
    --method eopd

echo "[$(date)] EOPD TRAIN+MERGE COMPLETE"
