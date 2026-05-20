# PW-OPSD: Position-Weighted On-Policy Self-Distillation for Reasoning

Official code release for the paper

> **When Are Teacher Tokens Reliable? Position-Weighted On-Policy
> Self-Distillation for Reasoning.**
> Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, Chaowei Xiao.

This repository reproduces the four training methods (OPSD, EOPD, REOPOLD,
PW-OPSD) and the 38k-token reasoning evaluation suite (MATH-500,
AIME 2024, AIME 2025, HMMT 2025) on three base models: Qwen3-4B,
DeepSeek-R1-Distill-Llama-8B, and Olmo-3-7B-Think.

## Main results (Avg@12, 38k-token regime, 3 evaluation seeds)

Entries report mean ± across-seed sample standard deviation over three
evaluation seeds. The `Avg@12` column is the equal-weight mean of the
four per-benchmark Avg@12 values.

### Qwen3-4B

| Method            | MATH-500       | AIME 2024      | AIME 2025      | HMMT 2025      | Avg@12  |
|-------------------|----------------|----------------|----------------|----------------|---------|
| OPSD              | 95.33 ± 0.08   | 75.19 ± 0.42   | 66.67 ± 1.27   | 43.89 ± 0.73   | 70.27   |
| EOPD              | 95.33 ± 0.08   | 73.61 ± 2.17   | 65.65 ± 0.89   | 41.94 ± 2.00   | 69.13   |
| REOPOLD           | 95.09 ± 0.09   | 73.98 ± 0.42   | 62.13 ± 1.67   | 41.39 ± 1.44   | 68.15   |
| PW-OPSD Moderate (**ours**) | 95.34 ± 0.10 | **76.20 ± 0.58** | **67.78 ± 1.27** | 43.33 ± 1.21 | 70.66 |
| PW-OPSD Aggressive (ours)   | **95.53 ± 0.04** | 75.19 ± 0.85 | 67.59 ± 0.85 | **45.37 ± 0.58** | **70.92** |

### DeepSeek-R1-Distill-Llama-8B

| Method            | MATH-500       | AIME 2024      | AIME 2025      | HMMT 2025      | Avg@12  |
|-------------------|----------------|----------------|----------------|----------------|---------|
| OPSD              | **89.02 ± 0.19** | 40.74 ± 2.58 | 31.57 ± 1.28   | 20.93 ± 0.80   | 45.56   |
| EOPD              | 88.84 ± 0.30   | 43.06 ± 1.69   | 30.65 ± 0.58   | 20.93 ± 1.43   | **45.87** |
| REOPOLD           | 88.59 ± 0.36   | **43.43 ± 2.74** | 30.65 ± 1.40 | 18.98 ± 0.42   | 45.41   |
| PW-OPSD Moderate (**ours**) | 88.72 ± 0.25 | 41.20 ± 0.32 | **32.22 ± 0.56** | **21.48 ± 0.42** | **45.91** |

Across both base models, PW-OPSD with the Moderate schedule
`(w_min, tau, s) = (0.25, 0.30, 0.10)` (held fixed across models)
delivers the highest Avg@12 among the four compared methods.

## Methods

| Method     | `--method` flag | What it does                                                |
|------------|-----------------|-------------------------------------------------------------|
| OPSD       | `opsd`          | Uniform forward-KL self-distillation (Zhao et al., 2026).   |
| EOPD       | `eopd`          | Entropy-conditioned RKL/FKL mixture (Jin et al., 2026).     |
| REOPOLD    | `reopold`       | Relaxed on-policy distillation, policy-gradient form (Ko et al., 2026). |
| PW-OPSD    | `pwopsd`        | **Ours** — position-weighted FKL with per-sequence reduction. |

## Repository layout

```
public_release/
├── README.md
├── code/
│   ├── ea_opsd_train.py         # training entry point (all four methods)
│   ├── ea_opsd_trainer.py       # extended trainer with --method-based dispatch
│   ├── opsd_trainer.py          # base OPSD trainer (parent class)
│   ├── data_collator.py         # privileged-prompt + chat-template collator
│   ├── merge_lora.py            # LoRA -> merged-checkpoint utility
│   ├── accelerate_4gpu.yaml     # accelerate config (4-GPU DDP)
│   └── environment.yml          # conda env spec (creates env named `pwopsd`)
├── eval/
│   ├── evaluate_math.py         # vLLM-based evaluator (Pass@N / Avg@N / Maj@N)
│   └── recompute_majn.py        # offline Maj@N re-aggregation utility
└── launchers/
    ├── run_qwen4b_base_{opsd,eopd,pwopsd,reopold}_train.sh
    ├── run_dsr1_l8b_{opsd,eopd,pwopsd,reopold}_train.sh
    ├── run_olmo3_7b_think_{opsd,eopd,pwopsd,reopold}_train.sh
    └── run_eval.sh
```

## Setup

1. Create the conda env (Python 3.10 + PyTorch 2.8 + TRL 0.26 + vLLM 0.11):
   ```bash
   conda env create -f code/environment.yml
   conda activate pwopsd
   ```
2. Set the WandB API key, or run WandB in offline mode:
   ```bash
   export WANDB_API_KEY=<your wandb api key>
   # or, to skip the API-key check entirely:
   export WANDB_MODE=offline
   ```
   The launchers abort early with a clear error if neither `WANDB_API_KEY`
   nor `WANDB_MODE=offline|disabled` is set.
3. (Optional) Override the training dataset:
   ```bash
   export PWOPSD_TRAIN_DATASET="siyanzhao/Openthoughts_math_30k_opsd"  # default
   ```
   The trainer reads only the `problem` and `solution` string columns
   from the dataset's train split, so any Hugging Face dataset that
   exposes those two columns can be plugged in via `PWOPSD_TRAIN_DATASET`.
4. (Optional) For HMMT February 2025 evaluation, point at a local parquet
   derived from the MathArena `hmmt_feb_2025` release:
   ```bash
   export HMMT25_PARQUET_PATH=/path/to/hmmt25_clean/train.parquet
   ```

The launchers resolve `REPO_ROOT` automatically from the launcher script's
own location (one directory above `launchers/`), so they should be run
from the repository top level or via an absolute path:
```bash
bash launchers/run_qwen4b_base_pwopsd_train.sh
```

## Training

Each launcher invokes `ea_opsd_train.py` with the shared hyperparameters
(LoRA r=64, alpha=128, lr=5e-6, distillation temperature 1.1,
forward-KL clip 0.05, fixed teacher) and dispatches to the corresponding
method via the `--method` flag. The PW-OPSD launcher additionally
passes the Moderate schedule
`--position_w_min 0.25 --position_tau 0.30 --position_s 0.10`, and the
REOPOLD launcher passes the published reward-floor and phase-mask
hyperparameters.

All trainings use effective batch size 32 on 4 GPUs and run for 150
optimizer steps with a save every 25 steps. *The launchers merge and
evaluate `checkpoint-100`* — the paper protocol selects the step-100
snapshot for all reported numbers. The per-GPU batch / gradient-accumulation
split is chosen per base-model size (Qwen3-4B uses `bs=4 x ga=2`, DSR1-L8B
and Olmo-3-7B-Think use `bs=2 x ga=4`). The Moderate position schedule
itself is identical across the three models.

```bash
# Qwen3-4B base
bash launchers/run_qwen4b_base_opsd_train.sh
bash launchers/run_qwen4b_base_eopd_train.sh
bash launchers/run_qwen4b_base_reopold_train.sh
bash launchers/run_qwen4b_base_pwopsd_train.sh

# DeepSeek-R1-Distill-Llama-8B
bash launchers/run_dsr1_l8b_opsd_train.sh
bash launchers/run_dsr1_l8b_eopd_train.sh
bash launchers/run_dsr1_l8b_reopold_train.sh
bash launchers/run_dsr1_l8b_pwopsd_train.sh

# Olmo-3-7B-Think
bash launchers/run_olmo3_7b_think_opsd_train.sh
bash launchers/run_olmo3_7b_think_eopd_train.sh
bash launchers/run_olmo3_7b_think_reopold_train.sh
bash launchers/run_olmo3_7b_think_pwopsd_train.sh
```

Each launcher trains the LoRA adapter at `outputs/<run_name>/` and then
merges it into a single HuggingFace-loadable checkpoint at
`outputs/<run_name>_merged/`, then writes a `.ready_for_eval` sentinel.
`<run_name>` follows `{model_tag}_{method}` (e.g. `qwen3_4b_base_pwopsd`,
`dsr1_l8b_pwopsd`, `olmo3_7b_think_pwopsd`).

### Note on reasoning-mode bases

DSR1-Llama-8B and Olmo-3-7B-Think are released as reasoning-distilled
models whose chat templates unconditionally append an open
`<think>` tag and ignore `enable_thinking=False`. To match OPSD's
non-thinking-mode prompt format on these bases, the data collator
detects an unclosed trailing `<think>` and appends `</think>`,
restoring the intended `enable_thinking=False` semantics. The patch is
a no-op for templates (e.g. Qwen3-4B base) that already honor the flag.

## Evaluation (38k-token regime, three random seeds)

```bash
# Usage: run_eval.sh <method_name> <dataset> [tp] [seed]
bash launchers/run_eval.sh qwen3_4b_base_pwopsd       math500
bash launchers/run_eval.sh qwen3_4b_base_pwopsd       aime24
bash launchers/run_eval.sh dsr1_l8b_pwopsd            aime25  4  1
bash launchers/run_eval.sh olmo3_7b_think_pwopsd      hmmt25  4  2
```

Accepted `<method_name>` values are any of the `<run_name>`s listed
above; accepted `<dataset>` values are `math500`, `aime24`, `aime25`,
`hmmt25`. The eval launcher writes JSONs of the form
`eval_results/<method_name>_38k.json` (or
`eval_results/<method_name>_<dataset>_38k.json`) and per-run logs under
`eval_results/logs/`. Override `HOSTLABEL` to disambiguate logs across
parallel workers.

`evaluate_math.py` reports Pass@12 / Avg@12 / Maj@12 with
math-equivalence clustering (`sympy` + `math_verify`). The `INVALID`
cluster participates in the plurality count and is scored incorrect
when selected (formal definitions in the paper appendix).

`recompute_majn.py` re-aggregates Maj@12 over existing eval JSONs using
the same fixed clustering rule.

## Datasets

- **MATH-500** — `HuggingFaceH4/MATH-500`
- **AIME 2024** — `HuggingFaceH4/aime_2024`
- **AIME 2025** — `yentinglin/aime_2025`
- **HMMT February 2025** — locally cleaned parquet derived from the
  MathArena `hmmt_feb_2025` release (https://github.com/eth-sri/matharena).
  The evaluator expects a parquet with at least two string columns,
  `problem` and `answer`, holding the 30 HMMT February 2025 problems.
  Set `HMMT25_PARQUET_PATH` to point at it
  (default `./data/hmmt25_clean/train.parquet`).

### Downloading models and datasets

The launchers default `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0`,
so Hugging Face downloads happen on first use; export
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` to force the offline
cache once the assets are downloaded. The training launchers pull the
base model identified by `MODEL=...` inside the script
(`Qwen/Qwen3-4B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, or
`allenai/Olmo-3-7B-Think`) and the dataset selected by
`PWOPSD_TRAIN_DATASET`. The MATH-500, AIME 2024, and AIME 2025
benchmarks are downloaded automatically by `evaluate_math.py` from
their Hugging Face dataset IDs; only HMMT 2025 needs the local parquet
described above.

## Hardware

All paper experiments use 4×H100 80GB.

## License

This code is released under the MIT License; see `LICENSE` for the full
text. Upstream model and dataset assets remain under their original
licenses as declared on Hugging Face and GitHub.

## Citation

```bibtex
TBD
```
