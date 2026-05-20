"""
PW-OPSD Training Script.

Extends OPSD with Bayesian uncertainty decomposition to weight the distillation
loss per-token, suppressing noisy teacher signal and preserving task diversity.
"""
import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from ea_opsd_trainer import EAOPSDTrainer
from dataclasses import dataclass, field

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

@dataclass
class EAScriptArguments(ScriptArguments):
    """Extended script arguments for PW-OPSD."""

    # ---- Original OPSD args ----
    use_tinker_loss: bool = field(
        default=False,
        metadata={"help": "Use Thinking Machines style on-policy reverse KL loss."},
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={"help": "Use the initial policy (step 0) as a fixed teacher. Requires use_peft=True."},
    )
    run_config: str = field(
        default=None,
        metadata={"help": "Custom run name for output directory and WandB."},
    )
    presence_penalty: float = field(default=0.0, metadata={"help": "vLLM presence penalty."})
    reason_first: bool = field(
        default=False,
        metadata={"help": "Teacher first rationalizes the solution before evaluating."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD to top-k teacher tokens. 0 = full vocab."},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Clip per-token JSD to max value. 0 = no clipping."},
    )
    use_ema_teacher: bool = field(default=False, metadata={"help": "Use EMA teacher."})
    ema_decay: float = field(default=0.999, metadata={"help": "EMA decay factor."})

    # ---- Method selector (required) ----
    method: str = field(
        default=None,
        metadata={"help": "Required. One of: opsd | eopd | pwopsd | reopold. "
                          "Selects the per-token training objective."},
    )
    # ---- Position-weighted OPSD (PW-OPSD) ----
    position_w_min: float = field(
        default=0.25,
        metadata={"help": "Floor weight for early tokens (1=no down-weighting, 0=zero out)."},
    )
    position_tau: float = field(
        default=0.30,
        metadata={"help": "Sigmoid threshold (fraction of max_completion_length); below this, weight ~ w_min."},
    )
    position_s: float = field(
        default=0.10,
        metadata={"help": "Sigmoid sharpness (smaller = harder transition)."},
    )
    position_global_reduction: bool = field(
        default=False,
        metadata={"help": "2x2 ablation: when True, reduce the position-weighted loss with global token mean instead of per-sequence mean (then batch mean). Default False (paper PWOPSD)."},
    )
    # ---- REOPOLD: Relaxed On-Policy Distillation ----
    reopold_lambda: float = field(
        default=0.1,
        metadata={"help": "Mixture-clip coefficient lambda in (0,1). Floor for clipped reward = log(lambda)/(1-lambda). Smaller -> tighter clip, less aggressive negatives."},
    )
    reopold_beta: float = field(
        default=0.2,
        metadata={"help": "Top-fraction of high-entropy student tokens kept in refinement phase (Phase II mask). Paper recommended 0.2."},
    )
    reopold_t_switch: int = field(
        default=50,
        metadata={"help": "Global step at which Phase I (exploration) -> Phase II (refinement). For a 100-step run, 50 = halfway, mirroring the paper's 150/300 split."},
    )

if __name__ == "__main__":
    parser = TrlParser((EAScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # Resolve --method NAME to the internal training-objective dispatch code.
    METHOD_TO_CODE = {"opsd": 0, "eopd": -3, "pwopsd": -12, "reopold": -13}
    method = getattr(script_args, "method", None)
    if method is None:
        raise SystemExit(
            f"--method is required; pass one of {sorted(METHOD_TO_CODE)}"
        )
    if method not in METHOD_TO_CODE:
        raise SystemExit(
            f"--method must be one of {sorted(METHOD_TO_CODE)}, got {method!r}"
        )
    script_args.mc_samples = METHOD_TO_CODE[method]


    ################
    # WandB Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        mc_tag = f"_mc{script_args.mc_samples}" if script_args.mc_samples > 0 else ""
        fix_tag = "_fixteach" if script_args.fixed_teacher else ""

        full_wandb_run_config = (
            f"pwopsd_{model_name}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
            f"{mc_tag}{fix_tag}"
        )

    print(f"\n{'='*80}")
    print(f"PW-OPSD RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"MC Samples: {script_args.mc_samples}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                # Model
                "model_name": model_args.model_name_or_path,
                "method": script_args.method,
                # Training
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                # PEFT
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                # Teacher
                "fixed_teacher": script_args.fixed_teacher,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                # PW-OPSD
                "mc_samples": script_args.mc_samples,
                "position_w_min": script_args.position_w_min,
                "position_tau": script_args.position_tau,
                "position_s": script_args.position_s,
                "position_global_reduction": script_args.position_global_reduction,
                "reopold_lambda": script_args.reopold_lambda,
                "reopold_beta": script_args.reopold_beta,
                "reopold_t_switch": script_args.reopold_t_switch,
                "num_processes": num_processes,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "sdpa",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    training_args.presence_penalty = script_args.presence_penalty

    dataset_id = os.environ.get("PWOPSD_TRAIN_DATASET", "open-thoughts/OpenThoughts-Math-30K")
    dataset = load_dataset(dataset_id)
    train_dataset = dataset["train"]

    ################
    # Trainer
    ################
    trainer = EAOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        # PW-OPSD specific
        mc_samples=script_args.mc_samples,
        position_w_min=script_args.position_w_min,
        position_tau=script_args.position_tau,
        position_s=script_args.position_s,
        position_global_reduction=script_args.position_global_reduction,
        reopold_lambda=script_args.reopold_lambda,
        reopold_beta=script_args.reopold_beta,
        reopold_t_switch=script_args.reopold_t_switch,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
