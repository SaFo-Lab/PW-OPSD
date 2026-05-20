"""
PW-OPSD Trainer (Position-Weighted On-Policy Self-Distillation).

Extends OPSDTrainer with a per-token reliability weight. The four
production methods evaluated in the paper main results table are
dispatched inside compute_loss based on the internal method-code
selected by --method (opsd, eopd, pwopsd, reopold).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Any

from accelerate.utils import is_peft_model
from trl.trainer.utils import empty_cache
from opsd_trainer import OPSDTrainer

class EAOPSDTrainer(OPSDTrainer):
    """OPSDTrainer extended with epistemic-aware uncertainty weighting."""

    _tag_names = ["trl", "pwopsd"]
    _name = "PW-OPSD"

    def __init__(
        self,
        # PW-OPSD specific args
        mc_samples: int = 5,
        # ---- Position-weighted OPSD (PW-OPSD) ----
        # Weight per-token KL by sigmoid of token position. Down-weights early tokens
        # (path-decision branching) and keeps full weight on late tokens (execution).
        position_w_min: float = 0.25,
        position_tau: float = 0.30,
        position_s: float = 0.10,
        # 2x2 ablation: when True, use the legacy "global token mean" reduction
        # over the position-weighted per-token loss instead of per-sequence mean
        # then batch mean. Used to isolate the position schedule's effect from
        # the length-normalization effect of the per-sequence reduction.
        # Default False = current paper behavior (per-sequence mean).
        position_global_reduction: bool = False,
        # ---- REOPOLD: Relaxed On-Policy Distillation ----
        # Ko et al., 2026 (arXiv:2603.11137). Policy-gradient form with mixture-clip
        # reward + phase-dependent token mask (exploration -> refinement).
        # Defaults follow paper: lambda=0.1 (floor=log(0.1)/0.9 ~= -2.56),
        # beta=0.2 (top 20% high-entropy tokens in refinement),
        # t_switch=50 (half-way through a 100-step run; paper used 150/300).
        reopold_lambda: float = 0.1,
        reopold_beta: float = 0.2,
        reopold_t_switch: int = 50,
        # Pass-through to OPSDTrainer
        **kwargs,
    ):
        # All four methods are compatible with use_thinking_machines_loss; no
        # extra guard is needed here.

        super().__init__(**kwargs)

        self.mc_samples = mc_samples
        self.position_w_min = position_w_min
        self.position_tau = position_tau
        self.position_s = position_s
        self.position_global_reduction = position_global_reduction
        self.reopold_lambda = reopold_lambda
        self.reopold_beta = reopold_beta
        self.reopold_t_switch = reopold_t_switch


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute PW-OPSD loss.

        Modes:
        Dispatches on the internal method-code integer set by --method.
        """
        if self.mc_samples == 0:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        student_prompt_len = inputs["student_prompt_length"]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits_for_loss = outputs_student.logits[:, student_prompt_len - 1: -1, :]
        del outputs_student
        empty_cache()

        if self.mc_samples == -3:
            # === BASELINE: Entropy-Aware OPD (Jin et al., 2026) ===
            # L = L_RKL + I[H_teacher > tau] * L_FKL (top-k=16 approximation for FKL)
            # tau=0.8 as in EOPD paper
            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            teacher_prompt_len = inputs["teacher_prompt_length"]
            with torch.no_grad(), adapter_context:
                outputs_teacher = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1: -1, :].clone()
                del outputs_teacher
            empty_cache()

            # === EOPD per Jin et al. 2026 (arXiv:2603.07079), Eq 7-10 ===
            # L^EOPD_t = L^OPD_t + 1[H^te_t > τ] · L^FKL_t       (Eq 9)
            # L^OPD: PPO-clipped reverse-KL on the SAMPLED token   (Eq 7+8)
            # L^FKL: top-k forward KL with teacher renormalized over its top-k
            #        and STUDENT using its FULL-vocab probability  (Eq 10)
            # Paper defaults: τ = 0.8, top-k = 16; ε = 0.2 (standard PPO default;
            # paper Eq 8 uses ε without specifying its numeric value).
            s_t = student_logits_for_loss.float() / self.temperature
            t_t = teacher_logits.float() / self.temperature

            # Full-vocab log-probs (paper does NOT truncate at loss time;
            # truncation is only at sampling time)
            s_lp_full = F.log_softmax(s_t, dim=-1)                                   # [B,T,V] grad
            with torch.no_grad():
                t_lp_full = F.log_softmax(t_t, dim=-1)                               # [B,T,V] no grad
                t_probs_full = t_lp_full.exp()

            # --- L^OPD: PPO-clipped per-token loss on the sampled token (Eq 7+8) ---
            mask_label = (shifted_labels != -100)                                    # [B,T] bool
            safe_labels = shifted_labels.clamp(min=0)                                # [B,T] long
            log_p_S_new = s_lp_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T] grad
            log_p_S_old = log_p_S_new.detach()                                       # pure-on-policy: same θ
            with torch.no_grad():
                log_p_T = t_lp_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
            # Per-token advantage Â_t = log π_te − log π_θ_old, detached
            A_hat = (log_p_T - log_p_S_old).detach()
            # Importance ratio r_t = π_θ / π_θ_old, autograd through log π_θ
            r_t = torch.exp(log_p_S_new - log_p_S_old)                               # [B,T] grad
            ppo_eps = 0.2  # standard PPO default; paper Eq 8 uses ε without specifying its numeric value
            clipped_r = torch.clamp(r_t, 1.0 - ppo_eps, 1.0 + ppo_eps)
            # Eq 8: Ã_t = max(-r·Â, -clip(r,...)·Â) — pessimistic (= larger of the two negatives)
            L_OPD = torch.maximum(-r_t * A_hat, -clipped_r * A_hat)                  # [B,T]

            # --- L^FKL: top-k=16 forward KL with teacher renormalized over its top-k
            #     and student using FULL-vocab distribution (Eq 10) ---
            top_k = 16
            top_k_vals, top_k_idx = torch.topk(t_t, k=top_k, dim=-1)                 # [B,T,k]
            # π̃_te(x) = teacher renormalized over its own top-k support (no grad)
            with torch.no_grad():
                t_lp_topk_renorm = F.log_softmax(top_k_vals, dim=-1).detach()        # [B,T,k]
                t_probs_topk_renorm = t_lp_topk_renorm.exp()
            # log π_θ(x) at teacher's top-k indices, gathered from FULL student log-softmax
            # (NOT renormalized over top-k — this is the paper's denominator)
            s_lp_at_topk = s_lp_full.gather(-1, top_k_idx)                           # [B,T,k] grad
            # L^FKL = Σ π̃_te(x) · (log π̃_te(x) − log π_θ(x))
            L_FKL = (t_probs_topk_renorm * (t_lp_topk_renorm - s_lp_at_topk)).sum(-1)  # [B,T]

            # --- Teacher entropy gate (Eq 9, paper τ = 0.8) ---
            with torch.no_grad():
                teacher_entropy = -(t_probs_full * t_lp_full).sum(-1)                # [B,T]
            tau = 0.8
            high_entropy = (teacher_entropy > tau).to(L_OPD.dtype)

            # Per-token EOPD loss (no extra clipping — paper does not specify any
            # outer clamp on L^EOPD; PPO clip on r_t inside L^OPD is the only clip).
            per_token_loss = L_OPD + high_entropy * L_FKL

            # Reduction: paper Algorithm 1 line 13 sums over all valid tokens then
            # divides by the total number of valid tokens (global-token mean).
            if shifted_labels is not None:
                mask = shifted_labels != -100
                loss = per_token_loss[mask].sum() / mask.sum() if mask.any() else torch.tensor(0.0, device=per_token_loss.device, requires_grad=True)
            else:
                loss = per_token_loss.mean()

            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                with torch.no_grad():
                    m = shifted_labels != -100 if shifted_labels is not None else None
                    if m is not None and m.any():
                        self._metrics["train"]["eopd/teacher_entropy_mean"].append(float(teacher_entropy[m].mean().item()))
                        self._metrics["train"]["eopd/pct_high_entropy"].append(float(high_entropy[m].mean().item()))
                        self._metrics["train"]["eopd/L_OPD_mean"].append(float(L_OPD[m].mean().item()))
                        self._metrics["train"]["eopd/L_FKL_mean"].append(float(L_FKL[m].mean().item()))
                        self._metrics["train"]["eopd/r_mean"].append(float(r_t[m].mean().item()))
                        clip_active = ((r_t < 1.0 - ppo_eps) | (r_t > 1.0 + ppo_eps))
                        self._metrics["train"]["eopd/clip_fraction"].append(float(clip_active[m].float().mean().item()))

            del student_logits_for_loss, teacher_logits
            empty_cache()

        elif self.mc_samples == -12:
            # === MODE: Position-weighted OPSD ===
            # Loss = mean(w_t * KL(teacher || student)) where
            #   w_t = w_min + (1 - w_min) * sigmoid((t/T - tau) / s)
            # Early tokens (high-uncertainty branch decisions) are down-weighted;
            # late tokens (execution/calculation) keep full weight. No MC dropout.
            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            teacher_prompt_len = inputs["teacher_prompt_length"]
            with torch.no_grad(), adapter_context:
                outputs_teacher = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1: -1, :].clone()
                del outputs_teacher
            empty_cache()

            s_t = student_logits_for_loss.float() / self.temperature
            t_t = teacher_logits.float() / self.temperature

            if self.top_k_loss is not None and self.top_k_loss > 0:
                _, idx = torch.topk(t_t, k=self.top_k_loss, dim=-1)
                s_t = torch.gather(s_t, dim=-1, index=idx)
                t_t = torch.gather(t_t, dim=-1, index=idx)

            s_lp = F.log_softmax(s_t, dim=-1)
            t_lp = F.log_softmax(t_t, dim=-1)

            if self.beta == 0:
                # forward KL: KL(teacher || student)
                div = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
            elif self.beta == 1:
                # reverse KL: KL(student || teacher)
                div = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
            else:
                beta = torch.tensor(self.beta, dtype=s_lp.dtype, device=s_lp.device)
                mix = torch.logsumexp(
                    torch.stack([s_lp + torch.log1p(-beta), t_lp + torch.log(beta)]),
                    dim=0,
                )
                div = beta * F.kl_div(mix, t_lp, reduction="none", log_target=True)
                div = div + (1 - beta) * F.kl_div(mix, s_lp, reduction="none", log_target=True)

            if self.jsd_token_clip is not None:
                div = div.clamp(max=self.jsd_token_clip)

            per_token_loss = div.sum(-1)  # [B, T]
            mask = shifted_labels != -100  # [B, T]

            # FIX: position weight uses per-SEQUENCE response length, not the batch's
            # padded max length. Otherwise short responses get mostly low weights and
            # long responses mostly high weights, regardless of where the token sits
            # within its own response. We compute the within-row token index via cumsum
            # over the valid mask, and divide by the per-row response length.
            mask_f = mask.to(torch.float32)
            length_per_row = mask_f.sum(dim=1).clamp(min=1.0)               # [B]
            # Within-row 1-based index of each valid token; padded/invalid positions
            # get index 0 but are masked out below so their weight value doesn't matter.
            within_row_idx = mask_f.cumsum(dim=1) * mask_f                  # [B, T]
            frac = (within_row_idx - 0.5).clamp(min=0.0) / length_per_row.unsqueeze(1)
            weight = self.position_w_min + (1.0 - self.position_w_min) * torch.sigmoid(
                (frac - self.position_tau) / max(self.position_s, 1e-6)
            )
            weight = weight.to(per_token_loss.dtype)                        # [B, T]
            weighted = per_token_loss * weight

            # Reduction. Default (per-sequence mean then batch mean over valid
            # sequences) is the paper's PWOPSD reduction; setting
            # position_global_reduction=True selects the legacy "global token
            # mean" reduction, used by the 2x2 ablation that isolates the
            # position schedule from the length-normalization effect.
            if mask.any():
                if self.position_global_reduction:
                    # Cell (3): global token mean (length-normalization OFF).
                    weighted_masked = weighted * mask_f.to(per_token_loss.dtype)
                    loss = weighted_masked.sum() / mask.sum().clamp(min=1).to(per_token_loss.dtype)
                else:
                    # Default (cell 2/4): per-sequence mean then batch mean.
                    per_seq_loss = (weighted * mask_f.to(per_token_loss.dtype)).sum(dim=1) / length_per_row.to(per_token_loss.dtype)  # [B]
                    valid_seq_mask = (mask_f.sum(dim=1) > 0).to(per_token_loss.dtype)            # [B]
                    n_valid_seq = valid_seq_mask.sum().clamp(min=1.0)
                    loss = (per_seq_loss * valid_seq_mask).sum() / n_valid_seq
            else:
                loss = torch.tensor(0.0, device=per_token_loss.device, requires_grad=True)

            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                with torch.no_grad():
                    self._metrics["train"]["ea/position_weight_mean"].append(weight[mask].mean().item() if mask.any() else 0.0)
                    self._metrics["train"]["ea/position_weight_min"].append(weight[mask].min().item() if mask.any() else 0.0)
                    self._metrics["train"]["ea/position_weight_max"].append(weight[mask].max().item() if mask.any() else 1.0)

            del student_logits_for_loss, teacher_logits, per_token_loss, weighted
            empty_cache()

        elif self.mc_samples == -13:
            # === REOPOLD: Relaxed On-Policy Distillation (Ko et al., 2026, arXiv:2603.11137) ===
            # Policy-gradient form of on-policy distillation:
            #   J = (1/Σ M_t) · Σ_t  ρ_t · R̂^λ_t · M_t           (Eq. 4)
            # In our pure on-policy setup (no replay buffer), π_θ_old = π_θ at every
            # step, so ρ_t = 1 numerically.  We therefore use the equivalent gradient
            # form
            #   loss = -(1/Σ M_t) · Σ_t  R̂^λ_t · M_t · log π_θ(o_t)
            # which produces the same gradient (autograd through log π_θ).
            #
            # Three components from the paper:
            #   1) R̂^λ = max(sg(R), log(λ)/(1-λ))         — mixture-based clipping (Eq. 7)
            #      where R = log p_T(o_t) - log p_S(o_t)   (token-level log-ratio reward)
            #   2) Phase mask M:
            #        Phase I (k < T_switch): M = 𝟙[R ≥ floor]    — exploration  (Eq. 9)
            #        Phase II (k ≥ T_switch): M = 𝟙[H_t ≥ τ_β]   — refinement   (Eq. 10)
            #      where τ_β = top β-percentile of student entropy H_t in batch.
            #   3) The mask gets multiplied by the standard label-validity mask
            #      (shifted_labels != -100).
            #
            # Hyperparams: reopold_lambda (λ), reopold_beta (β), reopold_t_switch.
            import math

            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            teacher_prompt_len = inputs["teacher_prompt_length"]
            with torch.no_grad(), adapter_context:
                outputs_teacher = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1: -1, :].clone()
                del outputs_teacher
            empty_cache()

            # --- Per-position log-prob distributions (paper-faithful, Ko et al. 2026 Eq 7) ---
            # The REOPOLD reward is R_t = log pi_T(o_t) - log pi_theta(o_t), where both
            # log-probs come from the FULL vocabulary distribution. We previously
            # truncated to match vLLM sampling support; that turns out to deviate from
            # the paper algorithm (it rewrites "reward" at out-of-support tokens and
            # introduces NaN/inf paths in the gather). Removing the truncation makes
            # the loss faithful to the published reward and removes the need for the
            # student-finite NaN safety mask further down (log_softmax of finite
            # logits is always finite).
            student_lp_full = F.log_softmax(
                student_logits_for_loss.float() / self.temperature, dim=-1
            )  # [B,T,V] grad
            with torch.no_grad():
                teacher_lp_full = F.log_softmax(
                    teacher_logits.float() / self.temperature, dim=-1
                )  # [B,T,V] no grad

            # --- Gather log π(o_t) at the actual rolled-out token id ---
            # NaN safety: padded positions have label=-100 → safe_labels=0; token id 0
            # is usually OUTSIDE the truncated support, so log_p_{S,T}=-inf there. We
            # zero those positions out BEFORE forming R, so R is finite everywhere and
            # M_phase masking (which sets contributions to 0 at invalid positions) does
            # not produce `NaN * 0 = NaN` later in the loss.
            mask_label = (shifted_labels != -100)                                      # [B,T] bool
            safe_labels = shifted_labels.clamp(min=0)                                  # [B,T] long
            log_p_S_raw = student_lp_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T] grad, may be -inf
            with torch.no_grad():
                log_p_T_raw = teacher_lp_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T] may be -inf
            zero_like_S = torch.zeros_like(log_p_S_raw)
            log_p_S = torch.where(mask_label, log_p_S_raw, zero_like_S)
            with torch.no_grad():
                log_p_T = torch.where(mask_label, log_p_T_raw, torch.zeros_like(log_p_T_raw))
            del log_p_S_raw, log_p_T_raw, zero_like_S

            # Under the full-vocab paper-faithful formulation, log_softmax of finite
            # logits is finite, so log_p_S and log_p_T are finite at all valid
            # positions. We retain student_finite as an all-True mask only so that
            # the downstream logging code (which reports student_mismatch_fraction)
            # continues to work without code changes; it should always be 0 now.
            student_finite = torch.ones_like(log_p_S, dtype=torch.bool).detach()

            # --- Reward R = log p_T - log p_S, treated as fixed (stop-gradient) ---
            # Under the full-vocab paper-faithful formulation, R is finite at all
            # valid (mask_label=True) positions. At invalid (mask_label=False)
            # positions both log-probs were zeroed above, so R = 0; M masks these
            # out anyway in the final reduction.
            R = (log_p_T - log_p_S.detach())                                           # [B,T] no grad
            # Defensive: if numerical edge cases still make R non-finite, clamp via
            # nan_to_num so the sum below stays finite. nan→0 covers any 0×∞-style
            # leftovers; -inf→-1e30 lets the floor clamp pull it back to floor.
            R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=-1e30)

            # --- Mixture-based clipping floor: log(λ)/(1-λ) (Eq. 7) ---
            lam = float(self.reopold_lambda)
            assert 0.0 < lam < 1.0, f"reopold_lambda must be in (0,1), got {lam}"
            floor = math.log(lam) / (1.0 - lam)
            R_hat = torch.clamp(R, min=floor)                                          # [B,T] no grad

            # --- Student entropy per token (over the truncated support) for the
            # refinement-phase mask. Use the convention 0·log 0 = 0 (i.e., excluded
            # tokens with p=0 contribute nothing) by zeroing out non-finite log-probs.
            with torch.no_grad():
                student_probs = student_lp_full.exp()                                  # 0 where lp=-inf
                lp_safe = torch.where(
                    torch.isfinite(student_lp_full),
                    student_lp_full,
                    torch.zeros_like(student_lp_full),
                )
                H_t = -(student_probs * lp_safe).sum(dim=-1).detach()                  # [B,T]
                del student_probs, lp_safe

            # --- Phase-dependent mask (M_t) ---
            current_step = self.state.global_step
            if current_step < self.reopold_t_switch:
                # Phase I — Exploration:  M = 𝟙[R ≥ floor]
                # Removes tokens whose reward got clipped (i.e. teacher gives ~0 prob);
                # prevents heavy-tail negative updates that drive entropy collapse.
                M_phase = (R >= floor).to(R_hat.dtype)
                phase_id = 0.0
            else:
                # Phase II — Refinement: top β-percentile entropy within valid positions
                beta = float(self.reopold_beta)
                assert 0.0 < beta <= 1.0, f"reopold_beta must be in (0,1], got {beta}"
                if mask_label.any():
                    H_valid = H_t[mask_label]
                    # We want top β fraction (highest H), so threshold = (1-β)-quantile
                    tau_beta = torch.quantile(H_valid.float(), 1.0 - beta).to(H_t.dtype)
                else:
                    tau_beta = torch.tensor(0.0, device=H_t.device, dtype=H_t.dtype)
                M_phase = (H_t >= tau_beta).to(R_hat.dtype)
                phase_id = 1.0

            # Combine with label-validity mask AND drop student-numerics-mismatch
            # positions (where vLLM sampled an o_t we'd have excluded). Those have
            # log_p_S replaced by 0 above, so removing them from M is necessary so
            # they don't contribute spurious 0 gradients and don't inflate M_sum.
            M = (M_phase * mask_label.to(M_phase.dtype) * student_finite.to(M_phase.dtype)).detach()  # [B,T] no grad

            # --- Loss: -(1/Σ M) · Σ R̂ · M · log π_θ(o_t) ---
            # log_p_S has gradient; R_hat and M are stop-grad. ρ_t = 1 in pure on-policy.
            M_sum = M.sum().clamp(min=1.0)
            loss = -(R_hat * M * log_p_S).sum() / M_sum

            # --- Wandb logging ---
            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                with torch.no_grad():
                    valid_n = float(mask_label.sum().item())
                    self._metrics["train"]["reopold/floor"].append(floor)
                    self._metrics["train"]["reopold/phase"].append(phase_id)
                    # Restrict R metrics to (valid AND student-finite) so the
                    # mean/min isn't dominated by the -1e30 sentinel values that
                    # student-mismatch positions get from nan_to_num.
                    R_clean_mask = mask_label & student_finite
                    n_clean = float(R_clean_mask.sum().item())
                    self._metrics["train"]["reopold/R_mean"].append(
                        float(R[R_clean_mask].mean().item()) if n_clean > 0 else 0.0
                    )
                    self._metrics["train"]["reopold/R_min"].append(
                        float(R[R_clean_mask].min().item()) if n_clean > 0 else 0.0
                    )
                    self._metrics["train"]["reopold/clip_fraction"].append(
                        float(((R < floor) & mask_label).float().sum().item() / max(valid_n, 1.0))
                    )
                    # Fraction of valid positions where vLLM-sampled o_t fell outside
                    # our re-truncated student support (numerics mismatch). Should be
                    # very small (~0.1-1%); large values indicate truncation/temp
                    # disagreement with rollout.
                    self._metrics["train"]["reopold/student_mismatch_fraction"].append(
                        float(((~student_finite) & mask_label).float().sum().item() / max(valid_n, 1.0))
                    )
                    self._metrics["train"]["reopold/H_mean"].append(
                        float(H_t[mask_label].mean().item()) if valid_n > 0 else 0.0
                    )
                    self._metrics["train"]["reopold/M_fraction"].append(
                        float(M.sum().item() / max(valid_n, 1.0))
                    )

            del student_logits_for_loss, teacher_logits, student_lp_full, teacher_lp_full
            del log_p_S, log_p_T, R, R_hat, H_t, M, M_phase
            empty_cache()


        else:
            raise NotImplementedError(
                f"Internal training-objective code {self.mc_samples!r} is not handled. "
                "Use --method opsd|eopd|pwopsd|reopold from ea_opsd_train.py."
            )

        return loss
