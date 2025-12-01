"""Greedy information-gain sampler for diffusion decoding."""

from __future__ import annotations

import math
from typing import Optional

import torch

from models.info_gain_critic import CriticConfig, InfoGainCritic
from utils.feature_extractor import FeatureExtractorConfig, extract_token_features


def _load_critic(checkpoint: str, device: torch.device) -> tuple[InfoGainCritic, FeatureExtractorConfig, CriticConfig]:
    state = torch.load(checkpoint, map_location=device)
    critic_cfg = CriticConfig(**state["critic_config"])
    critic = InfoGainCritic(critic_cfg).to(device)
    critic.load_state_dict(state["critic_state_dict"])
    critic.eval()

    feature_cfg_dict = state.get("feature_config", {})
    feature_cfg = FeatureExtractorConfig(**feature_cfg_dict)
    return critic, feature_cfg, critic_cfg


class InfoGainSampler:
    """Implements the greedy selection policy described in the specification."""

    def __init__(self, model, tokenizer, diff_args):
        if diff_args.critic_checkpoint is None:
            raise ValueError("`critic_checkpoint` must be provided for info-gain decoding.")

        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.critic, self.feature_cfg, self.critic_config = _load_critic(diff_args.critic_checkpoint, self.device)
        self.alpha = diff_args.info_gain_alpha
        self.tau_util = diff_args.info_gain_tau_util
        self.tau_conf = diff_args.info_gain_tau_conf
        budget = getattr(diff_args, "info_gain_budget", None)
        self.max_tokens_per_step: Optional[int] = budget if isinstance(budget, int) and budget > 0 else None
        self.total_steps = diff_args.diffusion_steps
        self.max_entropy = math.log(tokenizer.vocab_size)
        strategy = getattr(diff_args, "decoding_strategy", "deterministic-linear")
        self.schedule = strategy.split("-")[-1]

    @torch.no_grad()
    def _forward(self, input_ids, attention_mask):
        embeds = self.model.get_embeds(input_ids)
        outputs = self.model.denoise_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.model.get_logits(hidden)
        return logits, hidden, outputs.attentions

    @torch.no_grad()
    def greedy_decode(
        self,
        xt: torch.Tensor,
        attention_mask: torch.Tensor,
        src_mask: torch.Tensor,
        timestep: int,
        init_maskable_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens_committed = 0
        total_slots = init_maskable_mask.sum(dim=1)
        rate = self._compute_rate(timestep)
        cutoff = torch.floor(total_slots.float() * rate).long()
        target_total = (total_slots - cutoff).clamp(min=0)
        mask = (xt == self.tokenizer.mask_token_id) & (~src_mask)
        current_unmasked = (init_maskable_mask & ~mask).sum(dim=1)
        remaining_targets = (target_total - current_unmasked).clamp(min=0).long()

        while True:
            mask = (xt == self.tokenizer.mask_token_id) & (~src_mask)
            if mask.sum() == 0:
                break
            if remaining_targets.sum() == 0:
                break

            logits, hidden, attentions = self._forward(xt, attention_mask)
            t_tensor = torch.full((xt.size(0),), timestep, device=xt.device)
            features = extract_token_features(
                logits=logits,
                hidden_states=hidden,
                attentions=attentions,
                mask=mask,
                timestep=t_tensor,
                total_steps=self.total_steps,
                config=self.feature_cfg,
            )

            if features.features.numel() == 0:
                break

            entropy_masked = features.entropy[mask]
            confidences = (1.0 - entropy_masked / self.max_entropy).clamp(min=0.0, max=1.0)
            logits_masked = logits[mask]
            predicted_gain = self.critic(features.features)
            utilities = torch.pow(confidences.clamp(min=1e-6), self.alpha) * predicted_gain
            batch_indices = features.masked_indices[:, 0]
            valid_positions = remaining_targets[batch_indices] > 0
            if not valid_positions.any():
                break
            utilities = utilities.masked_fill(~valid_positions, float("-inf"))
            best_idx = int(torch.argmax(utilities).item())
            best_util = float(utilities[best_idx].item())
            best_conf = float(confidences[best_idx].item())

            if best_util < self.tau_util or best_conf < self.tau_conf:
                break

            chosen_logits = logits_masked[best_idx]
            predicted_token = int(torch.argmax(chosen_logits).item())
            token_coords = features.masked_indices[best_idx]
            batch_idx = int(token_coords[0].item())
            pos_idx = int(token_coords[1].item())
            xt[batch_idx, pos_idx] = predicted_token
            tokens_committed += 1
            remaining_targets[batch_idx] = torch.clamp(remaining_targets[batch_idx] - 1, min=0)

            if self.max_tokens_per_step and tokens_committed >= self.max_tokens_per_step:
                break

        return xt

    def _compute_rate(self, timestep: int) -> float:
        max_step = max(1, self.total_steps)
        if self.schedule == "linear":
            return max(0.0, min(1.0, timestep / max_step))
        if self.schedule == "cosine":
            return math.cos(((max_step - timestep) / max_step) * (math.pi * 0.5))
        raise ValueError(f"Unsupported schedule: {self.schedule}")
