"""Feature extraction utilities for information-gain decoding.

The helper functions in this module operate on hidden states and logits
produced by the denoiser (``p_\theta``) and build the feature vector ``z_i``
specified in :mod:`spec.md` / :mod:`info_gain_diffusion.tex`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FeatureExtractorConfig:
    include_entropy: bool = True
    include_margin: bool = True
    include_propagation: bool = True
    include_global: bool = True
    include_context: bool = True


@dataclass
class FeatureOutput:
    features: torch.Tensor
    masked_indices: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor


def _ensure_tensor(attentions: Optional[Iterable[torch.Tensor]]) -> Optional[torch.Tensor]:
    if attentions is None:
        return None
    if isinstance(attentions, torch.Tensor):
        return attentions
    attentions = list(attentions)
    if not attentions:
        return None
    return attentions[-1]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def compute_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    top2 = probs.topk(k=2, dim=-1).values
    if top2.size(-1) == 1:
        return top2[..., 0]
    return top2[..., 0] - top2[..., 1]


def extract_token_features(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    attentions: Optional[Iterable[torch.Tensor]],
    mask: torch.Tensor,
    timestep: torch.Tensor,
    total_steps: int,
    config: FeatureExtractorConfig | None = None,
) -> FeatureOutput:
    """Build the per-token feature vectors.

    Args:
        logits: ``(B, N, V)`` tensor of denoiser logits.
        hidden_states: ``(B, N, H)`` tensor from the final transformer layer.
        attentions: Iterable of attention tensors as returned by HF models.
        mask: Boolean tensor selecting the masked (yet-to-be-decoded) tokens.
        timestep: Tensor of shape ``(B,)`` containing the current diffusion step.
        total_steps: Total number of diffusion steps ``T``.
        config: Optional :class:`FeatureExtractorConfig` to control which
            components are included.

    Returns:
        :class:`FeatureOutput` with flattened features only for masked tokens.
    """

    if config is None:
        config = FeatureExtractorConfig()

    attn = _ensure_tensor(attentions)

    entropy = compute_entropy(logits)
    margin = compute_margin(logits)

    hidden_norm = F.layer_norm(hidden_states, (hidden_states.size(-1),))

    components = [hidden_norm] if config.include_context else []

    if config.include_entropy:
        components.append(entropy.unsqueeze(-1))
    if config.include_margin:
        components.append(margin.unsqueeze(-1))

    if config.include_propagation:
        if attn is not None:
            attn_avg = attn.mean(dim=1) if attn.dim() == 4 else attn
            mask_float = mask.float()
            d_out = (attn_avg * mask_float.unsqueeze(-1)).sum(dim=1)
            neighbor_entropy = (attn_avg * (mask_float * entropy).unsqueeze(-1)).sum(dim=1)
        else:
            zero_template = torch.zeros_like(entropy)
            d_out = zero_template
            neighbor_entropy = zero_template
        components.append(d_out.unsqueeze(-1))
        components.append(neighbor_entropy.unsqueeze(-1))

    if config.include_global:
        mask_ratio = mask.float().sum(dim=1, keepdim=True) / mask.size(1)
        mask_ratio = mask_ratio.unsqueeze(-1).expand(-1, mask.size(1), 1)
        t_normalized = (timestep.float() / float(total_steps)).unsqueeze(-1).unsqueeze(-1)
        t_normalized = t_normalized.expand(-1, mask.size(1), 1)
        components.append(mask_ratio)
        components.append(t_normalized)

    feature_stack = torch.cat(components, dim=-1)
    masked_indices = mask.nonzero(as_tuple=False)
    masked_features = feature_stack[mask]

    return FeatureOutput(
        features=masked_features,
        masked_indices=masked_indices,
        entropy=entropy,
        margin=margin,
    )
