from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    decoding_strategy: str = field(
        default="stochastic0.5-linear",
        metadata={"help": "<topk_mode>-<schedule>"}
    )
    token_reweighting: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "for focal loss"}
    )
    gamma: float = field(
        default=2,
        metadata={"help": "for focal loss"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    topk_decoding: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    use_info_gain_ordering: bool = field(
        default=False,
        metadata={"help": "Enable greedy information-gain decoding."}
    )
    critic_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the trained critic checkpoint (.pt)."}
    )
    info_gain_alpha: float = field(
        default=1.0,
        metadata={"help": "Confidence exponent used in utility computation."}
    )
    info_gain_tau_util: float = field(
        default=0.0,
        metadata={"help": "Utility threshold for committing a token."}
    )
    info_gain_tau_conf: float = field(
        default=0.0,
        metadata={"help": "Confidence threshold for committing a token."}
    )
    info_gain_budget: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of tokens revealed per step."}
    )

    def __post_init__(self):
        pass
