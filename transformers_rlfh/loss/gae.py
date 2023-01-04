from dataclasses import dataclass

import torch
from torchtyping import TensorType

__all__ = ["calculate_gae", "ReturnAndAdvantage"]


@dataclass
class ReturnAndAdvantage:
    returns: TensorType["response_size", "seq_len"]
    advantages: TensorType["response_size", "seq_len"]


def whiten(x: torch.Tensor) -> torch.Tensor:
    var, mean = torch.var_mean(x)
    var += 1e-8
    return (x - mean) * torch.rsqrt(var)


def calculate_gae(
        value: TensorType["response_size", "seq_len"],
        reward: TensorType["response_size", "seq_len"],
        whitening: bool = True,
        gamma: float = 0.99,
        tau: float = 0.95,
) -> ReturnAndAdvantage:
    n = reward.shape[1]
    gae = 0
    returns = []
    for step in reversed(range(n)):
        next_value = value[:, step + 1] if step < n - 1 else 0
        delta = reward[:, step] + gamma * next_value - value[:, step]
        gae = delta + gamma * tau * gae
        returns.append(gae + value[:, step])

    returns = torch.stack(returns[::-1], dim=1)
    advantages = returns - value
    if whitening:
        advantages = whiten(advantages)
    return ReturnAndAdvantage(returns=returns, advantages=advantages)
