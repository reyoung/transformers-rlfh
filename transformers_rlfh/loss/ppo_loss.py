from torchtyping import TensorType
import torch

__all__ = ['ppo_lm_loss']


def ppo_lm_loss(
        logprobs: TensorType["batch_size", "response_size", torch.float],
        values: TensorType["batch_size", "response_size", torch.float],
        old_logprobs: TensorType["batch_size", "response_size", torch.float],
        old_values: TensorType["batch_size", "response_size", torch.float],
        advantages: TensorType["batch_size", "response_size", torch.float],
        returns: TensorType["batch_size", "response_size", torch.float],
        mask: TensorType["batch_size", "response_size"],
        cliprange_value: float = 0.2,
        cliprange_ratio: float = 0.2,
        vf_coef: float = 0.5,
) -> TensorType[torch.float]:
    """
    PPO Language Model loss

    :param logprobs: 当前step response token的log probs
    :param values: 当前step critic 网络的输出
    :param old_logprobs: 生成时的log probs
    :param old_values: 生成时的values
    :param advantages: advantages
    :param returns: returns
    :param mask: mask
    :param cliprange_value: value的cliprange
    :param cliprange_ratio: log prob ratio的cliprange
    :param vf_coef: value loss的系数
    :return: loss tensor of shape (1,)
    """
    values_clipped = torch.clamp(values, old_values - cliprange_value, old_values + cliprange_value)
    n = mask.sum()

    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask)

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_ratio, 1.0 + cliprange_ratio)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask)

    return (vf_loss * vf_coef + pg_loss) / n
