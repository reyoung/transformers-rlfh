from typing import Optional

import torch
from torchtyping import TensorType
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions

from transformers_rlfh.models.actor_critic import ValueHead

__all__ = ['GPTBestOfN']


class GPTBestOfN(torch.nn.Module):
    """
    GPT 排序模型，返回best of n

    参考 OpenAI LM-From-Human-preference
    """

    def __init__(self, base: GPT2Model, value_dropout=0.1):
        super().__init__()
        self.base = base
        self.value = ValueHead(feature_dim=base.config.hidden_size, dropout=value_dropout)
        self.value.to(self.base.device)
        self.reward_gain = torch.ones(size=(1,), dtype=torch.float32, device=self.base.device)
        self.reward_bias = torch.zeros(size=(1,), dtype=torch.float32, device=self.base.device)

    @property
    def device(self):
        return self.base.device

    def forward(self,
                input_ids: TensorType["batch_size", "n_best", "sequence_length", torch.long],
                best: TensorType["batch_size", torch.long],
                last_token_pos: Optional[TensorType["batch_size"]] = None,
                pad_token_id: Optional[int] = None,
                loss_reduction="mean",
                ):
        """

        :param input_ids: batch_size x n_best x query_length
        :param best: best sample id
        :param last_token_pos: batch_size, 每个sample最后一个token的id. 包含special token。只用来计算last token reward
        :param pad_token_id: pad token id, Padding token id
        :param loss_reduction: loss 的reduction方法。用于单元测试

        :note: last_token_ids, pad_token_id 要么全部设置，要么全部不设置。如果设置，会计算last token reward。
               否则直接计算最后一个padding的reward

        :return: loss
        """
        n_best = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[2]
        reward: TensorType["batch_size*n_best"] = self.get_reward(
            input_ids=input_ids.reshape(batch_size * n_best, sequence_length),
            pad_token_id=pad_token_id,
            last_token_pos=None if last_token_pos is None else last_token_pos.reshape(batch_size * n_best),
        )
        reward: TensorType["batch_size", "n_best"] = reward.reshape(batch_size, n_best)

        return torch.nn.functional.nll_loss(input=torch.nn.functional.log_softmax(reward, dim=1), target=best,
                                            reduction=loss_reduction)

    def get_reward(self,
                   input_ids: TensorType["batch_size", "sequence_length", torch.long],
                   last_token_pos: Optional[TensorType["batch_size"]] = None,
                   pad_token_id: Optional[int] = None) -> TensorType["batch_size", torch.float32]:
        """

        :param input_ids: batch_size x sequence_length query和response已经拼好了
        :param last_token_pos: batch_size, 每个sample最后一个token的id. 包含special token。只用来计算last token reward
        :param pad_token_id: pad token id, Padding token id

        :note: last_token_ids, pad_token_id 要么全部设置，要么全部不设置。如果设置，会计算last token reward。
               否则直接计算最后一个padding的reward
        :return: reward
        """
        # check (last_token_pos, pad_token_id) are both set or both not set
        if self.reward_gain.device != input_ids.device:
            self.reward_gain = self.reward_gain.to(input_ids.device)
            self.reward_bias = self.reward_bias.to(input_ids.device)


        if last_token_pos is None and pad_token_id is None:
            reward_style = "last_padding"
        elif last_token_pos is not None and pad_token_id is not None:
            reward_style = "last_token"
        else:
            raise ValueError("last_token_ids, pad_token_id must be all set or all not set")

        if reward_style == "last_padding":
            att_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.bool)
        else:
            att_mask = torch.ne(input_ids, pad_token_id)

        output: BaseModelOutputWithCrossAttentions = self.base(input_ids=input_ids, attention_mask=att_mask)
        last_hidden_state: ["batch_size", "sequence_length"] = output.last_hidden_state.squeeze(-1)

        if reward_style == "last_padding":
            last_hidden_state = last_hidden_state[:, -1, :]
            reward: TensorType["batch_size"] = self.value(last_hidden_state)
        else:
            mask = torch.nn.functional.one_hot(last_token_pos,
                                               num_classes=last_hidden_state.shape[1]).unsqueeze(-1).float()
            last_hidden_state = last_hidden_state * mask
            last_hidden_state = torch.sum(last_hidden_state, dim=1)
            reward: TensorType["batch_size"] = self.value(last_hidden_state)

        reward = self.reward_gain * reward + self.reward_bias
        return reward
