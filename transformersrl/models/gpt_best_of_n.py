from typing import Optional

import torch
from torchtyping import TensorType
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions

from transformersrl.models.actor_critic import ValueHead

__all__ = ['GPTBestOfN']


class GPTBestOfN(torch.nn.Module):
    """
    GPT 排序模型，返回best of n

    参考 OpenAI LM-From-Human-preference
    """

    def __init__(self, base: GPT2Model):
        super().__init__()
        self.base = base
        self.value = ValueHead(feature_dim=base.config.hidden_size)
        self.value.to(self.base.device)

        self.reward_gain = torch.nn.Parameter(
            torch.ones(size=(1,), dtype=torch.float32, device=self.base.device)
        )
        self.reward_bias = torch.nn.Parameter(
            torch.zeros(size=(1,), dtype=torch.float32, device=self.base.device)
        )

    @property
    def device(self):
        return self.base.device

    def forward(self,
                query: TensorType["batch_size", "query_length", torch.long],
                samples: TensorType["batch_size", "n_best", "sample_length", torch.long],
                best: TensorType["batch_size", torch.long],
                last_token_ids: Optional[TensorType["batch_size"]] = None,
                pad_token_id: Optional[int] = None,
                ):
        """

        :param query: batch_size x query_length， 同时每个query后会追加至少一个特殊token。
        :param samples: batch_size x n_best x sample_length， 每个sample后会追加至少一个特殊token
        :param best: best sample id
        :param last_token_ids: batch_size, 每个sample最后一个token的id. 包含special token。只用来计算last token reward
        :param pad_token_id: pad token id, Padding token id

        :note: last_token_ids, pad_token_id 要么全部设置，要么全部不设置。如果设置，会计算last token reward。
               否则直接计算最后一个padding的reward

        :return: loss
        """

        n_best = samples.shape[1]
        batch_size = query.shape[0]
        query_length = query.shape[1]
        sample_length = samples.shape[2]

        query: TensorType["batch_size", "n_best", "query_length"] = torch.repeat_interleave(query,
                                                                                            n_best,
                                                                                            dim=0).reshape(
            batch_size, n_best, query_length)

        input_ids: TensorType["batch_size", "n_best", "sequence_length"] = torch.cat((samples, query), dim=2)
        sequence_length = input_ids.shape[2]
        assert query_length + sample_length == sequence_length

        reward: TensorType["batch_size*n_best"] = self.get_reward(
            input_ids=input_ids.reshape(batch_size * n_best, sequence_length),
            pad_token_id=pad_token_id, last_token_ids=last_token_ids,
        )
        reward: TensorType["batch_size", "n_best"] = reward.reshape(batch_size, n_best)

        return torch.nn.functional.nll_loss(input=torch.nn.functional.log_softmax(reward, dim=1), target=best)

    def get_reward(self,
                   input_ids: TensorType["batch_size", "sequence_length", torch.long],
                   last_token_ids: Optional[TensorType["batch_size"]] = None,
                   pad_token_id: Optional[int] = None) -> TensorType["batch_size", torch.float32]:
        """

        :param input_ids: batch_size x sequence_length query和response已经拼好了
        :param last_token_ids: batch_size, 每个sample最后一个token的id. 包含special token。只用来计算last token reward
        :param pad_token_id: pad token id, Padding token id

        :note: last_token_ids, pad_token_id 要么全部设置，要么全部不设置。如果设置，会计算last token reward。
               否则直接计算最后一个padding的reward
        :return: reward
        """
        # check (last_token_ids, pad_token_id) are both set or both not set
        if last_token_ids is None and pad_token_id is None:
            reward_style = "last_padding"
        elif last_token_ids is not None and pad_token_id is not None:
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
            reward: TensorType["batch_size"] = self.value(last_hidden_state[:, -1])
        else:
            last_hidden_state = last_hidden_state * torch.nn.functional.one_hot(last_token_ids,
                                                                                num_classes=last_hidden_state.shape[
                                                                                    -1]).float()
            reward: TensorType["batch_size", "sequence_length"] = self.value(last_hidden_state)
            reward: TensorType["batch_size"] = torch.sum(reward, dim=1)

        reward = self.reward_gain * reward + self.reward_bias
        return reward
