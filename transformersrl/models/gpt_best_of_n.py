from typing import Union, Tuple

import torch
from torch import nn
from torchtyping import TensorType
from transformers import GPT2Model, AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions

from transformersrl.models.actor_critic import ValueHead

__all__ = ['GPTBestOfN']


class GPTBestOfN(PreTrainedModel):
    """
    GPT 排序模型，返回best of n

    参考 OpenAI LM-From-Human-preference
    """

    def _reorder_cache(self, past, beam_idx):
        self.base._reorder_cache(past, beam_idx)

    def _init_weights(self, module):
        self.reward_gain.data.fill_(1.0)
        self.reward_bias.data.fill_(0.0)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.base.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return self.base.get_position_embeddings()

    def __init__(self, base: Union[GPT2Model, AutoModel]):
        super().__init__(base.config)
        print(type(base.config))
        self.base = base
        self.value = ValueHead(feature_dim=base.config.hidden_size)
        self.value.to(self.base.device)

        self.reward_gain = torch.nn.Parameter(
            torch.ones(size=(1,), dtype=torch.float32, device=self.base.device)
        )
        self.reward_bias = torch.nn.Parameter(
            torch.zeros(size=(1,), dtype=torch.float32, device=self.base.device)
        )

    def forward(self,
                query: TensorType["batch_size", "query_length", torch.long],
                samples: TensorType["batch_size", "n_best", "sample_length", torch.long],
                best: TensorType["batch_size", torch.long],
                ):
        """

        :param query: batch_size x query_length， 同时每个query后会追加至少一个特殊token。
        :param samples: batch_size x n_best x sample_length， 每个sample后会追加至少一个特殊token
        :param best: best sample id
        :return:
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

        input_ids.resize_(n_best * batch_size, sequence_length)
        att_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.bool)

        output: BaseModelOutputWithCrossAttentions = self.base(input_ids=input_ids,
                                                               attention_mask=att_mask)
        reward: TensorType["batch_size", "n_best"] = self.value(output.last_hidden_state.squeeze(-1)[:, -1]).reshape(
            (batch_size, n_best))
        reward: TensorType["batch_size", "n_best"] = self.reward_gain * reward + self.reward_bias

        logits = torch.nn.functional.log_softmax(reward, dim=1)
        return torch.nn.functional.nll_loss(input=logits, target=best)
