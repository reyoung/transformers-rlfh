import json

import torch
from torchtyping import TensorType
import dataclasses

__all__ = ['PPOSample', 'PPOBatch', 'ActorCriticOutput']


@dataclasses.dataclass
class PPOSample:
    """
    query: Query的tensor。query length每个sample可以不同
    response: Response的tensor。response length每个sample可以不同
    reward: 每一个token的reward。其实大多数情况下，只有sequence结尾的时候给一个score就行了。不过为了增加 kl散度的惩罚项。所以每一个token
            初始化成KL散度的惩罚项。
    values: actor-critic网络中，critic的输出。这个values是 ref model的values
    logits: actor-critic网络中，actor的输出。即不同token的概率。 这个probs是ref model的probs
    """
    query: TensorType["query_length", torch.long]
    response: TensorType["response_length", torch.long]
    reward: TensorType["response_length", torch.float]
    values: TensorType["response_length", torch.float]
    log_probs: TensorType["response_length", torch.float]

    def as_dict(self):
        return dataclasses.asdict(self)

    def to(self, device: torch.device) -> 'PPOSample':
        return PPOSample(
            query=self.query.to(device),
            response=self.response.to(device),
            reward=self.reward.to(device),
            values=self.values.to(device),
            log_probs=self.log_probs.to(device),
        )


@dataclasses.dataclass
class PPOBatchModelInput:
    """
    送给GPT模型的输入

    input_ids: 组合query和response的tensor
    attention_mask: attention的mask
    """
    input_ids: TensorType["batch_size", "query_length+response_size", torch.long]
    attention_mask: TensorType["batch_size", "query_length+response_size", torch.long]


@dataclasses.dataclass
class PPOBatch:
    """
    values: actor-critic网络中，critic的输出。这个values是 ref model的values
    logits: actor-critic网络中，actor的输出。即不同token的概率。 这个probs是ref model的probs
    """
    query: TensorType["batch_size", "query_length", torch.long]  # left padding
    response: TensorType["batch_size", "response_length", torch.long]  # right padding
    log_probs: TensorType["batch_size", "response_size", torch.float]
    values: TensorType["batch_size", "response_size", torch.float]
    rewards: TensorType["batch_size", "response_size", torch.float]
    padding_value: int

    def to_model_input(self) -> PPOBatchModelInput:
        input_ids = torch.cat([self.query, self.response], dim=1)
        attention_mask = input_ids.not_equal(self.padding_value)
        return PPOBatchModelInput(input_ids=input_ids, attention_mask=attention_mask)

    def to(self, device: torch.device) -> 'PPOBatch':
        return PPOBatch(
            query=self.query.to(device),
            response=self.response.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            rewards=self.rewards.to(device),
            padding_value=self.padding_value,
        )

    def pin_memory(self) -> 'PPOBatch':
        return PPOBatch(
            query=self.query.pin_memory(),
            response=self.response.pin_memory(),
            log_probs=self.log_probs.pin_memory(),
            values=self.values.pin_memory(),
            rewards=self.rewards.pin_memory(),
            padding_value=self.padding_value,
        )


@dataclasses.dataclass
class ActorCriticOutput:
    logits: TensorType["batch_size", "sequence_length", "vocab_size"]
    values: TensorType["batch_size", "sequence_length"]

    def slice_response(self, query_length: int) -> 'ActorCriticOutput':
        return ActorCriticOutput(
            logits=self.logits[:, query_length:, :],
            values=self.values[:, query_length:],
        )
