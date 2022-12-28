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
    logits: TensorType["response_length", "vocab_size", torch.float]

    def to(self, device: torch.device) -> 'PPOSample':
        return PPOSample(
            query=self.query.to(device),
            response=self.response.to(device),
            reward=self.reward.to(device),
            values=self.values.to(device),
            logits=self.logits.to(device),
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
    logits: TensorType["batch_size", "response_size", "vocab_size", torch.long]
    values: TensorType["batch_size", "response_size", torch.float]
    rewards: TensorType["batch_size", "response_size", torch.float]
    pad_token: int

    def to_model_input(self) -> PPOBatchModelInput:
        input_ids = torch.cat([self.query, self.response], dim=1)
        attention_mask = input_ids.not_equal(self.pad_token)
        return PPOBatchModelInput(input_ids=input_ids, attention_mask=attention_mask)

    def to(self, device: torch.device) -> 'PPOBatch':
        return PPOBatch(
            query=self.query.to(device),
            response=self.response.to(device),
            logits=self.logits.to(device),
            values=self.values.to(device),
            rewards=self.rewards.to(device),
        )

    def pin_memory(self) -> 'PPOBatch':
        return PPOBatch(
            query=self.query.pin_memory(),
            response=self.response.pin_memory(),
            logits=self.logits.pin_memory(),
            values=self.values.pin_memory(),
            rewards=self.rewards.pin_memory(),
        )


@dataclasses.dataclass
class ActorCriticOutput:
    logits: TensorType["batch_size", "sequence_length", "vocab_size"]
    values: TensorType["batch_size", "sequence_length"]
