import torch
from torchtyping import TensorType
import dataclasses


@dataclasses.dataclass
class PPOSample:
    query: TensorType["query_length", torch.long]
    response: TensorType["response_length", torch.long]
    reward: TensorType["response_length", torch.float]
    values: TensorType["response_length", torch.float]
    log_probs: TensorType["response_length", "vocab_size", torch.float]

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
    input_ids: TensorType["batch_size", "query_length+response_size", torch.long]
    attention_mask: TensorType["batch_size", "query_length+response_size", torch.long]


@dataclasses.dataclass
class PPOBatch:
    query: TensorType["batch_size", "query_length", torch.long]  # left padding
    response: TensorType["batch_size", "response_length", torch.long]  # right padding
    log_probs: TensorType["batch_size", "response_size", "vocab_size", torch.long]
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
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            rewards=self.rewards.to(device),
        )

    def pin_memory(self) -> 'PPOBatch':
        return PPOBatch(
            query=self.query.pin_memory(),
            response=self.response.pin_memory(),
            log_probs=self.log_probs.pin_memory(),
            values=self.values.pin_memory(),
            rewards=self.rewards.pin_memory(),
        )
