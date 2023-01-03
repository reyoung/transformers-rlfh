import torch.nn
from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers_rlfh.types import ActorCriticOutput


class ValueHead(torch.nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()

        layers = []
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(feature_dim, 1))

        self.head = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class ActorCriticLM(torch.nn.Module):
    def __init__(self, base: GPTNeoForCausalLM):
        super().__init__()
        self.base = base
        self.value_head = ValueHead(base.config.hidden_size)
        self.value_head.to(self.base.device)

    @property
    def device(self):
        return self.base.device

    def forward(self, input_ids, attention_mask):
        output: CausalLMOutputWithPast = self.base(input_ids=input_ids, attention_mask=attention_mask,
                                                   output_hidden_states=True)
        value = self.value_head(output.hidden_states[-1]).squeeze(-1)
        return ActorCriticOutput(logits=output.logits, values=value)
