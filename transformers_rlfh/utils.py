import json
from typing import Any
import torch

__all__ = ['JSONEncoder']


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        else:
            return super().default(o)
