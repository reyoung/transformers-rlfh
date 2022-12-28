from typing import List, Optional

import torch

__all__ = ['left_pad_sequence']


def left_pad_sequence(sequence, padding_value: int):
    max_length = max(seq.shape[0] for seq in sequence)
    stacks: List[Optional[torch.Tensor]] = [None] * len(sequence)
    for i, seq in enumerate(sequence):
        pad_length = max_length - seq.shape[0]
        if pad_length == 0:
            stacks[i] = seq
        else:
            with torch.no_grad():
                pad = torch.nn.ConstantPad1d((pad_length, 0), padding_value)
                seq = pad(seq)
                stacks[i] = seq
    return torch.stack(stacks)
