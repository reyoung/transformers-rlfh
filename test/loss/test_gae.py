import torch

from transformers_rlfh.loss.gae import calculate_gae

def test_gae():
    generator = torch.Generator(device="cuda").manual_seed(42)

    value = torch.rand(size=(16, 128), generator=generator, dtype=torch.float32, device="cuda")
    reward = torch.rand(size=(16, 128), generator=generator, dtype=torch.float32, device="cuda")

    result = calculate_gae(value, reward)
    assert result.returns.shape == (16, 128)
    assert result.advantages.shape == (16, 128)
