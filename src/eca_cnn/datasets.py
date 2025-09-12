import torch

from eca_cnn.eca_core import rule_table, jump_ahead


def make_batch(B: int, N: int, rule: int, H: int, device: str = "cpu"):
    x_bits = torch.randint(0, 2, (B, N), dtype=torch.long, device=device)
    y_bits = jump_ahead(x_bits.clone(), rule_table(rule).to(device), H)
    x = x_bits.unsqueeze(1).float()
    y = y_bits.unsqueeze(1).float()
    return x, y


