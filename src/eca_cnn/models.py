import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self, H: int, hidden: int = 16):
        super().__init__()
        k = 2 * H + 1
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=k, padding=k // 2, padding_mode="circular", bias=False)
        self.act = nn.ReLU()
        self.out = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.conv1(x)
        a = self.act(h)
        logits = self.out(a)
        return logits


