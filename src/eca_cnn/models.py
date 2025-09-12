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


class DeepRolloutCNN(nn.Module):
    """
    Rollout-style CNN: stack of width-3 circular Conv1d layers to emulate sequential composition.
    The depth parameter controls the number of composition steps. A final 1x1 conv produces logits.
    """

    def __init__(self, depth: int, hidden: int = 32, activation: str = "relu"):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(depth):
            conv = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, padding_mode="circular", bias=False)
            layers.append(conv)
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "identity":
                layers.append(nn.Identity())
            else:
                layers.append(nn.ReLU())
            in_channels = hidden
        self.features = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.features(x)
        logits = self.out(h)
        return logits


def build_model(name: str, *, H: int | None = None, depth: int | None = None, hidden: int = 32, linear: bool = False):
    """
    Construct a model by name.
    - name: "shallow" or "deep"
    - H: horizon for shallow receptive field (required for shallow)
    - depth: number of composition layers (required for deep)
    - hidden: hidden channels
    - linear: if True, replace activations with Identity (for shallow only)
    """
    name = name.lower()
    if name == "shallow":
        if H is None:
            raise ValueError("H must be provided for shallow model")
        model = ShallowCNN(H, hidden=hidden)
        if linear:
            model.act = nn.Identity()
        return model
    elif name == "deep":
        if depth is None:
            raise ValueError("depth must be provided for deep model")
        activation = "identity" if linear else "relu"
        return DeepRolloutCNN(depth=depth, hidden=hidden, activation=activation)
    else:
        raise ValueError(f"Unknown model name: {name}")

