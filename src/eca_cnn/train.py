import argparse
from typing import Optional

import torch
import torch.nn as nn

from eca_cnn.datasets import make_batch
from eca_cnn.models import ShallowCNN


def train_rule(
    rule: int,
    H: int = 16,
    N: int = 256,
    steps: int = 500,
    B: int = 64,
    device: str = "cuda",
    hidden: int = 16,
    lr: float = 1e-3,
    log_every: int = 25,
    seed: Optional[int] = None,
    linear: bool = False,
):
    if seed is not None:
        torch.manual_seed(seed)

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    model = ShallowCNN(H, hidden=hidden).to(device)
    if linear:
        model.act = nn.Identity()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(steps):
        x, y = make_batch(B, N, rule, H, device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                acc = ((logits.sigmoid() > 0.5) == (y > 0.5)).float().mean().item()
            print(f"rule {rule:3d} | H={H:2d} | step {step+1:3d} | loss {loss:.3f} | acc {acc:.3f}")

    return model


def main():
    p = argparse.ArgumentParser(description="Train ShallowCNN on ECA jump-ahead task.")
    p.add_argument("--rule", type=int, default=150)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--linear", action="store_true", help="Use linear model (Identity activation)")
    args = p.parse_args()

    train_rule(
        rule=args.rule,
        H=args.H,
        N=args.N,
        steps=args.steps,
        B=args.batch,
        device=args.device,
        hidden=args.hidden,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
        linear=args.linear,
    )


if __name__ == "__main__":
    main()


