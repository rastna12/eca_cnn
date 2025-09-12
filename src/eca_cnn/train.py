import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from eca_cnn.datasets import make_batch
from eca_cnn.models import build_model


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
    # new parameters for experiments and model selection
    model: str = "shallow",
    depth: Optional[int] = None,
    runs_dir: str = "runs",
    tag: Optional[str] = None,
    save_every: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    mdl = build_model(
        model,
        H=H if model == "shallow" else None,
        depth=depth if model == "deep" else None,
        hidden=hidden,
        linear=linear,
    ).to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # prepare run directory and save config
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_parts = [
        f"rule{rule}",
        f"mdl-{model}",
    ]
    if model == "shallow":
        run_name_parts.append(f"H{H}")
    if model == "deep" and depth is not None:
        run_name_parts.append(f"D{depth}")
    if seed is not None:
        run_name_parts.append(f"seed{seed}")
    if tag:
        run_name_parts.append(tag)
    run_name = "_".join(run_name_parts)
    run_dir = Path(runs_dir) / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "rule": rule,
        "H": H,
        "N": N,
        "steps": steps,
        "batch": B,
        "device": device,
        "hidden": hidden,
        "lr": lr,
        "log_every": log_every,
        "seed": seed,
        "linear": linear,
        "model": model,
        "depth": depth,
        "runs_dir": runs_dir,
        "tag": tag,
        "save_every": save_every,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("step,loss,acc\n")

    for step in range(steps):
        x, y = make_batch(B, N, rule, H, device)
        logits = mdl(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                acc = ((logits.sigmoid() > 0.5) == (y > 0.5)).float().mean().item()
            print(
                f"rule {rule:3d} | {model:7s} | H={H:2d} | step {step+1:4d}/{steps} | loss {loss:.3f} | acc {acc:.3f}"
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(f"{step+1},{loss.item():.6f},{acc:.6f}\n")

        if save_every is not None and (step + 1) % save_every == 0:
            ckpt = {
                "model_state": mdl.state_dict(),
                "optimizer_state": opt.state_dict(),
                "step": step + 1,
                "config": config,
            }
            torch.save(ckpt, run_dir / f"checkpoint_step{step+1}.pt")

    # final checkpoint
    ckpt = {
        "model_state": mdl.state_dict(),
        "optimizer_state": opt.state_dict(),
        "step": steps,
        "config": config,
    }
    torch.save(ckpt, run_dir / "checkpoint_final.pt")

    return str(run_dir)


def main():
    p = argparse.ArgumentParser(description="Train ECA models on jump-ahead task and save artifacts.")
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
    p.add_argument("--linear", action="store_true", help="Use linear activations (Identity)")
    p.add_argument("--model", type=str, default="shallow", choices=["shallow", "deep"], help="Model type")
    p.add_argument("--depth", type=int, default=None, help="Depth for deep model")
    p.add_argument("--runs-dir", type=str, default="runs", help="Base directory for run artifacts")
    p.add_argument("--tag", type=str, default=None, help="Optional tag to append to run name")
    p.add_argument("--save-every", type=int, default=None, help="Checkpoint every k steps")
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
        model=args.model,
        depth=args.depth,
        runs_dir=args.runs_dir,
        tag=args.tag,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()


