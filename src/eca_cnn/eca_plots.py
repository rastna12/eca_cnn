import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from eca_cnn.plot_style import apply_ieee_style, savefig_ieee

from eca_cnn.eca_core import simulate


def plot_spacetime(grid: torch.Tensor, title: str, outpath: Path, dpi=220):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    ax.imshow(grid.numpy(), aspect='auto', interpolation='nearest', cmap='binary')
    ax.set_title(title)
    ax.set_xlabel("Space")
    ax.set_ylabel("Time")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    savefig_ieee(fig, outpath)
    plt.close(fig)


def plot_k_mask(mask: np.ndarray, title: str, outpath: Path, dpi=220):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    img = mask[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(3.35, 0.8))
    ax.imshow(img, aspect='auto', interpolation='nearest', cmap='binary')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    savefig_ieee(fig, outpath)
    plt.close(fig)


def kernel_power_gf2(base: np.ndarray, H: int) -> np.ndarray:
    if H == 0:
        return np.array([1], dtype=np.uint8)
    k = base.astype(np.uint8)
    acc = base.astype(np.uint8)
    for _ in range(1, H):
        acc = np.convolve(acc, k) % 2
    return acc.astype(np.uint8)


def k_mask_rule(rule: int, H: int) -> np.ndarray:
    if rule == 150:
        base = np.array([1, 1, 1], dtype=np.uint8)
    elif rule == 90:
        base = np.array([1, 0, 1], dtype=np.uint8)
    else:
        raise ValueError("k_mask_rule only defined for Rules 150 and 90.")
    return kernel_power_gf2(base, H)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for rule, label in [(150, "Rule 150 (reducible)"), (90, "Rule 90 (reducible)"), (30, "Rule 30 (irreducible)")]:
        grid = simulate(rule, width=args.width, steps=args.steps, p_init=args.p_init, seed=args.seed, init="random")
        plot_spacetime(grid, f"{label}: random IC", outdir / f"rule{rule}_spacetime_random.png")

        grid_single = simulate(rule, width=args.width, steps=args.steps, seed=args.seed_single, init="single")
        plot_spacetime(grid_single, f"{label}: single-cell IC", outdir / f"rule{rule}_spacetime_single.png")

    for rule in (150, 90):
        K = k_mask_rule(rule, args.Hmask)
        assert len(K) == 2 * args.Hmask + 1
        plot_k_mask(K, title=f"K^({args.Hmask}) — Rule {rule}", outpath=outdir / f"K_rule{rule}_H{args.Hmask}.png")

    print("Saved figures to:", outdir.resolve())


def main():
    p = argparse.ArgumentParser(description="Make ECA plots and parity masks.")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--p-init", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-single", type=int, default=1)
    p.add_argument("--Hmask", type=int, default=16)
    p.add_argument("--outdir", type=str, default="figs")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()


