# eca_plots.py
# Generate space–time diagrams for ECA rules 150, 90, 30
# and parity masks K^(H) for reducible rules (150, 90).

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# --------------------------- ECA simulation (Torch) ---------------------------

def rule_table(rule: int) -> torch.Tensor:
    """Return 8-entry lookup table for a Wolfram rule as uint8 tensor."""
    return torch.tensor([(rule >> i) & 1 for i in range(8)], dtype=torch.uint8)

def evolve_once(state: torch.Tensor, tbl: torch.Tensor) -> torch.Tensor:
    """
    One CA step with periodic boundaries.
    state: [1, N] in {0,1} (uint8)
    returns: [1, N] in {0,1}
    """
    L = torch.roll(state,  1, dims=-1)
    R = torch.roll(state, -1, dims=-1)
    idx = ((L << 2) | (state << 1) | R).long()  # Long for indexing
    return tbl[idx]

def simulate(rule: int, width=256, steps=256, p_init=0.5, seed=0, init="random") -> torch.Tensor:
    """
    Simulate an elementary CA with periodic boundaries.
    Returns grid of shape [steps, width] (uint8). Row 0 is t=0.
    """
    g = torch.Generator().manual_seed(seed)
    if init == "random":
        state = (torch.rand(width, generator=g) < p_init).to(torch.uint8).unsqueeze(0)  # [1,N]
    elif init == "single":
        state = torch.zeros(width, dtype=torch.uint8).unsqueeze(0)
        state[0, width // 2] = 1
    else:
        raise ValueError("init must be 'random' or 'single'")

    tbl = rule_table(rule)
    grid = torch.zeros((steps, width), dtype=torch.uint8)
    grid[0] = state[0]
    for t in range(1, steps):
        state = evolve_once(state, tbl)
        grid[t] = state[0]
    return grid


# ------------------------ Plotting helpers (Matplotlib) -----------------------

def plot_spacetime(grid: torch.Tensor, title: str, outpath: Path, dpi=220):
    """
    Space–time diagram: rows=time (top=0), columns=space; values in {0,1}.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(grid.numpy(), aspect='auto', interpolation='nearest', cmap='binary')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Space")
    ax.set_ylabel("Time")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

def plot_k_mask(mask: np.ndarray, title: str, outpath: Path, dpi=220):
    """
    Visualize a binary mask K^(H) (1D) as a thin horizontal image.
    mask: shape [2H+1], entries in {0,1}.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    img = mask[np.newaxis, :]  # make it 2D: [1, W]
    fig, ax = plt.subplots(figsize=(6, 1.0))
    ax.imshow(img, aspect='auto', interpolation='nearest', cmap='binary')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ------------------- GF(2) parity masks K^(H) via convolution -----------------

def kernel_power_gf2(base: np.ndarray, H: int) -> np.ndarray:
    """
    Repeated convolution over GF(2) of a base kernel (e.g., [1,1,1] for Rule 150),
    producing K^(H) of length (2H+1).
    """
    if H == 0:
        return np.array([1], dtype=np.uint8)
    k = base.astype(np.uint8)
    acc = base.astype(np.uint8)
    for _ in range(1, H):
        # ordinary integer convolution, then reduce mod 2
        acc = np.convolve(acc, k) % 2
    return acc.astype(np.uint8)

def k_mask_rule(rule: int, H: int) -> np.ndarray:
    """
    Return K^(H) for Rule 150 or Rule 90 via GF(2) convolution.
    (For other rules this "single-step kernel" shortcut generally doesn't exist.)
    """
    if rule == 150:
        base = np.array([1, 1, 1], dtype=np.uint8)  # z^{-1} + 1 + z
    elif rule == 90:
        base = np.array([1, 0, 1], dtype=np.uint8)  # z^{-1} + z
    else:
        raise ValueError("k_mask_rule only defined for Rules 150 and 90.")
    return kernel_power_gf2(base, H)


# ------------------------------- Main / CLI ----------------------------------

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Space–time diagrams for three rules, two ICs each
    for rule, label in [(150, "Rule 150 (reducible)"),
                        ( 90, "Rule 90 (reducible)"),
                        ( 30, "Rule 30 (irreducible)")]:
        # Random IC
        grid = simulate(rule, width=args.width, steps=args.steps,
                        p_init=args.p_init, seed=args.seed, init="random")
        plot_spacetime(grid, f"{label}: random IC",
                       outdir / f"rule{rule}_spacetime_random.png")

        # Single-cell IC
        grid_single = simulate(rule, width=args.width, steps=args.steps,
                               seed=args.seed_single, init="single")
        plot_spacetime(grid_single, f"{label}: single-cell IC",
                       outdir / f"rule{rule}_spacetime_single.png")

    # 2) Parity masks K^(H) for reducible rules (150/90)
    for rule in (150, 90):
        K = k_mask_rule(rule, args.Hmask)
        # Sanity: length should be 2H+1
        assert len(K) == 2 * args.Hmask + 1
        plot_k_mask(
            K,
            title=f"K^({args.Hmask}) — Rule {rule}",
            outpath=outdir / f"K_rule{rule}_H{args.Hmask}.png"
        )

    print("Saved figures to:", outdir.resolve())
    print("Examples:")
    print("  -", (outdir / "rule150_spacetime_random.png").name)
    print("  -", (outdir / "rule90_spacetime_random.png").name)
    print("  -", (outdir / "rule30_spacetime_random.png").name)
    print("  -", (outdir / f"K_rule150_H{args.Hmask}.png").name)
    print("  -", (outdir / f"K_rule90_H{args.Hmask}.png").name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make ECA plots and parity masks.")
    p.add_argument("--width", type=int, default=128, help="Lattice width N")
    p.add_argument("--steps", type=int, default=64, help="Time steps")
    p.add_argument("--p-init", type=float, default=0.5, help="Bernoulli p for random IC")
    p.add_argument("--seed", type=int, default=42, help="Seed for random IC")
    p.add_argument("--seed-single", type=int, default=1, help="Seed for single IC (center=1)")
    p.add_argument("--Hmask", type=int, default=16, help="H for K^(H) masks (Rules 150/90)")
    p.add_argument("--outdir", type=str, default="figs", help="Output directory")
    args = p.parse_args()
    main(args)
