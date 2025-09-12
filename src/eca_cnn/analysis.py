import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_metrics_csv(path: Path) -> Dict[str, np.ndarray]:
    steps = []
    losses = []
    accs = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            s, l, a = parts
            steps.append(int(s))
            losses.append(float(l))
            accs.append(float(a))
    return {"step": np.array(steps), "loss": np.array(losses), "acc": np.array(accs)}


def collect_runs(runs_dir: Path) -> List[Dict]:
    runs = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        cfg_path = run_path / "config.json"
        metrics_path = run_path / "metrics.csv"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        metrics = read_metrics_csv(metrics_path)
        runs.append({"path": run_path, "config": cfg, "metrics": metrics})
    return runs


def plot_acc_vs_H(runs: List[Dict], *, model: str, rule: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # group by H, aggregate final accuracy across seeds (and depths if deep is filtered separately)
    H_to_accs: Dict[int, List[float]] = {}
    for r in runs:
        cfg = r["config"]
        if cfg.get("model") != model or cfg.get("rule") != rule:
            continue
        H = int(cfg.get("H"))
        acc = float(r["metrics"]["acc"][-1]) if len(r["metrics"]["acc"]) else np.nan
        H_to_accs.setdefault(H, []).append(acc)

    Hs = sorted(H_to_accs.keys())
    means = [np.nanmean(H_to_accs[H]) for H in Hs]
    stds = [np.nanstd(H_to_accs[H]) for H in Hs]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(Hs, means, yerr=stds, fmt="-o", capsize=3)
    ax.set_title(f"Accuracy vs H — model={model}, rule={rule}")
    ax.set_xlabel("H")
    ax.set_ylabel("Final accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"acc_vs_H_model-{model}_rule-{rule}.png", dpi=220)
    plt.close(fig)


def plot_acc_vs_depth(runs: List[Dict], *, rule: int, H: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    depth_to_accs: Dict[int, List[float]] = {}
    for r in runs:
        cfg = r["config"]
        if cfg.get("model") != "deep" or cfg.get("rule") != rule or cfg.get("H") != H:
            continue
        depth = int(cfg.get("depth")) if cfg.get("depth") is not None else 0
        acc = float(r["metrics"]["acc"][-1]) if len(r["metrics"]["acc"]) else np.nan
        depth_to_accs.setdefault(depth, []).append(acc)

    depths = sorted(depth_to_accs.keys())
    means = [np.nanmean(depth_to_accs[d]) for d in depths]
    stds = [np.nanstd(depth_to_accs[d]) for d in depths]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(depths, means, yerr=stds, fmt="-o", capsize=3)
    ax.set_title(f"Accuracy vs depth — rule={rule}, H={H}")
    ax.set_xlabel("depth")
    ax.set_ylabel("Final accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"acc_vs_depth_rule-{rule}_H-{H}.png", dpi=220)
    plt.close(fig)


def plot_training_curves_for_run(run: Dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = run["config"]
    metrics = run["metrics"]
    if len(metrics["step"]) == 0:
        return
    steps = metrics["step"]
    loss = metrics["loss"]
    acc = metrics["acc"]

    title_bits = [
        f"rule={cfg.get('rule')}",
        f"model={cfg.get('model')}",
        f"H={cfg.get('H')}",
    ]
    if cfg.get("model") == "deep" and cfg.get("depth") is not None:
        title_bits.append(f"D={cfg.get('depth')}")
    if cfg.get("seed") is not None:
        title_bits.append(f"seed={cfg.get('seed')}")
    title = ", ".join(title_bits)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    axes[0].plot(steps, loss, label="loss")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(steps, acc, label="accuracy", color="tab:green")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = run["path"].name + "_curves.png"
    fig.savefig(outdir / fname, dpi=220)
    plt.close(fig)


def plot_final_acc_vs_H_all_models(runs: List[Dict], *, rule: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # aggregate by (model, depth) then H
    key_to_H_to_accs: Dict[Tuple[str, int], Dict[int, List[float]]] = {}
    for r in runs:
        cfg = r["config"]
        if cfg.get("rule") != rule:
            continue
        model = str(cfg.get("model"))
        depth = int(cfg.get("depth")) if (model == "deep" and cfg.get("depth") is not None) else 0
        H = int(cfg.get("H"))
        acc = float(r["metrics"]["acc"][-1]) if len(r["metrics"]["acc"]) else np.nan
        key = (model, depth)
        key_to_H_to_accs.setdefault(key, {}).setdefault(H, []).append(acc)

    if not key_to_H_to_accs:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for (model, depth), H_to_accs in sorted(key_to_H_to_accs.items()):
        Hs = sorted(H_to_accs.keys())
        means = [np.nanmean(H_to_accs[H]) for H in Hs]
        stds = [np.nanstd(H_to_accs[H]) for H in Hs]
        label = f"{model}" if model != "deep" else f"deep-D{depth}"
        ax.errorbar(Hs, means, yerr=stds, fmt="-o", capsize=3, label=label)
    ax.set_title(f"Final accuracy vs H — rule={rule}")
    ax.set_xlabel("H")
    ax.set_ylabel("Final accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"final_acc_vs_H_rule-{rule}.png", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Aggregate run artifacts and make summary plots.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--outdir", type=str, default="figs/analysis")
    p.add_argument("--rules", type=str, default="150,90,30")
    p.add_argument("--Hs", type=str, default="8,16,32")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    runs = collect_runs(runs_dir)

    rules = [int(x) for x in args.rules.split(",") if x]
    Hs = [int(x) for x in args.Hs.split(",") if x]

    # Per-run training curves
    curves_out = outdir / "curves"
    for r in runs:
        plot_training_curves_for_run(r, outdir=curves_out)

    # Final accuracy vs H, one plot per model and rule (existing)
    for rule in rules:
        for model in ("shallow", "deep"):
            plot_acc_vs_H(runs, model=model, rule=rule, outdir=outdir)

    # Final accuracy vs H, all models overlayed per rule
    for rule in rules:
        plot_final_acc_vs_H_all_models(runs, rule=rule, outdir=outdir)

    # Accuracy vs depth for each rule and H (existing)
    for rule in rules:
        for H in Hs:
            plot_acc_vs_depth(runs, rule=rule, H=H, outdir=outdir)

    print(f"Saved analysis figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()


