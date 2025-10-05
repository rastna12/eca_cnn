import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from eca_cnn.models import build_model
from eca_cnn.plot_style import apply_ieee_style, savefig_ieee, use_ieee_style
from eca_cnn.eca_core import rule_table, jump_ahead


def _rule_category(rule: int) -> str:
    return "Reducible" if rule in (90, 150) else "Irreducible"


def _rule_style(rule: int):
    # Colors chosen to be colorblind-friendly and distinct
    color_map = {30: "tab:orange", 110: "tab:red", 90: "tab:blue", 150: "tab:cyan"}
    marker_map = {30: "o", 110: "o", 90: "s", 150: "s"}
    linestyle_map = {30: "-", 110: "-", 90: "--", 150: "--"}
    return {
        "color": color_map.get(rule, "black"),
        "marker": marker_map.get(rule, "o"),
        "linestyle": linestyle_map.get(rule, "-")
    }


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

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    style = _rule_style(rule)
    label = f"Rule {rule} ({_rule_category(rule)})"
    ax.errorbar(Hs, means, yerr=stds, capsize=3, label=label, **style)
    ax.set_title(f"Final Accuracy vs Prediction Horizon $H$ — Rule={rule}")
    ax.set_xlabel("Prediction Horizon $H$")
    ax.set_ylabel("Final Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig_ieee(fig, outdir / f"acc_vs_H_rule-{rule}.png")
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

    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    style = _rule_style(rule)
    label = f"Rule {rule} ({_rule_category(rule)})"
    ax.errorbar(depths, means, yerr=stds, capsize=3, label=label, **style)
    ax.set_title(f"Accuracy vs Depth — Rule={rule}")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Final Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig_ieee(fig, outdir / f"acc_vs_depth_rule-{rule}.png")
    plt.close(fig)


def plot_training_curves_for_run(run: Dict, outdir: Path, *, ylims: Optional[Dict[str, Tuple[float, float]]] = None):
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = run["config"]
    metrics = run["metrics"]
    if len(metrics["step"]) == 0:
        return
    steps = metrics["step"]
    loss = metrics["loss"]
    acc = metrics["acc"]

    rule_val = cfg.get("rule")
    model_val = cfg.get("model")
    depth_val = cfg.get("depth")
    H_val = cfg.get("H")
    seed_val = cfg.get("seed")

    parts = [f"Rule {rule_val}", f"H: {H_val}"]
    if model_val == "deep" and depth_val is not None:
        parts.insert(2, f"Depth: {depth_val}")
    if seed_val is not None:
        parts.append(f"Seed: {seed_val}")
    title = "Training Curves — " + "  |  ".join(parts)

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.2), sharex=True)
    axes[0].plot(steps, loss, label="Loss")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(steps, acc, label="Accuracy", color="tab:green")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    if ylims is not None:
        loss_lim = ylims.get("loss")
        acc_lim = ylims.get("acc")
        if loss_lim is not None:
            axes[0].set_ylim(*loss_lim)
        if acc_lim is not None:
            axes[1].set_ylim(*acc_lim)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = run["path"].name + "_curves.png"
    savefig_ieee(fig, outdir / fname)
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

    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    for (model, depth), H_to_accs in sorted(key_to_H_to_accs.items()):
        Hs = sorted(H_to_accs.keys())
        means = [np.nanmean(H_to_accs[H]) for H in Hs]
        stds = [np.nanstd(H_to_accs[H]) for H in Hs]
        label = f"{model}" if model != "deep" else f"deep-D{depth}"
        ax.errorbar(Hs, means, yerr=stds, fmt="-o", capsize=3, label=label)
    ax.set_title(f"Final Accuracy vs Prediction Horizon $H$ — Rule={rule}")
    ax.set_xlabel("Prediction Horizon $H$")
    ax.set_ylabel("Final Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    savefig_ieee(fig, outdir / f"final_acc_vs_H_rule-{rule}.png")
    plt.close(fig)


def _select_best_run(runs: List[Dict], *, rule: int, H: int) -> Optional[Dict]:
    """
    Among runs matching (rule, H), pick the one with the highest final accuracy.
    Returns the run dict or None if not found.
    """
    candidates: List[Tuple[float, Dict]] = []
    for r in runs:
        cfg = r["config"]
        if int(cfg.get("rule")) != rule or int(cfg.get("H")) != H:
            continue
        acc_series = r["metrics"].get("acc", np.array([]))
        if acc_series.size == 0:
            continue
        final_acc = float(acc_series[-1])
        candidates.append((final_acc, r))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_model_for_run(run: Dict, device: str = "cpu"):
    cfg = run["config"]
    model_name = str(cfg.get("model"))
    H = int(cfg.get("H")) if cfg.get("H") is not None else None
    depth = int(cfg.get("depth")) if cfg.get("depth") is not None else None
    hidden = int(cfg.get("hidden", 32))

    mdl = build_model(
        model_name,
        H=H if model_name == "shallow" else None,
        depth=depth if model_name == "deep" else None,
        hidden=hidden,
        linear=bool(cfg.get("linear", False)),
    ).to(device)

    ckpt_path = run["path"] / "checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    mdl.load_state_dict(ckpt["model_state"])
    mdl.eval()
    return mdl


def _aggregate_shallow_means(runs: List[Dict], rules: List[int], Hs: List[int]) -> Dict[int, Dict[str, any]]:
    """
    Aggregate shallow model metrics by computing means across seeds for each (rule, H).
    Returns dict: rule -> {"H_means": {H: mean_acc}, "overall_bce": float, "overall_acc": float}
    """
    result = {}
    for rule in rules:
        # Filter shallow runs for this rule
        rule_runs = [r for r in runs if r["config"].get("model") == "shallow" and r["config"].get("rule") == rule]
        
        if not rule_runs:
            result[rule] = {"H_means": {}, "overall_bce": np.nan, "overall_acc": np.nan}
            continue
        
        # Per-H means
        H_means = {}
        for H in Hs:
            H_accs = []
            for r in rule_runs:
                if r["config"].get("H") == H:
                    acc_series = r["metrics"].get("acc", np.array([]))
                    if acc_series.size > 0:
                        H_accs.append(float(acc_series[-1]))
            H_means[H] = np.nanmean(H_accs) if H_accs else np.nan
        
        # Overall means across all Hs and seeds for this rule
        all_losses = []
        all_accs = []
        for r in rule_runs:
            if r["config"].get("H") in Hs:
                loss_series = r["metrics"].get("loss", np.array([]))
                acc_series = r["metrics"].get("acc", np.array([]))
                if loss_series.size > 0:
                    all_losses.append(float(loss_series[-1]))
                if acc_series.size > 0:
                    all_accs.append(float(acc_series[-1]))
        
        result[rule] = {
            "H_means": H_means,
            "overall_bce": np.nanmean(all_losses) if all_losses else np.nan,
            "overall_acc": np.nanmean(all_accs) if all_accs else np.nan,
        }
    
    return result


def render_latex_table_shallow_means(
    agg: Dict[int, Dict[str, any]],
    Hs: List[int],
    caption: str,
    label: str,
    precision: int
) -> str:
    """
    Render LaTeX tabularx table for shallow model performance summary.
    """
    n_H_cols = len(Hs)
    # Build column spec: l for Rule, then n_H_cols centered X columns
    col_spec = "l" + "*{%d}{>{\\centering\\arraybackslash}X}" % n_H_cols
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append("  \\setlength{\\tabcolsep}{3pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.05}")
    lines.append("  \\scriptsize")
    lines.append(f"  \\begin{{tabularx}}{{\\columnwidth}}{{@{{}}{col_spec}@{{}}}}")
    lines.append("    \\toprule")
    
    # Header: Rule | H1 H2 ...
    H_headers = " & ".join(str(H) for H in Hs)
    lines.append(f"    & \\multicolumn{{{n_H_cols}}}{{c}}{{Accuracy @ Horizon $H$}} \\\\")
    lines.append(f"    \\cmidrule(l){{2-{1+n_H_cols}}}")
    lines.append(f"    Rule & {H_headers} \\\\")
    lines.append("    \\midrule")
    
    # Data rows
    for rule in sorted(agg.keys()):
        row_data = agg[rule]
        H_means = row_data["H_means"]
        
        row_parts = [str(rule)]
        for H in Hs:
            val = H_means.get(H, np.nan)
            if np.isnan(val):
                row_parts.append("--")
            else:
                row_parts.append(f"{val:.{precision}f}")
        
        lines.append("    " + " & ".join(row_parts) + " \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabularx}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def plot_truth_vs_prediction(
    run: Dict,
    *,
    H: int,
    seed: int = 0,
    outdir: Path,
):
    """
    Create a two-row binary visualization: [ground truth; model prediction] at t+H,
    with compact but non-overlapping spacing.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = run["config"]
    rule = int(cfg.get("rule"))
    N = int(cfg.get("N", 256))
    model = str(cfg.get("model"))
    depth = int(cfg.get("depth")) if cfg.get("depth") is not None else None

    device = "cpu"
    mdl = _load_model_for_run(run, device=device)

    g = torch.Generator().manual_seed(seed)
    x_bits = torch.randint(0, 2, (1, N), dtype=torch.long, generator=g)
    x = x_bits.unsqueeze(1).float()  # (1,1,N)

    with torch.no_grad():
        logits = mdl(x)
        pred_bits = (logits.sigmoid() > 0.5).to(torch.uint8)[0, 0]

    # Ground truth via ECA simulator jump-ahead
    tbl = rule_table(rule)
    y_true_bits = jump_ahead(x_bits.clone(), tbl, H)[0]

    # Prepare images for subplots
    truth_img = y_true_bits.unsqueeze(0).cpu().numpy().astype(np.uint8)
    pred_img = pred_bits.unsqueeze(0).cpu().numpy().astype(np.uint8)

    # --- Plot with compact, non-overlapping layout ---
    # Keep a modest height; let constrained_layout allocate space for titles.
    fig, axes = plt.subplots(
        2, 1, figsize=(3.35, 1.75), sharex=True, constrained_layout=True
    )

    axes[0].imshow(truth_img, aspect="auto", interpolation="nearest",
                   cmap="binary", vmin=0, vmax=1)
    axes[0].set_title(r"Ground Truth $(t+H)$", pad=2)

    axes[1].imshow(pred_img, aspect="auto", interpolation="nearest",
                   cmap="binary", vmin=0, vmax=1)
    axes[1].set_title(r"Prediction $(t+H)$", pad=2)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # Slightly reduce inter-row spacing while keeping headroom for the suptitle
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02)

    # Let constrained_layout position this (do NOT force y).
    title = f"Rule {rule} ({_rule_category(rule)}) — $H={H}$"
    fig.suptitle(title)

    # --- Save ---
    fname = f"qual_rule-{rule}_H-{H}_model-{model}"
    if model == "deep" and depth is not None:
        fname += f"_D{depth}"
    fname += f"_seed{seed}.png"
    savefig_ieee(fig, outdir / fname)
    plt.close(fig)




def main():
    p = argparse.ArgumentParser(description="Aggregate run artifacts and make summary plots.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--outdir", type=str, default="figs/analysis")
    p.add_argument("--rules", type=str, default="150,90,30")
    p.add_argument("--Hs", type=str, default="8,16,32")
    p.add_argument("--qual-H", type=int, default=64, help="H for qualitative truth vs prediction plots")
    p.add_argument("--qual-seed", type=int, default=0, help="Seed for input vector for qualitative plots")
    p.add_argument("--latex-table", action="store_true", help="Generate LaTeX performance table")
    p.add_argument("--latex-out", type=str, default=None, help="Path to save LaTeX table (default: print to stdout)")
    p.add_argument("--table-caption", type=str, default="Prediction performance across rules and horizons", 
                    help="Caption for LaTeX table")
    p.add_argument("--table-label", type=str, default="tab:metrics", help="Label for LaTeX table")
    p.add_argument("--table-precision", type=int, default=2, help="Decimal precision for table values")
    args = p.parse_args()

    # Apply IEEE style up-front to cover any figures created below
    apply_ieee_style()

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    runs = collect_runs(runs_dir)

    rules = [int(x) for x in args.rules.split(",") if x]
    Hs = [int(x) for x in args.Hs.split(",") if x]

    # Generate LaTeX table if requested
    if args.latex_table:
        agg = _aggregate_shallow_means(runs, rules, Hs)
        latex = render_latex_table_shallow_means(
            agg, 
            Hs, 
            args.table_caption, 
            args.table_label, 
            args.table_precision
        )
        
        if args.latex_out:
            latex_path = Path(args.latex_out)
            latex_path.parent.mkdir(parents=True, exist_ok=True)
            with latex_path.open("w", encoding="utf-8") as f:
                f.write(latex)
            print(f"LaTeX table saved to {latex_path.resolve()}")
        else:
            print(latex)

    # Per-run training curves with consistent y-limits across runs
    curves_out = outdir / "curves"
    # Compute global y-limits
    all_losses = []
    all_accs = []
    for r in runs:
        m = r["metrics"]
        if len(m.get("loss", [])):
            all_losses.extend(m["loss"])  # type: ignore[arg-type]
        if len(m.get("acc", [])):
            all_accs.extend(m["acc"])  # type: ignore[arg-type]
    loss_ylim: Optional[Tuple[float, float]] = None
    acc_ylim: Optional[Tuple[float, float]] = None
    if all_losses:
        # pad a little for headroom
        lmin = float(min(all_losses))
        lmax = float(max(all_losses))
        pad = 0.05 * (lmax - lmin if lmax > lmin else 1.0)
        loss_ylim = (max(0.0, lmin - pad), lmax + pad)
    if all_accs:
        amin = float(min(all_accs))
        amax = float(max(all_accs))
        pad = 0.02 * (amax - amin if amax > amin else 1.0)
        acc_ylim = (max(0.0, amin - pad), min(1.0, amax + pad))
    ylims = {"loss": loss_ylim, "acc": acc_ylim}

    for r in runs:
        plot_training_curves_for_run(r, outdir=curves_out, ylims=ylims)

    # Final accuracy vs H, one plot per model and rule (existing)
    for rule in rules:
        for model in ("shallow", "deep"):
            plot_acc_vs_H(runs, model=model, rule=rule, outdir=outdir)

    # Final accuracy vs H, all models overlayed per rule (kept for reference)
    for rule in rules:
        plot_final_acc_vs_H_all_models(runs, rule=rule, outdir=outdir)

    # Accuracy vs depth for each rule and H (existing)
    for rule in rules:
        for H in Hs:
            plot_acc_vs_depth(runs, rule=rule, H=H, outdir=outdir)

    # New: For each model variant (e.g., shallow, deep-Dk), plot all rules as curves on one figure
    # Build set of model variants present
    model_keys: List[Tuple[str, int]] = []
    seen = set()
    for r in runs:
        cfg = r["config"]
        model = str(cfg.get("model"))
        depth = int(cfg.get("depth")) if (model == "deep" and cfg.get("depth") is not None) else 0
        key = (model, depth)
        if key not in seen:
            seen.add(key)
            model_keys.append(key)

    for (model, depth) in model_keys:
        # Aggregate by rule, then H
        rule_to_H_to_accs: Dict[int, Dict[int, List[float]]] = {}
        for r in runs:
            cfg = r["config"]
            if str(cfg.get("model")) != model:
                continue
            d = int(cfg.get("depth")) if (model == "deep" and cfg.get("depth") is not None) else 0
            if d != depth:
                continue
            rule = int(cfg.get("rule"))
            if rules and rule not in rules:
                continue
            H = int(cfg.get("H"))
            acc = float(r["metrics"]["acc"][-1]) if len(r["metrics"]["acc"]) else np.nan
            rule_to_H_to_accs.setdefault(rule, {}).setdefault(H, []).append(acc)

        if not rule_to_H_to_accs:
            continue

        fig, ax = plt.subplots(figsize=(3.35, 2.3))
        for rule in sorted(rule_to_H_to_accs.keys()):
            H_to_accs = rule_to_H_to_accs[rule]
            Hs_sorted = sorted(H_to_accs.keys())
            means = [np.nanmean(H_to_accs[H]) for H in Hs_sorted]
            stds = [np.nanstd(H_to_accs[H]) for H in Hs_sorted]
            style = _rule_style(rule)
            label = f"Rule {rule} ({_rule_category(rule)})"
            ax.errorbar(Hs_sorted, means, yerr=stds, capsize=3, label=label, **style)
        title = f"Final Accuracy vs Prediction Horizon $H$"
        if model == "deep":
            title += f", Depth={depth}"
        ax.set_title(title)
        ax.set_xlabel("Prediction Horizon $H$")
        ax.set_ylabel("Final Accuracy")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="best")
        fig.tight_layout()
        fname = f"final_acc_vs_H_by_rule_model-{model}"
        if model == "deep":
            fname += f"_D{depth}"
        savefig_ieee(fig, outdir / f"{fname}.png")
        plt.close(fig)

    # Qualitative: ground truth vs prediction rows for H=64 (by default), one per rule
    qual_out = outdir / "qualitative"
    qual_out.mkdir(parents=True, exist_ok=True)
    for rule in rules:
        best = _select_best_run(runs, rule=rule, H=args.qual_H)
        if best is None:
            continue
        plot_truth_vs_prediction(best, H=args.qual_H, seed=args.qual_seed, outdir=qual_out)

    print(f"Saved analysis figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()


