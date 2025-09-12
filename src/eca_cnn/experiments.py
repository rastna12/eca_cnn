import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List

from eca_cnn.train import train_rule


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_sweeps(
    rules: Iterable[int],
    horizons: Iterable[int],
    models: Iterable[str],
    depths: Iterable[int | None],
    *,
    steps: int,
    N: int,
    batch: int,
    device: str,
    hidden: int,
    lr: float,
    seeds: Iterable[int],
    runs_dir: str,
    log_every: int,
    save_every: int | None,
):
    runs_index = []
    for rule, H, model, depth, seed in itertools.product(rules, horizons, models, depths, seeds):
        if model == "deep" and depth is None:
            # skip invalid combos
            continue
        tag = None
        run_dir = train_rule(
            rule=rule,
            H=H,
            N=N,
            steps=steps,
            B=batch,
            device=device,
            hidden=hidden,
            lr=lr,
            log_every=log_every,
            seed=seed,
            linear=False,
            model=model,
            depth=depth,
            runs_dir=runs_dir,
            tag=tag,
            save_every=save_every,
        )
        runs_index.append(
            {
                "rule": rule,
                "H": H,
                "model": model,
                "depth": depth,
                "seed": seed,
                "run_dir": run_dir,
            }
        )

    # Save an index file in runs_dir for convenience
    runs_dir_path = Path(runs_dir)
    runs_dir_path.mkdir(parents=True, exist_ok=True)
    index_path = runs_dir_path / "runs_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(runs_index, f, indent=2)
    print(f"Saved runs index to {index_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Run experiment sweeps and save artifacts for later analysis.")
    p.add_argument("--rules", type=str, default="150,90,30", help="Comma-separated rules")
    p.add_argument("--Hs", type=str, default="8,16,32", help="Comma-separated horizons H")
    p.add_argument(
        "--models",
        type=str,
        default="shallow,deep",
        help="Comma-separated model names from {shallow,deep}",
    )
    p.add_argument(
        "--depths",
        type=str,
        default="2,4,8",
        help="Comma-separated depths for deep model (ignored for shallow)",
    )
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seeds")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=None)
    args = p.parse_args()

    rules = parse_csv_ints(args.rules)
    Hs = parse_csv_ints(args.Hs)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    depths = parse_csv_ints(args.depths)
    seeds = parse_csv_ints(args.seeds)

    # include None for depths so shallow combos iterate once
    depths_with_none = list({None} | set(depths))

    run_sweeps(
        rules=rules,
        horizons=Hs,
        models=models,
        depths=depths_with_none,
        steps=args.steps,
        N=args.N,
        batch=args.batch,
        device=args.device,
        hidden=args.hidden,
        lr=args.lr,
        seeds=seeds,
        runs_dir=args.runs_dir,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()


