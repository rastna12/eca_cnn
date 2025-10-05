## eca-cnn (v0.1.0)
Elementary Cellular Automata experiments with shallow CNNs and plotting utilities.

## Project layout
- src package: `src/eca_cnn/`
  - `eca_core.py`: CA simulation utilities (`simulate`, `evolve_once`, ...)
  - `datasets.py`: `make_batch` for training
  - `models.py`: `ShallowCNN`
  - `train.py`: CLI entrypoint and `train_rule`
  - `eca_plots.py`: plotting CLI and helpers
- Notebooks: `notebooks/`
- Figures: `figs/` (outputs)

## Install (uv, editable)
```bash
uv pip install -e .
```

This installs the package `eca-cnn` and exposes console scripts:
- `eca-cnn-train`
- `eca-cnn-plots`

### Alternative: pip
```bash
pip install -e .
```
Then run with your Python:
```bash
python -m eca_cnn.train --help
python -m eca_cnn.eca_plots --help
```

## CLI usage
### Train
```bash
uv run eca-cnn-train --help
uv run eca-cnn-train --rule 150 --model shallow --H 128 --steps 500 --device cuda
uv run eca-cnn-train --rule 90  --model shallow --H 128 --linear
uv run eca-cnn-train --rule 30  --model deep    --H 64 --depth 16 --device cpu
# Artifacts (config, metrics.csv, checkpoints) saved under runs/<timestamp>_rule.../ 
```

### Plots
```bash
uv run eca-cnn-plots --outdir figs --width 256 --steps 256 --Hmask 16
```

### Experiments (sweeps)
```bash
uv run eca-cnn-experiments --rules 150,90,30 --Hs 8,16,32 \
  --models shallow,deep --depths 2,4,8 --seeds 1,2,3 --steps 500 --device cuda
# Index of runs saved to runs/runs_index.json
```

### Analysis (from saved runs)
```bash
uv run eca-cnn-analysis --runs-dir runs --outdir figs/analysis --rules 150,90,30 --Hs 8,16,32
# Produces per-run training curves (loss/acc), acc_vs_H_* per model, final_acc_vs_H_* overlays per rule,
# and acc_vs_depth_* plots — all from saved metrics (no retraining)
```

#### LaTeX table export
Generate a formatted LaTeX performance table for paper inclusion:
```bash
# Print to console (for copy/paste)
uv run eca-cnn-analysis --runs-dir runs --rules 90,150,30,110 --Hs 8,16,32 --latex-table

# Save to file
uv run eca-cnn-analysis --runs-dir runs --rules 90,150,30,110 --Hs 8,16,32 \
  --latex-table --latex-out figs/analysis/metrics_table.tex

# Customize formatting
uv run eca-cnn-analysis --runs-dir runs --rules 90,150,30,110 --Hs 8,16,32 \
  --latex-table --table-precision 3 --table-caption "Model Performance Summary"
```
The table summarizes shallow model performance (mean across seeds) with columns for each horizon H.

### One-shot: PowerShell orchestrator
```powershell
# From repo root (Windows PowerShell)
./run_experiments.ps1 -Device cuda -Steps 500 -Rules "150,90,30" -Hs "8,16,32" -Models "shallow,deep" -Depths "2,4,8" -Seeds "1,2,3"
# Script auto-detects uv; falls back to python -m if uv is not found
```

## Python API quickstart
```python
from eca_cnn.eca_core import simulate

grid = simulate(150, width=128, steps=64, init="random", seed=42)
```

## Requirements
- Python >= 3.13
- Torch and torchvision (configured via `pyproject.toml`; CUDA 12.8 wheels via uv index)

## Notes
- For notebooks, after installing (`pip install -e .` or `uv pip install -e .`), you can `import eca_cnn` without modifying `sys.path`.