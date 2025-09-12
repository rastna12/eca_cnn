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
uv run eca-cnn-train --rule 150 --H 128 --steps 500 --device cuda
uv run eca-cnn-train --rule 90  --H 128 --linear
uv run eca-cnn-train --rule 30  --H 64  --device cpu
```

### Plots
```bash
uv run eca-cnn-plots --outdir figs --width 256 --steps 256 --Hmask 16
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