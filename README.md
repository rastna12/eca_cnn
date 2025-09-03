# CA Thin-Light-Cone Datasets — README

This README describes how we generate **supervised learning datasets** from **elementary cellular automata (ECA)** for probing **computational reducibility vs. irreducibility** with sequence models (e.g., LSTMs). The core idea is to slice each CA run into many training samples using a **thin light cone**—a narrow spatial strip centered at a fixed column over the last `history` time steps—and to predict the **center bit** at a future time.

---

## 1) Concept

* **Elementary CA (radius r=1):** Each cell updates from its 3-bit neighborhood at the previous time step:

  $$
  x_{t+1}(i) = f\big(x_t(i-1),\,x_t(i),\,x_t(i+1)\big)
  $$
* **Thin light cone (width `w`):** For a fixed center column `i_0`, we take a strip of columns $[i_0-w, \dots, i_0+w]$ over the last `history=d` rows as the **input** $X$.
* **Target (`y`):** The bit at the **center column** at the **target time**.
* **Crucial non-triviality choice:**
  If you include the **current row** (the row right before the target) and set `w ≥ 1` with 1-step forecasting, prediction is **trivial** for *all* ECA rules. To reveal differences (e.g., Rule 150 vs Rule 30), **exclude** the current row and/or forecast multiple steps ahead with **insufficient width** (`w < H`).

---

## 2) Dataset definition

Each dataset is a compressed `.npz` with:

* `X_train, y_train, X_val, y_val, X_test, y_test`
* `meta_json` — a JSON string storing all generation parameters.

### Shapes & types

* `X_*`: `float32` with shape `(N, history, 2*w + 1)`
  (You can optionally flatten to `(N, history*(2*w+1))` if you set `flatten=True` when building.)
* `y_*`: `uint8` with shape `(N,)` (values in `{0,1}`)

### Target time and gaps

* Let `L` be the last time index included in the input window.
* Define **gap** = `0` if `include_current_row=True` (current row included), else **gap** = `1` (excluded).
* With forecast horizon `H`, the **target** time is `t_target = L + gap + H`.

**Recommended for non-trivial 1-step prediction:** `include_current_row=False` (gap=1), `H=0`.

---

## 3) Parameters (what’s in `meta_json`)

* **Rule & lattice**

  * `rule`: Wolfram rule number (0–255), e.g., 30, 54, 110, 150, 184
  * `width`: number of columns
  * `steps`: number of rows (time steps)
  * `periodic`: spatial periodic boundary (wrap-around), typically `True`
  * `p_init`: probability of a 1 in the random initial row
* **Slicing**

  * `half_width` = `w` → features per row = `2*w + 1`
  * `history` = `d` → number of past rows in each input
  * `include_current_row`: include/exclude the row right before target (`True`→gap=0, `False`→gap=1)
  * `horizon` = `H`: steps ahead to forecast
  * `stride`: temporal stride when sampling windows
  * `flatten`: whether `X` is flattened
* **Splitting**

  * `episodes`: how many independent simulations (different random initial rows)
  * `split_by_episode`: if `True`, whole episodes go to train/val/test (prevents leakage)
  * `split_fracs`: e.g., `(0.8, 0.1, 0.1)`
  * `seed`: RNG seed used to draw episode seeds + splits

---

## 4) File naming convention

`eca_ds_rule{rule}_w{w}_d{history}_gap{gap}_H{H}_ep{episodes}.npz`

* `gap` is `0` if `include_current_row=True`, else `1`.

Example:

```
eca_ds_rule150_w1_d64_gap1_H0_ep9.npz
```

→ Rule 150, w=1, history=64, exclude current row (gap=1), 1-step ahead (H=0), 9 episodes.

---

## 5) Expected behavior (why this is useful)

* **Rule 150 (additive/linear):** Even with missing information (e.g., center-only or thin cone excluding current row), there are **learnable correlations**; accuracy typically rises with `history` and/or `w`.
* **Rule 30 (chaotic):** Center column (and thin cones) look pseudo-random under these constraints; models tend to hover near chance unless you give the full, sufficient light cone.

This contrast makes reducibility vs. irreducibility visible in learning curves.

---

## 6) Quick start (Python API)

Assuming you have the functions:

* `run_ca`, `extract_thin_light_cone`
* `build_thin_cone_dataset_from_grid`
* `make_rule_dataset`, `save_dataset_npz`

### Generate non-trivial 1-step datasets (exclude current row)

```python
from ca_datasets import make_rule_dataset, save_dataset_npz

cfg = dict(
    width=256, steps=256, episodes=9,
    half_width=1,      # thin cone w=1 → 3 features per row
    history=64,        # past rows
    horizon=0,         # 1-step forecasting
    include_current_row=False,  # exclude current row to avoid triviality
    seed=123
)

data150 = make_rule_dataset(rule=150, **cfg)
path150 = save_dataset_npz("eca_ds_rule150_w1_d64_gap1_H0_ep9.npz", data150)

data030 = make_rule_dataset(rule=30, **cfg)
path030 = save_dataset_npz("eca_ds_rule30_w1_d64_gap1_H0_ep9.npz", data030)
```

### Load into PyTorch

```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader

data = np.load("eca_ds_rule150_w1_d64_gap1_H0_ep9.npz")
Xtr = torch.from_numpy(data["X_train"]).float()   # (N, 64, 3)
ytr = torch.from_numpy(data["y_train"]).long()    # (N,)
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
```

---

## 7) Recommended experiment checklist

1. **Center-only baseline:**
   `w=0`, vary `history` (e.g., 32, 64, 128).
   Expect: Rule 150 > Rule 30.

2. **Thin cone excluding current row:**
   `w=1 or 2`, `include_current_row=False`, `H=0`.
   Expect: Rule 150 improves with `history`/`w`; Rule 30 \~ chance.

3. **H-step ahead with limited width:**
   `include_current_row=True`, set `H > 0`, but keep `w < H`.
   Expect: information-limited regime; Rule 150 degrades gracefully; Rule 30 remains hard.

4. **Ablations:**

   * Increase `width`/`steps` to check stationarity and leakage.
   * Try `rule=184` (traffic), `54/110` (edge-of-chaos) for intermediate difficulty.

---

## 8) Best practices & pitfalls

* **Split by episode** to avoid train/test leakage through temporally adjacent slices from the same CA run.
* **Periodic boundaries** reduce edge artifacts; report boundary conditions in your methods.
* **Report `w`, `history`, `H`, `gap`** clearly—these define the information set available to the model.
* **Avoid trivial settings** for 1-step tasks: if `include_current_row=True` and `w ≥ 1`, the model can learn the 3-bit truth table and hit \~100% for *any* rule.
* **Reproducibility:** fix the seed; store `meta_json` with your artifacts.

---

## 9) Extending

* **Other rules / neighborhoods:** Works for any 1D binary ECA; can be extended to larger radius or multi-state rules by adjusting the simulator.
* **Alternate targets:** Predict a different column, a vector of columns, or multiple steps jointly.
* **Different boundaries:** Implement absorbing/reflection if your application requires walls or reservoirs.

---

## 10) Minimal API reference

```python
def run_ca(rule, width, steps, p_init=0.5, seed=None, init="random", periodic=True) -> np.ndarray:
    """Simulate an ECA. Returns grid of shape (steps, width), dtype=uint8."""

def extract_thin_light_cone(grid, center, half_width, periodic=True) -> np.ndarray:
    """Return (T, 2*half_width+1) strip centered at 'center'."""

def build_thin_cone_dataset_from_grid(
    grid, half_width, history, horizon=0,
    include_current_row=False, center=None, periodic=True,
    stride=1, flatten=False, return_times=False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice grid into (X, y) (and optionally target time indices)."""

def make_rule_dataset(
    rule, width=256, steps=256, episodes=9,
    half_width=1, history=64, horizon=0, include_current_row=False,
    p_init=0.5, periodic=True, stride=1, seed=123,
    split_by_episode=True, split_fracs=(0.8,0.1,0.1), flatten=False
) -> dict:
    """Return dict with X_train, y_train, X_val, y_val, X_test, y_test, meta."""

def save_dataset_npz(path, data) -> pathlib.Path:
    """Save arrays + meta_json to compressed .npz."""
```

---

## 11) License & citation

Use and modify freely for research. If you publish results, please cite the dataset generation procedure (this README or your repo’s generator script) and specify `rule`, `w`, `history`, `H`, `gap`, boundaries, lattice size, and splitting policy.
