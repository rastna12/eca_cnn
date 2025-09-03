# ca_datasets.py
# Elementary CA + "thin light cone" dataset builder
# Produces (X, y) where X has shape (N, history, 2*half_width+1) and y are center bits.

from __future__ import annotations
import numpy as np
from pathlib import Path
import json

# ---------------------- Core CA ----------------------

def rule_to_lut(rule: int) -> np.ndarray:
    """
    Convert Wolfram rule number (0..255) to an 8-entry lookup table (uint8).
    Index encodes neighborhood (l,c,r) as i=4*l+2*c+r; value is next-state bit.
    """
    if not (0 <= rule <= 255):
        raise ValueError("rule must be in [0,255]")
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)

def step_ca(state: np.ndarray, lut: np.ndarray, periodic: bool = True) -> np.ndarray:
    """
    One step of a radius-1 elementary CA on a 1D binary state.
    state: (W,) uint8 in {0,1}
    returns: (W,) uint8
    """
    if periodic:
        left  = np.roll(state,  1)
        right = np.roll(state, -1)
    else:
        left  = np.zeros_like(state); left[1:]  = state[:-1]
        right = np.zeros_like(state); right[:-1] = state[1:]
    idx = (left << 2) | (state << 1) | right
    return lut[idx].astype(np.uint8)

def run_ca(rule: int, width: int, steps: int, p_init: float = 0.5, seed: int | None = None,
           init: str = "random", periodic: bool = True) -> np.ndarray:
    """
    Simulate an elementary CA.
    returns: grid with shape (steps, width), dtype uint8, time-first (grid[t] is row t).
    """
    rng = np.random.default_rng(seed)
    if init == "random":
        state0 = (rng.random(width) < p_init).astype(np.uint8)
    elif init == "single":
        state0 = np.zeros(width, dtype=np.uint8)
        state0[width // 2] = 1
    else:
        raise ValueError("init must be 'random' or 'single'")
    lut = rule_to_lut(rule)
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0] = state0
    for t in range(1, steps):
        grid[t] = step_ca(grid[t-1], lut, periodic=periodic)
    return grid

# ---------------------- Thin light-cone extraction ----------------------

def extract_thin_light_cone(grid: np.ndarray, center: int, half_width: int,
                            periodic: bool = True) -> np.ndarray:
    """
    Take a narrow spatial strip around 'center' for all times.
    returns: (T, 2*half_width+1) uint8
    """
    T, W = grid.shape
    xs = np.arange(center - half_width, center + half_width + 1)
    if periodic:
        xs = np.mod(xs, W)
        return grid[:, xs]
    # absorbing (non-periodic)
    sub = np.zeros((T, len(xs)), dtype=grid.dtype)
    valid = (xs >= 0) & (xs < W)
    sub[:, valid] = grid[:, xs[valid]]
    return sub

# ---------------------- Dataset builders ----------------------

def build_thin_cone_dataset_from_grid(
    grid: np.ndarray,
    half_width: int,
    history: int,
    horizon: int = 0,
    include_current_row: bool = False,
    center: int | None = None,
    periodic: bool = True,
    stride: int = 1,
    flatten: bool = False,
    return_times: bool = False,
):
    """
    Slice a CA grid into supervised samples using a thin spatiotemporal light cone.

    Inputs
    ------
    grid : (T, W) uint8  CA space-time (time-first).
    half_width : int     Spatial half-width w (features per row = 2*w+1).
    history : int        Number of past frames d included in X.
    horizon : int        Forecast offset beyond the gap to target (default 0).
    include_current_row : bool
        If False (default), we EXCLUDE the row just before the target (gap=1).
        If True, we INCLUDE it (gap=0). For radius-1 ECAs, horizon=0 with include_current_row=True
        makes 1-step prediction trivial for all rules.
    center : int | None  Center column index (default: W//2).
    periodic : bool      Spatial wrap-around for strip extraction.
    stride : int         Temporal stride between consecutive samples.
    flatten : bool       If True, returns X shape (N, history*(2*w+1)) instead of (N, history, F).
    return_times : bool  If True, also returns target time indices.

    Outputs
    -------
    X : float32 (N, history, 2*w+1)  or (N, history*(2*w+1)) if flatten
    y : uint8   (N,)                 Target: center bit at target time
    t_targets : int (N,)             [optional] absolute target times
    """
    T, W = grid.shape
    c = center if center is not None else (W // 2)
    sub = extract_thin_light_cone(grid, c, half_width, periodic=periodic)  # (T, F)
    gap = 0 if include_current_row else 1

    L_min = history - 1
    L_max = (T - 1) - (gap + horizon)  # ensure target time exists
    if L_max < L_min:
        raise ValueError("Not enough time steps for requested history/horizon settings.")

    last_idxs = np.arange(L_min, L_max + 1, stride)
    F = sub.shape[1]
    N = len(last_idxs)
    X = np.empty((N, history, F), dtype=np.float32)
    y = np.empty(N, dtype=np.uint8)
    t_targets = np.empty(N, dtype=np.int32)

    for j, L in enumerate(last_idxs):
        X[j] = sub[L - history + 1:L + 1, :]
        tgt_t = L + gap + horizon
        y[j] = grid[tgt_t, c]
        t_targets[j] = tgt_t

    if flatten:
        X = X.reshape(N, -1)
    return (X, y, t_targets) if return_times else (X, y)

def make_rule_dataset(
    rule: int,
    width: int = 256,
    steps: int = 256,
    episodes: int = 9,
    half_width: int = 1,
    history: int = 64,
    horizon: int = 0,
    include_current_row: bool = False,
    p_init: float = 0.5,
    periodic: bool = True,
    stride: int = 1,
    seed: int | None = 42,
    split_by_episode: bool = True,
    split_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
    flatten: bool = False,
):
    """
    Simulate multiple random-IC episodes of a given rule, slice each into (X,y),
    and return a dict with train/val/test splits + metadata.

    Splitting:
    - If split_by_episode=True (default): assign entire episodes to splits (no leakage).
    - Else: concatenate all samples and split randomly by sample index.
    """
    rng = np.random.default_rng(seed)
    epi_seeds = rng.integers(0, 2**32 - 1, size=episodes, dtype=np.uint64)

    splits = ["train", "val", "test"]
    p_train, p_val, p_test = split_fracs
    assign = rng.choice(splits, size=episodes, p=[p_train, p_val, p_test]) if split_by_episode else None

    buckets = {s: {"X": [], "y": []} for s in splits}
    all_X, all_y = [], []

    for i, s in enumerate(epi_seeds):
        grid = run_ca(rule, width, steps, p_init=p_init, seed=int(s), init="random", periodic=periodic)
        X, y = build_thin_cone_dataset_from_grid(
            grid, half_width=half_width, history=history, horizon=horizon,
            include_current_row=include_current_row, center=width//2,
            periodic=periodic, stride=stride, flatten=flatten
        )
        if split_by_episode:
            buckets[assign[i]]["X"].append(X)
            buckets[assign[i]]["y"].append(y)
        else:
            all_X.append(X); all_y.append(y)

    if split_by_episode:
        data = {}
        for s in splits:
            if buckets[s]["X"]:
                data[f"X_{s}"] = np.concatenate(buckets[s]["X"], axis=0)
                data[f"y_{s}"] = np.concatenate(buckets[s]["y"], axis=0)
            else:
                # empty split placeholder
                feat = history*(2*half_width+1) if flatten else (history, 2*half_width+1)
                data[f"X_{s}"] = np.empty((0,) if flatten else (0, *feat), dtype=np.float32)
                data[f"y_{s}"] = np.empty((0,), dtype=np.uint8)
    else:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        N = X_all.shape[0]
        idx = rng.permutation(N)
        n_train = int(p_train * N)
        n_val = int(p_val * N)
        tr = idx[:n_train]
        va = idx[n_train:n_train+n_val]
        te = idx[n_train+n_val:]
        data = dict(
            X_train=X_all[tr], y_train=y_all[tr],
            X_val=X_all[va],   y_val=y_all[va],
            X_test=X_all[te],  y_test=y_all[te],
        )

    data["meta"] = dict(
        rule=rule, width=width, steps=steps, episodes=episodes,
        half_width=half_width, history=history, horizon=horizon,
        include_current_row=include_current_row, p_init=p_init,
        periodic=periodic, stride=stride, seed=int(seed) if seed is not None else None,
        split_by_episode=split_by_episode, split_fracs=split_fracs, flatten=flatten,
    )
    return data

def save_dataset_npz(path: str | Path, data: dict) -> Path:
    """
    Save train/val/test arrays + JSON metadata to a compressed .npz file.
    Keys: X_train, y_train, X_val, y_val, X_test, y_test, meta_json
    """
    path = Path(path)
    meta_json = json.dumps(data.get("meta", {}))
    np.savez_compressed(
        path,
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"],     y_val=data["y_val"],
        X_test=data["X_test"],   y_test=data["y_test"],
        meta_json=meta_json
    )
    return path

# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    # Non-trivial setting: exclude current row (gap=1), horizon=0
    cfg = dict(width=256, steps=256, episodes=9, half_width=1, history=64,
               horizon=0, include_current_row=False, seed=123)

    data150 = make_rule_dataset(rule=150, **cfg)
    out150 = save_dataset_npz("eca_ds_rule150_w1_d64_gap1_H0_ep9.npz", data150)

    data030 = make_rule_dataset(rule=30, **cfg)
    out030 = save_dataset_npz("eca_ds_rule30_w1_d64_gap1_H0_ep9.npz", data030)

    print("Saved:", out150)
    print("Saved:", out030)
