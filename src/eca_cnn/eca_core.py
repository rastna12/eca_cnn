import torch


def rule_table(rule: int) -> torch.Tensor:
    return torch.tensor([(rule >> i) & 1 for i in range(8)], dtype=torch.uint8)


def evolve_once(state: torch.Tensor, tbl: torch.Tensor) -> torch.Tensor:
    L = torch.roll(state,  1, dims=-1)
    R = torch.roll(state, -1, dims=-1)
    idx = ((L << 2) | (state << 1) | R).long()
    return tbl[idx]


def jump_ahead(state: torch.Tensor, tbl: torch.Tensor, H: int) -> torch.Tensor:
    for _ in range(H):
        state = evolve_once(state, tbl)
    return state


def simulate(rule: int, width=256, steps=256, p_init=0.5, seed=0, init="random") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if init == "random":
        state = (torch.rand(width, generator=g) < p_init).to(torch.uint8).unsqueeze(0)
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


