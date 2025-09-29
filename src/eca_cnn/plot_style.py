from contextlib import contextmanager

import matplotlib as mpl


IEEE_RCPARAMS = {
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    # Lines and markers
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    # Axes
    "axes.spines.right": False,
    "axes.spines.top": False,
    # Grid
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    # Savefig
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_ieee_style() -> None:
    """Apply IEEE-like matplotlib rcParams globally."""
    mpl.rcParams.update(IEEE_RCPARAMS)


@contextmanager
def use_ieee_style():
    """Context manager to temporarily apply IEEE style within a with-block."""
    with mpl.rc_context(IEEE_RCPARAMS):
        yield


def savefig_ieee(fig, outpath, *, dpi: int | None = None, bbox_inches: str | None = None, pad_inches: float | None = None):
    """
    Save figure with IEEE defaults unless explicitly overridden.
    """
    fig.savefig(
        outpath,
        dpi=(dpi if dpi is not None else mpl.rcParams.get("savefig.dpi", 300)),
        bbox_inches=(bbox_inches if bbox_inches is not None else mpl.rcParams.get("savefig.bbox", "tight")),
        pad_inches=(pad_inches if pad_inches is not None else mpl.rcParams.get("savefig.pad_inches", 0.02)),
    )


