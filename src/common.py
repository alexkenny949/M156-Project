"""Shared data loading, the fixed city-grouped split, CV folds, and plot style."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings"
FIG = ROOT / "figures"
RES = ROOT / "results"
SEED = 0

# Plot style (slide-readable). Colors from the dataviz reference palette.
TRUE_C, FALSE_C = "#257de0", "#f47230"  # categorical slots 1 and 2
LR_C, RF_C, CHAT_C = "#257de0", "#f47230", "#00b894"
GRAY, TEXT, TEXT2 = "#c9c9c9", "#191919", "#666666"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
        "ytick.color": TEXT2, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8, "axes.axisbelow": True, "text.color": TEXT,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def savefig(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def load(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings (n x 4096) and metadata with region columns added."""
    X = np.load(EMB / f"{name}.npy")
    meta = pd.read_csv(EMB / f"{name}_meta.csv")
    reg = pd.read_csv(ROOT / "data" / "country_region.csv")
    meta = meta.merge(reg.rename(columns={"country": "correct_country"}),
                      on="correct_country", how="left", validate="many_to_one")
    assert meta["region"].notna().all() and len(meta) == len(X)
    return X, meta


def get_split() -> pd.DataFrame:
    """Fixed 80/20 test split grouped by city (seed 0). Created once, then read from disk."""
    path = ROOT / "data" / "split.csv"
    if path.exists():
        return pd.read_csv(path)
    _, meta = load("cities")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(meta, groups=meta["city"]))
    split = pd.DataFrame({"city": meta["city"].unique()})
    split["split"] = np.where(split["city"].isin(meta["city"].iloc[te]), "test", "train")
    split.to_csv(path, index=False)
    return split


def split_mask(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = get_split().set_index("city")["split"]
    side = meta["city"].map(s)
    assert side.notna().all()
    tr, te = (side == "train").to_numpy(), (side == "test").to_numpy()
    assert not set(meta.city[tr]) & set(meta.city[te]), "LEAKAGE: city on both sides of split"
    return tr, te


def cv_folds(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """5-fold CV grouped by city, fixed seed; same folds for every method."""
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(gkf.split(np.zeros(len(groups)), groups=groups))
    g = groups.to_numpy()
    for tr, va in folds:
        assert not set(g[tr]) & set(g[va]), "LEAKAGE: city in both CV train and validation"
    return folds
