"""Learning curve with hyperparameters re-tuned on each training subset (fair version).

For each training size and repeat (same subsets as extra_analyses.py, same seeds), LR and RF are
re-tuned over their section 4 grids by grouped inner CV on that subset only:
3-fold GroupKFold, or leave-one-city-out when the subset has fewer than 15 cities. The chosen
model is refit on the whole subset and scored on the fixed test set. No API calls.
Run: .venv/bin/python -W ignore src/learning_curve_retuned.py [--lr-only]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

from common import LR_C, RES, RF_C, SEED, TEXT2, load, savefig, set_style, split_mask
from tune import LR_GRID, RF_GRID, make_lr, make_rf, pca_label

set_style()
SIZES = [10, 25, 50, 100, 200]
REPEATS = 5


def inner_cv(groups):
    n = len(np.unique(groups))
    if n < 15:
        return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)), "leave-one-city-out"
    return list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        np.zeros(len(groups)), groups=groups)), "3-fold grouped"


def main(lr_only=False, plot_only=False):
    if plot_only:
        agg = pd.read_csv(RES / "learning_curve_retuned.csv")
        n_all = int(agg.n_cities.max())
        return plot(agg, n_all, lr_only)
    X, meta = load("cities")
    tr, te = split_mask(meta)
    y = meta["label"].to_numpy()
    Xtr, ytr, ctr = X[tr], y[tr], meta["city"].to_numpy()[tr]
    Xte, yte = X[te], y[te]
    train_cities = np.unique(ctr)
    n_all = len(train_cities)
    assert not set(train_cities) & set(meta.city[te])

    models = [("LR", make_lr, LR_GRID)] + ([] if lr_only else [("RF", make_rf, RF_GRID)])
    runs = []
    for size in SIZES + [n_all]:
        for r in range(REPEATS):
            rng = np.random.default_rng(1000 * size + r)  # identical subsets to extra_analyses.py
            sub = rng.choice(train_cities, size, replace=False) if size < n_all else train_cities
            idx = np.isin(ctr, sub)
            folds, scheme = inner_cv(ctr[idx])
            for c in folds:  # no city on both sides of an inner fold
                assert not set(ctr[idx][c[0]]) & set(ctr[idx][c[1]])
            for name, make, grid in models:
                t0 = time.time()
                gs = GridSearchCV(make(), grid, scoring="accuracy", cv=folds, n_jobs=-1, refit=True)
                gs.fit(Xtr[idx], ytr[idx])
                acc = accuracy_score(yte, gs.predict(Xte))
                bp = {k: (pca_label(v) if k == "pca" else v) for k, v in gs.best_params_.items()}
                row = {"model": name, "n_cities": size, "repeat": r, "inner_cv": scheme,
                       "n_inner_folds": len(folds), "inner_cv_acc": gs.best_score_,
                       "test_accuracy": acc, "best_params": json.dumps(bp, default=float),
                       "runtime_sec": round(time.time() - t0, 1)}
                if name == "LR":
                    row["nonzero_weights"] = int((gs.best_estimator_["clf"].coef_ != 0).sum())
                runs.append(row)
                print(f"{name} {size:>3} cities rep {r}: test {acc:.4f}  {bp}  ({row['runtime_sec']}s)", flush=True)
            if size == n_all:
                break  # all cities: one run (subset is the whole training split)
    runs = pd.DataFrame(runs)
    runs.to_csv(RES / "learning_curve_retuned_runs.csv", index=False)
    agg = runs.groupby(["model", "n_cities"])["test_accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg.to_csv(RES / "learning_curve_retuned.csv", index=False)

    rows = []
    for _, a in agg.iterrows():
        rows.append({"section": "Learning curve (re-tuned)", "dataset": "cities test", "method": f"{a.model} (re-tuned)",
                     "metric": f"test_accuracy_{a.n_cities}_cities", "value": round(a["mean"], 4),
                     "ci_low": np.nan, "ci_high": np.nan,
                     "notes": f"mean over {a['count']} subsets; sd = {a['std']:.4f}; grid re-tuned by inner grouped CV "
                              f"({'leave-one-city-out' if a.n_cities < 15 else '3-fold'}) on each subset"})
    lr = runs[runs.model == "LR"]
    for size in (10, 25):
        d = lr[lr.n_cities == size]
        rows.append({"section": "Learning curve (re-tuned)", "dataset": "cities train subsets", "method": "LR (re-tuned)",
                     "metric": f"nonzero_weights_{size}_cities", "value": round(d.nonzero_weights.mean(), 1),
                     "ci_low": np.nan, "ci_high": np.nan,
                     "notes": f"mean over 5 subsets; per subset: {d.nonzero_weights.tolist()}; "
                              f"chosen: {d.best_params.tolist()}"})
    for name in agg.model.unique():
        rows.append({"section": "Learning curve (re-tuned)", "dataset": "cities", "method": f"{name} (re-tuned)",
                     "metric": "total_runtime_sec", "value": round(runs[runs.model == name].runtime_sec.sum(), 1),
                     "ci_low": np.nan, "ci_high": np.nan, "notes": ""})
    pd.DataFrame(rows).to_csv(RES / "lc_retuned_results.csv", index=False)
    plot(agg, n_all, lr_only)
    print(agg.to_string())


def plot(agg, n_all, lr_only):

    # Figure: LR re-tuned, LR fixed C = 0.1 (dashed), RF (re-tuned if run, else fixed)
    fixed = pd.read_csv(RES / "learning_curve.csv")
    km = json.loads((RES / "tuning.json").read_text())["kmeans"]["PCA2"]["cluster_accuracy"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    rf_src = agg if not lr_only else fixed
    lines = [(agg[agg.model == "LR"], LR_C, "-", "Logistic regression (re-tuned per subset)"),
             (fixed[fixed.model == "LR"], LR_C, "--", "Logistic regression (fixed C = 0.1, L1)"),
             (rf_src[rf_src.model == "RF"], RF_C, "-",
              "Random forest (re-tuned per subset)" if not lr_only else "Random forest (fixed settings)")]
    for d, c, ls, lab in lines:
        if ls == "-":
            ax.fill_between(d.n_cities, d["mean"] - d["std"], np.minimum(d["mean"] + d["std"], 1.0),
                            color=c, alpha=0.15, lw=0)
        ax.plot(d.n_cities, d["mean"], ls, marker="o", color=c, lw=2, ms=7, label=lab,
                mec="white" if ls == "-" else c, mew=1.2 if ls == "-" else 1.8, mfc=c if ls == "-" else "white")
    ax.axhline(km, color=TEXT2, lw=1.2, ls=":")
    ax.text(n_all, km - 0.0007, f"k-means, no labels (training set): {km:.1%}",
            color=TEXT2, fontsize=10, ha="right", va="top")
    f10 = fixed[(fixed.model == "LR") & (fixed.n_cities == 10)]["mean"].iloc[0]
    if f10 < 0.97:
        ax.annotate(f"fixed C at 10 cities: {f10:.0%} (off scale)\nL1 zeroes every weight, predicts all False",
                    (10, 0.9705), xytext=(12, 0.9725), color=TEXT2, fontsize=10, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=TEXT2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks(SIZES + [n_all], [str(s) for s in SIZES + [n_all]])
    ax.minorticks_off()
    ax.set_ylim(0.97, 1.001)
    ax.set_xlabel("Number of training cities (2 statements each)")
    ax.set_ylabel("Test accuracy (mean ± sd)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("How many labelled cities does it take?")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.text(0.0, -0.09, "Y axis starts at 97%. 5 random subsets of training cities per size (all 598: one run); fixed 300-statement test set. "
             "Re-tuned = section 4 grid,\ninner grouped CV on the subset only (leave-one-city-out below 15 cities, "
             "else 3-fold). Dashed = main-result setting, never re-tuned.", color=TEXT2, fontsize=10)
    savefig(fig, "results_learning_curve.png")


if __name__ == "__main__":
    main(lr_only="--lr-only" in sys.argv, plot_only="--plot-only" in sys.argv)
