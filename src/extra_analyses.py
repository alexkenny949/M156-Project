"""Follow-up analyses (no API calls, existing results unchanged):
1. Learning curve: LR and RF (tuned settings) trained on random subsets of training cities.
2. Hard negatives: accuracy on same-region vs cross-region false statements vs true statements.

Same split, folds, seeds and tuned hyperparameters as evaluate.py. These analyses score the
fixed test set many times with FIXED hyperparameters; nothing here is used for model selection.
Run: .venv/bin/python -W ignore src/extra_analyses.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from common import (CHAT_C, GRAY, LR_C, RES, RF_C, ROOT, SEED, TEXT, TEXT2, load, savefig,
                    set_style, split_mask)
from evaluate import B, build, city_bootstrap

set_style()
SIZES = [10, 25, 50, 100, 200]
REPEATS = 5
rows = []


def rec(section, dataset, method, metric, value, lo=np.nan, hi=np.nan, notes=""):
    rows.append({"section": section, "dataset": dataset, "method": method, "metric": metric,
                 "value": value, "ci_low": lo, "ci_high": hi, "notes": notes})


def main():
    tuning = json.loads((RES / "tuning.json").read_text())
    X, meta = load("cities")
    tr, te = split_mask(meta)
    y = meta["label"].to_numpy()
    Xtr, ytr, ctr = X[tr], y[tr], meta["city"].to_numpy()[tr]
    Xte, yte = X[te], y[te]
    train_cities = np.unique(ctr)
    n_all = len(train_cities)
    assert not set(train_cities) & set(meta.city[te])

    # ---------------- 1. Learning curve
    lc = []
    for size in SIZES + [n_all]:
        for r in range(REPEATS):
            rng = np.random.default_rng(1000 * size + r)  # fixed seed per (size, repeat)
            sub = rng.choice(train_cities, size, replace=False) if size < n_all else train_cities
            idx = np.isin(ctr, sub)
            for name in ("LR", "RF"):
                m = build(name, tuning[name]["best_params"]).fit(Xtr[idx], ytr[idx])
                lc.append({"model": name, "n_cities": size, "repeat": r, "n_statements": int(idx.sum()),
                           "test_accuracy": accuracy_score(yte, m.predict(Xte))})
            if size == n_all:
                break  # all cities: every repeat is identical (same data, fixed model seeds)
    lc = pd.DataFrame(lc)
    lc.to_csv(RES / "learning_curve_runs.csv", index=False)
    agg = lc.groupby(["model", "n_cities"])["test_accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg.to_csv(RES / "learning_curve.csv", index=False)
    for _, a in agg.iterrows():
        rec("Learning curve", "cities test", a.model, f"test_accuracy_{a.n_cities}_cities", round(a["mean"], 4),
            notes=f"mean over {a['count']} random subsets; sd = {a['std']:.4f}")
    km = tuning["kmeans"]
    km_acc = km["PCA2"]["cluster_accuracy"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, c, lab in [("LR", LR_C, "Logistic regression"), ("RF", RF_C, "Random forest")]:
        d = agg[agg.model == name]
        ax.fill_between(d.n_cities, d["mean"] - d["std"], np.minimum(d["mean"] + d["std"], 1.0),
                        color=c, alpha=0.15, lw=0)
        ax.plot(d.n_cities, d["mean"], "-o", color=c, lw=2, ms=7, label=lab, mec="white", mew=1.2)
    ax.axhline(km_acc, color=TEXT2, lw=1.2, ls="--")
    ax.annotate(f"k-means, no labels\n(training set): {km_acc:.1%}", (60, km_acc), xytext=(60, 0.88),
                color=TEXT2, fontsize=11, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    ax.annotate("L1 penalty zeroes every weight\nat 20 statements: predicts all False", (10, 0.5),
                xytext=(13, 0.6), color=TEXT2, fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks(SIZES + [n_all], [str(s) for s in SIZES + [n_all]])
    ax.minorticks_off()
    ax.set_xlabel("Number of training cities (2 statements each)")
    ax.set_ylabel("Test accuracy (mean ± sd)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    lo = max(0.5, (agg["mean"] - agg["std"]).min() - 0.02)
    ax.set_ylim(lo, 1.005)
    ax.set_title("How many labelled cities does it take?")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.text(0.0, -0.04, "5 random subsets of training cities per size (all 598: one run). "
             "Fixed 300-statement test set; tuned hyperparameters.", color=TEXT2, fontsize=11)
    savefig(fig, "results_learning_curve.png")

    # ---------------- 2. Hard negatives
    reg = pd.read_csv(ROOT / "data" / "country_region.csv").set_index("country")["region"]
    tm = meta[te].reset_index(drop=True)
    tm["stated_region"] = tm["country"].map(reg)
    tm["group"] = np.where(tm.label == 1, "True statements",
                           np.where(tm.stated_region == tm.region, "False, same region", "False, cross-region"))
    final = {n: build(n, tuning[n]["best_params"]).fit(Xtr, ytr) for n in ("LR", "RF")}
    for n, m in final.items():
        tm[f"{n}_correct"] = m.predict(Xte) == yte
    chat = pd.read_csv(RES / "baseline_predictions_cities.csv").set_index("statement")
    tm["Chat_correct"] = chat.loc[tm.statement, "correct"].to_numpy().astype(bool)
    tm.to_csv(RES / "hard_negatives_test_predictions.csv", index=False)
    # sanity: overall accuracy matches the main evaluation
    ev = json.loads((RES / "eval_summary.json").read_text())
    for n in ("LR", "RF"):
        assert abs(tm[f"{n}_correct"].mean() - ev[n]["test_acc"]) < 1e-9, "refit differs from main result"

    boots = city_bootstrap(tm["city"].to_numpy(), np.random.default_rng(SEED))
    groups = ["False, same region", "False, cross-region", "True statements"]
    hn = []
    for g in groups:
        gmask = (tm.group == g).to_numpy()
        for n, lab in [("LR", "Logistic regression"), ("RF", "Random forest"), ("Chat", "Chat model (baseline)")]:
            corr = tm[f"{n}_correct"].to_numpy()
            accs = [corr[i][gmask[i]].mean() for i in boots if gmask[i].any()]
            lo, hi = np.percentile(accs, [2.5, 97.5])
            acc = corr[gmask].mean()
            hn.append({"group": g, "model": n, "n": int(gmask.sum()), "errors": int((~corr[gmask]).sum()),
                       "accuracy": acc, "ci_low": lo, "ci_high": hi})
            rec("Hard negatives", "cities test", n if n != "Chat" else "Chat baseline", f"accuracy_{g}",
                round(acc, 4), round(lo, 4), round(hi, 4),
                f"n = {gmask.sum()}, errors = {(~corr[gmask]).sum()}; 95% CI city-level bootstrap ({B})")
    hn = pd.DataFrame(hn)
    hn.to_csv(RES / "hard_negatives.csv", index=False)
    # which regions do same-region false statements come from
    sr = tm[tm.group == "False, same region"]["region"].value_counts()
    for r, c in sr.items():
        rec("Hard negatives", "cities test", "data", f"n_same_region_false_{r}", int(c))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    models = [("LR", LR_C, "Logistic regression"), ("RF", RF_C, "Random forest"),
              ("Chat", CHAT_C, "Chat model (baseline)")]
    for gi, g in enumerate(groups):
        for k, (n, c, lab) in enumerate(models):
            r = hn[(hn.group == g) & (hn.model == n)].iloc[0]
            yv = (len(groups) - 1 - gi) + (1 - k) * 0.22
            ax.plot([r.ci_low, r.ci_high], [yv, yv], color=c, lw=2.5, solid_capstyle="round")
            ax.plot(r.accuracy, yv, "o", color=c, ms=9, mec="white", mew=1.2, label=lab if gi == 0 else None)
            ax.text(0.9505, yv, f"{r.accuracy:.1%}  ({r.errors} wrong / {r.n})", va="center",
                    fontsize=10, color=TEXT)
    ax.set_yticks(range(len(groups)),
                  [f"{g}\n(n = {int(hn[hn.group == g].n.iloc[0])})" for g in groups][::-1])
    ax.set_xlim(0.95, 1.003)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Test accuracy (cities test split)")
    ax.set_title("Are false statements harder when the wrong country is nearby?", pad=34)
    ax.legend(frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 0.98))
    ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.06, "Axis starts at 95%. Same region = wrong country in the same UN region as the true one. "
             "Lines: 95% CI, resampling test cities.", color=TEXT2, fontsize=11)
    savefig(fig, "results_hard_negatives.png")

    pd.DataFrame(rows).to_csv(RES / "extra_results.csv", index=False)
    print(agg.to_string())
    print(hn.round(4).to_string())
    print(sr.to_string())


if __name__ == "__main__":
    main()
