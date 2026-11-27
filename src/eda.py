"""Handoff step 4: EDA figures on cities embeddings.

Run: .venv/bin/python -I src/eda.py
PCA here is fit on the TRAINING split of cities only (so nothing about the test set
informs these plots); training points are plotted.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import (FALSE_C, GRAY, RES, SEED, TEXT2, TRUE_C, load, savefig, set_style,
                    split_mask)

set_style()
rows = []


def rec(metric, value, notes=""):
    rows.append({"section": "EDA", "dataset": "cities", "method": "PCA", "metric": metric,
                 "value": value, "ci_low": np.nan, "ci_high": np.nan, "notes": notes})


X, meta = load("cities")
tr, _ = split_mask(meta)
Xtr, mtr = X[tr], meta[tr].reset_index(drop=True)
pca = PCA(n_components=50, random_state=SEED).fit(Xtr)
Z = pca.transform(Xtr)
evr = pca.explained_variance_ratio_

# How much do the top PCs track truth vs region? |corr| with label; eta^2 for region.
for k in range(10):
    r = abs(np.corrcoef(Z[:, k], mtr["label"])[0, 1])
    grp = pd.Series(Z[:, k]).groupby(mtr["region"])
    eta2 = (grp.count() * (grp.mean() - Z[:, k].mean()) ** 2).sum() / ((Z[:, k] - Z[:, k].mean()) ** 2).sum()
    rec(f"PC{k+1}_abs_corr_with_label", round(r, 4), "PCA fit on cities training split")
    rec(f"PC{k+1}_eta2_region", round(float(eta2), 4), "share of PC variance explained by region")
    rec(f"PC{k+1}_explained_var", round(float(evr[k]), 4))
for k in (2, 10, 20, 50):
    rec(f"cum_explained_var_{k}PCs", round(float(evr[:k].sum()), 4))

# 1. PC1 vs PC2 by label
fig, ax = plt.subplots(figsize=(7.5, 6))
for lab, c, name in [(0, FALSE_C, "False"), (1, TRUE_C, "True")]:
    m = mtr["label"] == lab
    ax.scatter(Z[m, 0], Z[m, 1], s=14, c=c, alpha=0.7, lw=0.3, edgecolor="white", label=name)
ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
ax.set_title("Top two principal components, colored by truth label")
ax.legend(frameon=False, loc="upper right", markerscale=2)
fig.text(0.0, -0.02, f"cities training split, n = {len(Z)}. PCA fit on training split only.",
         color=TEXT2, fontsize=11)
savefig(fig, "eda_pca_by_label.png")

# 2. PC1 vs PC2 by region: small multiples, one highlighted region per panel
regions = ["Asia", "Americas", "Africa", "Europe"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True)
for ax, reg in zip(axes, regions):
    m = (mtr["region"] == reg).to_numpy()
    ax.scatter(Z[~m, 0], Z[~m, 1], s=8, c=GRAY, alpha=0.5, lw=0)
    ax.scatter(Z[m, 0], Z[m, 1], s=12, c=TRUE_C, alpha=0.8, lw=0.3, edgecolor="white")
    ax.set_title(f"{reg} (n = {m.sum()})", fontsize=14)
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
fig.suptitle("Same PCs, colored by region of the city's true country", x=0.06, ha="left",
             fontweight="bold", fontsize=16, y=1.03)
fig.text(0.06, -0.06, "Gray = all other training statements. Oceania (1 city) shown in gray. "
         "UN geoscheme regions.", color=TEXT2, fontsize=11)
savefig(fig, "eda_pca_by_region.png")

# 3. Explained variance, individual + cumulative, first 50 PCs
fig, ax = plt.subplots(figsize=(8, 5))
k = np.arange(1, 51)
ax.bar(k, evr, color=TRUE_C, width=0.8, label="Individual")
ax.plot(k, np.cumsum(evr), color=FALSE_C, lw=2, marker="o", ms=3, label="Cumulative")
ax.set_xlabel("Principal component")
ax.set_ylabel("Fraction of variance explained")
ax.set_ylim(0, 1)
ax.set_title("PCA explained variance (first 50 components)")
ax.legend(frameon=False, loc="center right")
ax.annotate(f"50 PCs: {evr.sum():.0%}", (50, evr.sum()), xytext=(-10, 10),
            textcoords="offset points", ha="right", color=TEXT2, fontsize=12)
savefig(fig, "eda_explained_variance.png")

# 4. Cosine similarity: true/false twin pairs vs random pairs of different cities
rng = np.random.default_rng(SEED)
idx_true = np.where(meta["label"] == 1)[0]
twin = meta.groupby("city").indices
pairs = np.array([twin[c] for c in meta["city"].unique()])
cos_twin = np.einsum("ij,ij->i", X[pairs[:, 0]], X[pairs[:, 1]])  # vectors are unit length
a, b = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
keep = meta["city"].to_numpy()[a] != meta["city"].to_numpy()[b]
cos_rand = np.einsum("ij,ij->i", X[a[keep]][:5000], X[b[keep]][:5000])
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0.1, 1.0, 46)
ax.hist(cos_rand, bins=bins, density=True, color=GRAY, label=f"Random pairs, different cities (n = 5000)")
ax.hist(cos_twin, bins=bins, density=True, color=TRUE_C, alpha=0.85,
        label=f"True/false twins, same city (n = {len(cos_twin)})")
ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Density")
ax.set_title("A true statement and its false twin are close in embedding space")
ax.legend(frameon=False, loc="upper left")
savefig(fig, "eda_cosine_twins.png")
for nm, v in [("twin", cos_twin), ("random", cos_rand)]:
    rows.append({"section": "EDA", "dataset": "cities", "method": "cosine", "metric": f"cosine_{nm}_median",
                 "value": round(float(np.median(v)), 4), "ci_low": round(float(np.percentile(v, 2.5)), 4),
                 "ci_high": round(float(np.percentile(v, 97.5)), 4), "notes": "ci columns = 2.5/97.5 percentiles of the distribution"})

pd.DataFrame(rows).to_csv(RES / "eda_results.csv", index=False)
print(pd.DataFrame(rows).to_string())
