"""Step 5a: hyperparameter tuning by 5-fold city-grouped CV on the cities TRAINING split only.

The test split is never loaded into a model here. Writes results/tuning.json and
results/cv_results_{lr,rf}.csv. Also runs PCA + k-means (unsupervised) on the training split.
Run: .venv/bin/python src/tune.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import RES, SEED, cv_folds, load, split_mask


def make_lr():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED))])


def make_rf():
    return Pipeline([("pca", "passthrough"),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])


LR_GRID = {"clf__C": list(np.logspace(-4, 2, 13)), "clf__penalty": ["l1", "l2"]}
RF_GRID = {"pca": ["passthrough", PCA(20, random_state=SEED), PCA(50, random_state=SEED),
                   PCA(100, random_state=SEED)],
           "clf__n_estimators": [300, 500], "clf__max_depth": [None, 10, 20],
           "clf__max_features": ["sqrt", 0.1]}


def pca_label(p):
    return "none" if p == "passthrough" else f"PCA{p.n_components}"


def main():
    X, meta = load("cities")
    tr, _ = split_mask(meta)
    Xtr, ytr, gtr = X[tr], meta["label"].to_numpy()[tr], meta["city"][tr].reset_index(drop=True)
    folds = cv_folds(gtr)
    out = {"n_train": int(tr.sum()), "n_train_cities": int(gtr.nunique()),
           "folds_val_sizes": [len(v) for _, v in folds]}

    for name, make, grid in [("LR", make_lr, LR_GRID), ("RF", make_rf, RF_GRID)]:
        n_cfg = int(np.prod([len(v) for v in grid.values()]))
        t0 = time.time()
        gs = GridSearchCV(make(), grid, scoring="accuracy", cv=folds, n_jobs=-1, refit=False)
        gs.fit(Xtr, ytr)
        secs = time.time() - t0
        cv = pd.DataFrame(gs.cv_results_)
        if name == "RF":
            cv["param_pca"] = cv["param_pca"].map(pca_label)
        keep = [c for c in cv.columns if c.startswith("param_") or c.startswith("split")
                or c in ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]
        cv[keep].to_csv(RES / f"cv_results_{name.lower()}.csv", index=False)
        best = gs.best_index_
        params = {k: (pca_label(v) if k == "pca" else (float(v) if isinstance(v, (float, np.floating)) else v))
                  for k, v in gs.best_params_.items()}
        out[name] = {"grid_size": n_cfg, "n_fits": n_cfg * len(folds), "runtime_sec": round(secs, 1),
                     "best_params": params,
                     "cv_mean": float(cv.loc[best, "mean_test_score"]),
                     "cv_sd": float(cv.loc[best, "std_test_score"])}
        # best per sub-variant (L1 vs L2; raw vs each PCA size)
        sub = "param_clf__penalty" if name == "LR" else "param_pca"
        out[name]["best_by_variant"] = {
            str(k): {"cv_mean": float(g["mean_test_score"].max()),
                     "cv_sd": float(g.loc[g["mean_test_score"].idxmax(), "std_test_score"]),
                     "params": {c.replace("param_", ""): (str(g.loc[g["mean_test_score"].idxmax(), c]))
                                for c in g.columns if c.startswith("param_")}}
            for k, g in cv.groupby(sub)}
        print(name, json.dumps(out[name], indent=1, default=str))

    # L1 sparsity: refit best L1 config on training split (no test data) and count nonzero weights
    l1 = out["LR"]["best_by_variant"]["l1"]["params"]
    m = make_lr().set_params(clf__C=float(l1["clf__C"]), clf__penalty="l1").fit(Xtr, ytr)
    out["LR"]["best_l1_nonzero_coefs"] = int((m["clf"].coef_ != 0).sum())

    # Unsupervised: PCA + k-means (k=2) on training split, labels used only for scoring
    out["kmeans"] = {}
    t0 = time.time()
    for k in (2, 10, 50):
        pipe = Pipeline([("pca", PCA(k, random_state=SEED)),
                         ("km", KMeans(2, n_init=10, random_state=SEED))])
        cl = pipe.fit_predict(Xtr)
        acc = accuracy_score(ytr, cl)
        reg = meta["region"][tr].to_numpy()
        out["kmeans"][f"PCA{k}"] = {"ARI_label": float(adjusted_rand_score(ytr, cl)),
                                    "cluster_accuracy": float(max(acc, 1 - acc)),
                                    "ARI_region": float(adjusted_rand_score(reg, cl))}
    out["kmeans"]["runtime_sec"] = round(time.time() - t0, 1)
    out["kmeans"]["grid_size"] = 3
    print("kmeans", json.dumps(out["kmeans"], indent=1))
    (RES / "tuning.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
