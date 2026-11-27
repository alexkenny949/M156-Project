"""Steps 5b-6: final test evaluation, uncertainty, transfer tests, error analysis, figures.

Uses the hyperparameters chosen in tune.py (results/tuning.json). Each final model is
refit on the full cities training split and scored on the test split exactly once
(enforced by TestGuard). Transfer models reuse the cities-tuned hyperparameters.
Run: .venv/bin/python src/evaluate.py
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from common import (CHAT_C, GRAY, LR_C, RES, RF_C, SEED, TEXT, TEXT2, load, savefig, set_style,
                    split_mask)
from tune import make_lr, make_rf

set_style()
B = 1000
rows = []


def rec(section, dataset, method, metric, value, lo=np.nan, hi=np.nan, notes=""):
    rows.append({"section": section, "dataset": dataset, "method": method, "metric": metric,
                 "value": value, "ci_low": lo, "ci_high": hi, "notes": notes})


class TestGuard:
    """Raises if any (model, evaluation set) pair is scored more than once."""
    def __init__(self):
        self.used = set()

    def score(self, key, model, X):
        assert key not in self.used, f"test set {key} already used"
        self.used.add(key)
        return model.predict(X), model.predict_proba(X)[:, 1]


def build(name, params):
    if name == "LR":
        return make_lr().set_params(clf__C=float(params["clf__C"]), clf__penalty=params["clf__penalty"])
    p = params["pca"]
    pca = "passthrough" if p == "none" else PCA(int(p.replace("PCA", "")), random_state=SEED)
    md = params["clf__max_depth"]
    mf = params["clf__max_features"]
    return make_rf().set_params(pca=pca, clf__n_estimators=int(params["clf__n_estimators"]),
                                clf__max_depth=None if md in (None, "None") else int(md),
                                clf__max_features=mf if mf == "sqrt" else float(mf), clf__n_jobs=-1)


def city_bootstrap(cities: np.ndarray, rng) -> list[np.ndarray]:
    """Resample test cities with replacement; each resample returns row indices (both statements)."""
    uniq = np.unique(cities)
    rows_of = {c: np.where(cities == c)[0] for c in uniq}
    return [np.concatenate([rows_of[c] for c in rng.choice(uniq, len(uniq), replace=True)])
            for _ in range(B)]


def main():
    tuning = json.loads((RES / "tuning.json").read_text())
    Xc, mc = load("cities")
    Xn, mn = load("neg_cities")
    trc, tec = split_mask(mc)
    trn, ten = split_mask(mn)
    yc, yn = mc["label"].to_numpy(), mn["label"].to_numpy()
    # leakage: no test city in the training split (also asserted inside split_mask/cv_folds)
    assert not set(mc.city[tec]) & set(mc.city[trc])
    assert set(mc.city[tec]) == set(mn.city[ten]), "cities/neg_cities test cities differ"
    guard = TestGuard()
    out = {}

    # ---- Final models: refit on cities training split, score once on cities test split
    preds, probs, models, fit_secs = {}, {}, {}, {}
    for name in ("LR", "RF"):
        t0 = time.time()
        models[name] = build(name, tuning[name]["best_params"]).fit(Xc[trc], yc[trc])
        fit_secs[name] = time.time() - t0
        preds[name], probs[name] = guard.score(("cities_test", name), models[name], Xc[tec])
    yt = yc[tec]
    test_cities = mc.city[tec].to_numpy()
    rng = np.random.default_rng(SEED)
    boots = city_bootstrap(test_cities, rng)

    chat = pd.read_csv(RES / "baseline_predictions_cities.csv")
    chat = chat.set_index("statement").loc[mc.statement[tec]]  # align to test row order
    chat_correct = chat["correct"].to_numpy().astype(bool)

    for name in ("LR", "RF"):
        p, pr = preds[name], probs[name]
        acc, auc = accuracy_score(yt, p), roc_auc_score(yt, pr)
        accs = [accuracy_score(yt[i], p[i]) for i in boots]
        aucs = [roc_auc_score(yt[i], pr[i]) for i in boots]
        lo, hi = np.percentile(accs, [2.5, 97.5])
        alo, ahi = np.percentile(aucs, [2.5, 97.5])
        cm = confusion_matrix(yt, p, labels=[0, 1])
        t = tuning[name]
        out[name] = {"test_acc": acc, "acc_ci": [lo, hi], "test_auc": auc, "auc_ci": [alo, ahi],
                     "cm": cm.tolist(), "cv_mean": t["cv_mean"], "cv_sd": t["cv_sd"]}
        rec("Test", "cities", name, "test_accuracy", round(acc, 4), round(lo, 4), round(hi, 4),
            "95% CI: city-level bootstrap, 1000 resamples")
        rec("Test", "cities", name, "test_roc_auc", round(auc, 4), round(alo, 4), round(ahi, 4),
            "95% CI: city-level bootstrap, 1000 resamples")
        rec("CV", "cities", name, "cv_accuracy_mean", round(t["cv_mean"], 4), notes=f"sd = {t['cv_sd']:.4f}; 5-fold city-grouped CV on training split")
        rec("CV", "cities", name, "cv_accuracy_sd", round(t["cv_sd"], 4))
        rec("Test", "cities", name, "confusion_matrix", json.dumps(cm.tolist()),
            notes="rows = true label [False, True], cols = predicted [False, True]")
        rec("Tuning", "cities", name, "grid_size", t["grid_size"], notes=f"{t['n_fits']} CV fits")
        rec("Tuning", "cities", name, "tuning_runtime_sec", t["runtime_sec"])
        rec("Tuning", "cities", name, "final_fit_runtime_sec", round(fit_secs[name], 1))
        rec("Tuning", "cities", name, "best_params", json.dumps(t["best_params"]))
        for v, d in t["best_by_variant"].items():
            rec("CV", "cities", f"{name} ({v})", "cv_accuracy_mean", round(d["cv_mean"], 4),
                notes=f"sd = {d['cv_sd']:.4f}; best config within this variant: {json.dumps(d['params'])}")
        if abs(acc - t["cv_mean"]) > 0.10:
            raise SystemExit(f"STOP: {name} CV and test differ by more than 10 points")
    rec("Tuning", "cities", "LR (l1)", "nonzero_coefficients", tuning["LR"]["best_l1_nonzero_coefs"],
        notes="best L1 model refit on training split; out of 4096")
    rec("Test", "cities", "Majority class", "test_accuracy", 0.5, notes="test split is exactly 150 true / 150 false")
    ca = [chat_correct[i].mean() for i in boots]
    lo, hi = np.percentile(ca, [2.5, 97.5])
    out["Chat"] = {"test_acc": float(chat_correct.mean()), "acc_ci": [lo, hi]}
    rec("Test", "cities", "Chat baseline (Nemotron-3-Super-120B)", "test_accuracy",
        round(chat_correct.mean(), 4), round(lo, 4), round(hi, 4), "zero-shot; baseline only, not a course method")
    for name in ("LR", "RF"):
        if out[name]["test_acc"] > 0.99:
            print(f"NOTE: {name} test accuracy {out[name]['test_acc']:.4f} > 0.99 (user approved proceeding)")

    # ---- Paired LR vs RF
    cl, cr = preds["LR"] == yt, preds["RF"] == yt
    diffs = [cl[i].mean() - cr[i].mean() for i in boots]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    b, c = int((cl & ~cr).sum()), int((~cl & cr).sum())
    p_mc = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    out["paired"] = {"diff": cl.mean() - cr.mean(), "ci": [lo, hi], "lr_only_right": b, "rf_only_right": c, "mcnemar_p": p_mc}
    rec("Paired", "cities", "LR minus RF", "test_accuracy_difference", round(cl.mean() - cr.mean(), 4),
        round(lo, 4), round(hi, 4), "95% CI: paired city-level bootstrap")
    rec("Paired", "cities", "LR vs RF", "mcnemar_exact_p", round(p_mc, 4),
        notes=f"discordant: LR right/RF wrong = {b}, RF right/LR wrong = {c}")

    # ---- Unsupervised (from tuning, training split)
    for k, d in tuning["kmeans"].items():
        if k.startswith("PCA"):
            for m in ("ARI_label", "cluster_accuracy", "ARI_region"):
                rec("Unsupervised", "cities (train split)", f"PCA{k[3:]} + k-means(2)", m, round(d[m], 4))
    rec("Tuning", "cities", "PCA + k-means", "grid_size", 3, notes="PCs in {2, 10, 50}")

    # ---- Transfer tests (cities-tuned hyperparameters for every model)
    tr_rows = []
    for name in ("LR", "RF"):
        base = build(name, tuning[name]["best_params"])
        settings = [
            ("within cities", "cities", models[name], None, None, None),  # already scored above
            ("(a) city-disjoint: cities -> neg_cities", "neg_cities_test", None, (Xc[trc], yc[trc]), Xn[ten], yn[ten]),
            ("(b) all cities: cities -> neg_cities", "neg_cities_all", None, (Xc, yc), Xn, yn),
            ("within neg_cities", "neg_cities_test_within", None, (Xn[trn], yn[trn]), Xn[ten], yn[ten]),
            ("(a) city-disjoint: neg_cities -> cities", "cities_test_from_neg", None, (Xn[trn], yn[trn]), Xc[tec], yc[tec]),
            ("(b) all cities: neg_cities -> cities", "cities_all", None, (Xn, yn), Xc, yc),
        ]
        for label, key, fitted, train, Xe, ye in settings:
            if label == "within cities":
                acc, auc = out[name]["test_acc"], out[name]["test_auc"]
                ptrue = float(preds[name].mean())
            else:
                m = fitted or clone(base).fit(*train)
                p, pr = guard.score((key, name), m, Xe)
                acc, auc = accuracy_score(ye, p), roc_auc_score(ye, pr)
                ptrue = float(p.mean())
            tr_rows.append({"model": name, "setting": label, "accuracy": acc, "auc": auc,
                            "share_predicted_true": ptrue})
            rec("Transfer", label, name, "accuracy", round(acc, 4), notes="cities-tuned hyperparameters")
            rec("Transfer", label, name, "roc_auc", round(auc, 4))
            rec("Transfer", label, name, "share_predicted_true", round(ptrue, 4),
                notes="true share is 0.5; deviation = threshold shift")
    trdf = pd.DataFrame(tr_rows)
    trdf.to_csv(RES / "transfer.csv", index=False)

    # ---- Error analysis for the best model (higher test accuracy; LR on tie)
    best = "LR" if out["LR"]["test_acc"] >= out["RF"]["test_acc"] else "RF"
    te = mc[tec].reset_index(drop=True).assign(pred=preds[best], p_true=probs[best])
    te["correct"] = te["pred"] == te["label"]
    errs = te[~te["correct"]]
    errs[["statement", "label", "pred", "p_true", "city", "country", "correct_country", "region"]].to_csv(
        RES / f"errors_{best}.csv", index=False)
    by_reg = te.groupby("region").agg(n=("correct", "size"), errors=("correct", lambda s: int((~s).sum())))
    by_reg["error_rate"] = by_reg["errors"] / by_reg["n"]
    top = te["correct_country"].value_counts().head(10).index
    by_cty = te[te.correct_country.isin(top)].groupby("correct_country").agg(
        n=("correct", "size"), errors=("correct", lambda s: int((~s).sum())))
    by_cty["error_rate"] = by_cty["errors"] / by_cty["n"]
    by_cty = by_cty.sort_values("n", ascending=False)
    by_reg.to_csv(RES / f"errors_by_region_{best}.csv")
    by_cty.to_csv(RES / f"errors_by_country_top10_{best}.csv")
    rec("Errors", "cities test", best, "n_errors", len(errs), notes=f"out of {len(te)}; list in results/errors_{best}.csv")
    for r, d in by_reg.iterrows():
        rec("Errors", "cities test", best, f"error_rate_region_{r}", round(d.error_rate, 4), notes=f"{d.errors}/{d.n}")
    for r, d in by_cty.iterrows():
        rec("Errors", "cities test", best, f"error_rate_country_{r}", round(d.error_rate, 4), notes=f"{d.errors}/{d.n}")
    # Statements both final models get wrong (and the chat model)
    te["rf_correct"] = preds["RF"] == yt
    te["lr_correct"] = preds["LR"] == yt
    te["chat_correct"] = chat_correct
    te[~(te.lr_correct & te.rf_correct & te.chat_correct)][
        ["statement", "label", "region", "lr_correct", "rf_correct", "chat_correct"]].to_csv(
        RES / "errors_all_models.csv", index=False)

    pd.DataFrame(rows).to_csv(RES / "eval_results.csv", index=False)
    (RES / "eval_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(pd.DataFrame(rows).to_string())
    print(trdf.to_string())
    print("best model:", best); print(errs[["statement", "p_true"]].to_string())

    figures(out, trdf, tuning)


def figures(out, trdf, tuning):
    # 1. Test accuracy with 95% CIs
    fig, ax = plt.subplots(figsize=(8, 3.8))
    items = [("Logistic regression", out["LR"], LR_C), ("Random forest", out["RF"], RF_C),
             ("Chat model, zero-shot\n(baseline)", out["Chat"], CHAT_C)]
    for i, (lab, d, c) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot(d["acc_ci"], [y, y], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(d["test_acc"], y, "o", color=c, ms=10, mec="white", mew=1.5)
        ax.annotate(f"{d['test_acc']:.1%}  [{d['acc_ci'][0]:.1%}, {d['acc_ci'][1]:.1%}]",
                    (d["acc_ci"][0], y), xytext=(0, 12), textcoords="offset points",
                    color=TEXT, fontsize=12)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items][::-1])
    ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_xlim(0.95, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Test accuracy (cities, 300 statements)")
    ax.set_title("Test accuracy with 95% bootstrap CIs")
    ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.06, "Axis starts at 95%. Majority-class baseline = 50%. CIs resample the 150 test cities.",
             color=TEXT2, fontsize=11)
    savefig(fig, "results_test_accuracy.png")

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"wspace": 0.35})
    for ax, name, title in [(axes[0], "LR", "Logistic regression"), (axes[1], "RF", "Random forest")]:
        cm = np.array(out[name]["cm"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=20,
                    color="white" if v > cm.max() / 2 else TEXT)
        ax.set_xticks([0, 1], ["False", "True"])
        ax.set_yticks([0, 1], ["False", "True"])
        ax.set_xlabel("Predicted")
        if name == "LR":
            ax.set_ylabel("Actual")
        ax.set_title(f"{title}\naccuracy {out[name]['test_acc']:.1%}", fontsize=14)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices, cities test split (n = 300)", x=0.02, ha="left",
                 fontweight="bold", fontsize=16, y=1.06)
    savefig(fig, "results_confusion_matrices.png")

    # 3. Transfer bar chart
    order = ["within cities", "(a) city-disjoint: cities -> neg_cities", "(b) all cities: cities -> neg_cities",
             "within neg_cities", "(a) city-disjoint: neg_cities -> cities", "(b) all cities: neg_cities -> cities"]
    short = ["cities\n→ cities", "cities → neg\n(a) disjoint", "cities → neg\n(b) all",
             "neg\n→ neg", "neg → cities\n(a) disjoint", "neg → cities\n(b) all"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.38
    for k, (name, c, lab) in enumerate([("LR", LR_C, "Logistic regression"), ("RF", RF_C, "Random forest")]):
        v = trdf[trdf.model == name].set_index("setting").loc[order, "accuracy"].to_numpy()
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=c, label=lab)
        for xi, vi in zip(x + (k - 0.5) * (w + 0.02), v):
            ax.text(xi, vi + 0.015, f"{vi:.1%}", ha="center", fontsize=10, color=TEXT)
    ax.axhline(0.5, color=TEXT2, lw=1, ls="--")
    ax.text(len(order) - 0.5, 0.51, "chance", color=TEXT2, fontsize=11, ha="right", va="bottom")
    ax.axvline(2.5, color=GRAY, lw=1)
    ax.set_xticks(x, short, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Accuracy")
    ax.set_title("Does a truth classifier transfer across negation?", pad=34)
    ax.legend(frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 0.98))
    ax.grid(axis="x", visible=False)
    auc_min = trdf["auc"].min()
    fig.text(0.0, -0.04, f"Cities-tuned hyperparameters throughout. ROC AUC stays >= {auc_min:.3f} in every setting: "
             "the ranking transfers, the decision threshold does not.", color=TEXT2, fontsize=11)
    savefig(fig, "results_transfer.png")

    # 4. LR tuning curve
    cv = pd.read_csv(RES / "cv_results_lr.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pen, c, lab in [("l2", LR_C, "L2 (ridge)"), ("l1", RF_C, "L1 (lasso)")]:
        d = cv[cv.param_clf__penalty == pen].sort_values("param_clf__C")
        ax.plot(d.param_clf__C, d.mean_test_score, "-o", color=c, lw=2, ms=6, label=lab)
        ax.fill_between(d.param_clf__C, d.mean_test_score - d.std_test_score,
                        d.mean_test_score + d.std_test_score, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("CV accuracy (mean ± sd)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Logistic regression tuning: 5-fold city-grouped CV")
    ax.legend(frameon=False, loc="lower right")
    bp = tuning["LR"]["best_params"]
    ax.annotate(f"selected: C = {float(bp['clf__C']):g}, {bp['clf__penalty'].upper()}",
                (float(bp["clf__C"]), tuning["LR"]["cv_mean"]), xytext=(-20, -45),
                textcoords="offset points", ha="right", fontsize=11, color=TEXT2,
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    savefig(fig, "results_lr_tuning.png")


if __name__ == "__main__":
    main()
