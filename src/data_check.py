"""Handoff step 2: load cities / neg_cities and report basic data facts.

Run from the project root:  .venv/bin/python -I src/data_check.py
Writes results/data_check.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "geometry-of-truth" / "datasets"
OUT = ROOT / "results" / "data_check.json"


def length_stats(s: pd.Series) -> dict:
    return {"mean": round(float(s.mean()), 3), "sd": round(float(s.std()), 3),
            "min": int(s.min()), "max": int(s.max())}


def check(name: str) -> tuple[dict, pd.DataFrame]:
    path = DATA / f"{name}.csv"
    df = pd.read_csv(path)
    df["n_chars"] = df["statement"].str.len()
    df["n_words"] = df["statement"].str.split().str.len()

    # Per city: how many statements, and which labels
    per_city = df.groupby("city")["label"].agg(["size", "nunique"])
    both = int((per_city["nunique"] == 2).sum())

    # Is the stated country in the true statement always correct_country?
    true_rows = df[df["label"] == (1 if name == "cities" else 0)]
    consistent = bool((true_rows["country"] == true_rows["correct_country"]).all())

    length = {}
    for col in ["n_chars", "n_words"]:
        g1, g0 = df.loc[df.label == 1, col], df.loc[df.label == 0, col]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        length[col] = {"true": length_stats(g1), "false": length_stats(g0),
                       "welch_t": round(float(t.statistic), 3),
                       "p_value": round(float(t.pvalue), 4)}

    # Country frequency in true vs false statements (does country predict label?)
    ct = pd.crosstab(df["country"], df["label"])
    chi2 = stats.chi2_contingency(ct)

    info = {
        "file": str(path.relative_to(ROOT)),
        "format": "CSV, comma-separated, header row",
        "columns": {c: str(t) for c, t in df.drop(columns=["n_chars", "n_words"]).dtypes.items()},
        "n_rows": int(len(df)),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicate_statements": int(df["statement"].duplicated().sum()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "label_prop_true": round(float(df["label"].mean()), 4),
        "n_unique_cities": int(df["city"].nunique()),
        "n_unique_countries_stated": int(df["country"].nunique()),
        "n_unique_correct_countries": int(df["correct_country"].nunique()),
        "statements_per_city": {str(k): int(v) for k, v in per_city["size"].value_counts().sort_index().items()},
        "n_cities_in_both_true_and_false": both,
        "country_column_matches_correct_for_affirmed_rows": consistent,
        "length_by_label": length,
        "country_vs_label_chi2": {"chi2": round(float(chi2.statistic), 2), "dof": int(chi2.dof),
                                  "p_value": round(float(chi2.pvalue), 4)},
        "top_correct_countries": {k: int(v) for k, v in
                                  df.drop_duplicates("city")["correct_country"].value_counts().head(10).items()},
        "examples_true": df.loc[df.label == 1, "statement"].head(3).tolist(),
        "examples_false": df.loc[df.label == 0, "statement"].head(3).tolist(),
    }
    return info, df


def main() -> None:
    results = {"source_commit": "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"}
    frames = {}
    for name in ["cities", "neg_cities"]:
        results[name], frames[name] = check(name)

    c, n = frames["cities"], frames["neg_cities"]
    key = ["city", "country"]
    merged = c.merge(n, on=key, suffixes=("_aff", "_neg"))
    results["cities_vs_neg_cities"] = {
        "same_city_set": set(c.city) == set(n.city),
        "n_shared_cities": len(set(c.city) & set(n.city)),
        "n_rows_matching_on_city_and_country": int(len(merged)),
        "labels_exactly_flipped": bool((merged.label_aff == 1 - merged.label_neg).all()),
        "neg_statement_equals_aff_with_not": bool(
            (merged.statement_aff.str.replace(" is in ", " is not in ", regex=False)
             == merged.statement_neg).all()),
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    np.random.seed(0)
    main()
