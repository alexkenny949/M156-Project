"""Combine all result pieces into results/results_table.csv.
Run: .venv/bin/python src/make_table.py
"""
import json
import pandas as pd
from common import RES

parts = [pd.read_csv(RES / "eda_results.csv"), pd.read_csv(RES / "eval_results.csv"),
         pd.read_csv(RES / "extra_results.csv")]
b = json.loads((RES / "baseline.json").read_text())
emb = json.loads((RES / "embedding_check.json").read_text())
extra = []
for ds in ("cities", "neg_cities"):
    extra.append({"section": "Baseline", "dataset": f"{ds} test", "method": f"Chat baseline ({b['model']})",
                  "metric": "accuracy", "value": round(b[ds]["accuracy"], 4),
                  "notes": f"zero-shot, temperature 0, max_tokens 2, thinking disabled; unparseable = {b[ds]['n_unparseable']}"})
extra.append({"section": "Cost", "dataset": "cities test + neg_cities test", "method": "Chat baseline",
              "metric": "api_cost_usd", "value": b["tokens_this_run"]["cost_usd"],
              "notes": f"{b['tokens_this_run']['input']} input + {b['tokens_this_run']['output']} output tokens; "
                       f"${b['price_per_1M_in']}/${b['price_per_1M_out']} per 1M in/out"})
for ds in ("cities", "neg_cities"):
    extra.append({"section": "Cost", "dataset": ds, "method": "Embedding (Qwen3-Embedding-8B)",
                  "metric": "prompt_tokens", "value": emb[ds]["prompt_tokens"], "notes": "$0.01 per 1M tokens"})
df = pd.concat(parts + [pd.DataFrame(extra)], ignore_index=True)
order = ["EDA", "Tuning", "CV", "Test", "Paired", "Baseline", "Unsupervised", "Transfer", "Learning curve", "Hard negatives", "Errors", "Cost"]
df["_o"] = df["section"].map({s: i for i, s in enumerate(order)})
df = df.sort_values("_o", kind="stable").drop(columns="_o")
df.to_csv(RES / "results_table.csv", index=False)
print(len(df), "rows")
