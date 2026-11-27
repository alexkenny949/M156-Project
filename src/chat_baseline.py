"""Step 7: zero-shot chat-model True/False baseline on the test-split statements.

Baseline only (not a course method). Responses cached to data/baseline/{dataset}.csv;
re-running only calls the API for statements not yet cached.
Run: .venv/bin/python src/chat_baseline.py
"""
import json
import time

import pandas as pd

from common import ROOT, RES, load, split_mask
from embed import BASE_URL, load_key
from openai import OpenAI

MODEL = "nvidia/nemotron-3-super-120b-a12b"
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}  # answer directly, no hidden reasoning
PRICE_IN, PRICE_OUT = 0.30, 0.90  # USD per 1M tokens, from /v1/models?verbose=true on 2026-11-26
PROMPT = ("Is the following statement true or false? Answer with exactly one word: True or False."
          "\n\nStatement: {s}")
COST_CAP = 2.00
OUT = ROOT / "data" / "baseline"


def parse(ans):
    if ans is None:
        return None
    a = ans.strip().strip(".").strip().lower()
    return {"true": 1, "false": 0}.get(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    work = {}
    for name in ("cities", "neg_cities"):
        _, meta = load(name)
        _, te = split_mask(meta)
        work[name] = meta[te].reset_index(drop=True)
    n = sum(len(d) for d in work.values())
    est = n * (60 * PRICE_IN + 2 * PRICE_OUT) / 1e6  # ~60 input tokens incl. chat template
    print(f"{n} statements, estimated cost ${est:.4f} (cap ${COST_CAP})")
    if est > COST_CAP:
        raise SystemExit("Cost estimate exceeds cap; stopping.")

    client = OpenAI(base_url=BASE_URL, api_key=load_key())
    summary = {"model": MODEL, "extra_body": EXTRA, "price_per_1M_in": PRICE_IN,
               "price_per_1M_out": PRICE_OUT, "prompt": PROMPT, "temperature": 0, "max_tokens": 2,
               "estimated_cost_usd": round(est, 4)}
    tok_in = tok_out = 0
    for name, df in work.items():
        path = OUT / f"{name}.csv"
        cache = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["statement", "raw_answer"])
        done = dict(zip(cache["statement"], cache["raw_answer"]))
        new = []
        for s in df["statement"]:
            if s in done:
                continue
            for attempt in range(5):
                try:
                    r = client.chat.completions.create(
                        model=MODEL, messages=[{"role": "user", "content": PROMPT.format(s=s)}],
                        temperature=0, max_tokens=2, extra_body=EXTRA)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            tok_in += r.usage.prompt_tokens
            tok_out += r.usage.completion_tokens
            done[s] = r.choices[0].message.content
            new.append(s)
        print(f"{name}: {len(new)} new API calls")
        df = df.assign(raw_answer=df["statement"].map(done))
        df[["statement", "raw_answer"]].to_csv(path, index=False)
        df["pred"] = df["raw_answer"].map(parse)
        unparse = df["pred"].isna()
        correct = (df["pred"] == df["label"]) & ~unparse  # unparseable scored as wrong
        summary[name] = {"n": len(df), "n_unparseable": int(unparse.sum()),
                         "accuracy": float(correct.mean()),
                         "pred_true_rate": float((df["pred"] == 1).mean())}
        df.assign(correct=correct)[["statement", "label", "city", "correct_country", "region",
                                    "raw_answer", "pred", "correct"]].to_csv(
            RES / f"baseline_predictions_{name}.csv", index=False)
    summary["tokens_this_run"] = {"input": tok_in, "output": tok_out,
                                  "cost_usd": round((tok_in * PRICE_IN + tok_out * PRICE_OUT) / 1e6, 5)}
    (RES / "baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
