"""Handoff step 3: embed cities / neg_cities with Qwen3-Embedding-8B via Nebius Token Factory.

Run from the project root:  .venv/bin/python -I src/embed.py
Reads NEBIUS_API_KEY from the environment or from .env. Skips any dataset already cached.
Writes data/embeddings/{name}.npy (n x 4096, float32), {name}_meta.csv, and
results/embedding_check.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "geometry-of-truth" / "datasets"
OUT = ROOT / "data" / "embeddings"
MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
BATCH = 64
DATASETS = ["cities", "neg_cities"]


def load_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k == "NEBIUS_API_KEY":
                key = v.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("NEBIUS_API_KEY not set (env var or .env)")
    return key


def embed_all(client: OpenAI, texts: list[str]) -> tuple[np.ndarray, int]:
    vecs, tokens = [], 0
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for attempt in range(5):
            try:
                r = client.embeddings.create(model=MODEL, input=batch, encoding_format="float")
                break
            except Exception as e:  # retry transient API errors
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  batch {i}: {type(e).__name__}, retrying in {wait}s")
                time.sleep(wait)
        data = sorted(r.data, key=lambda d: d.index)
        assert len(data) == len(batch)
        vecs.extend(d.embedding for d in data)
        tokens += r.usage.prompt_tokens
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}")
    return np.asarray(vecs, dtype=np.float32), tokens


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=load_key())
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "results" / "embedding_check.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"model": MODEL, "base_url": BASE_URL, "input": "plain statement text, no instruction prefix"})

    for name in DATASETS:
        npy = OUT / f"{name}.npy"
        df = pd.read_csv(DATA / f"{name}.csv")
        if npy.exists():
            print(f"{name}: cached, skipping API call")
            X = np.load(npy)
            tokens = report.get(name, {}).get("prompt_tokens")
        else:
            print(f"{name}: embedding {len(df)} statements")
            X, tokens = embed_all(client, df["statement"].tolist())
            np.save(npy, X)
            df[["statement", "label", "city", "country", "correct_country"]].to_csv(
                OUT / f"{name}_meta.csv", index=False)
        norms = np.linalg.norm(X, axis=1)
        report[name] = {
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "rows_match_csv": int(X.shape[0]) == len(df),
            "any_nan": bool(np.isnan(X).any()),
            "l2_norm_min": round(float(norms.min()), 6),
            "l2_norm_max": round(float(norms.max()), 6),
            "l2_normalized": bool(np.allclose(norms, 1.0, atol=1e-4)),
            "n_duplicate_vectors": int(len(X) - len(np.unique(X, axis=0))),
            "prompt_tokens": tokens,
        }

    # Determinism check: re-embed 3 cities statements and compare with the cache
    X = np.load(OUT / "cities.npy")
    texts = pd.read_csv(DATA / "cities.csv")["statement"].tolist()[:3]
    r = client.embeddings.create(model=MODEL, input=texts, encoding_format="float")
    again = np.asarray([d.embedding for d in sorted(r.data, key=lambda d: d.index)], dtype=np.float32)
    report["determinism_max_abs_diff_first3"] = float(np.abs(again - X[:3]).max())

    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
