"""
OTTO Multi-Objective Session-Based Recommendation - Baseline.

Co-visitation matrix candidate generator. Items that co-occur within a
1-hour window in the same session are accumulated in three weighted
counters (clicks, carts, orders). At inference time we take the last K
events of a session, look up their top-N co-visited items per matrix,
aggregate, and return the top-20.

This is the canonical strong baseline from the public 2022 OTTO Kaggle
competition leaderboard. No neural component, runs on CPU, fits in
memory if items are pre-filtered to the top 200K most-active aids.

Run:
    python src/model_baseline.py --data ../data --out ../deliverables

The script is NOT executed in scaffold phase (Phase 1). The user runs
it later in the main session once the OTTO dataset has been downloaded
to ../data/.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# OTTO competition score weights
EVENT_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}

# Co-visitation aggregation weights (from public Kaggle leaderboard convention)
COVIS_WEIGHTS = {"clicks": 1.0, "carts": 6.0, "orders": 3.0}

# 1-hour co-visitation window in milliseconds
WINDOW_MS = 60 * 60 * 1000

TOP_N_CANDIDATES = 20
TOP_K_RECENT_EVENTS = 30


def load_sessions(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Load OTTO sessions as a flat DataFrame of (session, aid, ts, type).

    Prefers the parquet shards from the radek1/otto-full-optimized-memory-footprint
    Kaggle dataset. Falls back to raw jsonl if parquet is absent.
    """
    parquet_path = data_dir / f"{split}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No {split} data found in {data_dir}")
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["session"]
            for ev in rec["events"]:
                rows.append((sid, ev["aid"], ev["ts"], ev["type"]))
    return pd.DataFrame(rows, columns=["session", "aid", "ts", "type"])


def build_covisitation(df: pd.DataFrame, top_n: int = 40) -> dict:
    """
    Build three sparse co-visitation dictionaries: aid -> [top-N co-visited aids].

    For each session, for each pair of events (a, b) within WINDOW_MS,
    increment a counter weighted by the type of event b. After all
    sessions are processed, retain only the top-N targets per source aid.
    """
    matrices = {"clicks": defaultdict(Counter),
                "carts": defaultdict(Counter),
                "orders": defaultdict(Counter)}
    df = df.sort_values(["session", "ts"], kind="mergesort")
    for sid, g in df.groupby("session", sort=False):
        aids = g["aid"].to_numpy()
        ts = g["ts"].to_numpy()
        types = g["type"].to_numpy()
        n = len(aids)
        for i in range(n):
            for j in range(i + 1, n):
                if ts[j] - ts[i] > WINDOW_MS:
                    break
                w = COVIS_WEIGHTS.get(types[j], 1.0)
                matrices[types[j]][aids[i]][aids[j]] += w
                matrices[types[j]][aids[j]][aids[i]] += w
    pruned = {k: {src: [a for a, _ in c.most_common(top_n)]
                  for src, c in m.items()} for k, m in matrices.items()}
    return pruned


def predict_session(events: pd.DataFrame, covis: dict, top_k: int = TOP_K_RECENT_EVENTS) -> dict:
    """Generate top-20 candidates for each of clicks / carts / orders for one session."""
    recent = events.tail(top_k)["aid"].tolist()
    out = {}
    for tgt in ("clicks", "carts", "orders"):
        scores: Counter = Counter()
        m = covis[tgt]
        for rank, aid in enumerate(reversed(recent)):
            decay = 1.0 / (1 + rank)
            for cand in m.get(aid, []):
                scores[cand] += decay
        out[tgt] = [a for a, _ in scores.most_common(20)]
    return out


def recall_at_k(pred: list, truth: list, k: int = 20) -> float:
    if not truth:
        return 0.0
    pred_set = set(pred[:k])
    return len(pred_set & set(truth)) / min(len(truth), k)


def evaluate(predictions: dict, labels: pd.DataFrame) -> dict:
    """Compute weighted Recall@20 across the three event types."""
    scores = {}
    for evt in ("clicks", "carts", "orders"):
        sub = labels[labels["type"] == evt]
        rs = []
        for _, row in sub.iterrows():
            rs.append(recall_at_k(predictions.get(row["session"], {}).get(evt, []),
                                  row["ground_truth"]))
        scores[evt] = float(np.mean(rs)) if rs else 0.0
    scores["weighted"] = sum(EVENT_WEIGHTS[k] * scores[k] for k in EVENT_WEIGHTS)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../deliverables"))
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Loading train sessions...")
    train = load_sessions(args.data, "train")
    print(f"  {len(train):,} events across {train['session'].nunique():,} sessions")

    print("Building co-visitation matrices...")
    covis = build_covisitation(train, top_n=args.top_n)
    n_keys = sum(len(m) for m in covis.values())
    print(f"  {n_keys:,} (event_type, source_aid) keys total")

    covis_path = args.out / "covisitation.pkl"
    with covis_path.open("wb") as f:
        pickle.dump(covis, f)
    print(f"Saved co-visitation matrix to {covis_path}")

    print("Generating predictions for test sessions...")
    test = load_sessions(args.data, "test")
    preds: dict = {}
    for sid, g in test.groupby("session", sort=False):
        preds[sid] = predict_session(g, covis)

    pred_path = args.out / "baseline_predictions.json"
    with pred_path.open("w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f)
    print(f"Saved {len(preds):,} session predictions to {pred_path}")

    label_path = args.data / "test_labels.parquet"
    if label_path.exists():
        labels = pd.read_parquet(label_path)
        scores = evaluate(preds, labels)
        with (args.out / "baseline_metrics.json").open("w") as f:
            json.dump(scores, f, indent=2)
        print(f"Weighted Recall@20: {scores['weighted']:.4f}")
        print(f"  clicks={scores['clicks']:.4f}  carts={scores['carts']:.4f}  orders={scores['orders']:.4f}")
    else:
        print(f"(no labels at {label_path}, skipping evaluation)")

    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
