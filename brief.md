# Project 17 - OTTO Multi-Objective Session-Based Recommendation

**Track:** Machine Learning Engineer / Recommender Systems
**Difficulty:** 8/10
**Domain:** E-commerce session-based recommendation
**Phase:** Phase 1 - scaffolded code-only

## Goal

Predict the next set of items a user will click, add to cart, and order within an active browsing session on the OTTO e-commerce platform (Germany's largest online retailer, second only to Amazon in DACH). The task is multi-objective: a single ranked list of 20 recommended items per session must be optimised against three target events (clicks, carts, orders) with weighted Recall@20, where orders carry the highest weight.

## Why this matters

Session-based recommendation is the dominant production pattern at large e-commerce platforms because most user traffic is anonymous or only weakly identified, so collaborative-filtering models that rely on persistent user profiles cannot be applied. The OTTO challenge captures the realistic constraints: cold-start sessions, short sequences (median 3-5 events), heavy-tailed item catalog (1.85M items), and three downstream business metrics that conflict (clicks are abundant and noisy, orders are sparse and decisive).

This project pairs with project #13 (H&M personalised recommendations) as a deliberate twin: H&M is collaborative-filtering on user-level history, OTTO is sequence modelling on within-session events. Together they cover the two production paradigms at the core of every modern recsys stack.

## Dataset

OTTO Recommender Systems Dataset (Kaggle, 2022 challenge).

| Item | Value |
|------|-------|
| Sessions | 12,899,779 train sessions, 1,671,803 test sessions |
| Events | 220+ million |
| Distinct items (aids) | 1,855,603 |
| Event types | click, cart, order |
| Format | parquet shards (jsonl alternative) |
| Total size on disk | ~7 GB |
| Time window | 4 weeks of real OTTO traffic (anonymised) |

The dataset is gated by Kaggle competition acceptance and exceeds the 2 GB download threshold in the agent rules, so this scaffold documents the download command in `data/README.md` and does not pull data.

## Modeling plan

**Baseline (`src/model_baseline.py`).** Co-visitation matrix. For every pair of items (a, b) seen in the same session within a 1-hour window, increment a weighted counter where the weight depends on event type (click=1, cart=6, order=3, following the public OTTO competition leaderboard convention). At inference, for each session take the last K events, look up top-N co-visited items per event, aggregate scores, and return top-20. This is a strong non-neural baseline; the public Kaggle leaderboard had co-visitation solutions inside the top 20 percent.

**Advanced (`src/model_advanced.py`).** SASRec / Transformer4Rec. A causal self-attention encoder over the within-session item sequence with three multi-task heads (click, cart, order) sharing a backbone. Item embeddings are learned end-to-end, positional encodings capture recency. Loss is a weighted sum of per-task sampled-softmax. This mirrors the Transformer4Rec NVIDIA Merlin recipe (de Souza Pereira Moreira et al., 2021) and the original SASRec architecture (Kang and McAuley, 2018).

## Evaluation

Weighted Recall@20 per the OTTO competition rule:

```
score = 0.10 * Recall@20(clicks) + 0.30 * Recall@20(carts) + 0.60 * Recall@20(orders)
```

Local validation uses the last 7 days of the train split as a held-out simulated test set, since the official test labels are private.

## Deliverables

- `data/README.md` - Kaggle CLI command and setup notes
- `notebooks/01_EDA.ipynb` - skeleton EDA, not executed
- `reports/references.md` - 22+ verified academic references
- `src/model_baseline.py` - co-visitation candidate generator
- `src/model_advanced.py` - SASRec multi-task transformer
- `manuscripts/manuscript.md` - IMRaD writeup, 4000+ words, results placeholders
- `deliverables/presentation.html` - 9-section static HTML
- `checkpoint.json` - status manifest

## Open questions for Phase 2

1. Single-stage ranking vs two-stage retrieve-then-rerank (YouTube pattern, Covington 2016). Plan to build candidate generation first and reranker second.
2. How aggressively to filter the long tail. 1.85M items but the top 100K cover ~80 percent of orders. Embedding table memory is the binding constraint.
3. Evaluation leakage: making sure no future events from the same session bleed into the training context.
