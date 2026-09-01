> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# OTTO Multi-Objective Session-Based Recommendation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch) ![Transformers](https://img.shields.io/badge/SASRec-Transformer-purple) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

Production-grade session-based recommendation system on the **OTTO e-commerce dataset** (12.9M sessions, 220M events, 1.85M items). Implements a co-visitation matrix baseline and a SASRec multi-task transformer optimised for three simultaneous objectives: clicks, add-to-carts, and orders.

---

## Architecture

```
User Session Events  [click, cart, order, ...]
         │
         ▼
┌────────────────────────────────────┐
│   Stage 1: Candidate Retrieval     │
│   Co-visitation Matrix             │
│   (click=1, cart=6, order=3 weights│
│    1-hour co-occurrence window)    │
└────────────┬───────────────────────┘
             │  Top-K candidates
             ▼
┌────────────────────────────────────┐
│   Stage 2: Re-ranking              │
│   SASRec Transformer               │
│   ┌────────────────────────┐       │
│   │ Item Embedding (d=64)  │       │
│   │ + Position Encoding    │       │
│   │ N × Self-Attention     │       │
│   │ + FFN blocks           │       │
│   └─────────┬──────────────┘       │
│             │                      │
│    ┌────────┼────────┐             │
│    ▼        ▼        ▼             │
│  Click    Cart    Order            │
│  Head     Head    Head             │
└────────────────────────────────────┘
             │
             ▼
    Weighted Recall@20
    0.10×clicks + 0.30×carts + 0.60×orders
```

---

## Key Features

- **Two-stage retrieval + re-ranking** — co-visitation candidate generation followed by SASRec transformer re-ranking
- **Multi-task learning** — single backbone, three prediction heads (click/cart/order) with weighted sampled-softmax loss
- **Causal self-attention** — no future event leakage within session; positional encodings capture recency
- **Weighted evaluation metric** — `Recall@20` weighted 0.10/0.30/0.60 (click/cart/order) per OTTO competition specification
- **Cold-start ready** — session-based; no persistent user history required, works for anonymous traffic
- **Heavy-tail handling** — top-100K items cover ~80% of orders; configurable item embedding table size

---

## Dataset

[OTTO Recommender Systems Dataset](https://www.kaggle.com/competitions/otto-recommender-system) (Kaggle, 2022).

```bash
kaggle competitions download -c otto-recommender-system
```

See `data/README.md` for full setup. Dataset acceptance required via Kaggle.

| Split | Sessions | Events |
|-------|----------|--------|
| Train | 12,899,779 | 220M+ |
| Test | 1,671,803 | - |
| Distinct items | 1,855,603 | - |
| Time window | 4 weeks real OTTO traffic | - |

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Co-visitation matrix candidate generator
│   └── model_advanced.py      # SASRec multi-task transformer
├── notebooks/
│   └── 01_EDA.ipynb           # Session length distributions, item frequency, event mix
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup (~4,000 words)
├── reports/
│   └── references.md          # 22+ verified academic references
├── deliverables/
│   └── presentation.html      # Self-contained HTML presentation
├── data/
│   └── README.md              # Dataset download and preprocessing instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/otto-session-recsys.git
cd otto-session-recsys
pip install -r requirements.txt

# Download dataset first (see data/README.md)

# Run co-visitation baseline
python src/model_baseline.py

# Run SASRec transformer (GPU recommended, ~6-12h on RTX 3090)
python src/model_advanced.py --data data/ --out deliverables/ --epochs 3
```

---

## Results

| Model | Recall@20 (clicks) | Recall@20 (carts) | Recall@20 (orders) | Weighted Score |
|-------|-------------------|-------------------|-------------------|----------------|
| Co-visitation baseline | TBD | TBD | TBD | TBD |
| SASRec multi-task | TBD | TBD | TBD | TBD |

Top public Kaggle solutions achieved ~0.60 weighted Recall@20. Co-visitation alone placed in the top 20%.

---

## Evaluation

Local validation uses the last 7 days of the train split as a held-out simulated test set (official test labels are private):

```python
score = 0.10 * Recall20(clicks) + 0.30 * Recall20(carts) + 0.60 * Recall20(orders)
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Transformer model | PyTorch |
| Candidate retrieval | NumPy / Polars (co-visitation) |
| Data loading | PyArrow / Parquet |
| Session datasets | PyTorch DataLoader |
| Evaluation | Custom Recall@20 |

---

## References

Key papers: SASRec (Kang & McAuley 2018), Transformer4Rec (de Souza Pereira Moreira et al. 2021), BERT4Rec (Sun et al. 2019). Full list in `reports/references.md`.

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Germany.

---

## License

MIT
