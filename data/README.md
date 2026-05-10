# OTTO Dataset - download instructions

The OTTO Recommender Systems dataset is gated by Kaggle competition acceptance and is approximately 7 GB on disk, well above the 2 GB implementation-only threshold. This folder is empty by design. Download is a manual step.

## Step 1. Accept the competition rules

Visit https://www.kaggle.com/competitions/otto-recommender-system in a logged-in browser and click "Join Competition". Without this acceptance the Kaggle CLI returns 403.

## Step 2. Kaggle CLI auth

The Kaggle credential file should already exist on this machine at:

```
~/.kaggle/kaggle.json
~/.kaggle/access_token
```

If not, download `kaggle.json` from https://www.kaggle.com/settings/account (Create New API Token), drop it into `~/.kaggle/`, and `chmod 600 ~/.kaggle/kaggle.json`.

## Step 3. Download

```bash
cd data/
kaggle competitions download -c otto-recommender-system
unzip otto-recommender-system.zip
```

Expected files after unzip:

| File | Size | Rows |
|------|------|------|
| `train.jsonl` | ~12 GB jsonl, ~6 GB parquet | 12,899,779 sessions |
| `test.jsonl` | ~1.5 GB | 1,671,803 sessions |
| `sample_submission.csv` | ~80 MB | sessions x 3 event types |

For faster iteration, the community-curated parquet shards from Radek Osmulski are recommended over the raw jsonl:

```bash
kaggle datasets download -d radek1/otto-full-optimized-memory-footprint
```

This gives `train.parquet`, `test.parquet`, `train_labels.parquet`, totaling about 2.5 GB.

## Step 4. Schema

Each row in `train.jsonl` is a session:

```json
{
  "session": 12345,
  "events": [
    {"aid": 1234567, "ts": 1659304800021, "type": "clicks"},
    {"aid": 9876543, "ts": 1659304850114, "type": "carts"},
    {"aid": 9876543, "ts": 1659305001899, "type": "orders"}
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `session` | int64 | session id, anonymised |
| `aid` | int32 | item id, 0 to 1,855,602 |
| `ts` | int64 | unix timestamp in milliseconds |
| `type` | string | one of `clicks`, `carts`, `orders` |

## Step 5. Memory footprint

The full train set has ~220M events. On a 32 GB RAM box load only the parquet shards and use polars/duckdb streaming. Avoid `pd.read_json(lines=True)` on the raw file (peaks at 90+ GB RAM).

## Source citations

- OTTO Group, Kaggle Competition (2022): https://www.kaggle.com/competitions/otto-recommender-system
- Reiss-Mirzaei M, Schifferer B, Souza Pereira Moreira G, Karatzoglou A, et al. OTTO Recommender Systems Dataset: a real-world e-commerce dataset for session-based recommender systems. arXiv:2307.14906, 2023.
- Osmulski R. otto-full-optimized-memory-footprint. https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint
