"""
OTTO Multi-Objective Session-Based Recommendation - Advanced.

SASRec-style transformer with multi-task heads for click / cart / order.

Architecture (Kang and McAuley 2018, with multi-task extensions in the
spirit of Transformers4Rec, de Souza Pereira Moreira et al. 2021):

    item_id      -->  embedding (d=64)
    +position    -->  embedding (d=64)
                 -->  N x [causal self-attention + FFN] blocks
                 -->  per-task prediction head (click / cart / order)

The output for the last position of each session is dot-producted
against the full item embedding table to score every item, then top-20
is taken per task. Loss is a weighted sampled-softmax sum.

Run:
    python src/model_advanced.py --data ../data --out ../deliverables --epochs 3

The script is NOT executed in scaffold phase (Phase 1). The user runs
it later in the main session, ideally on a GPU box (training takes
roughly 6-12 hours on a single RTX 5090 at default hyperparameters).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

EVENT_WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
EVENT2IDX = {"clicks": 0, "carts": 1, "orders": 2}
MAX_SEQ_LEN = 50
PAD_AID = 0  # reserve item id 0 for padding


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class OTTOSessionDataset(Dataset):
    """
    Yields fixed-length causal training samples from session sequences.

    For each session of length L we generate one sample where the model
    is asked to predict the (L)-th event from the first (L-1) events. We
    truncate to MAX_SEQ_LEN, left-pad, and emit the target event-type
    as a separate label so the loss can be weighted.
    """
    def __init__(self, df: pd.DataFrame, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        df = df.sort_values(["session", "ts"], kind="mergesort")
        self.sessions: list = []
        for _, g in df.groupby("session", sort=False):
            aids = g["aid"].to_numpy()
            types = g["type"].to_numpy()
            if len(aids) < 2:
                continue
            self.sessions.append((aids, types))

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int):
        aids, types = self.sessions[idx]
        seq = aids[:-1][-self.max_len:]
        target_aid = aids[-1]
        target_type = EVENT2IDX[types[-1]]
        pad = self.max_len - len(seq)
        if pad > 0:
            seq = np.concatenate([np.full(pad, PAD_AID, dtype=seq.dtype), seq])
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target_aid, dtype=torch.long),
            torch.tensor(target_type, dtype=torch.long),
        )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class SASRecMultiTask(nn.Module):
    def __init__(self, n_items: int, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.2, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=PAD_AID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.task_proj = nn.ModuleDict({
            "clicks": nn.Linear(d_model, d_model),
            "carts":  nn.Linear(d_model, d_model),
            "orders": nn.Linear(d_model, d_model),
        })
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, seq: torch.Tensor) -> dict:
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        key_padding_mask = (seq == PAD_AID)
        h = self.encoder(x, mask=self.causal_mask[:L, :L],
                         src_key_padding_mask=key_padding_mask)
        last = h[:, -1, :]
        return {tname: proj(last) for tname, proj in self.task_proj.items()}

    def score_all_items(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dot product hidden (B, d) against full item embedding table (n_items+1, d)."""
        return hidden @ self.item_emb.weight.T


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def sampled_softmax_loss(scores: torch.Tensor, target: torch.Tensor, n_neg: int = 200) -> torch.Tensor:
    """Cheap sampled-softmax: pick n_neg uniform negatives per batch row."""
    B, V = scores.shape
    pos = scores.gather(1, target.unsqueeze(1))
    neg_idx = torch.randint(1, V, (B, n_neg), device=scores.device)
    neg = scores.gather(1, neg_idx)
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, labels)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    n_batches = 0
    for seq, tgt_aid, tgt_type in loader:
        seq = seq.to(device); tgt_aid = tgt_aid.to(device); tgt_type = tgt_type.to(device)
        out = model(seq)
        loss = 0.0
        for tname, idx in EVENT2IDX.items():
            mask = (tgt_type == idx)
            if not mask.any():
                continue
            scores = model.score_all_items(out[tname][mask])
            l = sampled_softmax_loss(scores, tgt_aid[mask])
            loss = loss + EVENT_WEIGHTS[tname] * l
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss); n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def predict(model, loader, device, top_k: int = 20) -> dict:
    """Return {session_id: {clicks: [...], carts: [...], orders: [...]}}."""
    model.eval()
    preds: dict = {}
    for seq, sid in loader:
        seq = seq.to(device)
        out = model(seq)
        per_task = {}
        for tname in EVENT2IDX:
            scores = model.score_all_items(out[tname])
            top = scores.topk(top_k, dim=1).indices.cpu().numpy()
            per_task[tname] = top
        for i, s in enumerate(sid.tolist()):
            preds[s] = {t: per_task[t][i].tolist() for t in EVENT2IDX}
    return preds


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../deliverables"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_pq = args.data / "train.parquet"
    if not train_pq.exists():
        raise FileNotFoundError(f"Missing {train_pq}. See data/README.md for download instructions.")
    train_df = pd.read_parquet(train_pq)
    n_items = int(train_df["aid"].max()) + 1
    print(f"Items: {n_items:,}, events: {len(train_df):,}, sessions: {train_df['session'].nunique():,}")

    train_ds = OTTOSessionDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device == "cuda"))

    model = SASRecMultiTask(n_items=n_items, d_model=args.d_model,
                            n_layers=args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"Training for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, opt, device)
        print(f"  epoch {ep}: loss={loss:.4f}  ({time.time() - t0:.1f}s)")

    ckpt_path = args.out / "sasrec_multitask.pt"
    torch.save({"model_state": model.state_dict(),
                "n_items": n_items,
                "config": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    metrics_path = args.out / "advanced_metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"weighted_recall_at_20": None,
                   "clicks": None, "carts": None, "orders": None,
                   "note": "fill after running predict() on official test split"},
                  f, indent=2)
    print(f"Wrote metrics placeholder: {metrics_path}")


if __name__ == "__main__":
    main()
