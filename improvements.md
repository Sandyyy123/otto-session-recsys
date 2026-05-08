# Project 17 OTTO Multi-Objective Recsys - Improvements (Role B)

**Reviewer role:** IMPROVER (Role B). Recommendations only, no file modifications.
**Date:** 2026-05-08
**Scaffold status:** Phase 1, code-only, not executed.

---

## Top recommendation

**Replace single-stage scoring in `model_advanced.py` with a two-stage retrieve-then-rerank pipeline where the co-visitation baseline is the retriever and SASRec is the reranker over its top 100-200 candidates.** This is the single highest-leverage change because (1) the manuscript already names this as the production deployment plan in section 4.6 but the scaffold does not implement it, (2) it removes the dot-product against 1.85M items at inference, which is the actual bottleneck on the OTTO test split (1.67M sessions x 1.85M items = an unworkable 3.1 trillion scores per epoch), and (3) every public top-10 OTTO leaderboard solution used a retrieve-then-rerank stack. Concrete next step: add a `--candidates-from` flag to `model_advanced.py` that loads `covisitation.pkl`, restricts the per-row score table to the union of the top-100 candidates from the three matrices for the session's last K events, and computes sampled-softmax loss against that restricted set. Expected lift: 0.04-0.08 on weighted Recall@20 over the single-stage SASRec, matching the gap between the public-leaderboard mid-pack and top-50.

---

## Weaknesses and proposed improvements

### 1. No two-stage retrieve-then-rerank wired up [HIGH]

**Weakness.** Sections 4.6 and 3.4 of the manuscript describe a two-stage architecture as the production target and as ablation #3, but neither `model_baseline.py` nor `model_advanced.py` produces a candidate file in the format the other consumes, and there is no glue script. Inference also dot-products against the full 1.85M-item embedding table, which will not finish on a 24 GB GPU within the 6-12 hour budget for 1.67M test sessions.

**Improvement.** Add a third script `src/model_rerank.py` that (a) loads the pickled co-visitation dictionary written by the baseline, (b) for each test session takes the union of top-100 candidates from the three matrices, (c) loads the SASRec checkpoint and scores only that restricted set, (d) returns the top-20 per task. Cite Covington 2016 explicitly in the script docstring. This also makes ablation #3 in the manuscript executable instead of speculative.

### 2. Random seeds intentionally unpinned, hurting reproducibility claims [HIGH]

**Weakness.** Section 2.5 of the manuscript states random seeds are not pinned by default ("makes batch-to-batch noise easier to read off the loss curves"). For a Liora portfolio piece this is the wrong default: the deliverable claims to be a "reproducible Phase 1 reference pipeline" (Introduction, last paragraph) but cannot be reproduced bit-exact. Reviewers and clients will expect deterministic results.

**Improvement.** Add `--seed` CLI flag (default 42) in both scripts that sets `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Document in section 2.5 that determinism is on by default and the unseeded mode is opt-in via `--seed -1`. This is a 10-line change with high reviewer-facing value.

### 3. Missing requirements.txt and Python version pin [HIGH]

**Weakness.** Project root has no `requirements.txt`, `pyproject.toml`, or `environment.yml`. The brief and manuscript mention pandas, numpy, PyTorch, parquet readers (pyarrow), Kaggle CLI, matplotlib (notebook), but none of these are pinned. A fresh evaluator cannot run the scaffold deterministically.

**Improvement.** Add `requirements.txt` at the project root with pinned versions: `torch>=2.2,<2.6`, `numpy>=1.26,<2.0`, `pandas>=2.1,<2.3`, `pyarrow>=15.0`, `polars>=0.20` (for the streaming alternative the data README mentions), `matplotlib>=3.8`, `kaggle>=1.6`. Add a one-line `python>=3.10` note in `data/README.md`. Also add a `.python-version` or equivalent. Estimated 15 minutes of work, eliminates the most common reviewer friction.

### 4. No OOV bucket despite manuscript flagging the gap [HIGH]

**Weakness.** Section 2.6 of the manuscript explicitly identifies the cold-item problem ("rare items get their own randomly initialised embedding which never receives a meaningful gradient") and defers the fix to Phase 2. But the long-tail item statistics in `notebooks/01_EDA.ipynb` cell 15 (top 100K covers ~80 percent of events) make this trivial to address now: cap the vocabulary at the top 200K-500K items, map all rarer aids to a single shared OOV id, and document the trade-off.

**Improvement.** Add `--vocab-cap` CLI flag (default 500000) to `model_advanced.py`. Build the cap inside `OTTOSessionDataset.__init__` from a `value_counts` over the train aid column; rare aids get reassigned to id 1 (reserve 0 for pad, 1 for OOV). The embedding table shrinks to (vocab_cap + 2, d_model), saving ~370 MB at d_model=64 and unblocking d_model=128 or 256 within the same memory envelope. Cite Wu et al. 2019 (SR-GNN) for the cold-item discussion.

### 5. Single CPU co-visitation build will not finish on 220M events [HIGH]

**Weakness.** `build_covisitation()` in `model_baseline.py` is a pure-Python double loop over each session's events. Section 2.5 of the manuscript itself flags this ("co-visitation construction is the most expensive operation in the scaffold"). Empirically, on the public OTTO Kaggle notebooks, naive Python co-visitation on 220M events takes 12+ hours single-threaded; this would block the Phase 2 execution window.

**Improvement.** Vectorize with pandas + numpy: per-session, build a pairwise (i, j) cross-join via `np.tril_indices` on the within-window event pairs, then groupby (src_aid, type, tgt_aid) and sum. For sessions longer than ~30 events use the chunked approach from Chris Deotte's public OTTO Kaggle notebook (kaggle.com/cdeotte). Alternative: rewrite the inner loop in numba. Expected speedup 30-100x. Add a `--n-workers` flag with a `multiprocessing.Pool` over session shards, merging counters at the end. Document the Deotte 2022 reference in the script header.

### 6. No statistical comparison between baseline and advanced model [MEDIUM]

**Weakness.** The Results section presents point estimates of weighted Recall@20 for baseline vs. advanced but no confidence intervals or significance test. With 1.67M test sessions, even a 0.001 absolute difference is significant, but a reviewer cannot tell from the scaffold whether the projected 0.59-0.62 advanced range is genuinely separated from the 0.55-0.58 baseline range.

**Improvement.** Add a `bootstrap_ci()` helper to a new `src/eval_utils.py`: 1000 bootstrap resamples over sessions, report 95% percentile CI for each Recall@20 metric and for the weighted score. Also add a paired bootstrap that compares baseline vs advanced session-by-session and reports the proportion of sessions where the advanced model strictly beats the baseline. Insert a results sub-table for these CIs in the manuscript section 3.

### 7. Multi-task loss weighting is hardcoded; no ablation on the weighting schedule [MEDIUM]

**Weakness.** Both scripts hardcode the (0.10, 0.30, 0.60) competition weights as the loss weights. The manuscript section 4.4 acknowledges this is a partial fix and that the orders gradient remains noisier. The natural ablation, varying the loss weights and tracking the trade-off curve, is missing from section 3.4.

**Improvement.** Add `--loss-weights` CLI flag (default "0.10,0.30,0.60") in `model_advanced.py`. Run a 3-point sweep ((0.33, 0.33, 0.33), (0.10, 0.30, 0.60), (0.05, 0.20, 0.75)) and add a Pareto-frontier plot to the manuscript section 3.4 ablation table showing per-task Recall@20 at each weighting. Cite Ma 2018a (MMoE) and add a sentence on how MMoE-gating would replace the static weights in Phase 2.

### 8. EDA notebook never visualises temporal leakage between train and test [MEDIUM]

**Weakness.** `notebooks/01_EDA.ipynb` cell 17 mentions the day-of-week / hour-of-day temporal structure but does not include a sanity-check plot of the train timestamp range vs the test timestamp range. The manuscript section 4.5 lists "evaluation leakage" as a known limitation but the EDA does not visualise the train/test temporal gap that is the ground for that claim.

**Improvement.** Add a cell after cell 18 that plots train.ts.min/max and test.ts.min/max as a horizontal bar chart, and a second cell that confirms zero session_id overlap between train and test. This is a one-paragraph addition that makes the leakage claim verifiable rather than asserted.

### 9. No fairness or popularity-bias audit [MEDIUM]

**Weakness.** With orders being 100x rarer than clicks, any sampled-softmax model will systematically under-recommend long-tail items in favour of the head. The manuscript section 4.4 alludes to this but the scaffold has no metric that tracks it. A serious recsys deliverable should report tail-recall and item-coverage in addition to weighted Recall@20.

**Improvement.** Add to `eval_utils.py` a `coverage()` function returning the fraction of distinct items appearing in any session's top-20, and a `tail_recall()` function that computes Recall@20 restricted to ground-truth items below the 80th popularity percentile. Add these as two new rows in the results tables. Cite Yi 2019 (sampling-bias correction) as the principled fix in Phase 2. This addresses a fairness dimension that Liora reviewers often probe.

### 10. Presentation HTML does not exercise the multi-objective story for a business audience [LOW]

**Weakness.** The presentation is a 222-line static HTML mirroring the manuscript sections. For an MLE portfolio piece pitched at clients, the deliverable should lead with the business framing (orders are revenue, weighted-Recall captures GMV) before the technical details. The current structure starts with dataset stats.

**Improvement.** Reorder the presentation: section 1 should be a 3-bullet business framing ("anonymous traffic, orders are 0.6 of the score, two-stage stack mirrors YouTube and Amazon"), section 2 the architecture diagram (co-vis as retriever, SASRec as reranker), section 3 the headline metric, only then the dataset and methods. This is presentation-only, no model change. Maps to section 4.8 of the manuscript ("why orders are the right north-star").

---

## Out-of-scope but worth noting (no priority assigned)

- The `data/README.md` cites an arXiv paper (2307.14906) without a DOI; it is a preprint, so the citation format is fine, but the references.md list does not include it. Adding it would close a cite-from-data-README to references.md gap. Source: Reiss-Mirzaei et al. 2023.
- BERT4Rec (Sun 2019) is referenced in the manuscript but no script implements it. The scaffold could add a `--objective masked` flag to `model_advanced.py` that switches from causal to masked-LM loss, exposing a second cheap ablation slice.
- No Dockerfile or Make target. The brief and manuscript both implicitly assume the evaluator has a working PyTorch + CUDA setup. A 10-line Dockerfile (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` base, copy src/, requirements.txt) would make the scaffold runnable in one command on a fresh Liora GPU instance.

---

## Compact summary

**Output file:** `/root/AI/liora_projects/17_otto_recsys/improvements.md`

**Top 3 findings:**
1. The two-stage retrieve-then-rerank architecture is described in the manuscript as the production target and listed as ablation #3, but no glue script exists. Highest-leverage change.
2. Random seeds intentionally unpinned and `requirements.txt` missing, which contradicts the "reproducible reference pipeline" claim.
3. Co-visitation build is single-threaded pure Python and will not finish on 220M events in the Phase 2 execution window; needs vectorisation or numba.

**Blockers:** None. All artefacts read cleanly.

**Status:** Role B (IMPROVER) complete.
