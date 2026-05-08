# Validation Report - Project 17 OTTO Multi-Objective Recsys

## Summary

**Overall: PASS**

All 11 validator tasks pass without warnings or failures. The notebook parses as valid JSON, both Python scripts compile cleanly, the manuscript is 4,218 words (inside the 4,000-5,000 target), the presentation HTML is fully self-contained with zero external resources, every IMRaD section is present, every method named in the Methods section maps to a concrete construct in `model_baseline.py` or `model_advanced.py`, all 19 inline citations resolve to entries in `reports/references.md`, 5 randomly sampled references returned CrossRef HTTP 200 with perfect title-token overlap, no em-dashes appear anywhere in the seven scaffold artefacts, no AI-tell phrases were found anywhere in the project tree, and `checkpoint.json` contains all four required keys (project_number, title, methodology, status). This scaffold is ready for Phase 2 main-session execution.

---

## Findings

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as valid JSON via `json.load`.

### 2. Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly under `ast.parse` (no SyntaxError).
- [PASS] `src/model_advanced.py` parses cleanly under `ast.parse` (no SyntaxError).

### 3. Manuscript word count
- [PASS] `manuscripts/manuscript.md` = 4,218 words (target 4,000-5,000, inside band).

### 4. Self-contained HTML
- [PASS] `deliverables/presentation.html` has 0 external `href="http..."` or `src="http..."` references. Inline-only.

### 5. IMRaD completeness
- [PASS] All required sections present: Title, Abstract, Introduction, Methods (with subsections 2.1-2.6), Results (with subsections 3.1-3.4), Discussion (with subsections 4.1-4.9), Conclusion, References.

### 6. Method drift (Methods section vs src)
- [PASS] co-visitation matrix -> `build_covisitation` in `model_baseline.py`.
- [PASS] 1-hour window -> `WINDOW_MS = 60 * 60 * 1000`.
- [PASS] click/cart/order weights 1.0/6.0/3.0 -> `COVIS_WEIGHTS`.
- [PASS] top-40 prune -> `top_n` arg with default 40.
- [PASS] recency decay 1/(1+rank) -> `1.0 / (1 + rank)` in `predict_session`.
- [PASS] last K=30 events -> `TOP_K_RECENT_EVENTS = 30`.
- [PASS] SASRec multi-task transformer -> `class SASRecMultiTask`.
- [PASS] 64-dim embedding, 2 heads, 2 layers -> `d_model=64, n_heads=2, n_layers=2`.
- [PASS] GELU + pre-LayerNorm -> `activation="gelu", norm_first=True`.
- [PASS] causal mask -> `register_buffer("causal_mask", ...)`.
- [PASS] three task heads -> `task_proj` ModuleDict (clicks/carts/orders).
- [PASS] sampled softmax with 200 negatives -> `sampled_softmax_loss(..., n_neg=200)`.
- [PASS] 0.10/0.30/0.60 task weights -> `EVENT_WEIGHTS`.
- [PASS] max seq 50 left-pad -> `MAX_SEQ_LEN=50` and left-padding in `__getitem__`.
- [PASS] AdamW lr=1e-3 wd=1e-5 -> `torch.optim.AdamW(... lr=args.lr=1e-3, weight_decay=1e-5)`.
- [PASS] batch size 512, 3 epochs -> `--batch-size 512 --epochs 3`.
- [PASS] Recall@20 weighted scoring -> `recall_at_k` and `evaluate` with `EVENT_WEIGHTS`.

### 7. Citation drift (inline vs references.md)
- [PASS] 19 unique inline citations extracted from manuscript body. All 19 resolve to a matching author+year in `reports/references.md`. Zero orphans.
- Citations checked: Afsar 2022, Cheng 2016, Covington 2016, Guo 2017, He 2017, Kang 2018, Koren 2009, Ma 2018a, Ma 2018b, Sarwar 2001, Souza 2021, Sun 2019, Tan 2016, Tang 2018, Wang 2019, Wu 2019, Yi 2019, Zhang 2019, Zhou 2018.

### 8. Re-verify 5 random references via CrossRef live
Random seed 17, 5 entries sampled from 22. All returned HTTP 200 with 1.00 title-token overlap.
- [PASS] DOI 10.1145/3219819.3219823 -> "Deep Interest Network for Click-Through Rate Prediction" (2018). HTTP 200.
- [PASS] DOI 10.1145/3298689.3346996 -> "Sampling-bias-corrected neural modeling for large corpus item recommendations" (2019). HTTP 200.
- [PASS] DOI 10.24963/ijcai.2019/600 -> "Feature-level Deeper Self-Attention Network for Sequential Recommendation" (2019). HTTP 200.
- [PASS] DOI 10.1145/2959100.2959190 -> "Deep Neural Networks for YouTube Recommendations" (2016). HTTP 200.
- [PASS] DOI 10.1145/3159652.3159656 -> "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" (2018). HTTP 200.

### 9. Em-dash scan
- [PASS] Total em-dash count across brief.md, notebook, references.md, both src files, manuscript, presentation HTML: **0**.

### 10. AI-tell scan
- [PASS] Recursive grep for `verified by N agents`, `AI-verified`, `cross-checked by Claude` across the entire project tree returned 0 hits.

### 11. Checkpoint schema
- [PASS] `checkpoint.json` parses cleanly.
- [PASS] Required fields present: `project_number`, `title`, `methodology`, `status`. Also includes `phase`, `needs_main_session_execution`, `blockers` (extra, not required).

---

## Notes

- This is a Phase 1 scaffold-only project; per QA rules, missing executed model artefacts in `deliverables/` are NOT applicable here (project #17 is outside the #1-#8 executed band). Only `presentation.html` is expected in `deliverables/`, and it is present.
- Results-section tables in the manuscript correctly use `<TBD after model run>` placeholders, in line with the no-fabrication rule.
- The references file states verification was done live on 2026-05-08; the random 5-of-22 spot-check on 2026-05-08 reproduces the reported HTTP 200 + matching titles, so the reference list is trustworthy.

Role A (Validator) complete.
