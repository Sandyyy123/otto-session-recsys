# Multi-Objective Session-Based Recommendation on the OTTO E-commerce Dataset

**Authors:** Sandeep Grover, Independent Research


---

## Abstract

Session-based recommendation is the dominant production paradigm at large e-commerce platforms because most browsing traffic is anonymous or only weakly identified, which makes the classical user-item matrix factorisation approach inapplicable. The OTTO 2022 Kaggle competition released a four-week trace of real traffic from Germany's second-largest online retailer: 12.9 million training sessions, 220 million events, 1.85 million distinct items, and three downstream business metrics that conflict with each other (clicks, add-to-carts, orders). The competition asks for a single ranked list of 20 items per active session that maximises a weighted Recall@20, with orders weighted six times higher than clicks. We implementation two reference systems for this task. The baseline is a co-visitation matrix that counts item-item co-occurrences within a one-hour window in the same session, weighted by event type, and aggregates the top neighbours of the most recent K events at inference time. The advanced model is a SASRec-style causal transformer with three multi-task heads, trained end-to-end with sampled-softmax loss using the competition's own event-type weights. The advanced model is expected to lift weighted Recall@20 by 15-25 percent over the baseline based on published numbers from comparable competitions, with the largest gains concentrated on the orders task. We discuss the trade-offs between single-stage ranking and two-stage retrieve-then-rerank, the memory cost of a 1.85 million item embedding table, and the leakage risks introduced by within-session label peeking.

**Keywords:** session-based recommendation, multi-objective learning, transformer, e-commerce, candidate generation.

---

## 1. Introduction

Recommender systems have moved from research curiosities to revenue-critical infrastructure in less than two decades. The earliest production systems used user-based collaborative filtering [Sarwar 2001] and matrix factorisation [Koren 2009] to predict missing entries in a sparse user-item rating matrix. Both methods assume the existence of a stable user identifier whose history accumulates over time. That assumption fails for a large fraction of present-day e-commerce traffic. On the OTTO platform, the median session is between three and five events long, most sessions belong to anonymous or first-time visitors, and even logged-in users browse from multiple devices without a unified identity. Predicting the next item in such a session is therefore a sequence-modelling problem that resembles language modelling more than collaborative filtering.

The first wave of session-based recommenders adapted recurrent neural networks. Hidasi et al.'s GRU4Rec, followed by Tan et al.'s improved variants [Tan 2016], showed that a gated recurrent network could comfortably beat item-based collaborative filtering on session-level next-click prediction. Tang and Wang's Caser introduced convolutional sequence embeddings [Tang 2018]. The decisive shift came with the application of self-attention. SASRec [Kang 2018] dropped the recurrent backbone entirely in favour of a stack of causal transformer blocks, making the model trivially parallel and lifting accuracy on every public benchmark at the time. BERT4Rec [Sun 2019] extended this to a masked-language-modelling objective; SR-GNN [Wu 2019] folded the same insight into a graph-neural-network frame; FDSA [Zhang 2019] explored deeper feature-level attention. NVIDIA Merlin's Transformers4Rec [Souza 2021] then made these architectures painless to deploy at industrial scale by integrating them with the rest of the data pipeline.

Production recommenders also have to worry about more than the next click. The downstream metric of interest in e-commerce is revenue, and the user actions that contribute most directly to revenue (orders) are several orders of magnitude rarer than the actions that are easiest to predict (clicks). This tension is the motivation for multi-task and multi-objective learning. Cheng et al.'s Wide and Deep architecture [Cheng 2016] composed memorisation and generalisation. Covington et al.'s YouTube system [Covington 2016] split the problem into a candidate-generation stage and a reranker stage, where the reranker optimises a different objective than the retriever. ESMM [Ma 2018b] shared a backbone between click and conversion heads to address the sample-selection bias of post-click conversion estimation. MMoE [Ma 2018a] introduced gating on top of mixture-of-experts to let unrelated tasks share parameters without interfering. Yi et al. [Yi 2019] formalised sampling-bias correction for two-tower retrieval.

The OTTO competition combined all three of these threads in a single benchmark. Sequences are short, sessions are anonymous, items are millions, and the scoring function explicitly weights three event types. This project implementations a reproducible v1.0 deliverable for that benchmark: a strong non-neural baseline that captures the competition's leaderboard floor, an advanced transformer model with multi-task heads, an end-to-end EDA notebook, and a manuscript-form report. Numbers in the Results section below are placeholders. Once the user executes the two scripts on a downloaded copy of the dataset, the placeholders will be replaced with the measured weighted Recall@20.

The contribution of this implementation is not a novel algorithm. It is a reproducible reference pipeline that pairs cleanly with project 13 (H&M personalised recommendations). H&M is collaborative filtering on the persistent user history; OTTO is sequence modelling on the within-session events. Together they cover the two production paradigms at the core of every modern recsys stack [Zhang 2019].

## 2. Methods

### 2.1 Dataset

The OTTO Recommender Systems dataset was released for the 2022 Kaggle competition of the same name. It contains four weeks of anonymised browsing and purchase events from the OTTO Group's German e-commerce site. Train and test splits are temporally disjoint: the model has access to the first 27 days and is asked to predict the 28th. Each row in the raw jsonl is a session, identified by an integer `session` field, with a list of `events`. Each event has an `aid` (anonymised item identifier in the range 0 to 1,855,602), a millisecond unix `ts`, and a `type` drawn from `{clicks, carts, orders}`. The official scoring rule is

> `score = 0.10 * Recall@20(clicks) + 0.30 * Recall@20(carts) + 0.60 * Recall@20(orders)`

per session and event type, where Recall@20 is the fraction of ground-truth items from the held-out window that appear in the top 20 predicted items. Submissions are required to provide a separate ranked list for each of the three event types per session, so the prediction problem is genuinely multi-output.

The full dataset is approximately 7 GB on disk in raw jsonl form. The community-curated parquet shards from Radek Osmulski compress this to 2.5 GB and load in minutes rather than hours, so the parquet path is preferred for both scripts. Download instructions and the schema are documented in `data/README.md`. Because the dataset exceeds the 2 GB implementation-only threshold and is gated by Kaggle competition acceptance, this Initial implementation does not download data.

### 2.2 Baseline: co-visitation matrix

The baseline (`src/model_baseline.py`) is a non-neural co-visitation candidate generator. For each ordered pair of events (a, b) inside the same session whose timestamps differ by no more than one hour, we increment three sparse counters keyed on the source aid and indexed by the target aid: a clicks counter, a carts counter, and an orders counter. Increments are weighted by the type of the target event, with a fixed schedule of 1.0 / 6.0 / 3.0 for clicks / carts / orders that follows the Kaggle leaderboard convention. After all training sessions are processed, each source aid is pruned to its top 40 most co-visited targets per matrix to keep memory bounded.

At inference, given a session, we take the K most recent events (K = 30), look up the top 40 co-visited items per matrix per source event with a recency decay 1 / (1 + rank), aggregate scores by item, and return the top 20. The same procedure is repeated for each of clicks, carts, orders. The model has no learned parameters and runs comfortably on a single CPU.

The baseline is non-trivial. The 2022 OTTO public leaderboard contained co-visitation submissions inside the top 20 percent. It also serves as a candidate-generation oracle for the advanced model: any item that the baseline does not surface as a candidate is unreachable by a downstream reranker that takes the baseline as input.

### 2.3 Advanced: SASRec multi-task transformer

The advanced model (`src/model_advanced.py`) is a SASRec-style causal transformer [Kang 2018] with three task-specific projection heads that share a backbone. Item ids are mapped to a 64-dimensional embedding through a learned table; positions inside the session are mapped to a separate 64-dimensional positional embedding. The two are summed, dropout is applied, and the result is passed through two transformer encoder blocks with two attention heads each, GELU activations, and pre-LayerNorm. A causal mask prevents each position from attending to its successors, so the prediction at the last position depends only on the history. The output of the last position is projected three times by separate per-task linear layers, one per event type.

Predictions over the 1.85 million item catalog are produced by dot-producting each task head's output against the full item embedding table. This is the SASRec convention and avoids learning a separate output projection. To keep training tractable we use a sampled softmax loss with 200 uniform random negatives per row, weighted by the same 0.10 / 0.30 / 0.60 schedule that the competition uses for evaluation. Training samples are constructed by truncating each session to the most recent 50 events, left-padding with item id 0, and asking the model to predict the (L)-th event from the first (L-1).

Optimisation uses AdamW with learning rate 1e-3 and weight decay 1e-5, batch size 512, three epochs, on a single GPU. Default hyperparameters are conservative: a 64-dimensional embedding table for 1.85 million items occupies roughly 460 MB of GPU memory in float32, which fits on a 24 GB device with room to spare. Larger embedding sizes and deeper stacks are deferred to v2.0.

### 2.4 Evaluation protocol

We use a temporal hold-out split. The last 7 days of the public train set become a simulated test set; the model is fit on the first 21 days. This mirrors the public-private leaderboard split that the competition used and avoids the leakage that would arise from a random split. For each session in the simulated test set, the ground-truth labels are the items that appeared as a click, cart, or order during the held-out window in that same session, and Recall@20 is computed per event type. The weighted aggregate uses the official competition coefficients.

We also report unweighted per-task Recall@20 to expose how the multi-task weighting redistributes effort across heads. A model that overfits clicks at the cost of orders should show a high clicks-Recall@20 but a low orders-Recall@20 and a comparatively poor weighted score.

### 2.5 Implementation details

Both scripts are written against numpy, pandas, and PyTorch only, with no dependence on a specialised recsys framework. The deliberate intention is that an evaluator with a fresh Python environment can run them end-to-end in under an hour of setup time. Co-visitation construction in the baseline is the most expensive operation in the implementation, but it is embarrassingly parallel: each session contributes independent counter increments. A future optimisation would shard sessions across workers and merge counters at the end, which the current implementation does not do.

The transformer model uses PyTorch's built-in `TransformerEncoderLayer` with `batch_first=True` and `norm_first=True`. Pre-LayerNorm is preferred to post-LayerNorm because it stabilises training at higher learning rates and avoids the warmup schedule that the original Transformer architecture required. The causal mask is precomputed once at model construction time and registered as a non-persistent buffer so it lives on the same device as the parameters without polluting the saved state dictionary.

Random seeds are not pinned by default. The user can pin them by adding `torch.manual_seed(0)` at the top of `model_advanced.py` if exact reproducibility is required for an ablation. We chose the unseeded default because it makes batch-to-batch noise easier to read off the loss curves, which is more informative during the initial training runs than perfect reproducibility.

### 2.6 Data engineering risks

Three risks were considered before writing the scripts. First, the raw jsonl is large enough to peak above 80 GB of resident memory if naively loaded with `pd.read_json(lines=True)`. The mitigation in `load_sessions` is to prefer the parquet shards from the community-curated mirror and to stream the jsonl line by line as a fallback. Second, the timestamps in the dataset are millisecond unix integers, which are easy to misread as seconds and produce a 1000x off-by-one factor in the co-visitation window. The constant `WINDOW_MS = 60 * 60 * 1000` is named explicitly to make this explicit. Third, the test set contains sessions whose items have never appeared in train. The baseline handles this gracefully (the lookup simply returns the empty list), but the advanced model has to assign these items the OOV embedding. The current implementation does not include an OOV bucket; rare items get their own randomly initialised embedding which never receives a meaningful gradient. Mitigating this through an OOV mapping is a v2.0 task.

## 3. Results

Numbers in this section are placeholders. They will be replaced with measured values after `src/model_baseline.py` and `src/model_advanced.py` are executed in the main session.

### 3.1 Dataset profile

A full EDA is in `notebooks/01_EDA.ipynb`. Headline statistics expected on the full train set:

| Quantity | Value |
|----------|-------|
| Sessions | <TBD after model run> |
| Events | <TBD after model run> |
| Distinct items | <TBD after model run> |
| Median session length | <TBD after model run> |
| Click share | <TBD after model run> |
| Cart share | <TBD after model run> |
| Order share | <TBD after model run> |

### 3.2 Baseline performance

| Metric | Value |
|--------|-------|
| Recall@20 clicks | <TBD after model run> |
| Recall@20 carts | <TBD after model run> |
| Recall@20 orders | <TBD after model run> |
| Weighted Recall@20 | <TBD after model run> |

The baseline is expected to land between 0.55 and 0.58 weighted Recall@20 based on community reports of co-visitation submissions on the public leaderboard.

### 3.3 Advanced model performance

| Metric | Value |
|--------|-------|
| Recall@20 clicks | <TBD after model run> |
| Recall@20 carts | <TBD after model run> |
| Recall@20 orders | <TBD after model run> |
| Weighted Recall@20 | <TBD after model run> |

The advanced model is expected to land in the 0.59 to 0.62 weighted Recall@20 range, lifted by 0.03 to 0.05 over the baseline. The lift is expected to be largest on the orders task because the multi-task head explicitly trades off click density for order signal.

### 3.4 Ablations

We anticipate three ablation slices, all `<TBD after model run>`:

1. Single-task SASRec on orders only versus multi-task SASRec, to quantify the value of the shared backbone.
2. SASRec without positional embeddings, to confirm that recency is being absorbed by positional rather than item-co-occurrence signal.
3. Co-visitation as candidate generator with the SASRec model used as a reranker over its top 100 candidates, to test the two-stage YouTube-style architecture [Covington 2016].

## 4. Discussion

### 4.1 Why the baseline is competitive

Co-visitation captures the strongest signal in the dataset: items that are bought together by the same user in the same session are likely to be the next bought together. This is also the signal that two-decade-old item-based collaborative filtering captured [Sarwar 2001]. The advance over Sarwar is small in spirit but large in practice: weighting the counters by the target event type rather than the source event type lets the same matrix serve all three Recall@20 prediction tasks.

The baseline also has the production virtue of being entirely interpretable. Each recommendation is justified by a direct path through the co-visitation graph: "we recommended item B because you just clicked item A, and item B was added to a cart in the same session as item A 1,243 times in the training window." Neural models lose this property unless they are augmented with attention-based explanations [Wang 2019].

### 4.2 Why the advanced model is expected to win

A transformer over the session sequence gives three things the co-visitation baseline does not. First, item embeddings are end-to-end optimised for the prediction loss rather than fixed at the row of a count matrix. Second, position embeddings let the model reason about recency in a continuous way; co-visitation reduces recency to a hand-crafted linear decay. Third, the multi-task heads share a backbone, so signal that helps predict orders also helps predict carts and vice versa. He et al. [He 2017] showed the same backbone-sharing principle for neural collaborative filtering; Zhou et al. [Zhou 2018] extended it to attention over the user's history for click-through-rate prediction; Guo et al. [Guo 2017] combined factorisation-machine memorisation with deep generalisation. SASRec is the cleanest of these in code and runs fast.

### 4.3 Memory and compute trade-offs

The 1.85 million item catalog is the binding memory constraint at training time. A 64-dimensional float32 embedding occupies 256 bytes per item, so the full table is roughly 460 MB. With the default batch size of 512 and a max sequence length of 50, the activation memory for the transformer stack is small. Inference scoring over the full catalog is the more expensive operation: a single dot product against 1.85 million item vectors is the bottleneck, not the transformer pass. Approximate nearest-neighbour libraries such as FAISS or ScaNN reduce this by an order of magnitude in production, at the cost of a small recall hit. We defer ANN integration to v2.0.

### 4.4 Multi-objective trade-offs

The clicks task has roughly two orders of magnitude more training examples than the orders task. A naive shared-loss model therefore over-fits the clicks head and under-fits the orders head. The 0.10 / 0.30 / 0.60 loss weighting partially compensates, but the gradient noise on the orders head remains larger because the per-batch sample count is smaller. The principled fix is per-task gradient surgery (gradient projection or PCGrad-style routing). MMoE [Ma 2018a] is the most popular industrial answer: separate expert subnetworks plus a learned gating that decides which expert to use for which task. ESMM [Ma 2018b] addresses a different problem (sample-selection bias from clicks-only training data) but uses a similar architectural pattern. Both are deferred to v2.0.

### 4.5 Limitations

This implementation is intentionally narrow. It does not include: a reranker over the candidate generator, time-bucketed negative sampling, BPR or pairwise margin losses, two-tower retrieval [Yi 2019], graph approaches such as SR-GNN [Wu 2019] or KGAT [Wang 2019], reinforcement learning [Afsar 2022], session-level sequential pretraining in the BERT4Rec [Sun 2019] mould, or a comparison against the specialised feature-level attention of FDSA [Zhang 2019]. Each of these is a sensible v2.0 direction, and the references support that work.

The evaluation protocol also has a known limitation. The competition test labels are private; our simulated hold-out is the last 7 days of the train set. A model that overfits to the temporal structure of the train period (specific weekday patterns, specific promotional events) will look better on the hold-out than on the official leaderboard. This caveat applies to every published OTTO solution and is not specific to this implementation.

### 4.6 Production deployment notes

If the advanced model is ever deployed live, the architectural choice would be a two-stage retrieve-then-rerank stack [Covington 2016]. The retriever is the co-visitation baseline (or a two-tower neural retriever with sampled-softmax negatives [Yi 2019]). The reranker is the SASRec multi-task model scoring the top 100 candidates from the retriever. This buys two things: tractable inference (the dot product against the full catalog is replaced by a 100-item dot product) and operational separation between the high-recall stage and the high-precision stage. Cheng et al.'s Wide and Deep [Cheng 2016] and Zhou et al.'s DIN [Zhou 2018] are also obvious upgrades to the reranker.

### 4.7 Comparison to project 13 (H&M)

Project 13 attacks the H&M Personalized Fashion Recommendations Kaggle competition with a collaborative-filtering approach. Users have a persistent identity, the training history runs across years rather than minutes, and the evaluation metric is Mean Average Precision at 12 over a 7-day held-out window of customer-item transactions. The two projects are deliberately complementary. H&M tests how well a model can rank items for a known user given a long history; OTTO tests how well a model can rank items for an anonymous session given a short context. The architectures that win the two competitions are different in kind. H&M leaderboard solutions are dominated by tabular gradient-boosted ranking with rich hand-engineered features (item recency, customer buying frequency, color-and-size affinity). OTTO leaderboard solutions are dominated by sequence transformers and co-visitation candidates. Together the two implementations expose the modelling vocabulary that any production recsys engineer needs.

### 4.8 Why orders are the right north-star metric

The multi-objective weighting is not arbitrary. Orders are the only event type that closes the loop between recommendation and revenue. A click that does not lead to a cart adds zero to the merchant. A cart that does not lead to an order is comparatively cheap to acquire (the user already showed strong intent). Orders capture the last-mile commitment that the merchant actually wants to maximise. In OTTO's own production stack the equivalent metric is gross merchandise value attributable to a recommendation impression. Recall@20 on orders is a tractable proxy for that: any item that the model fails to surface in the top 20 cannot become an attributed order, so increasing Recall@20 caps the upside of the rest of the personalisation funnel. The 0.60 weight on orders in the competition score is a direct encoding of this, and any model that outperforms on the weighted metric is implicitly outperforming on the metric the merchant cares about.

### 4.9 Negative sampling alternatives

Sampled softmax with uniform negatives is the baseline negative-sampling strategy for recommendation models at scale. It is fast and unbiased only in the limit of infinite negatives. With finite negatives it systematically under-corrects for popular items, which receive too few negative gradients relative to their training frequency. Yi et al. [Yi 2019] introduced importance-weighted sampling that corrects for this bias by reweighting each negative by its inverse log frequency. Their approach lifts retrieval recall on YouTube by a measurable margin and is the recommended v2.0 upgrade for the advanced model. Two-tower retrieval architectures benefit even more from this correction because they have to score the full catalog at inference, where popularity bias compounds.

A second negative-sampling axis is hardness. Uniform negatives are easy: most random items are obviously irrelevant, so the gradient signal is small once training has progressed past the first few thousand steps. Hard-negative mining picks negatives that the current model already ranks highly but that should be ranked lower. The trade-off is that hard negatives can amplify training instability if the model is still in an early stage. A practical pattern is to start with uniform negatives for the first epoch and gradually mix in hard negatives sampled from the model's own top predictions in later epochs. This is also deferred to v2.0.

## 5. Conclusion

We implementationed a reproducible v1.0 deliverable for the OTTO multi-objective session-based recommendation task. The baseline is a co-visitation matrix that captures the strongest signal in the dataset and serves as a candidate-generation oracle for downstream rerankers. The advanced model is a SASRec-style causal transformer with three multi-task heads, trained with sampled-softmax loss and the competition's own event-type weights. Both scripts are runnable, neither has been executed yet, and the manuscript is ready to absorb measured numbers once they exist. v2.0 directions are well-defined: a two-stage retrieve-then-rerank architecture, MMoE-style multi-task routing, ANN serving with FAISS, BERT4Rec masked-language-modelling pretraining, and graph approaches such as SR-GNN or KGAT.

The system pairs cleanly with project 13 (H&M personalised recommendations). Together they cover the two production paradigms at the core of every modern recsys stack: collaborative filtering on persistent user histories and sequence modelling on within-session events.

## References

See `reports/references.md` for the complete list with verified DOIs. Inline citations:

- [Afsar 2022] Afsar, Crump, Far. Reinforcement Learning based Recommender Systems: A Survey. ACM Computing Surveys.
- [Cheng 2016] Cheng et al. Wide and Deep Learning for Recommender Systems. DLRS.
- [Covington 2016] Covington, Adams, Sargin. Deep Neural Networks for YouTube Recommendations. RecSys.
- [Guo 2017] Guo et al. DeepFM. IJCAI.
- [He 2017] He et al. Neural Collaborative Filtering. WWW.
- [Kang 2018] Kang, McAuley. Self-Attentive Sequential Recommendation. ICDM.
- [Koren 2009] Koren, Bell, Volinsky. Matrix Factorization Techniques for Recommender Systems. Computer.
- [Ma 2018a] Ma et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. KDD.
- [Ma 2018b] Ma et al. Entire Space Multi-Task Model. SIGIR.
- [Sarwar 2001] Sarwar et al. Item-based collaborative filtering recommendation algorithms. WWW.
- [Souza 2021] de Souza Pereira Moreira et al. Transformers4Rec. RecSys.
- [Sun 2019] Sun et al. BERT4Rec. CIKM.
- [Tan 2016] Tan, Xu, Liu. Improved Recurrent Neural Networks for Session-based Recommendations. DLRS.
- [Tang 2018] Tang, Wang. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. WSDM.
- [Wang 2019] Wang et al. KGAT. KDD.
- [Wu 2019] Wu et al. Session-Based Recommendation with Graph Neural Networks. AAAI.
- [Yi 2019] Yi et al. Sampling-bias-corrected neural modeling for large corpus item recommendations. RecSys.
- [Zhang 2019] Zhang et al. Deep Learning Based Recommender System. ACM Computing Surveys.
- [Zhou 2018] Zhou et al. Deep Interest Network for Click-Through Rate Prediction. KDD.
