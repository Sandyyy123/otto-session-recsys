# References - Project 17 OTTO Multi-Objective Session-Based Recommendation

All entries below were verified live against CrossRef (https://api.crossref.org/works/{doi}). Per project rule, only author / title / journal / year / DOI are kept; volume, issue, and pages are stripped to avoid hallucination.

---

## Background and foundations

1. Sarwar B, Karypis G, Konstan J, Riedl J. Item-based collaborative filtering recommendation algorithms. Proceedings of the 10th international conference on World Wide Web. 2001. DOI:10.1145/371920.372071

2. Koren Y, Bell R, Volinsky C. Matrix Factorization Techniques for Recommender Systems. Computer. 2009. DOI:10.1109/MC.2009.263

3. Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation. 1997. DOI:10.1162/neco.1997.9.8.1735

4. Zhang S, Yao L, Sun A, Tay Y. Deep Learning Based Recommender System. ACM Computing Surveys. 2019. DOI:10.1145/3285029

## Session-based and sequential recommendation

5. Tan Y, Xu X, Liu Y. Improved Recurrent Neural Networks for Session-based Recommendations. Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 2016. DOI:10.1145/2988450.2988452

6. Tang J, Wang K. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 2018. DOI:10.1145/3159652.3159656

7. Kang W, McAuley J. Self-Attentive Sequential Recommendation. 2018 IEEE International Conference on Data Mining (ICDM). 2018. DOI:10.1109/ICDM.2018.00035

8. Sun F, Liu J, Wu J, Pei C, Lin X, Ou W. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019. DOI:10.1145/3357384.3357895

9. Wu S, Tang Y, Zhu Y, Wang L, Xie X, Tan T. Session-Based Recommendation with Graph Neural Networks. Proceedings of the AAAI Conference on Artificial Intelligence. 2019. DOI:10.1609/aaai.v33i01.3301346

10. Zhang T, Zhao P, Liu Y, Sheng V, Xu J, Wang D. Feature-level Deeper Self-Attention Network for Sequential Recommendation. Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence. 2019. DOI:10.24963/ijcai.2019/600

11. de Souza Pereira Moreira G, Rabhi S, Lee J, Ak R, Oldridge E. Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation. Fifteenth ACM Conference on Recommender Systems. 2021. DOI:10.1145/3460231.3474255

## Industrial recommender systems and candidate generation

12. Covington P, Adams J, Sargin E. Deep Neural Networks for YouTube Recommendations. Proceedings of the 10th ACM Conference on Recommender Systems. 2016. DOI:10.1145/2959100.2959190

13. Cheng H, Koc L, Harmsen J, Shaked T, Chandra T, Aradhye H. Wide and Deep Learning for Recommender Systems. Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 2016. DOI:10.1145/2988450.2988454

14. Yi X, Yang J, Hong L, Cheng D, Heldt L, Kumthekar A. Sampling-bias-corrected neural modeling for large corpus item recommendations. Proceedings of the 13th ACM Conference on Recommender Systems. 2019. DOI:10.1145/3298689.3346996

## Click-through rate and feature crossing

15. He X, Liao L, Zhang H, Nie L, Hu X, Chua T. Neural Collaborative Filtering. Proceedings of the 26th International Conference on World Wide Web. 2017. DOI:10.1145/3038912.3052569

16. Guo H, Tang R, Ye Y, Li Z, He X. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence. 2017. DOI:10.24963/ijcai.2017/239

17. Zhou G, Zhu X, Song C, Fan Y, Zhu H, Ma X. Deep Interest Network for Click-Through Rate Prediction. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2018. DOI:10.1145/3219819.3219823

## Multi-objective and multi-task recsys

18. Ma J, Zhao Z, Yi X, Chen J, Hong L, Chi E. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2018. DOI:10.1145/3219819.3220007

19. Ma X, Zhao L, Huang G, Wang Z, Hu Z, Zhu X. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval. 2018. DOI:10.1145/3209978.3210104

## Pretraining and language model methodology used in BERT4Rec

20. Devlin J, Chang M, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics. 2019. DOI:10.18653/v1/N19-1423

## Reinforcement learning and graph approaches

21. Afsar M, Crump T, Far B. Reinforcement Learning based Recommender Systems: A Survey. ACM Computing Surveys. 2022. DOI:10.1145/3543846

22. Wang X, He X, Cao Y, Liu M, Chua T. KGAT: Knowledge Graph Attention Network for Recommendation. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2019. DOI:10.1145/3292500.3330989

---

**Verification protocol.** Each DOI was queried at https://api.crossref.org/works/{doi} on 2026-05-08; all 22 returned 200 with matching title and year. Volume, issue, and page numbers were intentionally stripped because the same dataset would be hand-typed into the manuscript otherwise, which violates the project no-fabrication rule.


---

## 2024-2026 additions (post-QA literature scout)

# Additional References - Project 17 OTTO Multi-Objective Session-Based Recommendation

Independent literature scout sweep, 2024 to 2026, prioritising recent peer-reviewed and conference work that the existing `reports/references.md` (which stops at 2022) does not cover. Every DOI below was queried live at `https://api.crossref.org/works/{doi}` on 2026-05-08; entries that did not return a 200 with matching title or year were dropped from this list rather than padded.

Format follows the project rule: Authors. Title. Journal. Year. DOI. No volume, issue, or pages.

---

## State-of-the-art callout (gaps in current `references.md`)

The current `references.md` lists 22 entries that stop at 2022 (the most recent are Afsar 2022 on RL recommenders and Transformers4Rec 2021). The OTTO challenge has moved on substantially since then. The following five lines are the highest-leverage gaps that the project SHOULD cite, all live and verified:

1. **Sampled softmax is not as well-behaved as the SASRec paper assumes.** Wu et al. 2024 in TOIS (DOI:10.1145/3637061) and Wu et al. 2024 at ICDE (DOI:10.1109/icde60146.2024.00068) directly address why the loss used in `model_advanced.py` is biased and propose drop-in replacements. Both are missing from the current bibliography and are directly implementable in the existing code.
2. **LogQ correction for large-corpus retrieval was revisited at RecSys 2025** (Khrylchenko et al., DOI:10.1145/3705328.3748033). For a 1.85M-item catalog with sampled-softmax this is the single most relevant recent paper for a v2.0 ablation.
3. **Generative retrieval with semantic identifiers** is the SOTA paradigm shift since 2023. The order-agnostic identifier work at SIGIR 2025 (Lin et al., DOI:10.1145/3726302.3730053) and the AAAI 2026 MusicRec paper (Zhao et al., DOI:10.1609/aaai.v40i19.38685) define the new direction. The current bibliography has no generative retrieval entries.
4. **LLM-based recommender system agents** (Carraro et al., RecSys 2025, DOI:10.1145/3705328.3759334) and the Wang/Zhang/Chua 2024 chapter on recommendation in the era of generative AI (DOI:10.1007/978-3-031-73147-1_8) capture the shift the field has taken; both are missing.
5. **Capsule and hypergraph GNN session models** have overtaken plain GNN session work in 2024-2025 benchmarks (El Alaoui et al. 2025 in Neural Networks, DOI:10.1016/j.neunet.2025.107176; Yang and Peng 2024 in Applied Intelligence, DOI:10.1007/s10489-024-05877-1). The current bibliography stops at SR-GNN 2019.

---

## Sampled softmax, retrieval loss, large-catalog training (2024-2025)

These are the most directly actionable for `model_advanced.py`, which currently uses 200 uniform random negatives without LogQ correction.

1. Wu J, Wang X, Gao X, Chen J. On the Effectiveness of Sampled Softmax Loss for Item Recommendation. ACM Transactions on Information Systems. 2024. DOI:10.1145/3637061

2. Wu J, Chen J, Wu J, Shi W. BSL: Understanding and Improving Softmax Loss for Recommendation. 2024 IEEE 40th International Conference on Data Engineering. 2024. DOI:10.1109/icde60146.2024.00068

3. Khrylchenko K, Baikalov V, Makeev S, Matveev A. Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval. Proceedings of the Nineteenth ACM Conference on Recommender Systems. 2025. DOI:10.1145/3705328.3748033

## Session-based and sequential recommendation - architectures (2024-2026)

Recent architectures that extend or supersede the SASRec / BERT4Rec lineage cited in the existing references.md.

4. Yang F, Peng D. A graph neural network with topic relation heterogeneous multi-level cross-item information for session-based recommendation. Information Systems. 2024. DOI:10.1016/j.is.2024.102380

5. El Alaoui D, Riffi J, Sabri A, Aghoutane B. A novel session-based recommendation system using capsule graph neural network. Neural Networks. 2025. DOI:10.1016/j.neunet.2025.107176

6. Wang L, Jin D. A Time-Sensitive Graph Neural Network for Session-Based New Item Recommendation. Electronics. 2024. DOI:10.3390/electronics13010223

7. Celik E, Ilhan Omurca S. Skip-Gram and Transformer Model for Session-Based Recommendation. Applied Sciences. 2024. DOI:10.3390/app14146353

8. Wang J, Zhang S. Sentiment-Time Heterogeneous Residual Graph Attention Transformer for Session-Based Recommendation. International Journal of Software Engineering and Knowledge Engineering. 2024. DOI:10.1142/s0218194024500037

9. Jiang K, Pan A, Jiang Y, Cheng S. Dynamic intent-aware and cross-session integration for session-based recommendation. Discover Artificial Intelligence. 2025. DOI:10.1007/s44163-025-00650-w

## Contrastive learning for sequential and session-based recsys

This whole strand of work is missing from `references.md` and is now the dominant regulariser for sequence models.

10. Xie Z, Li J. Simple Debiased Contrastive Learning for Sequential Recommendation. Knowledge-Based Systems. 2024. DOI:10.1016/j.knosys.2024.112257

11. Yang F, Peng D. MVC-HGAT: multi-view contrastive hypergraph attention network for session-based recommendation. Applied Intelligence. 2024. DOI:10.1007/s10489-024-05877-1

12. Yang F, Peng D. Spatio-Temporal Contrastive Heterogeneous Graph Attention Networks for Session-Based Recommendation. Mathematics. 2024. DOI:10.3390/math12081193

13. Jin K, Blanco-Encomienda F. HiCORE: Enhancing consumer intent prediction via hybrid attention and contrastive learning in sequential recommendation. Expert Systems with Applications. 2026. DOI:10.1016/j.eswa.2025.130860

14. Zhu N, Sun L, Luo X, Cao J. Exploitation or Exploration Next? User Behavior Decoupling and Emerging Intent Modeling for Next-Item Recommendation. 2024 IEEE International Conference on Data Mining (ICDM). 2024. DOI:10.1109/icdm59182.2024.00123

## Generative retrieval and LLM-based recommendation (2024-2026)

This is the new paradigm since the existing bibliography was last updated. None of the existing 22 references cover it.

15. Lin X, Shi H, Wang W, Feng F. Order-agnostic Identifier for Large Language Model-based Generative Recommendation. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2025. DOI:10.1145/3726302.3730053

16. Zhao Y, Shi L, Zhong Y, Kou F. MusicRec: Multi-modal Semantic-Enhanced Identifier with Collaborative Signals for Generative Recommendation. Proceedings of the AAAI Conference on Artificial Intelligence. 2026. DOI:10.1609/aaai.v40i19.38685

17. Mekonnen K, Tang Y, de Rijke M. Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2025. DOI:10.1145/3726302.3730023

18. Carraro T, Singh B, Pedanekar N. Large Language Model-based Recommendation System Agents. Proceedings of the Nineteenth ACM Conference on Recommender Systems. 2025. DOI:10.1145/3705328.3759334

19. Wang W, Zhang Y, Chua T. Recommendation in the Era of Generative Artificial Intelligence. Information Access in the Era of Generative AI. 2024. DOI:10.1007/978-3-031-73147-1_8

## Multi-objective and multi-task recsys (post-MMoE / ESMM era)

The existing references.md cites MMoE (2018) and ESMM (2018) but nothing from the last three years on this thread, even though OTTO is explicitly multi-objective.

20. Jin T. A Graph Bottleneck and Masked Feature Interaction Framework for Multi-Objective Optimization in E-commerce Recommendation Systems. Proceedings of the 2025 International Conference on Management Science and Computer Engineering. 2025. DOI:10.1145/3760023.3760094

21. Yu W, Liu B, Xia B, Xu X. Unsupervised Ranking Ensemble Model for Recommendation. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024. DOI:10.1145/3637528.3671598

## Knowledge distillation and efficient session models

Useful for the v2.0 retrieve-then-rerank discussion in the brief.

22. Yang Y, He J, Yang Y. On Adaptive Knowledge Distillation with Generalized KL-Divergence Loss for Ranking Model Refinement. Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information Retrieval. 2024. DOI:10.1145/3664190.3672522

---

**Verification protocol.** All 22 entries above were resolved live against `https://api.crossref.org/works/{doi}` on 2026-05-08. Title strings and publication years matched. Candidates that did not resolve, or whose Crossref title disagreed with the search result, were dropped rather than guessed. No volume, issue, or page numbers are recorded, in line with the project no-fabrication rule.

