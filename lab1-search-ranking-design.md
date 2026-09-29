# Lab 1 — Search Ranking Design (E-commerce)

## 1. Goal

Given a query, rank catalog items so the top results maximize purchase relevance — not just textual match. Two-stage architecture: **retrieval (recall) → ranking (precision)**.

## 2. Data Sources


| Source                | Contents                                                          | Use                                     |
| --------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| Product catalog       | title, description, brand, category, price, attributes, images    | Text/index features, retrieval          |
| Query logs            | raw queries, normalized/corrected queries                         | Training pairs, spell-correct, synonyms |
| Interaction logs      | impressions, clicks, add-to-cart, purchases, dwell time, position | Behavioral features, labels             |
| Inventory & logistics | stock, delivery ETA, seller rating                                | Business-rule features                  |
| User profile          | purchase/Click history, segment, device, geo                      | Personalization features                |




## 3. Features

- **Query–item match**: BM25/semantic similarity (embedding dot product), title/category match, exact-phrase flag, query coverage.
- **Item**: popularity (CTR/CVR), sales velocity, rating, review count, price, freshness, stock, return rate.
- **User**: past category/brand affinity, purchase history, session intent, geo/language.
- **Context**: device, time, traffic source, A/B bucket.
- **Position/debias features**: logged rank (for position-bias correction).

**Labels**: purchase > add-to-cart > click (weighted); implicit labels corrected for position bias.

## 4. Ranking Model

1. **Retrieval**: BM25 (keyword) + dual-encoder embeddings (query↔item) → top ~1,000 candidates.
2. **Ranking**: gradient-boosted trees (LightGBM/XGBoost) with learning-to-rank objective (LambdaMART) over the features above; strong baseline, fast, interpretable.
3. **Optional later stage**: neural re-ranker (e.g., cross-encoder or small transformer) on top ~50 for personalization.
4. **Business layer**: deterministic rules on top (out-of-stock demotion, seller policy, diversity constraints).



## 5. Evaluation Metrics

- **Offline**: NDCG@10, MRR, Recall@k (retrieval stage), pairwise accuracy; offline replay on held-out query logs.
- **Online (A/B)**: CTR, **CVR, revenue per search**, add-to-cart rate, zero-result rate, result-page abandonment. Primary metric: revenue per search (guardrails: latency p95 < 200 ms, relevance human eval).



## 6. Monitoring & Retraining

- **Monitoring dashboards**: metric deltas by query segment/head-torso-tail, zero-result rate, latency, feature distribution drift, label leakage alerts.
- **Drift detection**: population-stability index on features; alert on catalog or behavior shifts (seasonal, new sellers).
- **Retraining**: weekly full retrain of ranker; embedding index refreshed daily; popularity features updated hourly (near-real-time via streaming).
- **Governance**: every model change goes through offline eval → shadow mode → A/B before full rollout; auto-rollback if online guardrails breach.
- **Human loop**: bad-result sampling for relevance judgment; query review feeds synonym/stopword dictionaries.

