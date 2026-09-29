# Lab 2 — Ad Click Prediction: Short Explanation

**Notebook:** `lab2-ad-click-prediction.ipynb` (self-executing; auto-downloads the OpenML `Click_prediction_small` dataset, ~29 MB, 399k ad impressions)

## 1. Model

- **Data**: sponsored-search impressions with a `click` label (observed CTR 16.8% — negatives are subsampled, so this is far more balanced than production traffic).
- **Features**: categorical IDs (ad, advertiser, query, keyword, title, description — each capped to top-250 train values + "other" bucket) plus numeric `depth` and `position`. `user_id`/`url_hash` are dropped as identifiers.
- **Model**: `HistGradientBoostingClassifier` with native categorical splits; logistic regression on one-hot features as the baseline.



## 2. ROC-AUC + Calibration


| Metric      | HistGB | LogReg | HistGB + isotonic |
| ----------- | ------ | ------ | ----------------- |
| ROC-AUC     | 0.680  | 0.674  | 0.682             |
| Brier score | 0.1291 | 0.1296 | 0.1290            |


- **ROC-AUC** measures ranking quality only: probability that a random clicked impression outranks a random non-clicked one. Threshold-free, so it's the right first check for a CTR ranker.
- **Calibration** is what an ad auction actually consumes: bids are `p(click) × value`, so predicted scores must be *probabilities*. We validate with reliability curves and Brier score, and recalibrate with isotonic regression (`CalibratedClassifierCV`).
- Key insight: isotonic calibration is monotone — it reshapes scores without re-ranking, so **AUC is unchanged while calibration error drops**. A model can win on AUC and still lose money if its probabilities are biased.



## 3. Business tradeoffs — false positives vs false negatives

- **False positive** (predict click, impression doesn't click): wasted media spend and worse user experience; in an auction you paid for that impression.
- **False negative** (real clicker not served): lost revenue that was pure upside.
- With revenue `R` per click and cost `C` per impression, expected profit per served impression is `p·R − C`, so the **break-even threshold is** `C/R`. In the notebook's sweep (R=$0.50, C=$0.02 → break-even 0.04), the profit-maximizing threshold lands at ~0.05, close to theory.
- At a naive 0.5 threshold, precision is 0.65 but recall only 0.07 — most clicks are missed. The right threshold is an economics decision (bid prices, budget caps, UX cost), not an ML one.

