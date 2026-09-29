# Lab 3 — Spam Detection: Metric Analysis

**Task checklist**: ✅ spam classifier → ✅ precision @ recall constraint → ✅ drift monitoring plan
Companion notebook: `lab3-spam-detection.ipynb` (UCI SMS Spam Collection, 5,574 messages, 13.4% spam).

## 1. Why these metrics

Accuracy and ROC-AUC are the wrong headline metrics for spam: the majority class is ham (86.6%), so "mark everything ham" scores 86.6% accuracy, and ROC-AUC's false-positive rate denominator makes performance look better than it is on a rare positive class. The operating metric is **precision at a recall constraint**, read off the PR curve — the only view where the ham-flagged-as-spam (FP) cost the filter exists to avoid is visible.

## 2. Model comparison (test set, 20% stratified hold-out)


| Model                                      | ROC-AUC   | PR-AUC    | Best precision @ recall ≥ 0.90 | Operating threshold |
| ------------------------------------------ | --------- | --------- | ------------------------------ | ------------------- |
| **Logistic Regression** (TF-IDF 1–2 grams) | **0.992** | **0.979** | **1.000**                      | **0.558**           |
| Multinomial Naive Bayes (baseline)         | 0.988     | 0.970     | 0.993                          | 0.388               |
| All-ham baseline                           | 0.500     | 0.134     | — (recall = 0)                 | —                   |


Selection rule: maximize precision subject to recall ≥ 0.90 → **Logistic Regression at threshold 0.558**.

## 3. The chosen operating point

At threshold 0.558: **TP = 135, FP = 0, FN = 14, TN = 966** → precision 1.000, recall 0.906, 12.1% of messages flagged.

- The FP = 0 result is a property of an easy, near-separable dataset (random hold-out, 1,115 messages) — not a promise about production. Both models clear the floor with ≥ 0.99 precision; LogReg wins by the declared tie-break (max precision at the constraint) and higher PR-AUC, which is the more informative of the two aggregate scores here.
- The threshold is a product decision encoded as a constraint ("let at most 10% of spam through"), deliberately *not* F1-maximizing — F1 would buy recall with precision, pushing legitimate messages into the junk folder, which is the failure mode spam filtering must not have. Raising precision is also cheap insurance for drift: when scores degrade, headroom at the precision ceiling absorbs some of the hit before the floor is breached.
- Re-tuning is threshold-only: the sweep is minutes of compute and no retraining, so the constraint can be re-fit whenever the product requirement changes.



## 4. Monitoring plan for drift

Spam is adversarial, so degradation is guaranteed, not a risk. The plan:

- **Monitor**: score-distribution PSI (primary), vocabulary/length feature drift + OOD-token share, weekly spam rate (prior shift), delayed precision proxies (user "not spam" reports, junk-folder appeals), and volume anomalies as a silent-failure alarm.
- **Alert thresholds**: PSI < 0.1 normal and 0.1–0.25 watch and > 0.25 page; false-positive reports above SLA (e.g., > 0.2% of traffic) page immediately.
- **Demonstrated in notebook**: PSI of healthy test scores vs. train reference = **0.028** (healthy); synthetic drifted scores (adversary adapts) = **7.863** (alert).
- **Response playbook**: re-tune threshold on recent labeled data → targeted retrain with newly labeled spam (same precision@recall evaluation before promotion) → escalate to feature work if PR-AUC degrades structurally.
- **Hygiene**: validate temporally (train on older messages, test on newer — random splits flatter the model), log per-message score attributions for post-hoc FP debugging, run challengers in shadow mode.



## 5. Caveats

- Single random split; production evaluation must be temporal and adversarial-aware.
- No user feedback loop in the dataset — the precision proxies that drive real monitoring can't be simulated here, only designed for.
- 13.4% spam rate is far higher than real inboxes (<1%); the recall floor hides more FP risk at real priors, strengthening the case for high-precision thresholds.

