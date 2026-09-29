# Three Labs — Reflection

Labs: 1 · search ranking design — 2 · ad click prediction (CTR) — 3 · spam detection.

## 1. Which problem was hardest, and why?

**Lab 2 (CTR) — specifically calibration, not prediction.** Ranking clicks is a standard ML problem (0.66 AUC, position features do most of the work); the hard part was realizing that a good ranker still loses money. In the auction, `p(click)` is multiplied by bids, so a reliability curve that sags below the diagonal means systematically overpaying — isotonic recalibration fixed it, but only after recognizing that AUC said nothing about it. The other labs were hard in conventional ways (Lab 1 was a design-space problem, Lab 3 was mostly metric discipline); Lab 2 required questioning which *evaluator* is correct, which is a harder skill.

## 2. How would I explain my solution to a product manager?

Skip the models; explain the decision each one makes:

- **Spam filter (Lab 3)**: "Users should see at most 10% of spam. Given that rule, this filter blocks 90% of spam with **zero legitimate messages sent to junk** in testing — and when it degrades, we have a dial (the threshold) we can turn in minutes without retraining."
- **CTR (Lab 2)**: "We don't guess which ad wins — we pay for what a click is *actually worth*. The model outputs a probability that behaves like a real probability (we checked and corrected it), so the auction math `p(click) × bid` is trustworthy."
- **Search ranking (Lab 1)**: "Better ranking means the right result is on top, and we measure that with a rank-sensitive score so 'somewhere on page 1' never counts as success."

The common sentence: *"I translated your business constraint into the number the model optimizes — and gave you a monitoring plan so the number stays honest after launch."*

## 3. What would I improve in production?

1. **Temporal validation everywhere** — all three labs use random splits; production models must train on older data and be scored on newer data (this is the caveat most likely to bite first).
2. **Drift monitoring from day one** — Lab 3's plan (PSI alerting at 0.1/0.25, delayed precision proxies, re-tune → retrain → escalate) should apply to Lab 2 too; CTR distributions shift with campaigns and seasons just as spam shifts with adversaries.
3. **Score + attribution logging** — persist every prediction with its top contributing features, so false positives can be debugged after the fact instead of becoming folklore.
4. **Calibration per segment** for Lab 2 (per-position, per-slice reliability curves), and a **champion/challenger shadow-mode** pipeline so retrains compete on the same constrained metric before promotion.

