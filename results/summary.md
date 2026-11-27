# Results summary (for slide makers)

All numbers are in `results/results_table.csv`; figures are in `figures/` (300 dpi PNG).
Seed 0 everywhere. Data split: `data/split.csv` (598 training cities / 150 test cities; both statements of a city are always on the same side).

## Headline

- **Primary question:** Yes. Classifiers on Qwen3-Embedding-8B vectors separate true from false city statements almost perfectly: **99.7% test accuracy for both logistic regression and random forest** (1 error out of 300 each, the same statement). The nonlinear method does **not** beat the linear one (difference 0.0 points, McNemar p = 1.0).
- **Secondary 1 (no labels):** Truth *is* the dominant direction. PC1 alone correlates 0.84 with the label, and k-means with k = 2 recovers the labels at 98.4% cluster accuracy (ARI 0.94) without ever seeing them.
- **Secondary 2 (negation transfer):** Partial. Accuracy drops to 73-87% for LR and 93-94% for RF when moving between affirmative and negated statements, but ROC AUC stays at 0.99 or above. The truth *ranking* transfers; the decision *threshold* shifts.
- **Secondary 3 (chat model):** A 120B chat model answering zero-shot gets 99.3% (cities) and 100% (neg_cities). It's about as accurate as the classifiers.

## Matching against CLAUDE.md section 6 (decided before seeing results)

| Outcome row | Match? | Evidence |
|---|---|---|
| LR high (> 90%), RF not better | **Yes** | LR 99.7% [99.0, 100], RF 99.7% [99.0, 100]; LR-RF difference 0.0, CI [0.0, 0.0], McNemar p = 1.0 |
| RF clearly beats LR | No | Difference CI does not exclude 0 |
| Both near 50% | No | |
| High within-dataset, near/below 50% on neg_cities | **No** (degraded, not collapsed) | Transfer accuracy 73-94%, AUC >= 0.99 |
| k-means ARI near 0, top PCs track country | **No, the opposite** | k-means ARI with label 0.94; ARI with region ~0. PC1 tracks truth (abs r = 0.84), region explains < 2% of PC1 |

**Conclusion supported:** truth is largely linearly separable in this embedding space. It's also the largest single direction of variation, so even unsupervised clustering finds it.

---

## Figures

### `figures/eda_pca_by_label.png`: Top two PCs, colored by label
PCA fit on the cities training split (1196 statements). True (blue) and false (orange) statements fall into clearly separate regions, mostly along PC1. This is the "aha" slide: you can almost see the answer before training any classifier.

### `figures/eda_pca_by_region.png`: Same PCs, one panel per region
Each region's statements (blue) are spread across both the true and the false clusters. Region explains under 2% of PC1's variance and 6% of PC2's. Region shows up in later components instead: it explains 11-22% of each of PC3-PC8. So the main direction is truth, not geography. (This was the opposite of what we expected in section 4.)

### `figures/eda_explained_variance.png`: PCA explained variance
No single component dominates: PC1 explains 7.7% of variance, 10 PCs explain 36%, 50 PCs explain 69%. The embedding is high-dimensional, but the *truth-relevant* part is concentrated in PC1.

### `figures/eda_cosine_twins.png`: Twin pairs vs random pairs
A true statement and its false twin (same city, different country) have median cosine similarity 0.71. Two random statements about different cities have median 0.48. The embedding mostly encodes *which city* is being discussed. It still keeps a consistent truth signal on top of that.

### `figures/results_lr_tuning.png`: LR tuning curve
5-fold city-grouped CV accuracy against C, for L1 and L2. L2 is above 99% across the whole grid. L1 collapses to 50% when C is very small (all weights forced to zero) and recovers by C = 0.01. Selected: C = 0.1, L1 (L1 and L2 tie at 99.67% CV). **The selected L1 model uses only 39 of the 4096 dimensions.**

### `figures/results_test_accuracy.png`: Test accuracy with 95% CIs
LR 99.7% [99.0, 100], RF 99.7% [99.0, 100], chat model 99.3% [98.3, 100]. The CIs come from resampling the 150 test cities 1000 times. The axis starts at 95%; majority-class baseline is 50%. All three overlap completely.

### `figures/results_confusion_matrices.png`: Confusion matrices
Both models: 150/150 false statements correct, 149/150 true statements correct. The single error (for both) is "The city of Tangerang is in Indonesia." (true, predicted false). Tangerang is a large satellite city of Jakarta, possibly less known by name.

### `figures/results_transfer.png`: Transfer across negation
Bars: accuracy for train → test combinations. (a) = city-disjoint (train cities ≠ test cities); (b) = all 748 cities on both sides (same cities, only the negation changes).
- LR: cities → neg 86.7% (a) / 83.4% (b); neg → cities 72.7% (a) / 77.5% (b).
- RF: 94.3% / 93.6%; 93.0% / 94.0%.
- **Why accuracy drops but AUC doesn't:** LR trained on affirmative statements labels only 37% of negated statements "true" (the real share is 50%). LR trained on negated statements labels 77% of affirmative statements "true". The direction that separates true from false is shared, but its zero point shifts with negation. In the paper's terms, part of what the classifier learned is tied to sentence form.
- (a) and (b) are similar, so unseen cities are not the issue. The drop comes from negation.

### `figures/results_learning_curve.png`: How many labelled cities does it take?
LR and RF (tuned settings) trained on random subsets of 10, 25, 50, 100, 200 and all 598 training cities. Each subset size is repeated 5 times with fixed seeds, and every run is scored on the same 300-statement test set. Mean test accuracy:

| Training cities | 10 | 25 | 50 | 100 | 200 | 598 (all) |
|---|---|---|---|---|---|---|
| Logistic regression | 50.0% ± 0.0 | 98.1% ± 1.0 | 98.7% ± 0.4 | 99.5% ± 0.2 | 99.5% ± 0.2 | 99.7% |
| Random forest | 98.4% ± 0.7 | 98.9% ± 0.5 | 98.9% ± 0.2 | 99.3% ± 0.4 | 99.5% ± 0.2 | 99.7% |

- **About 25 labelled cities (50 statements) are enough for 98%.** Beyond that, more data adds less than 2 points.
- **The LR 50% at 10 cities is a regularization artifact, not a failure to learn.** C = 0.1 was tuned for about 1,200 training statements. With only 20, the same L1 penalty outweighs the data and sets every weight to zero, so the model predicts "False" for everything. At 25 cities it keeps 5 nonzero weights and reaches 98%. This makes a good Q&A answer for "what does C control?".
- **The dashed line is k-means on the training set with no labels (98.4%).** A few dozen labels are needed to beat it. It's drawn only as a reference, because it's scored on training data rather than the test set.
- **Section 6 outcome:** consistent with "LR high, RF not better". The truth direction is so strong that very little labelled data is needed to find it.

### `figures/results_hard_negatives.png`: Are false statements harder when the wrong country is nearby?
Each false test statement is labelled **same-region** if the wrong country is in the same UN region as the true one (e.g. a Chinese city placed in India), and **cross-region** otherwise (e.g. a Russian city placed in South Africa). Accuracy with 95% city-bootstrap CIs:

| Group | n | Logistic regression | Random forest | Chat model |
|---|---|---|---|---|
| False, same region | 63 | 100% (0 wrong) | 100% (0 wrong) | 100% (0 wrong) |
| False, cross-region | 87 | 100% (0 wrong) | 100% (0 wrong) | 100% (0 wrong) |
| True statements | 150 | 99.3% [98.0, 100] (1 wrong) | 99.3% [98.0, 100] (1 wrong) | 98.7% [96.7, 100] (2 wrong) |

- **Same-region false statements are not harder:** every method gets all 63 right. All errors, for every method, are true statements called false.
- **Caveat:** 58 of the 63 same-region false statements are in Asia (49 of them involve China or India as the true or the wrong country, e.g. an Indian city placed in China), because the paper sampled wrong countries by frequency. "Same UN region" is also a coarse notion of "nearby". A finer split (e.g. UN subregion or neighbouring countries) might find harder cases, but the test set has no errors to explain here.
- The same-region group has 63 statements, above the 20 threshold, so no out-of-fold CV breakdown was needed.
- **Section 6 outcome:** consistent with "LR high, RF not better". Nothing suggests the classifiers rely on coarse geography: if they only detected "wrong continent", same-region false statements would fail.

## Tables

### Main results (cities test split, n = 300)

| Method | CV accuracy (mean ± sd) | Test accuracy [95% CI] | Test ROC AUC [95% CI] | Grid size | Tuning time |
|---|---|---|---|---|---|
| Majority class | | 50.0% | | | |
| Logistic regression (L1, C = 0.1) | 99.67 ± 0.17% | 99.7% [99.0, 100] | 0.999 [0.998, 1.0] | 26 | 14 s |
| Random forest (500 trees, max_features 0.1, no PCA) | 99.42 ± 0.42% | 99.7% [99.0, 100] | 0.999 [0.998, 1.0] | 48 | 223 s |
| Chat model zero-shot (baseline, not a course method) | n/a | 99.3% [98.3, 100] | n/a | n/a | n/a |

Paired LR vs RF: accuracy difference 0.0 [0.0, 0.0]; McNemar exact p = 1.0 (0 discordant statements). The two models are **not distinguishable**.

RF on PCA-reduced features did slightly *worse* in CV (98.7-98.8% for 20/50/100 PCs vs 99.4% raw). Our section 4 worry that RF needs PCA on dense features didn't hold here.

### Unsupervised (cities training split, PCA + k-means, k = 2)

| PCs | ARI vs truth label | Cluster accuracy | ARI vs region |
|---|---|---|---|
| 2 | 0.937 | 98.4% | -0.001 |
| 10 | 0.941 | 98.5% | -0.001 |
| 50 | 0.947 | 98.7% | -0.001 |

Not ranked against the supervised models (no labels used, scored on training data).

### Error analysis (LR, cities test)
1 error in 300. By region: Asia 1/194, Africa 0/38, Americas 0/32, Europe 0/36. By country (top 10 by test count): only Indonesia has an error (1/8). There are too few errors to find a pattern. Full lists: `results/errors_LR.csv`, `results/errors_all_models.csv`. The chat model's 2 errors are different statements: Nowrangapur (India) and New Kingston (Jamaica), both called false although true.

### Chat-model baseline details
`nvidia/nemotron-3-super-120b-a12b` ($0.30 / $0.90 per 1M input/output tokens), thinking disabled, temperature 0, max_tokens 2. Prompt: "Is the following statement true or false? Answer with exactly one word: True or False. Statement: ...". 0 unparseable answers out of 600. Total cost $0.009. Responses cached in `data/baseline/`.

## Talking points for limitations
- The result is near ceiling, so this dataset can't tell LR and RF apart. A harder dataset would be needed for that.
- The classifiers may be detecting "this city doesn't belong with this country" rather than truth in general. Separability is not knowledge (section 7). The negation threshold shift supports this caution.
- One embedding model, templated statements, 150 test cities.
- Transfer models reuse the hyperparameters tuned on cities; no separate tuning on neg_cities.
