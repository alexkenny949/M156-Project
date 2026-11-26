# MATH 156 Project Spec: Do LLM Embeddings Encode Truth?

## Instructions for Claude Code

- This file is the project spec. Follow it; do not change the research question, methods, or evaluation rules without asking.
- Work one handoff step (section 10) at a time. Stop after each step, report results, and wait for review before continuing.
- Never commit `.env` or the API key. Read the key from the `NEBIUS_API_KEY` environment variable.
- Items marked VERIFY must be checked and reported, not assumed.
- Fix random seeds. Save every number to the results table and every figure to `figures/`.

Status: plan v3 (option 2: API embeddings, rubric-checked). Group of 3. Items marked VERIFY must be checked in Claude Code before relying on them.

Suggested title: "Do LLM Embeddings Encode Truth? Classifying True and False Statements from Qwen3 Embeddings." The original title ("Probing an LLM's Hidden States for Truth") no longer fits, because the API returns only the embedding model's final output vector, not internal layers.

## 0. Rubric coverage map

| Rubric criterion (20 each) | Where covered |
|---|---|
| 1. Clear research question and understanding of the data | Sections 1, 2, 3 (incl. EDA) |
| 2. Understanding and justification of the methods | Section 4 (incl. how each works, why not others) |
| 3. Model evaluation: data splitting, tuning and fair comparison | Section 5 (incl. leakage rules, tuning grids, bootstrap CIs) |
| 4. Analysis of results, conclusions and limitations | Sections 6, 7 |
| 5. Clarity, organisation and timing | Section 8 (slide outline, timing, Q&A prep) |
| Guideline: compare 2-3 course methods | Section 4 |
| Guideline: cite data source on slides | Sections 2, 3, 8 |
| Guideline: every member speaks and answers questions | Section 8 |

## 1. Research question (criterion 1)

**Primary:** Can classifiers trained on embeddings from an LLM-based embedding model (Qwen3-Embedding-8B) distinguish true from false factual statements, and does a nonlinear method beat a regularized linear one?

**Secondary (drop from the bottom if short on time):**
1. Does truth appear without labels? (PCA + k-means)
2. Does a classifier trained on affirmative statements transfer to negated ones? (cities -> neg_cities)
3. Optional: how do the classifiers compare to a large chat model answering "True or False?" directly via the API?

**Why it matters (one slide):** LLMs state falsehoods confidently; if truth is recoverable from a model's representations with simple classifiers, cheap truth-checks become possible.

## 2. Data (criterion 1)

**Source:** Geometry of Truth datasets, Marks & Tegmark (COLM 2024). https://github.com/saprmarks/geometry-of-truth (paper: arXiv:2310.06824). Cite both on the slides.

| Dataset | Example | Rows (per paper) | Role |
|---|---|---|---|
| cities | "The city of [city] is in [country]." | 1496 | Main train/test set |
| neg_cities | Negated cities statements ("not") | 1496 | Transfer test |
| sp_en_trans | Spanish-English translation statements | VERIFY | Optional cross-topic test |
| larger_than / smaller_than | Number comparisons | VERIFY | Optional cross-topic test |

**Variables:** statement text (input), binary label true/false (target). After embedding: 4096 numeric features per statement.

**Data understanding (EDA) to present:**
- How the statements were constructed (templated from real city-country facts). VERIFY the underlying source of city data in the paper/repo and state it, since a grader may ask whether templated statements count as "real data."
- Label balance (expect ~50/50), number of unique cities and countries, example true and false rows.
- Whether the same city appears in both a true and a false statement. VERIFY. If yes, split by city (GroupKFold) to prevent leakage.
- Statement length distribution for true vs false (a length difference would be a confound).
- PCA scatter of embeddings colored by label, and separately colored by country/region, to show what the main directions of variation track.

Note: the guidelines list UCI/OpenML/Kaggle/Hugging Face as "possible sources," so GitHub should be acceptable. Confirm with the TA if unsure.

## 3. Features: embeddings via Nebius Token Factory

- **Model:** `Qwen/Qwen3-Embedding-8B`, 4096-dimensional output, $0.01 per 1M input tokens (per Token Factory's public model catalog, checked 11/26/2026). Cost for a few thousand statements: well under $1. Cite the model on the slides.
- **Endpoint:** OpenAI-compatible, base URL `https://api.tokenfactory.nebius.com/v1/`, `client.embeddings.create(model=..., input=[...], encoding_format="float")`. API key in env var `NEBIUS_API_KEY` (never commit it).
- Batch the requests; cache results to disk (one .npy of shape n x 4096 per dataset plus labels) so the API is called once.
- Feed statements as plain text, no instruction prefix (state this on the slides).
- VERIFY: returned vector length is 4096 and whether vectors come back L2-normalized.

Key point for slides: dimension (4096) exceeds training-set size (~1200), so unregularized models can fit the training data perfectly. This motivates regularization and PCA (curse of dimensionality, Lectures 1.2, 2.2, 6.3).

## 4. Methods (criterion 2)

Supervised comparison: logistic regression vs random forest (two course methods, compared head-to-head). Unsupervised check: PCA + k-means (third course method, answers a separate question, not ranked against the supervised ones).

| Method | How it works (be able to explain) | Why chosen for this question | Tuning |
|---|---|---|---|
| Logistic regression, L2 (and L1 variant) | Models P(true given x) = sigmoid(w.x + b); minimizes cross-entropy loss plus penalty on w (L2: squared norm, shrinks weights; L1: absolute norm, sets many to zero) | Tests whether truth is linearly separable; standard probe in the literature; regularization required since d > n; L1 shows how many features are needed | C over a log grid (e.g. 1e-4 to 1e2, 13 values); penalty L1 vs L2 |
| Random forest | Many decision trees, each grown on a bootstrap sample with a random subset of features per split; predictions averaged (bagging, Lectures 5.2-6.1) | Tests whether a nonlinear model finds structure the linear model misses; little tuning needed. Expected weakness: single-feature splits on 4096 dense features, so also run on PCA-reduced features | n_estimators (e.g. 300, 500), max_depth (None, 10, 20), max_features (sqrt, 0.1); PCA components (none, 20, 50, 100) |
| PCA + k-means (k=2) | PCA: project onto directions of maximum variance (eigenvectors of covariance). k-means: minimize within-cluster sum of squares | Tests whether truth is a dominant direction without labels. Embedding models are trained for semantic similarity, so top PCs may track topic (country/region) instead of truth; a result either way | Number of PCs (2, 10, 50); report ARI and cluster accuracy (best label assignment) |

**Why not other course methods (have one sentence each ready for Q&A):**
- Kernel methods / Gaussian processes: costly in high dimension relative to benefit at this n; logistic regression already covers the linear case.
- Boosting: a reasonable swap for random forest; random forest chosen for less tuning under time pressure.
- Feedforward neural network: more tuning and overfitting risk at n ~1200 with d = 4096.

Baselines: majority class (~50%). Optional: a Token Factory chat model answering "True or False?" zero-shot (not a course method; label it as a baseline only). VERIFY available chat models via the /v1/models endpoint.

## 5. Evaluation (criterion 3)

1. **Split:** hold out a stratified 20% test set (grouped by city if cities repeat). Touch it once, at the end.
2. **Leakage rules:** standardization and PCA are fit inside each CV fold only (sklearn Pipeline), never on the full data. Nothing is fit on the test set.
3. **Tuning:** 5-fold stratified CV on the training set, selection metric = accuracy, using the grids in section 4. Identical folds for every method (fixed seed).
4. **Fair comparison:** same splits, same features, each method tuned over its stated grid; report the grid sizes so tuning effort is visible.
5. **Final metrics on test set:** accuracy, ROC AUC, confusion matrix; CV mean +/- sd alongside.
6. **Uncertainty:** bootstrap 95% confidence intervals for test accuracy (resample test set 1000 times, Lecture 5.3), plus a paired comparison of LR vs RF (bootstrap the accuracy difference, or McNemar's test). With ~300 test statements, differences of a few points may be noise; state whether the difference is distinguishable.
7. **Transfer:** train the tuned models on all of cities, test on neg_cities (and reverse). Compare against within-dataset accuracy.
8. **Unsupervised:** k-means ARI and cluster accuracy on the training set; PCA explained-variance plot.
9. Fix random seeds; log every number to one results table.

## 6. Analysis plan (criterion 4)

**How to interpret possible outcomes (decide before seeing results):**

| Outcome | Conclusion it supports |
|---|---|
| LR high (e.g. > 90%), RF not better | Truth is largely linearly separable in embedding space |
| RF clearly beats LR (CI on difference excludes 0) | Truth information exists but is nonlinear |
| Both near 50% | Embedding does not encode truth for these statements (consistent with similarity-trained embeddings) |
| High within-dataset, near or below 50% on neg_cities | Classifier learned surface features of affirmative statements, not truth itself |
| k-means ARI near 0, top PCs track country | Truth is not the dominant variation; topic is |

**Error analysis:** list misclassified test statements; check whether errors cluster by country, region, or city fame.

**Conclusion slide:** answer the primary question in one sentence, then one sentence per secondary question.

## 7. Limitations (criterion 4)

- Embedding models are trained to place similar-meaning sentences together; "Paris is in France" and "Paris is in Spain" are near-identical in surface meaning, so the embedding may underrepresent truth. This probes the embedding model's output, not a generative LLM's internal layers.
- Only the final output vector is available; no layer-by-layer analysis.
- Separability is not the same as the model "knowing" the truth; classifiers may pick up correlates (specific city-country pairs, word frequency).
- One embedding model; simple, unambiguous, templated statements only, so results may not extend to real-world claims.
- Small test set limits precision (see CIs).
- k-means is unsupervised, so its scores are not directly comparable to supervised accuracy.

## 8. Presentation (criterion 5): 15 min + 5 min Q&A, group of 3

Target 14 minutes to leave buffer. About 13-15 slides.

| Speaker | Slides | Rubric | Time |
|---|---|---|---|
| 1 | Title; motivation; research question; data source + citation; EDA (balance, examples, length check); embedding pipeline + model citation; d > n point | 1 | 4.5 min |
| 2 | LR, RF, PCA+k-means: how each works and why chosen; why not others; split, leakage rules, CV and tuning grids, fair comparison | 2, 3 | 4.5 min |
| 3 | Results table with CIs; PCA scatter; transfer results; error analysis; conclusions; limitations; sources slide | 4 | 5 min |

**Rehearsal:** at least two timed full run-throughs; cut slides if over 14 min.

**Q&A prep (everyone must answer any of these):**
- Why is regularization needed here? What does C control?
- Why fit PCA inside the CV folds?
- How was the test set kept separate from tuning?
- Is the LR vs RF difference meaningful? (point to CI)
- Why might an embedding model fail to encode truth?
- What would change with a different model or with real-world claims?
- Why does k-means not count as a head-to-head competitor?
- Where does the data come from, and is templated data "real"?

## 9. Work split and timeline (assumes worst case: presenting Mon 11/30)

- Thu 11/26 - Fri 11/27: Member 1 runs Claude Code steps 1-4 (data check, EDA, embeddings). Members 2-3 build slide skeleton and write question, data and methods slides.
- Sat 11/28: Member 1 runs steps 5-6 (methods, evaluation, figures). Member 2 runs optional chat-model baseline. Member 3 drafts limitations, error analysis slide, and Q&A answers.
- Sun 11/29: fill in results slides together; at least two timed rehearsals; each member practices answering the Q&A list.

## 10. Claude Code handoff order

1. Environment: Python, openai, scikit-learn, numpy, pandas, matplotlib. Set `NEBIUS_API_KEY`.
2. Clone dataset repo; load CSVs; report rows, columns, label balance, unique cities/countries, city repetition, length by label.
3. Embed cities and neg_cities via Token Factory; confirm shape and normalization; cache to disk.
4. EDA figures: PCA scatter by label and by region; explained-variance plot.
5. Methods per section 4 inside sklearn Pipelines; evaluation per section 5, including bootstrap CIs and paired LR vs RF comparison.
6. Transfer test; error analysis list; results table and figures (confusion matrices, transfer bar chart).
7. Optional: chat-model True/False baseline.

## Open items

- Presentation day: unknown.
