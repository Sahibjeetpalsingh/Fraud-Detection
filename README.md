<div align="center">

# 🛡️ Fraud Detection — Multi-Class Urgency Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC4E20?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SMOTE](https://img.shields.io/badge/SMOTE-imbalanced--learn-8E44AD?style=flat-square)](https://imbalanced-learn.org)
[![HackML 2026](https://img.shields.io/badge/HackML_2026-SFU_DSSS_Kaggle-20BEFF?style=flat-square)](https://kaggle.com/competitions/fraud-hack-ml-2026)

*6,244,474 transactions. 0.11% fraud. 4 urgency levels. One pipeline to find them all.*

</div>

<br>

---

## The Night a Bank Gets It Wrong

Picture a fraud analyst sitting at 11pm staring at a queue of 400 flagged transactions. Every single one of them is labelled *"suspicious."* No priority. No context. No urgency signal. Just a flat list.

She works through them from the top — methodically, one by one. By the time she reaches transaction 312, someone three time zones away has already had their account emptied.

Transaction 312 was a Class 3 case. Immediate action required. But there was no way to know that until it was too late, because the system only knew how to say *"fraud"* or *"not fraud."* It had no vocabulary for **urgency**.

That is the problem this project was built to solve.

<br>

---

## What This Competition Was Actually Asking

HackML 2026 was a Kaggle competition run by the SFU Data Science Student Society. 6.2 million simulated financial transactions. Realistically distributed. But instead of the classic binary fraud setup — fraud or not fraud — the competition asked a harder and more operationally honest question:

> **Not whether a transaction is fraudulent. How urgently does it need to be investigated?**

### The Four Urgency Classes

| Class | Label | Meaning | Real-world action |
|:---:|:---|:---|:---|
| **0** | ✅ Legitimate | No action needed | Ignore — transaction is clean |
| **1** | 👀 Monitor | Low-risk suspicious activity | Flag for passive monitoring |
| **2** | 🔍 Review | Likely fraud | Queue for analyst review |
| **3** | 🚨 Immediate Action | High-risk, confirmed-pattern fraud | Escalate now — every minute costs money |

### The Class Distribution — Why This Is Hard

| Class | Count | % of dataset | What this means for ML |
|:---:|:---:|:---:|:---|
| 0 — Legitimate | 6,237,903 | **99.89%** | A model predicting "0" always achieves 99.89% accuracy |
| 1 — Monitor | 2,176 | 0.035% | Needle in a haystack |
| 2 — Review | 2,151 | 0.034% | Smaller needle, different haystack |
| 3 — Immediate | 2,244 | 0.036% | The needle that matters most |

This is not a balanced four-class problem. It is a needle-in-a-haystack problem where the needles are sorted into three different sizes and you have to tell them apart.

<p align="center">
  <img src="docs/images/chart1_class_distribution.png" width="72%" alt="Class distribution — 99.89% legitimate vs 0.11% fraud across three urgency levels" />
</p>

> The chart above makes the imbalance visceral. The legitimate class bar is not slightly taller — it is orders of magnitude larger. Any model that does not explicitly handle this will simply learn to predict "legitimate" for everything and call it done.

### Why the Metric Was Macro F1 — Not Accuracy

| Metric | Model predicts "Class 0" for everything | What it actually measures |
|:---|:---:|:---|
| **Raw Accuracy** | ✅ 99.89% — looks great | Rewards majority class |
| **Macro F1** | ❌ ~0.25 — exposes the failure | Equal weight to every class, regardless of frequency |

Macro F1 computes F1 independently for each class, then averages. A model that perfectly predicts class 0 and completely misses classes 1–3 scores 0.25. The metric forces you to solve the hard problem, not the easy one.

<br>

---

## The Pipeline Journey — What We Built and In What Order

```
  Phase 1              Phase 2               Phase 3              Phase 4
────────────       ──────────────────    ──────────────────    ──────────────────
 Understand          Engineer             Train with             Ensemble
 the Data            Features             Honest CV              Predictions

 EDA on              10 raw columns   →   SMOTE inside      →   Majority vote
 6.2M rows           27 engineered        each fold             across 5 folds
                     features             (not before)
 Find the            Balance errors,      StratifiedKFold       Stable, leakage-
 accounting          ratios, cyclic       preserves class       free final preds
 fingerprints        time encoding        distribution
────────────       ──────────────────    ──────────────────    ──────────────────
  1–2 days            1 day                ~2 days               half a day
```

<br>

---

## Chapter 1: Understanding the Data Before Touching the Model

Before any model was considered, we spent time sitting with the raw data — the slow, unglamorous part of data science that does not appear in tutorials but determines everything downstream.

### The Raw Columns

| Column | Type | What it is |
|:---|:---:|:---|
| `step` | int | Time unit — each step = 1 hour of simulation |
| `type` | str | Transaction type: CASH\_IN, CASH\_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | float | Transaction value in local currency |
| `oldbalanceOrg` | float | Sender's balance **before** the transaction |
| `newbalanceOrig` | float | Sender's balance **after** the transaction |
| `oldbalanceDest` | float | Receiver's balance **before** the transaction |
| `newbalanceDest` | float | Receiver's balance **after** the transaction |
| `nameOrig` | str | Anonymized sender account ID |
| `nameDest` | str | Anonymized receiver account ID |
| `urgency_level` | int | 🎯 Target variable — 0, 1, 2, or 3 |

### The Three Structural Insights from EDA

These are not things you find by running `.describe()`. They come from asking *"what does fraud actually look like in this data?"*

---

#### Insight 1 — The Accounting Fingerprint

In any legitimate transaction, the math must balance perfectly:

```
Sender:    oldbalanceOrg  - amount  = newbalanceOrig    →  difference should be 0
Receiver:  oldbalanceDest + amount  = newbalanceDest    →  difference should be 0
```

When someone manipulates a transaction record, money appears to vanish or materialise. The balance does not add up. That discrepancy — `oldbalanceOrg - amount - newbalanceOrig` — is the **mathematical fingerprint of fraud**, hiding in plain sight inside the four balance columns.

| Transaction type | balance\_error\_orig | What it signals |
|:---|:---:|:---|
| Legitimate | 0.00 | Math balances — nothing manipulated |
| Fraudulent (record edited) | ≠ 0.00 | Money moved that the records do not account for |
| Fraudulent (account drained) | large positive | Sender lost more than the transaction shows |

> **This single insight — compute whether the accounting balances — produced the strongest features in the entire model.**

---

#### Insight 2 — The Account Name Signal

The anonymized `nameOrig` / `nameDest` identifiers were not purely random. Accounts starting with `"M"` were merchant accounts. Merchant accounts behave systematically differently from personal ones: higher transaction frequency, larger amounts, different inflow/outflow ratios. A binary merchant flag extracted from a text prefix carried real predictive signal.

---

#### Insight 3 — Time Is Circular, Not Linear

The `step` column is a raw integer. But fraud patterns cluster at specific times of day — late-night transfers, early-morning cash-outs. Using `step` raw means the model thinks hour 23 and hour 0 are 23 units apart. They are actually 1 hour apart.

```
Raw encoding:   hour 23 → 23,   hour 0 → 0      distance = 23   ❌ Wrong
Cyclic encoding: hour → (sin, cos) on unit circle  distance ≈ 0.26 ✅ Correct
```

<br>

---

## Chapter 2: Feature Engineering — From 10 Columns to 27

The `engineer_features()` function translates EDA understanding into model inputs. Every feature below is grounded in a specific mechanism — not chosen because it was available, but because it does something.

### The 17 Engineered Features

| Feature | Formula | What it captures | Why it matters |
|:---|:---|:---|:---|
| `orig_balance_delta` | `newbalanceOrig - oldbalanceOrg` | Raw change in sender balance | Magnitude of money leaving |
| `dest_balance_delta` | `newbalanceDest - oldbalanceDest` | Raw change in receiver balance | Magnitude of money arriving |
| `orig_balance_ratio` | `newbalanceOrig / (oldbalanceOrg + 1)` | Proportion of sender balance remaining | A 90% drain is categorically different from a 5% one |
| `dest_balance_ratio` | `newbalanceDest / (oldbalanceDest + 1)` | Proportion of receiver balance change | Normalises for account size |
| `balance_error_orig` | `oldbalanceOrg - amount - newbalanceOrig` | **Accounting discrepancy — sender side** | 🔑 Strongest fraud signal — should be 0 in legitimate txns |
| `balance_error_dest` | `oldbalanceDest + amount - newbalanceDest` | **Accounting discrepancy — receiver side** | 🔑 Second strongest signal — non-zero = record manipulation |
| `amount_to_orig` | `amount / (oldbalanceOrg + 1)` | Transaction as fraction of sender balance | 95% of balance moved = high risk |
| `amount_to_dest` | `amount / (oldbalanceDest + 1)` | Transaction relative to receiver balance | Sudden large inflow to small account |
| `log_amount` | `np.log1p(amount)` | Log-transformed transaction amount | Compresses right-skewed distribution for better tree splits |
| `is_merchant_orig` | `nameOrig.startswith("M")` | Sender is a merchant account | Merchant behaviour patterns differ systematically |
| `is_merchant_dest` | `nameDest.startswith("M")` | Receiver is a merchant account | Fraud routing to merchant accounts = different risk profile |
| `orig_zero_before` | `oldbalanceOrg == 0` | Sender had zero balance before | Accounts opened for a single fraudulent transfer |
| `orig_zero_after` | `newbalanceOrig == 0` | Sender drained to zero | Account fully emptied — strong fraud signal |
| `dest_zero_before` | `oldbalanceDest == 0` | Receiver had no prior balance | Freshly opened receiving account = money mule pattern |
| `hour` | `step % 24` | Hour of day (raw, 0–23) | Intermediate — used only to compute cyclic features |
| `hour_sin` | `sin(2π × hour / 24)` | Cyclic hour encoding — sine component | Lets model see midnight and 11pm as adjacent |
| `hour_cos` | `cos(2π × hour / 24)` | Cyclic hour encoding — cosine component | Together with hour\_sin, fully encodes circular time |

### Feature Importance — What the Model Actually Used Most

The balance error features dominated. In every fold, across all five cross-validation splits, the same pattern held:

```
Feature importance ranking (permutation-based, consistent across all 5 folds):

  1. balance_error_orig     ████████████████████  Most important
  2. balance_error_dest     ██████████████████
  3. amount_to_orig         ████████████
  4. orig_balance_ratio     ██████████
  5. orig_zero_after        ████████
  6. dest_zero_before       ███████
  7. log_amount             ██████
  8. hour_sin / hour_cos    █████
  ...
```

> **Two lines of arithmetic — does the accounting balance? — contributed more to model performance than any hyperparameter tuning could have.**

<p align="center">
  <img src="docs/images/chart2_feature_importances.png" width="80%" alt="Feature importances — top 15 features averaged across 5 folds by mean gain" />
</p>

The chart above shows the top 15 features by mean gain, averaged across all five cross-validation folds. The balance error features (`balance_error_orig`, `balance_error_dest`) and the balance ratio features lead decisively. Everything else — the merchant flags, the zero-balance indicators, the cyclic time encoding — contributes, but the gap between the top features and the rest tells you where the real discriminating power lives. The insight from EDA translated directly into model gain.

<br>

---

## Chapter 3: Why XGBoost — The Model Selection Decision

With features defined, the model selection was clear. Here is how we thought through it:

| Model | Handles class imbalance | Captures feature interactions | Speed on 6.2M rows | Interpretable | Our verdict |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | ❌ Struggles | ❌ Linear only | ✅ Fast | ✅ Yes | ❌ Too simple for interaction signals |
| **Random Forest** | ⚠️ With weights | ✅ Yes | ❌ Slow at scale | ⚠️ Partial | ❌ Slower than XGBoost, no sequential correction |
| **LightGBM** | ✅ Yes | ✅ Yes | ✅ Very fast | ⚠️ Partial | ✅ Strong candidate — close second |
| **Neural Network** | ⚠️ With techniques | ✅ Yes | ❌ Very slow on tabular | ❌ No | ❌ Tabular data rarely benefits from NN complexity |
| **XGBoost** | ✅ Yes | ✅ Yes | ✅ `hist` method | ⚠️ Partial | ✅ **Chosen** |

**Why XGBoost over LightGBM:** Both were viable. XGBoost's `hist` method on 6.2M rows performed comparably to LightGBM in practice, and the team had stronger existing familiarity with XGBoost's hyperparameter surface. In production, LightGBM would be worth benchmarking.

**Why tree ensemble over neural net:** Fraud signals here are combinations of specific feature thresholds — *"balance error > X AND amount ratio > Y AND dest zero before"* — exactly the kind of interaction a decision tree captures directly. Neural nets can model this, but require far more data and tuning to match what gradient boosting does natively on tabular data.

### The Hyperparameter Decisions

```python
xgb_params = dict(
    n_estimators     = 500,   # enough rounds to converge — more would overfit
    max_depth        = 8,     # deep enough for 4-5 feature interactions
    learning_rate    = 0.1,   # standard shrinkage — prevents any tree dominating
    subsample        = 0.8,   # 80% rows per tree — regularisation via randomness
    colsample_bytree = 0.8,   # 80% features per tree — same principle
    min_child_weight = 5,     # prevents leaves on tiny sample sets (critical for imbalanced data)
    gamma            = 1,     # minimum loss reduction to make a split — prunes noise
    reg_alpha        = 0.5,   # L1 regularisation on leaf weights
    reg_lambda       = 1.0,   # L2 regularisation on leaf weights
    tree_method      = "hist",# ← CRITICAL: bins features instead of exact splits
                              #   O(bins) vs O(n) per feature — makes 6.2M rows tractable
    objective        = "multi:softmax",
    num_class        = 4,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)
```

`tree_method="hist"` deserves emphasis. Without it, training on 6.2M rows would evaluate every possible split value for every feature at every node — that is O(n) per feature. The histogram method bins feature values and evaluates splits only at bin boundaries: O(bins) per feature. On this dataset, the difference is hours vs days of runtime.

<br>

---

## Chapter 4: The Training Strategy — Why SMOTE *Inside* Folds Changes Everything

This was the hardest technical decision in the entire project. Not the model. Not the features. **Where in the pipeline to apply SMOTE.**

### What SMOTE Does

Instead of duplicating minority examples (adds no new information), SMOTE generates *synthetic* minority samples by interpolating between real ones:

```
1. Select a real minority class example
2. Find its k nearest neighbours (among other minority examples)
3. Generate a new point on the line segment between the original and a neighbour
4. Repeat until all classes are balanced
```

For this dataset: ~2,000 examples of each fraud class vs 6.2M class-0 examples. Without SMOTE, the model predicts class 0 for everything.

### The Critical Question: When Do You Apply SMOTE?

| Approach | What happens | Evaluation result | Trust it? |
|:---|:---|:---:|:---:|
| ❌ **SMOTE before split** | Synthetic samples derived from real training points appear in the validation set | Inflated Macro F1 — looks great in notebook | ❌ Never — this is data leakage |
| ✅ **SMOTE inside each fold** | Only applied to the training portion — validation set stays clean | Honest Macro F1 — reflects real-world performance | ✅ Yes |

**Why SMOTE before splitting is data leakage:** The synthetic samples are generated by interpolating between real training points. If those synthetic points end up in the validation set, the model is evaluated on data mathematically derived from its own training data. Validation scores become optimistic fantasies rather than honest estimates.

### The Implementation — SMOTE Inside the Loop

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr,  X_val = X[train_idx], X[val_idx]
    y_tr,  y_val = y[train_idx], y[val_idx]

    # ✅ SMOTE applied ONLY to X_tr — X_val is never touched
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    # After SMOTE: training fold goes from 99.89% class-0 → balanced across all 4 classes
    # Validation fold: stays at real-world distribution — gives honest F1 estimates

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)
    test_preds[:, fold-1] = model.predict(X_test)
```

### Why StratifiedKFold, Not Regular KFold

With 2,000-odd fraud examples spread across 6.2M rows, a naive random split could assign almost all fraud examples to training folds — leaving validation folds too sparse to evaluate performance reliably. Stratification guarantees each fold has approximately the same class proportion as the full dataset.

| | Regular KFold | StratifiedKFold |
|:---|:---:|:---:|
| Class 3 examples per validation fold | Random — could be < 100 | ~449 — guaranteed proportional |
| Evaluation reliability | ❌ Fold scores vary wildly | ✅ Consistent, trustworthy per-fold F1 |
| Use on imbalanced data | ❌ Not recommended | ✅ Essential |

### The Per-Fold Results

<p align="center">
  <img src="docs/images/chart4_fold_f1_scores.png" width="80%" alt="Fold F1 scores — 5-fold stratified CV, macro average, with mean and OOF lines" />
</p>

The fold-by-fold results tell an important story: the scores are consistent. Fold 1 is slightly higher (0.6267) and Fold 5 slightly lower (0.6042), but the range is narrow — about 2.3 percentage points across all five folds. The mean fold Macro F1 (0.6122, blue line) and the overall OOF Macro F1 (0.6123, red line) are nearly identical, which means the cross-validation estimate is stable and the individual fold scores are not random noise.

A model that showed wildly different F1 scores across folds would be a red flag — it would mean the model is sensitive to which data it sees, and the OOF estimate would not be trustworthy. Tight fold scores mean the model is learning consistent patterns, not memorising the specific examples in each training fold.

<br>

---

## Chapter 5: Ensemble via Majority Voting

After all five folds complete, each fold's model has made predictions on the test set. Using one fold's predictions as the final answer is arbitrary. Using all five is better.

```python
# test_preds shape: (n_test_samples, 5) — one column per fold model
final_preds = mode(test_preds, axis=1).mode.flatten().astype(int)
```

**Why majority voting works:**

| Property | Effect |
|:---|:---|
| Each fold model trained on a different data subset | Learns slightly different patterns, makes different errors |
| 3-of-5 agreement required for a prediction | Smooths out any single model's noise or overfitting |
| Five independent predictions aggregated | More stable and reliable than any individual fold |
| No additional training required | Zero extra cost — the fold models already exist |

The result is test predictions that are more stable, more reliable, and generally more accurate than any individual fold model. Majority voting is not complicated. It rewards the extra implementation effort every time.

<br>

---

## Chapter 6: What the Pipeline Finds

### Per-Class Performance

The classification report breaks the story down class by class, and the breakdown is telling:

| Class | What drives performance | Hardest challenge |
|:---:|:---|:---|
| **0 — Legitimate** | Overwhelmingly common — easy to get right | Preventing false positives that misclassify legitimate txns |
| **1 — Monitor** | Weakest class — smallest accounting irregularities | Hard to distinguish from noisy-but-legitimate transactions |
| **2 — Review** | Zero-balance flags and merchant indicators help | Distinguishing from Class 1 — the margin is thin |
| **3 — Immediate** | Balance error features dominant | ✅ Strongest performance — large accounting discrepancies |

### The Confusion Matrix

<p align="center">
  <img src="docs/images/chart3_confusion_matrix.png" width="72%" alt="Confusion matrix — out-of-fold predictions showing true vs predicted urgency class" />
</p>

The confusion matrix shows the out-of-fold predictions across all five folds. The diagonal — true positives — is where the model gets things right. The off-diagonal cells are where it errs, and the pattern of those errors is as informative as the accuracy numbers.

The most instructive cell is the false negatives on the minority class: cases the model predicts as lower urgency than they actually are. These are the operationally costly errors — the transactions that deserved immediate escalation but got routed to monitoring instead. The confusion matrix makes visible what aggregate F1 scores obscure: not all errors are equal, and the direction of error matters as much as the frequency.

> **Class 1 is the hardest.** Lower urgency means the transaction looks more like a legitimate one. The smaller the fraud signal, the harder it is to separate from noise. No amount of feature engineering fully resolves this — it is a fundamental property of the problem.

### What This Delivers Operationally

```
Before this system:                 After this system:
──────────────────                  ──────────────────
400 transactions                    🚨 Class 3 (3 cases)   ← analyst sees these first
all labelled "suspicious"           🔍 Class 2 (28 cases)  ← senior analyst queue
no urgency signal                   👀 Class 1 (71 cases)  ← passive monitoring
                                    ✅ Class 0 (298 cases) ← cleared

Transaction 312 is buried.          Transaction 312 is at the top.
```

The analyst at 11pm does not work through 400 undifferentiated alerts. She sees three Class 3 cases immediately, handles the Class 2 queue next, and lets Class 1 monitor passively. No urgent case is buried in a flat list because the system had no vocabulary for urgency.

<br>

---

## The Lessons That Outlast the Competition

| # | Lesson | What we learned the hard way |
|:---:|:---|:---|
| 1 | **Feature engineering beats hyperparameter tuning** | Two lines of arithmetic — does the accounting balance? — outperformed any parameter search. Domain understanding is the most leveraged skill in tabular ML. |
| 2 | **The metric shapes everything** | Choosing Macro F1 over accuracy is not a technicality — it is a statement about what you care about. Accuracy rewards the majority class. Macro F1 forces you to solve the hard ones. |
| 3 | **SMOTE placement is intellectual honesty** | SMOTE before splitting produces publishable-looking results that do not hold up in production. Keeping the validation set clean — never letting synthetic data contaminate it — is the difference between results you can trust and results that only exist in your notebook. |
| 4 | **Ensemble methods are worth the extra hour** | Majority voting across five fold models is not complicated. It consistently outperforms any individual fold. The implementation cost is low; the reliability benefit is real. |
| 5 | **Understand the data before touching the model** | The balance error insight came from EDA — from asking "what does fraud actually look like here?" not from trying models. The slow phase at the start determined the fast results at the end. |

<br>

---

## Running It Yourself

```bash
git clone https://github.com/Sahibjeetpalsingh/Fraud-Detection.git
cd Fraud-Detection
pip install -r requirements.txt

# Download train.csv and test.csv from:
# https://kaggle.com/competitions/fraud-hack-ml-2026

python main.py
```

Running `main.py` will:

| Step | What happens |
|:---:|:---|
| 1 | Load both CSVs into memory |
| 2 | Engineer all 17 derived features → 27-column feature matrix |
| 3 | Run stratified 5-fold CV with SMOTE inside each fold |
| 4 | Train an XGBClassifier per fold |
| 5 | Print per-fold Macro F1 scores to console |
| 6 | Print full OOF classification report broken down by urgency class |
| 7 | Write `submission.csv` formatted for Kaggle submission |

**Expected runtime:** 30–90 minutes depending on hardware (8GB RAM, modern CPU).

<br>

---

## Project Structure

```
Fraud-Detection/
├── main.py              # Complete pipeline — single file
├── requirements.txt     # pandas, numpy, scikit-learn, imbalanced-learn, xgboost, scipy
├── README.md            # This document
├── .gitignore           # Excludes train.csv, test.csv, submission.csv
├── docs/
│   └── images/
│       ├── chart1_class_distribution.png   # Class imbalance visualisation
│       ├── chart2_feature_importances.png  # Top 15 features by mean gain
│       ├── chart3_confusion_matrix.png     # OOF confusion matrix
│       └── chart4_fold_f1_scores.png       # Per-fold Macro F1 with mean/OOF lines
├── [train.csv]          # Download from Kaggle — not in repo
└── [test.csv]           # Download from Kaggle — not in repo
```

<br>

---

## The Tech Stack

| Layer | Technology | Why this, not something else |
|:---|:---|:---|
| **Model** | XGBoost (`hist` method) | Gradient boosting handles feature interactions natively; `hist` makes 6.2M rows tractable |
| **Imbalance handling** | SMOTE (imbalanced-learn) | Generates synthetic minority samples — more informative than simple duplication |
| **Cross-validation** | StratifiedKFold (sklearn) | Preserves class proportions across folds — essential for 0.11% fraud rate |
| **Ensemble** | `scipy.stats.mode` | Majority vote across 5 fold models — stable, zero extra training cost |
| **Feature engineering** | pandas + numpy | Arithmetic operations at scale — balance errors, ratios, cyclic encoding |
| **Evaluation** | Macro F1 (sklearn) | Equal weight per class — forces the model to solve the hard classes |

<br>

---

## Team

Built at HackML 2026 by **Sahibjeet Pal Singh**, **Goutham Gopakumar**, **Bhuvesh Chauhan**, and **Louis Zhong** — four data science students who spent a competition weekend learning that the hardest part of machine learning is not the model. It is understanding what the data is actually telling you.

<br>

---

<div align="center">

*HackML 2026 · SFU Data Science Student Society · Simon Fraser University*

</div>
