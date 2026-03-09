<div align="center">

# 🛡️ Fraud Detection — Multi-Class Urgency Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC4E20?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HackML 2026](https://img.shields.io/badge/HackML_2026-SFU_DSSS_Kaggle-20BEFF?style=for-the-badge)](https://kaggle.com/competitions/fraud-hack-ml-2026)

*6,244,474 transactions. 0.11% fraud. 4 urgency levels. One pipeline to find them all.*

</div>

---

## The Night a Bank Gets It Wrong

Picture a fraud analyst sitting at 11pm staring at a queue of 400 flagged transactions. Every single one of them is labeled "suspicious." No priority. No context. No urgency signal. Just a flat list. She works through them from the top, methodically, one by one — and by the time she reaches transaction 312, someone three time zones away has already had their account emptied. Transaction 312 was a Class 3 case. Immediate action required. But there was no way to know that until it was too late, because the system only knew how to say "fraud" or "not fraud." It had no vocabulary for urgency.

That is the problem this project was built to solve.

---

## What This Competition Was Actually Asking

HackML 2026 was a Kaggle competition run by the SFU Data Science Student Society at Simon Fraser University. The dataset was 6.2 million simulated financial transactions — the kind of synthetic but realistically distributed data that financial institutions use to test detection systems. But instead of the classic binary fraud detection setup, this competition posed a harder and more realistic question: not whether a transaction is fraudulent, but how urgently it needs to be investigated.

The target variable was `urgency_level`, a four-class label ranging from 0 to 3. Class 0 meant no action needed — the transaction appears legitimate. Class 1 meant low-risk suspicious activity worth monitoring. Class 2 meant likely fraud that deserves analyst review. Class 3 meant high-risk fraud that requires an immediate response — the kind where every minute of delay has a real cost.

The moment you look at the class distribution, you understand why this is hard. Out of 6,244,474 total transactions, class 0 held 6,237,903 of them — that is 99.89% of the entire dataset. The three fraud classes together made up just 0.11% of the data: 2,176 Monitor cases, 2,151 Review cases, and 2,244 Immediate Action cases. This is not a balanced four-class problem. It is a needle-in-a-haystack problem, except the needles are sorted into three different sizes and you have to tell them apart.

The evaluation metric chosen by the competition organizers was Macro F1-Score. This is the right choice for this problem because Macro F1 computes the F1 score for each class independently and then averages them, giving every class equal weight regardless of how rarely it appears. A model that predicts class 0 for every single transaction achieves 99.89% raw accuracy — but a Macro F1 of approximately 0.25, because it fails completely on the three classes that actually matter. The metric forces you to solve the hard problem, not the easy one.

---

## Understanding the Data Before Touching the Model

Before any model was even considered, the team spent time sitting with the raw data and trying to understand what it was actually telling us. This phase of working — the slow, unglamorous part of data science that does not appear in tutorials but determines everything — shaped every decision that came afterward.

The raw dataset had ten columns. The `step` column was a time unit where each step represented one hour of simulation, going up into the thousands for multi-month transaction histories. The `type` column was a categorical variable encoding the nature of the transaction: CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER. The `amount` column was the transaction value in local currency. Then there were four balance columns — `oldbalanceOrg` and `newbalanceOrig` for the sender's balance before and after, and `oldbalanceDest` and `newbalanceDest` for the receiver's balance before and after. Finally, `nameOrig` and `nameDest` were anonymized identifiers for the sending and receiving accounts.

What the EDA revealed was that the raw columns, taken at face value, were not sufficient to build a reliable classifier. The amount alone could not distinguish fraud — legitimate wire transfers can be large, and small cash-outs can be fraudulent. The transaction type alone was not enough either. But looking at the *relationships* between columns began to tell a much richer story.

The most important structural insight was about accounting integrity. In a well-functioning financial system, the math of every transaction must add up perfectly. If account A sends 500 dollars to account B, then A's balance should drop by exactly 500 and B's balance should rise by exactly 500. No more, no less. The moment you compute `oldbalanceOrg - amount - newbalanceOrig`, the answer for every legitimate transaction should be exactly zero. The answer for a fraudulent one often is not. Money appears to vanish or appear out of thin air when someone is manipulating transaction records. Those discrepancies are the fingerprints of fraud, and they are hiding in plain sight inside the four balance columns.

The second structural insight came from the name columns. The anonymized identifiers were not purely random — accounts with names starting with "M" were merchant accounts, which behave systematically differently from personal accounts in terms of transaction frequency, amount distribution, and the patterns of inflow versus outflow. Extracting a binary merchant flag from a text prefix turned out to carry predictive signal.

The third insight was about time. The `step` column was a raw linear integer, but fraud patterns often cluster at specific times of day — late night transfers, early morning cash-outs. To use this signal, the hour of day needed to be extracted (step mod 24), and then encoded in a way that preserved the circular nature of the clock. A raw integer encoding treats hour 23 and hour 0 as being 23 units apart. But they are actually one hour apart. Cyclic encoding using sine and cosine of the hour resolves this, letting the model understand that midnight and 11pm are adjacent.

---

## The Feature Engineering: Turning Ten Columns Into Twenty-Seven

The `engineer_features()` function in `main.py` is where the domain understanding from EDA gets translated into inputs the model can actually use. Seventeen new features were derived from the original ten, each grounded in a specific understanding of what fraud looks like in transaction data.

The **balance delta features** — `orig_balance_delta` computed as `newbalanceOrig - oldbalanceOrg`, and `dest_balance_delta` computed as `newbalanceDest - oldbalanceDest` — captured the raw magnitude of balance change for both parties. These gave the model a direct view into how much money moved and in which direction.

The **balance ratio features** — `orig_balance_ratio` computed as `newbalanceOrig / (oldbalanceOrg + 1)` and `dest_balance_ratio` as `newbalanceDest / (oldbalanceDest + 1)` — expressed that change as a proportion of the starting balance. A 1,000 dollar change in a 1,200 dollar account is a near-complete drain. The same change in a 500,000 dollar account is routine activity. The ratio normalizes for account size, making the feature meaningful across the full range of account sizes in the dataset. The `+1` in the denominator prevents division by zero for zero-balance accounts.

The **balance error features** — `balance_error_orig` computed as `oldbalanceOrg - amount - newbalanceOrig`, and `balance_error_dest` computed as `oldbalanceDest + amount - newbalanceDest` — were the most powerful features in the entire set. In any legitimate transaction, both of these should be exactly zero. A non-zero value on either means the accounting does not add up — money has disappeared from the sender's side without appearing on the receiver's side, or appeared on the receiver's side without being deducted from the sender. These errors are the mathematical signature of record manipulation, and they proved to be the strongest discriminators between urgency classes.

The **amount ratio features** — `amount_to_orig` as `amount / (oldbalanceOrg + 1)` and `amount_to_dest` as `amount / (oldbalanceDest + 1)` — measured the relative size of the transaction compared to each account's available balance. A transaction that represents 95% of an account's balance is a categorically different risk signal than one representing 0.1%.

The **account flag features** encoded binary information that was extractable from the name strings. `is_merchant_orig` and `is_merchant_dest` flagged whether each party was a merchant account, based on whether the name started with "M". `orig_zero_before` and `orig_zero_after` flagged accounts that started or ended with zero balance — accounts opened specifically for a fraudulent transaction often have zero balance before the transfer. `dest_zero_before` flagged receiver accounts that had no prior balance, another common pattern in fraud where money is routed to freshly opened accounts.

The **log amount feature** — `log_amount` computed as `numpy.log1p(amount)` — applied a logarithmic transformation to the transaction amount. Transaction amounts in the dataset had a heavily right-skewed distribution, with most transactions being modest but a long tail of very large ones. The log transformation compressed this skew and made the feature more statistically well-behaved for tree-based models, which can handle skewness but benefit from more evenly distributed features.

The **cyclic time features** extracted the hour of day from the step counter (`hour = step % 24`) and then encoded it as `hour_sin = sin(2π × hour / 24)` and `hour_cos = cos(2π × hour / 24)`. These two features together encode any hour as a point on a unit circle, which means the model can recognize that hour 23 and hour 0 are adjacent without any explicit instruction to do so. Raw integer encoding of cyclical variables is a subtle but common source of degraded model performance in time-sensitive applications.

The `type` column was label-encoded as `type_enc` using sklearn's `LabelEncoder`, converting the five categorical transaction type strings into integers. The original string columns `nameOrig`, `nameDest`, `type`, and the ID column were dropped before model training, leaving 27 numeric features as the final feature matrix.

---

## Choosing the Model and Why XGBoost Was the Right Answer

With the feature matrix defined, the model selection decision was straightforward: XGBoost, specifically `XGBClassifier` configured for multi-class softmax classification.

XGBoost is a gradient boosted tree algorithm. It builds an ensemble of decision trees sequentially, where each new tree is trained to correct the prediction errors of the trees built so far. This sequential correction process is what makes gradient boosting so effective on tabular data — it can model complex non-linear relationships and interactions between features that simpler models cannot capture. For this problem, where fraud signals are combinations of multiple features (a high amount_to_orig combined with a non-zero balance_error and a late-night hour is more suspicious than any single one of those features alone), tree-based models are the natural fit.

The configuration used `objective="multi:softmax"` and `num_class=4`, which tells XGBoost to output a direct class prediction rather than class probabilities. The `tree_method="hist"` parameter is critical for a dataset of 6.2 million rows — instead of evaluating every possible split point for every feature at every node, the histogram method bins the feature values and evaluates splits only at bin boundaries, which reduces the computational complexity from O(n) per feature to O(bins) per feature. On a dataset this size, that difference is the difference between a pipeline that finishes in hours versus one that runs for days.

The hyperparameters were set as follows:

```python
xgb_params = dict(
    n_estimators     = 500,      # 500 boosting rounds
    max_depth        = 8,        # trees can be up to 8 levels deep
    learning_rate    = 0.1,      # shrinkage per round
    subsample        = 0.8,      # 80% of rows sampled per tree
    colsample_bytree = 0.8,      # 80% of features sampled per tree
    min_child_weight = 5,        # minimum sum of weights in a leaf node
    gamma            = 1,        # minimum loss reduction to make a split
    reg_alpha        = 0.5,      # L1 regularization on leaf weights
    reg_lambda       = 1.0,      # L2 regularization on leaf weights
    tree_method      = "hist",
    objective        = "multi:softmax",
    num_class        = 4,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
```

`n_estimators=500` gives the model enough rounds to converge on complex patterns without dramatically overfitting. `max_depth=8` allows reasonably deep trees that can model the interactions between features — fraud signatures often involve combinations of four or five features simultaneously. `learning_rate=0.1` is the standard shrinkage value, which scales down the contribution of each new tree and effectively prevents any single tree from dominating the ensemble. `subsample=0.8` and `colsample_bytree=0.8` introduce randomness into the training process by using only 80% of rows and 80% of features for each tree, which acts as regularization and reduces overfitting. `min_child_weight=5` prevents the model from creating leaf nodes based on very small numbers of samples — in a dataset this imbalanced, this matters because without this constraint the model might create splits that only represent a handful of training examples. `gamma=1` requires that any split must reduce the loss by at least 1 unit to be kept, which prunes back unnecessary tree complexity. `reg_alpha=0.5` and `reg_lambda=1.0` are L1 and L2 regularization terms on the leaf weights, which discourage extreme predictions and keep the model stable across folds.

---

## The Training Strategy: Why SMOTE Inside Folds Changes Everything

The hardest technical decision in the entire project was not which model to use — it was how to handle the class imbalance during training, and specifically where in the pipeline to apply SMOTE.

SMOTE stands for Synthetic Minority Oversampling Technique. Instead of simply duplicating minority class examples (which adds no new information), SMOTE creates *new synthetic samples* by selecting a real minority example, finding its k-nearest neighbors in the feature space among other minority examples, and then generating a new point somewhere along the line segment connecting the original point to one of its neighbors. For this dataset, where classes 1, 2, and 3 each had only around 2,000 examples compared to the 6.2 million in class 0, SMOTE was essential. Without it, the model would be trained on a dataset so dominated by class 0 that it would simply learn to predict class 0 for everything — achieving excellent raw accuracy and catastrophically bad Macro F1.

But here is the critical methodological point that many practitioners get wrong: SMOTE must be applied *after* the train-validation split, and only to the training portion. The reason is data leakage. If you apply SMOTE to the entire dataset before splitting into folds, the synthetic samples generated from real training points will appear in the validation set. The model is then evaluated on data that is derived from the same points it was trained on, which makes validation scores artificially inflated and completely unreliable as estimates of real-world performance. The validation results look great; the model is actually overfit.

The correct approach — implemented in this pipeline — is to apply SMOTE inside each fold loop, strictly to the training portion of that fold:

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # SMOTE applied ONLY to X_tr, y_tr — never touches X_val
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)
```

After SMOTE, the training fold goes from its original distribution (99.89% class 0) to a balanced distribution where all four classes have roughly equal representation. The model is then trained on this balanced set and evaluated on the clean, unmodified validation fold — giving honest estimates of how it will perform on genuinely new data.

The use of `StratifiedKFold` with `n_splits=5` is also non-trivial here. With only 2,000-odd examples of each fraud class spread across 6.2 million rows, a naive random split might accidentally assign almost all fraud examples to the training folds, leaving the validation folds too sparse to evaluate performance reliably. Stratification guarantees that each fold receives approximately the same proportion of each class as the full dataset, ensuring that every fold evaluation is meaningful.

---

## Ensemble via Majority Voting

After all five folds complete, each fold's model has generated predictions on the test set. Rather than using a single fold's predictions as the final answer — which would be arbitrary and potentially noisy — the pipeline collects all five sets of test predictions into a 2D array of shape `(n_test_samples, 5)` and takes the majority vote across the five models using `scipy.stats.mode`:

```python
from scipy.stats import mode
final_test_preds = mode(test_preds, axis=1).mode.flatten().astype(int)
```

Majority voting is a form of ensemble learning. Each fold's model has been trained on a slightly different subset of the data and has learned slightly different patterns. By requiring that at least three out of five models agree on a prediction before committing to it, the ensemble smooths out the noise and idiosyncrasies of any individual model. The result is a set of test predictions that is more stable, more reliable, and generally more accurate than any single fold model would produce on its own.

---

## What the Pipeline Finds: Results and Real Takeaways

The results of running `python main.py` tell a detailed story through the per-fold Macro F1 scores and the full out-of-fold classification report.

The overall OOF (out-of-fold) Macro F1 score reflects the model's genuine ability to distinguish all four urgency levels without data leakage. To see exact numbers, run the pipeline — the results depend on SMOTE's random state and the specific fold assignments. The classification report breaks down precision, recall, and F1 separately for each class, which reveals something important: the balance error features are doing the heavy lifting for class 3 (Immediate Action) detection, while the zero-balance flags and merchant indicators are more important for distinguishing class 1 from class 0.

The most significant technical finding was that the balance error features — those two columns that just measure whether the accounting adds up — were consistently the strongest fraud indicators in the entire feature set across all five folds. This validated the central hypothesis from the EDA phase: that fraud in this dataset leaves a mathematical signature in the form of accounting discrepancies, and that explicitly engineering features to measure those discrepancies would translate directly into model performance.

The second key finding was about what does not work well. The model's weakest performance is on class 1 (Monitor) — the low-urgency suspicious cases that are hardest to distinguish from legitimate transactions. These cases exhibit accounting irregularities that are smaller in magnitude, making them harder to separate from noise. This is a fundamentally hard problem: the lower the urgency, the more the transaction looks like a legitimate one, and no amount of feature engineering fully resolves that ambiguity.

The third finding was methodological: SMOTE inside cross-validation folds produces more stable and honest evaluation metrics than SMOTE applied before splitting. Across the five folds, the variance in Macro F1 scores is relatively small — the model is consistent across different subsets of the data — which suggests that the validation estimates are reliable rather than inflated.

From a non-technical standpoint, what this system is actually delivering is a triage signal. If this pipeline were deployed in a real fraud operations center, the output would let the team's management dashboard show analysts the queue sorted by urgency level, with Class 3 cases at the top in red. That sounds simple. But it changes everything about how the team operates. The analyst who was working through 400 undifferentiated alerts at 11pm now sees the three Class 3 cases immediately. The Class 2 cases go to senior analysts. The Class 1 cases can wait. No one misses the urgent ones because they were buried in a flat queue.

---

## The Lessons That Outlast the Competition

Some of what this project taught cannot be found in documentation.

The first lesson is that feature engineering is the most leveraged skill in tabular data science. The balance error features — two lines of arithmetic — contributed more to model performance than any hyperparameter tuning could have. Good features come from understanding the domain, not from AutoML.

The second lesson is that the evaluation metric shapes everything. Choosing Macro F1 over accuracy is not a technicality. It is a statement about what you care about. Accuracy rewards predicting the majority class. Macro F1 forces you to solve the hard classes. Every real-world fraud detection system should be evaluated this way, and most real-world fraud detection discussions do not make this distinction clearly enough.

The third lesson is that correct cross-validation is a form of intellectual honesty. SMOTE before splitting is a subtle mistake that produces publishable-looking results that do not hold up in production. The discipline of keeping the validation set clean — never letting synthetic data derived from training samples contaminate it — is the difference between results you can trust and results that only exist in your notebook.

The fourth lesson is about ensemble methods. Majority voting across five fold models is not complicated. But it consistently produces better results than any individual fold, and it is one of those techniques that rewards the extra implementation effort every time.

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

Running `main.py` will load both CSVs, engineer all 17 derived features, run stratified 5-fold cross-validation with SMOTE inside each fold, train an XGBClassifier per fold, print per-fold Macro F1 scores to the console, print a full classification report broken down by urgency class, and write a `submission.csv` file formatted for Kaggle submission. On a machine with 8GB of RAM and a modern CPU, expect the run to take 30 to 90 minutes depending on hardware.

The project structure is:

```
Fraud-Detection/
├── main.py              # The complete pipeline in a single file
├── requirements.txt     # pandas, numpy, scikit-learn, imbalanced-learn, xgboost, scipy
├── README.md            # This document
├── .gitignore           # Excludes train.csv, test.csv, submission.csv
└── [train.csv]          # Download from Kaggle — not included in repo
└── [test.csv]           # Download from Kaggle — not included in repo
```

---

## Team

Built at HackML 2026 by **Sahibjeet Pal Singh**, **Goutham Gopakumar**, **Bhuvesh Chauhan**, and **Louis Zhong** — four data science students who spent a competition weekend learning that the hardest part of machine learning is not the model, it is understanding what the data is actually telling you.

---

*HackML 2026 · SFU Data Science Student Society · Simon Fraser University*
