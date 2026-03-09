<div align="center">

# 🛡️ Fraud Detection — Multi-Class Classification

### 🏆 HackML 2026 | SFU Data Science Student Society | Kaggle Competition

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/competitions/fraud-hack-ml-2026)

*Can we predict how urgently a suspicious transaction needs to be investigated?*

</div>

---

## The Story Behind This Project

Every single day, banks and financial platforms process hundreds of millions of transactions around the world. The overwhelming majority of those are completely normal — someone paying rent, buying groceries, or transferring money to a friend. But buried inside that enormous flood of activity is a small number of fraudulent transactions, and catching them before they cause real financial damage is one of the hardest problems in data science.

The traditional framing of fraud detection is a binary one: fraud or not fraud. But that is not how fraud response actually works in practice. A fraud analyst sitting at a desk cannot investigate every suspicious flag equally. Some cases demand an immediate phone call to the account holder right now. Others just need to be logged and watched for patterns over the next few days. Others might warrant a quiet automatic review. The urgency of the response is everything, and getting it wrong either wastes analyst time on low-risk cases or misses the ones that are genuinely on fire.
taset contained 6,244,474 real transaction records. Of those, only about 0.11% were any kind of fraud at all. That extreme imbalance was the defining challenge of the entire project.

---

## What This Project Tried to Solve

The financial industry loses billions of dollars annually to fraud, but the human cost of bad fraud detection is just as significant. Overly aggressive systems freeze legitimate accounts and frustrate innocent customers. Under-sensitive systems let fraudsters drain accounts before anyone notices. And systems that only give a binary answer give fraud teams no way to prioritize — every alert feels equally urgent, which means in practice none of them feel urgent enough.

The goal here was to build a machine learning pipeline that could look at a raw transaction record — just ten columns of numbers and labels — and predict not only whether it was fraudulent but how dangerous it was. That prediction, if accurate, lets a fraud operations team triage their workload intelligently: automate the low-risk monitoring, route mid-level cases to junior analysts, and put the highest-urgency ones in front of senior investigators immediately.

The four classes this model had to distinguish were:

Class 0 meant the transaction was completely legitimate and no action was needed. Class 1 meant low-risk suspicious activity that should be monitored. Class 2 meant likely fraud that deserved analyst review. Class 3 meant high-risk fraud requiring an immediate response.

---

## Understanding the Data

Before any model was trained, a thorough exploratory data analysis was done on the raw dataset to understand what signals might distinguish fraudulent from legitimate transactions.

The raw dataset had ten columns. The `step` column represented time in hours. The `type` column was a categorical label for the kind of transaction: CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER. The `amount` column was the transaction value. Four balance columns tracked the sender's and receiver's account balances before and after the transaction. Two name columns contained anonymized account identifiers. And the `urgency_level` column was the prediction target.

The most important structural insight from the EDA was this: in a legitimate, well-functioning financial system, the accounting must balance perfectly. If someone sends 100 dollars from Account A to Account B, Account A's balance must drop by exactly 100 dollars and Account B's balance must rise by exactly 100 dollars. Any discrepancy between what the balances should be and what they actually are is a serious red flag. Real fraud in this dataset was characterized precisely by these accounting inconsistencies — money that disappeared or appeared in ways the transaction records couldn't explain.

The second observation was that names starting with the letter "M" indicated merchant accounts, which behave differently from personal accounts in terms of expected transaction patterns. This was extractable from the anonymized name strings.

The third observation was that fraud patterns might correlate with the time of day, but the raw `step` column was a linear integer that did not capture the circular nature of the 24-hour clock. Hour 23 and hour 0 are one hour apart in reality but 23 units apart as raw integers. This needed to be addressed with cyclic encoding.

**Class Distribution**

| Class | Label | Count | Percentage |
|---|---|---|---|
| 0 | No Action | 6,237,903 | 99.89% |
| 1 | Monitor | 2,176 | 0.035% |
| 2 | Review | 2,151 | 0.034% |
| 3 | Immediate Action | 2,244 | 0.036% |

The fraud classes combined represent less than 0.11% of all data. This is the core difficulty of the entire problem.

---

## How It Was Built — The Technical Architecture

### Feature Engineering: 17 Derived Features

The most impactful work in this project was not the model tuning — it was the feature engineering. Seventeen new features were derived from the ten raw columns, each designed to capture a specific fraud signal.

**Balance Delta Features** (`orig_balance_delta`, `dest_balance_delta`) measured how much each party's balance actually changed during the transaction. These gave the model a direct view into whether the money moved as expected.

**Balance Ratio Features** (`orig_balance_ratio`, `dest_balance_ratio`) expressed the balance change as a proportion of the starting balance. A $1,000 movement means something very different in a $1,200 account versus a $1,200,000 account. Normalization here was essential.

**Balance Error Features** (`balance_error_orig`, `balance_error_dest`) were the single most important features. These were computed as `oldbalanceOrg - amount - newbalanceOrig` for the sender and `oldbalanceDest + amount - newbalanceDest` for the receiver. In any legitimate transaction these values should be exactly zero. Non-zero values indicate accounting discrepancies — the signature of fraud manipulation.

**Amount Ratio Features** (`amount_to_orig`, `amount_to_dest`) measured how large the transaction was relative to each account's balance, flagging transactions that were disproportionately large for the account involved.

**Account Flag Features** (`is_merchant_orig`, `is_merchant_dest`, `orig_zero_before`, `orig_zero_after`, `dest_zero_before`) captured account type (merchant vs personal) and zero-balance states, both common in fraud patterns where accounts are drained to zero or opened with zero balance specifically for a fraudulent transaction.

**Log Amount** (`log_amount`) applied a log1p transformation to the transaction amount, reducing the heavy right skew in the distribution and making the feature more useful for the tree-based model.

**Cyclic Time Features** (`hour`, `hour_sin`, `hour_cos`) extracted the hour-of-day from the raw step counter using modulo 24, then encoded it cyclically using sine and cosine. This preserved the circular structure of time so that the model understood that 11pm and midnight are close together.

### Model: XGBoost with 5-Fold Stratified Cross-Validation + SMOTE

XGBoost was chosen as the model for several reasons. It handles high-dimensional tabular data extremely well. Its gradient boosting framework builds trees sequentially, each one correcting the errors of the last. It has built-in L1 and L2 regularization to prevent overfitting. And its histogram-based tree method (`tree_method="hist"`) makes training feasible on datasets of 6+ million rows without excessive memory usage.

The training strategy was 5-fold stratified cross-validation. Stratified folds were critical here — with only 0.11% fraud samples, a random split might place almost no fraud cases in some folds. Stratification guaranteed that each fold had a representative proportion of all four classes.

The key methodological decision was where to apply SMOTE. SMOTE (Synthetic Minority Oversampling Technique) works by finding real minority-class samples, then generating new synthetic samples by interpolating between them and their k-nearest neighbors. The critical rule is that SMOTE must be applied only inside the training portion of each fold, never before the fold split. If SMOTE runs before splitting, synthetic versions of real training samples end up in the validation set, causing data leakage that makes validation scores look artificially better than they really are. In this pipeline, SMOTE was applied strictly within each training fold.

After SMOTE, each training fold went from heavily imbalanced (99.89% class 0) to balanced (roughly equal counts across all four classes), giving the model a fair chance to learn the patterns of the rare fraud classes.

For each of the five folds, the model trained and generated predictions on the held-out validation fold as well as the test set. After all five folds, the final test predictions were determined by majority voting across the five fold models using `scipy.stats.mode`. This ensemble approach smoothed out instability from any single fold.

**Hyperparameters**

| Parameter | Value | Reasoning |
|---|---|---|
| n_estimators | 500 | Enough boosting rounds to converge |
| max_depth | 8 | Deep enough to capture complex fraud patterns |
| learning_rate | 0.1 | Standard shrinkage to prevent overfitting |
| subsample | 0.8 | Row sampling per tree adds regularization |
| colsample_bytree | 0.8 | Feature sampling per tree adds regularization |
| min_child_weight | 5 | Prevents splits on very small node samples |
| gamma | 1 | Minimum loss reduction required to make a split |
| reg_alpha | 0.5 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |
| tree_method | hist | Fast histogram-based algorithm for large datasets |

### Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ | Core language |
| XGBoost 1.7+ | Gradient boosted tree classifier |
| imbalanced-learn | SMOTE oversampling |
| scikit-learn | Cross-validation, metrics, preprocessing |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| SciPy | Majority voting (mode) |

---

## Results and Findings

The evaluation metric for this competition was Macro F1-Score, which weights each class equally regardless of its frequency. This is the correct metric for this problem because it forces the model to perform well on the rare fraud classes, not just the dominant legitimate class. A naive classifier that predicts "No Action" for everything would achieve 99.89% accuracy but a Macro F1 of approximately 0.25 — completely useless.

To see the actual per-fold scores and the full classification report broken down by class, run the pipeline locally with `python main.py`. The results depend on the SMOTE random state and the specific fold splits, so running the full pipeline is the only way to see precise numbers.

**What Worked Well**

The balance error features proved to be the strongest fraud indicators in the entire feature set. The insight that legitimate systems should have perfect accounting — and that any discrepancy is a red flag — directly translated into the most predictive features. SMOTE effectively boosted minority class recall, giving the model enough examples of fraud patterns to learn from. The 5-fold stratified cross-validation provided robust evaluation that was not inflated by data leakage. And majority voting across five fold models stabilized the final test predictions.

**What Could Be Improved**

Hyperparameter tuning using Optuna or Bayesian optimization was not done due to the computational cost on 6.2M rows. Feature selection using SHAP importance scores could identify which of the 17 engineered features were actually carrying signal versus noise. Ensemble stacking with LightGBM and CatBoost as additional base models could push Macro F1 further. Undersampling the majority class as an alternative or complement to SMOTE is also worth exploring.

**Key Technical Lessons**

Class imbalance is the defining challenge of real-world fraud detection. A model with 99.89% accuracy can be completely worthless if that accuracy comes entirely from predicting the majority class. The evaluation metric must be chosen to reflect what actually matters — and in fraud, what matters is catching the rare cases.

SMOTE must go inside the cross-validation loop. Applying it before splitting is a subtle but serious form of data leakage that is easy to miss and will cause your validation results to be overoptimistic.

Cyclic encoding of time features is one of those details that gets glossed over in tutorials but matters in practice. Without it, the model treats midnight and 11pm as being 23 time units apart instead of 1.

Feature engineering consistently outperforms hyperparameter tuning. The balance error features were the single biggest driver of model performance, not the choice of regularization coefficients.

---

## Project Structure

```
Fraud-Detection/
├── main.py              # Full ML pipeline: load → engineer → train → predict
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .gitignore           # Excludes CSV files and Python artifacts
├── train.csv            # Training data (6.2M rows) — download from Kaggle
├── test.csv             # Test data (~118K rows) — download from Kaggle
└── submission.csv       # Generated predictions — created by running main.py
```

> Note: CSV files are excluded from the repo via .gitignore. Download the datasets from the Kaggle competition page linked below.
>
> ---
>
> ## Quick Start
>
> ```bash
> # 1. Clone the repository
> git clone https://github.com/Sahibjeetpalsingh/Fraud-Detection.git
> cd Fraud-Detection
>
> # 2. Install dependencies
> pip install -r requirements.txt
>
> # 3. Place train.csv and test.csv in the project root
> # Download from: https://kaggle.com/competitions/fraud-hack-ml-2026
>
> # 4. Run the full pipeline
> python main.py
> ```
>
> Running `main.py` will load the data, engineer all 17 features, run 5-fold stratified cross-validation with SMOTE inside each fold, print per-fold Macro F1 scores, print a full classification report broken down by urgency class, and save a `submission.csv` file ready for Kaggle upload.
>
> ---
>
> ## Team
>
> This project was built as a team for the HackML 2026 competition hosted by SFU DSSS (Data Science Student Society) at Simon Fraser University.
>
> **Sahibjeet Pal Singh · Goutham Gopakumar · Bhuvesh Chauhan · Louis Zhong**
>
> ---
>
> ## Citation
>
> ```
> @misc{hackml2026fraud,
>   title  = {FRAUD | HackML 2026},
>   author = {daniel06smith and StanleyS},
>   year   = {2026},
>   url    = {https://kaggle.com/competitions/fraud-hack-ml-2026},
>   note   = {Kaggle Competition}
> }
> ```
>
> ---
>
> *Built with dedication for HackML 2026 @ Simon Fraser University*
This project was built for HackML 2026, a Kaggle competition hosted by the SFU Data Science Student Society at Simon Fraser University. The competition asked exactly this harder, more realistic question: not "is this fraud?" but "how urgently does this transaction need to be investigated?" The task was to assign each suspicious transaction one of four urgency levels — No Action, Monitor, Review, or Immediate Action — and to do it accurately enough to be genuinely useful to a real fraud operations team.

The da
