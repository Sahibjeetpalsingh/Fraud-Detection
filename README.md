# 🏆 HackML 2026 — Fraud Detection  
### Multi-Class Classification | SFU DSSS Kaggle Competition

> **Team Competition** — Simon Fraser University, January 2026  
> **Metric:** Macro F1-Score &nbsp;|&nbsp; **Result:** see [Results](#-results) below

---

## 📌 Problem Statement

Financial institutions process millions of transactions daily, with only a tiny fraction being fraudulent. Rather than a simple binary *"Is this fraud?"*, this competition asks: **how urgently should a transaction be investigated?**

We built a **multi-class classifier** to assign each transaction one of four urgency levels:

| Label | Level | Business Context |
|:-----:|-------|------------------|
| **0** | No Action | Transaction appears legitimate |
| **1** | Monitor | Low-risk suspicious activity |
| **2** | Review | Likely fraud requiring analyst review |
| **3** | Immediate Action | High-risk fraud requiring urgent response |

### Key Challenge — Extreme Class Imbalance

| Class | Count | % of Total |
|:-----:|------:|-----------:|
| 0 | 6,237,903 | 99.89% |
| 1 | 2,176 | 0.035% |
| 2 | 2,151 | 0.034% |
| 3 | 2,244 | 0.036% |

Classes 1–3 together represent **< 0.11%** of the data — a classic real-world fraud distribution.

---

## 🔧 Approach

### 1. Feature Engineering

We derived **17 features** from the raw transaction data:

| Category | Features |
|----------|----------|
| **Balance Deltas** | `orig_balance_delta`, `dest_balance_delta` |
| **Balance Ratios** | `orig_balance_ratio`, `dest_balance_ratio` |
| **Balance Errors** | `balance_error_orig`, `balance_error_dest` — discrepancies between expected and actual post-transaction balances (strong fraud signals) |
| **Amount Ratios** | `amount_to_orig`, `amount_to_dest` |
| **Account Flags** | `is_merchant_orig`, `is_merchant_dest`, `orig_zero_before`, `orig_zero_after`, `dest_zero_before` |
| **Transformations** | `log_amount` |
| **Temporal** | `hour`, `hour_sin`, `hour_cos` — cyclic encoding of the 24-hour clock |

### 2. Handling Imbalance — SMOTE

We applied **SMOTE** (Synthetic Minority Oversampling Technique) on each training fold to synthetically balance the minority classes before training.

### 3. Model — XGBoost

- **Algorithm:** `XGBClassifier` with histogram-based tree method  
- **Validation:** 5-Fold Stratified Cross-Validation  
- **Inference:** Majority vote across all 5 fold models  

Key hyperparameters:

```
n_estimators    = 500
max_depth       = 8
learning_rate   = 0.1
subsample       = 0.8
colsample_bytree= 0.8
min_child_weight= 5
gamma           = 1
```

---

## 📊 Results

| Metric | Score |
|--------|------:|
| **Overall OOF Macro-F1** | *Run `main.py` to get score* |
| **Mean Fold Macro-F1** | *Run `main.py` to get score* |

> After running the pipeline, the classification report and per-fold scores are printed to the console.

---

## 📁 Repository Structure

```
hackathon10jan/
├── main.py            # Full ML pipeline (feature eng → train → predict)
├── requirements.txt   # Python dependencies
├── .gitignore         # Excludes data files & standard Python artifacts
├── README.md          # This file
├── train.csv          # Training data (6.2M rows) — not committed
├── test.csv           # Test data (118K rows) — not committed
└── submission.csv     # Generated predictions — not committed
```

> **Note:** CSV files are excluded from the repo via `.gitignore`. Download the data from the [Kaggle competition page](https://kaggle.com/competitions/fraud-hack-ml-2026).

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/hackml2026-fraud-detection.git
cd hackml2026-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place train.csv and test.csv in the project root

# 4. Run the pipeline
python main.py
```

The script will:
1. Load & engineer features  
2. Train 5-fold XGBoost with SMOTE  
3. Print per-fold and overall Macro F1-scores  
4. Save `submission.csv` for Kaggle upload  

---

## 📚 Dataset Description

Each row is a single transaction from a simulated payment system.

| Column | Description |
|--------|-------------|
| `step` | Time unit (1 step = 1 hour) |
| `type` | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount in local currency |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `nameOrig` | Anonymized sender identifier |
| `nameDest` | Anonymized receiver identifier |
| `urgency_level` | **Target** — investigation urgency (0–3) |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **XGBoost** — gradient boosted trees  
- **imbalanced-learn** — SMOTE oversampling  
- **scikit-learn** — evaluation, CV, preprocessing  
- **pandas / NumPy** — data wrangling  

---

## 📝 Citation

```
daniel06smith and StanleyS. FRAUD | HackML 2026.
https://kaggle.com/competitions/fraud-hack-ml-2026, 2026. Kaggle.
```

---

## 📜 License

This project was created for the **HackML 2026** competition hosted by **SFU DSSS**.
