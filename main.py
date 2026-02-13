"""
HackML 2026 — Fraud Detection: Multi-Class Classification
==========================================================
Predicts the urgency_level (0–3) for each transaction using XGBoost
with SMOTE oversampling to handle extreme class imbalance.

Metric: Macro F1-Score
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "urgency_level"
ID_COL = "id"

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"\nClass distribution:\n{train[TARGET].value_counts().sort_index()}")

# ──────────────────────────────────────────────
# 2. Feature Engineering
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw transaction data."""
    df = df.copy()

    # Encode transaction type
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    # Balance-change features
    df["orig_balance_delta"]  = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_delta"]  = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_ratio"]  = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1)
    df["dest_balance_ratio"]  = df["newbalanceDest"] / (df["oldbalanceDest"] + 1)

    # Error-based features (discrepancies hint at fraud)
    df["balance_error_orig"]  = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["balance_error_dest"]  = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Amount-to-balance ratios
    df["amount_to_orig"]      = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest"]      = df["amount"] / (df["oldbalanceDest"] + 1)

    # Whether origin or destination is a merchant (name prefix)
    df["is_merchant_orig"]    = df["nameOrig"].str.startswith("M").astype(int)
    df["is_merchant_dest"]    = df["nameDest"].str.startswith("M").astype(int)

    # Zero-balance flags
    df["orig_zero_before"]    = (df["oldbalanceOrg"] == 0).astype(int)
    df["orig_zero_after"]     = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_zero_before"]    = (df["oldbalanceDest"] == 0).astype(int)

    # Log-transformed amount
    df["log_amount"]          = np.log1p(df["amount"])

    # Step-based cyclic features (24-hour cycle)
    df["hour"]                = df["step"] % 24
    df["hour_sin"]            = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]            = np.cos(2 * np.pi * df["hour"] / 24)

    return df


print("\nEngineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# Columns to drop before modelling
DROP_COLS = ["nameOrig", "nameDest", "type", ID_COL]

FEATURES = [c for c in train.columns if c not in DROP_COLS + [TARGET]]
print(f"Feature count: {len(FEATURES)}")

X = train[FEATURES].values
y = train[TARGET].values
X_test = test[FEATURES].values

# ──────────────────────────────────────────────
# 3. Model Training  (Stratified K-Fold + SMOTE)
# ──────────────────────────────────────────────
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X), dtype=int)
test_preds = np.zeros((len(X_test), N_FOLDS), dtype=int)
fold_scores = []

xgb_params = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    tree_method="hist",          # fast histogram-based method
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

print(f"\nTraining {N_FOLDS}-fold XGBoost with SMOTE...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Apply SMOTE to training fold only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr_res, y_tr_res,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Validation predictions
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    fold_f1 = f1_score(y_val, val_pred, average="macro")
    fold_scores.append(fold_f1)
    print(f"  Fold {fold}  Macro-F1 = {fold_f1:.5f}")

    # Test predictions (majority vote later)
    test_preds[:, fold - 1] = model.predict(X_test)

# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────
overall_f1 = f1_score(y, oof_preds, average="macro")
print(f"\n{'='*50}")
print(f"  Overall OOF Macro-F1 = {overall_f1:.5f}")
print(f"  Mean Fold Macro-F1   = {np.mean(fold_scores):.5f}")
print(f"{'='*50}")
print(f"\nClassification Report (OOF):\n")
print(classification_report(y, oof_preds, target_names=[
    "0 – No Action", "1 – Monitor", "2 – Review", "3 – Immediate Action"
]))

# ──────────────────────────────────────────────
# 5. Generate Submission
# ──────────────────────────────────────────────
# Majority vote across folds
from scipy.stats import mode
final_test_preds = mode(test_preds, axis=1).mode.flatten().astype(int)

submission = pd.DataFrame({
    "id": test[ID_COL],
    "urgency_level": final_test_preds,
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved  →  submission.csv  ({len(submission)} rows)")
print(submission["urgency_level"].value_counts().sort_index())
