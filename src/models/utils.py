"""
utils.py
========
Shared helpers for both the ordinal‑logit and random‑forest pipelines.

Functions here are *pure* (no prints, no I/O) so they can be reused in tests,
notebooks, or other scripts without side‑effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────────
Q_WARD = "Which ward is this survey from?"
Q_AGE = "What is your age group?"
Q_GENDER = "What is your gender identity?"
Q_RACE = (
    "With what ethnicity or race(s) do you most identify? (Check all that apply)"
)
Q_EDUC = "What is your highest level of education?"
Q_INCOME = "What is your annual household income?"
Q_POL_PARTY = "How would you describe your political affiliation?"

Q_CONF_DIR_US = (
    "Overall, do you feel that the country is heading in the right direction or "
    "the wrong direction?"
)
Q_CONF_DIR_RI = (
    "Overall, do you feel that Rhode Island is heading in the right direction or "
    "the wrong direction?"
)
Q_CONF_POL_CARE = (
    "How confident are you that your Rhode Island politicians care about you?"
)
Q_CONF_ELECT_MGMT = (
    "How confident are you in the ability of local voting officials to manage "
    "elections fairly and accurately?"
)
Q_CONF_ELECT_RESULTS = (
    "How confident are you that the results of the national presidential election "
    "will accurately reflect the votes cast?"
)

Q_PRES_VOTE = "Who did you vote for President"
Q_EXCITED_VOTE = "Were you excited to vote for your presidential candidate?"
Q_INFO_SRC = (
    "Where do you primarily get your information about voting options, such as "
    "precinct locations, voting times, early voting, and mail-in ballot options? "
    "(Select all that apply)"
)
Q_INPERSON_REASON = (
    "Why did you choose to vote in-person on election day? (Select all that apply)"
)

POLICY_CONFIG = {
    "sdr": {
        "outcome": (
            "Would you support Rhode Island implementing same day voter registration, "
            "allowing unregistered voters to both register and vote on election day?"
        )
    },
    "rcv": {
        "outcome": (
            "Would you support Rhode Island implementing ranked choice voting? "
            "This would allow voters to rank candidates in order of preference."
        )
    },
}

ORDERS = {
    Q_AGE: [
        "18-24",
        "25-29",
        "30-39",
        "40-49",
        "50-64",
        "65 or older",
    ],
    Q_EDUC: [
        "Less than High School",
        "High school graduate / GED",
        "Some college",
        "2 year degree",
        "4 year degree",
        "Professional degree",
        "Doctorate",
    ],
    Q_INCOME: [
        "Less than $30,000",
        "30,000 - 49,999",
        "50,000 - 100,000",
        "100,000 - 199,999",
        "$200,000 or more",
    ],
    "CONF_5": [
        "Definitely Wrong",
        "Somewhat Wrong",
        "Unsure",
        "Somewhat Right",
        "Definitely Right",
    ],
    "EXCITE_5": [
        "Not at all Excited",
        "Not Very Excited",
        "Somewhat Excited",
        "Very Excited",
        "Extremely Excited",
    ],
}

SUPPORT_MAP = {
    "Definitely Not": 1,
    "Probably Not": 2,
    "I would need more information to make a decision": 3,
    "Probably": 4,
    "Definitely": 5,
}

REQ_COLUMNS: List[str] = [
    Q_WARD,
    Q_AGE,
    Q_GENDER,
    Q_RACE,
    Q_EDUC,
    Q_INCOME,
    Q_POL_PARTY,
    Q_CONF_DIR_US,
    Q_CONF_DIR_RI,
    Q_CONF_POL_CARE,
    Q_CONF_ELECT_MGMT,
    Q_CONF_ELECT_RESULTS,
    Q_PRES_VOTE,
    Q_EXCITED_VOTE,
    Q_INFO_SRC,
    Q_INPERSON_REASON,
]


# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING & TARGET ENGINEERING
# ────────────────────────────────────────────────────────────────────────────────
def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV and keep only the survey columns (16:37)."""
    raw = pd.read_csv(csv_path, header=1)
    return raw.iloc[:, 16:37].copy()


def prepare_target(df: pd.DataFrame, policy: str) -> pd.Series:
    """Return the binary support series for the chosen policy."""
    outcome_col = POLICY_CONFIG[policy]["outcome"]
    df[outcome_col] = df[outcome_col].replace("", np.nan)
    df["support_ord"] = df[outcome_col].map(SUPPORT_MAP)
    # Drop ambiguous rows
    df = df[df["support_ord"] != 3].copy()
    df["support_bin"] = df["support_ord"].apply(lambda x: 0 if x in (1, 2) else 1)
    return df["support_bin"]


# ────────────────────────────────────────────────────────────────────────────────
# ENCODERS
# ────────────────────────────────────────────────────────────────────────────────
def _ordinal(df: pd.DataFrame, col: str, categories: List[str]) -> np.ndarray:
    enc = OrdinalEncoder(
        categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1
    )
    return enc.fit_transform(df[[col]])


def _onehot(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, List[str]]:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    mat = enc.fit_transform(df[[col]])
    names = [f"{col}={c}" for c in enc.categories_[0]]
    return mat, names


def _multilabel(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, List[str]]:
    values = df[col].apply(
        lambda x: [item.strip() for item in x.split(",")] if pd.notnull(x) else []
    )
    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(values)
    names = [f"{col}={c}" for c in mlb.classes_]
    return mat, names


# ────────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX
# ────────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    parts: List[np.ndarray] = []
    names: List[str] = []

    # Ward
    mat, lbl = _onehot(df, Q_WARD)
    parts.append(mat)
    names.extend(lbl)

    # Age (ordinal)
    parts.append(_ordinal(df, Q_AGE, ORDERS[Q_AGE]))
    names.append("Age")

    # Gender
    mat, lbl = _onehot(df, Q_GENDER)
    parts.append(mat)
    names.extend(lbl)

    # Race (multi‑select)
    mat, lbl = _multilabel(df, Q_RACE)
    parts.append(mat)
    names.extend(lbl)

    # Education & Income
    parts.append(_ordinal(df, Q_EDUC, ORDERS[Q_EDUC]))
    names.append("Education")
    parts.append(_ordinal(df, Q_INCOME, ORDERS[Q_INCOME]))
    names.append("Income")

    # Political party
    mat, lbl = _onehot(df, Q_POL_PARTY)
    parts.append(mat)
    names.extend(lbl)

    # Confidence questions (5‑point ordinal)
    for q in [
        Q_CONF_DIR_US,
        Q_CONF_DIR_RI,
        Q_CONF_POL_CARE,
        Q_CONF_ELECT_MGMT,
        Q_CONF_ELECT_RESULTS,
    ]:
        parts.append(_ordinal(df, q, ORDERS["CONF_5"]))
        names.append(q[:25])  # truncated for readability

    # Behavior
    mat, lbl = _onehot(df, Q_PRES_VOTE)
    parts.append(mat)
    names.extend(lbl)

    parts.append(_ordinal(df, Q_EXCITED_VOTE, ORDERS["EXCITE_5"]))
    names.append("Excitement")

    mat, lbl = _multilabel(df, Q_INFO_SRC)
    parts.append(mat)
    names.extend(lbl)

    mat, lbl = _multilabel(df, Q_INPERSON_REASON)
    parts.append(mat)
    names.extend(lbl)

    X = np.hstack(parts)
    return X, names


# ────────────────────────────────────────────────────────────────────────────────
# RANDOM‑FOREST HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def train_random_forest(
    X: np.ndarray, y: np.ndarray, *, trees: int = 300, seed: int = 42
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=trees, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X, y)
    return rf


def evaluate(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 5,
    seed: int = 42,
) -> str:
    """Return a formatted report string (no prints)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    holdout = classification_report(
        y_te, model.predict(X_te), digits=2, output_dict=False
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    for tr, te in skf.split(X, y):
        rf = train_random_forest(X[tr], y[tr], trees=model.n_estimators, seed=seed)
        y_pred = rf.predict(X[te])
        y_true_all.extend(y[te])
        y_pred_all.extend(y_pred)
    cv_report = classification_report(y_true_all, y_pred_all, digits=2, output_dict=False)

    return (
        f"── Hold‑out (30%) ──\n{holdout}\n"
        f"── {cv}‑fold Stratified CV ──\n{cv_report}"
    )