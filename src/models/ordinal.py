"""
ordinal.py
==========

Fit *two* ordinal‑logistic (proportional‑odds) models that predict support for

1. Same‑Day Voter Registration (SDR)
2. Ranked‑Choice Voting (RCV)

Both models share the **same feature matrix**, built from:

* voter demographics
* voter‑confidence questions
* voter‑behavior questions

Run from the command line:

    python -m src.models.ordinal --csv survey_data.csv --save models/ordinal.pkl

The script will print model summaries and (optionally) pickle the fitted
`statsmodels` results objects for later inspection.

Author: Brown Every Vote Counts Analytics Team
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
)
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ────────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS ─ column names & ordered categories
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

OUTCOME_SDR = (
    "Would you support Rhode Island implementing same day voter registration, "
    "allowing unregistered voters to both register and vote on election day?"
)
OUTCOME_RCV = (
    "Would you support Rhode Island implementing ranked choice voting? This would "
    "allow voters to rank candidates in order of preference."
)

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
    # confidence & excitement share the same 5‑point scale
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

# ────────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & CLEANING
# ────────────────────────────────────────────────────────────────────────────────
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


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Read the CSV (skipping survey metadata columns) and drop rows with NaNs."""
    raw = pd.read_csv(csv_path, header=1)
    df = raw.iloc[:, 16:37].copy()  # keep survey questions only
    # Replace empty strings with NaN, then drop any row lacking required data
    df[REQ_COLUMNS + [OUTCOME_SDR, OUTCOME_RCV]] = df[
        REQ_COLUMNS + [OUTCOME_SDR, OUTCOME_RCV]
    ].replace("", np.nan)
    df.dropna(subset=REQ_COLUMNS + [OUTCOME_SDR, OUTCOME_RCV], inplace=True)
    print(f"[INFO] Cleaned dataframe shape: {df.shape}")
    return df


# ────────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def encode_single_column(
    df: pd.DataFrame, col: str, *, ordered: bool = False, categories: List[str] | None = None
) -> np.ndarray:
    """One‑hot encode nominal columns or ordinal‑encode ordered ones."""
    if ordered:
        enc = OrdinalEncoder(
            categories=[categories or ORDERS[col]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        return enc.fit_transform(df[[col]])
    # Nominal
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return enc.fit_transform(df[[col]])


def encode_multiselect(df: pd.DataFrame, col: str) -> np.ndarray:
    """Multi‑label binarize comma‑separated 'check all that apply' answers."""
    values = df[col].apply(
        lambda x: [item.strip() for item in x.split(",")] if pd.notnull(x) else []
    )
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(values)


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return X matrix and list of feature names (for interpretability)."""
    parts: List[np.ndarray] = []
    names: List[str] = []

    # 1. Demographics -----------------------------------------------------------
    # Ward
    ward_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ward = ward_enc.fit_transform(df[[Q_WARD]])
    parts.append(ward)
    names.extend([f"Ward={c}" for c in ward_enc.categories_[0]])

    # Age (ordinal)
    age = encode_single_column(df, Q_AGE, ordered=True)
    parts.append(age)
    names.append("Age_ordinal")

    # Gender (one‑hot)
    gender_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    gender = gender_enc.fit_transform(df[[Q_GENDER]])
    parts.append(gender)
    names.extend([f"Gender={c}" for c in gender_enc.categories_[0]])

    # Race (multi‑select)
    race = encode_multiselect(df, Q_RACE)
    parts.append(race)
    # classes_ populated by a fresh mlb:
    race_classes = sorted(
        {r.strip() for lst in df[Q_RACE].dropna().str.split(",") for r in lst}
    )
    names.extend([f"Race={c}" for c in race_classes])

    # Education (ordinal)
    educ = encode_single_column(df, Q_EDUC, ordered=True)
    parts.append(educ)
    names.append("Education_ordinal")

    # Income (ordinal)
    income = encode_single_column(df, Q_INCOME, ordered=True)
    parts.append(income)
    names.append("Income_ordinal")

    # Political party (one‑hot)
    party_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    party = party_enc.fit_transform(df[[Q_POL_PARTY]])
    parts.append(party)
    names.extend([f"Party={c}" for c in party_enc.categories_[0]])

    # 2. Confidence -------------------------------------------------------------
    for q in [
        Q_CONF_DIR_US,
        Q_CONF_DIR_RI,
        Q_CONF_POL_CARE,
        Q_CONF_ELECT_MGMT,
        Q_CONF_ELECT_RESULTS,
    ]:
        conf = encode_single_column(
            df,
            q,
            ordered=True,
            categories=ORDERS["CONF_5"],
        )
        parts.append(conf)
        names.append(f"{q[:30]}_ordinal")  # truncated for readability

    # 3. Behavior ---------------------------------------------------------------
    # Presidential vote
    vote_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    vote = vote_enc.fit_transform(df[[Q_PRES_VOTE]])
    parts.append(vote)
    names.extend([f"PresVote={c}" for c in vote_enc.categories_[0]])

    # Excitement (ordinal)
    excite = encode_single_column(
        df,
        Q_EXCITED_VOTE,
        ordered=True,
        categories=ORDERS["EXCITE_5"],
    )
    parts.append(excite)
    names.append("Excitement_ordinal")

    # Info sources (multi‑select)
    info = encode_multiselect(df, Q_INFO_SRC)
    parts.append(info)
    info_classes = sorted(
        {r.strip() for lst in df[Q_INFO_SRC].dropna().str.split(",") for r in lst}
    )
    names.extend([f"Info={c}" for c in info_classes])

    # In‑person reasons (multi‑select)
    reasons = encode_multiselect(df, Q_INPERSON_REASON)
    parts.append(reasons)
    reason_classes = sorted(
        {r.strip() for lst in df[Q_INPERSON_REASON].dropna().str.split(",") for r in lst}
    )
    names.extend([f"Reason={c}" for c in reason_classes])

    X = np.hstack(parts)
    return X, names


# ────────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ────────────────────────────────────────────────────────────────────────────────
def map_outcome(series: pd.Series) -> np.ndarray:
    """Convert 5‑point support answers to integers 1‑5."""
    return series.map(SUPPORT_MAP).values.astype(int)


def fit_ordinal_model(y: np.ndarray, X: np.ndarray) -> OrderedModel:
    """Fit a proportional‑odds logistic regression."""
    model = OrderedModel(endog=y, exog=X, distr="logit", hasconst=False)
    result = model.fit(method="bfgs", disp=False)
    return result


def train_models(df: pd.DataFrame) -> Dict[str, OrderedModel]:
    X, feature_names = build_feature_matrix(df)
    print(f"[INFO] Final feature matrix shape: {X.shape}")

    models: Dict[str, OrderedModel] = {}

    for outcome, label in [(OUTCOME_SDR, "SDR"), (OUTCOME_RCV, "RCV")]:
        y = map_outcome(df[outcome])
        result = fit_ordinal_model(y, X)
        print(f"\n{'='*80}\n{label} model summary\n{'='*80}")
        print(result.summary())
        models[label] = result

    return models


# ────────────────────────────────────────────────────────────────────────────────
# 5. MAIN CLI
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ordinal‑logit models for SDR/RCV.")
    p.add_argument("--csv", required=True, help="Path to survey_data.csv")
    p.add_argument(
        "--save",
        help="Where to pickle the fitted models (will store a dict with keys 'SDR' & 'RCV')",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_and_clean(Path(args.csv))
    models = train_models(df)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "wb") as f:
            pickle.dump(models, f)
        print(f"[INFO] Models saved to {args.save}")


if __name__ == "__main__":
    main()
