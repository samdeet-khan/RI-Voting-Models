# RI‑Voting‑Models 📊🗳️

Data‑science experiments on Rhode Island voters’ support for two election‑reform policies:

| Policy | Notebook | Script | Question ID |
|--------|----------|--------|-------------|
| **Same‑Day Voter Registration (SDR)** | `notebooks/02_rf_sdr.ipynb` | `src/models/train_random_forest.py --policy sdr` | Q19 |
| **Ranked‑Choice Voting (RCV)** | `notebooks/03_rf_rcv.ipynb` | `src/models/train_random_forest.py --policy rcv` | Q20 |
| **Ordinal comparison (SDR + RCV)** | `notebooks/01_ordinal_sdr_rcv.ipynb` | `src/models/ordinal.py` | Q19 + Q20 |

The goal is to uncover which demographics, attitudes, and behaviors best predict support for each policy.

---

## Directory map

```text
RI‑Voting‑Models/
│
├── .gitignore
├── environment.yml         ← reproducible Conda environment
├── LICENSE                 ← MIT
├── README.md               ← you’re here
│
├── notebooks/              ← narrative analysis & visuals
│   ├── 01_ordinal_sdr_rcv.ipynb
│   ├── 02_rf_sdr.ipynb
│   └── 03_rf_rcv.ipynb
│
├── src/                    ← source code for machine learning experiments (random forest and ordinal regression)
│   ├── __init__.py
│   └── models/
│        ├── utils.py
│        ├── train_random_forest.py   ← one CLI script: --policy sdr|rcv
│        └── ordinal.py              ← CLI script for ordinal‑logit models
│
├── figures/                ← graphs from final report that show relationships between significant features and support for election-reform policies
│   ├── same_day_registration
│   ├── ranked_choice_voting
│
├── models/                 ← stores ordinal regression and random forest model outputs in .pkl format
│   └── README.md          
│
└── docs/                
    └── Data‑Driven_Insights_Voter_Support.pdf            ← final report
```
---

## Quick-start (local)

> **Prerequisites:** Conda ≥ 4.10 installed

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/RI-Voting-Models.git
cd RI-Voting-Models

# 2. Duplicate the environment
conda env create -f environment.yml
conda activate ri_voting_models
```
---

## Re-generate Models

You can retrain and save each model using the following commands:

### Same-Day Registration – Random Forest

```bash
python -m src.models.train_random_forest \
       --csv raw_data/survey_data.csv \
       --policy sdr \
       --model models/rf_sdr.pkl
```
### Ranked Choice Voting – Random Forest

```bash
python -m src.models.train_random_forest \
       --csv raw_data/survey_data.csv \
       --policy rcv \
       --model models/rf_rcv.pkl
```
### Ordinal Regression Model (Both Policies)

```bash
python -m src.models.ordinal \
       --csv raw_data/survey_data.csv \
       --save models/ordinal.pkl
```

---

For questions about this project, feel free to reach out: samdeet_khan@brown.edu
