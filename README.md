# RI-Voting-Models 📊🗳️

Data science experiments conducted by the Analytics team at Brown Every Vote Counts to assess Rhode Island voters’ support for two election-reform policies: Same-Day Voter Registration and Ranked-Choice Voting.

| Policy | Notebook | Script | Question ID |
|--------|----------|--------|-------------|
| **Same-Day Voter Registration (SDR)** | `notebooks/02_rf_sdr.ipynb` | `src/models/train_random_forest.py --policy sdr` | Q19 |
| **Ranked-Choice Voting (RCV)** | `notebooks/03_rf_rcv.ipynb` | `src/models/train_random_forest.py --policy rcv` | Q20 |
| **Ordinal comparison (SDR + RCV)** | `notebooks/01_ordinal_sdr_rcv.ipynb` | `src/models/ordinal.py` | Q19 + Q20 |

The goal of the project was to uncover which demographics, attitudes, and behaviors best predict support for each policy.

---

## Directory map
```text
RI-Voting-Models/
│
├── docs/
│   └── Data-Driven_Insights_Voter_Support.pdf           ← final report
│
├── figures/                 ← graphs from final report that show relationships between significant features and support for election-reform policies   
│   ├── ranked_choice_voting
│   └── same_day_registration
│
├── models/                 ← stores ordinal regression and random forest model outputs in .pkl format
│   └── README.md          
│
├── notebooks/              ← narrative analysis & visuals
│   ├── 01_ordinal_sdr_rcv.ipynb
│   ├── 02_rf_sdr.ipynb
│   └── 03_rf_rcv.ipynb
│
├── raw_data/               ← raw exit-polling dataset collected on Election Day 2024 and variable definitions
│   ├── survey_dataset.csv
│   └── variables_def.txt
│
├── src/                    
│   ├── __init__.py
│   └── models/
│        ├── utils.py
│        ├── train_random_forest.py   ← one CLI script: --policy sdr|rcv
│        └── ordinal.py              ← CLI script for ordinal-logit models
│
├── LICENSE                 ← MIT
├── README.md               ← you’re here
└── environment.yml         ← reproducible Conda environment

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

## Project Team

| Role | Member |
|------|--------|
| **Analytics Lead** | Samdeet Khan (<samdeet_khan@brown.edu>) |
| **Research Supervisor** | Benjamin Buka (<benjamin_buka@brown.edu>) |
| **Analysts** | Zhaocheng Yang • Jason McDermott • Evan Luo • Caleb Schultz • Kevin Pan • Alexander Wang • Jason Boek |

---
