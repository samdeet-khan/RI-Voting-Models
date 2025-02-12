# RI-Voting-Models

This repository contains analytics projects focused on Rhode Island voter preferences. Currently, there are two projects:

- **Project 1:** Support for Same-Day Voter Registration (Q19 in our 2024 presidential election survey data)  
- **Project 2:** Support for Ranked Choice Voting (Q20 in our 2024 presidential election-day survey data)

Both projects aim to identify which factors (demographics, political affiliation, confidence, etc.) predict support for these voting policies, using a shared survey dataset.

## Repository Structure

```text
RI-Voting-Models/
├── LICENSE                  - Open-source license file
├── README.md                - This file
├── .gitignore               - Specifies files/directories to ignore in version control
├── raw_data/                - Folder for raw datasets
│   └── survey_data.csv      - The original survey dataset
├── docs/                    - Documentation (project overviews, meeting notes, etc.)
│   ├── pj1_one_pager.pdf    - One-page overview for Project 1
│   └── pj2_one_pager.pdf    - One-page overview for Project 2
├── environment/             - Environment setup files
├── requirements.txt         - List of Python dependencies
├── setup_instructions.md    - Instructions for installing Conda and dependencies
├── pj1_same_day_registration/
│   ├── notebooks/
│   │   ├── data_cleaning.ipynb        - Data cleaning and preprocessing
│   │   ├── exploratory_analysis.ipynb - Exploratory data analysis (EDA)
│   │   └── model_training.ipynb       - Prototyping and training predictive models
│   ├── src/
│   │   ├── data_preparation.py   - Functions for data cleaning and encoding
│   │   ├── train_model.py         - Script for model training (including cross-validation)
│   │   └── evaluate_model.py      - Functions for model evaluation (accuracy, F1-score, etc.)
│   └── results/                   - Output files, figures, and evaluation metrics
└── pj2_ranked_choice_voting/
    ├── notebooks/
    │   ├── data_cleaning.ipynb        - Data cleaning and preprocessing
    │   ├── exploratory_analysis.ipynb - Exploratory data analysis (EDA)
    │   └── model_training.ipynb       - Prototyping and training predictive models
    ├── src/
    │   ├── data_preparation.py   - Functions for data cleaning and encoding
    │   ├── train_model.py         - Script for model training (including cross-validation)
    │   └── evaluate_model.py      - Functions for model evaluation (accuracy, F1-score, etc.)
    └── results/                   - Output files, figures, and evaluation metrics
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<username>/RI-Voting-Models.git
   cd RI-Voting-Models
   ```

2. **Set Up the Environment**
   - Install Miniconda or Anaconda.
   - Create and activate a Conda environment:
     ```bash
     conda create -n ri_voting python=3.9
     conda activate ri_voting
     ```
   - Follow additional setup instructions in `environment/setup_instructions.md`.

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   - Navigate to either `pj1_same_day_registration/notebooks` or `pj2_ranked_choice_voting/notebooks` to begin your analysis.

## How to Contribute

- **Branch & Pull Request Workflow**:
  - Create a new branch for your changes.
  - Commit your changes with descriptive messages.
  - Open a pull request for review and merge into the main branch.

- **Coding Guidelines**:
  - Use the notebooks for prototyping and interactive exploration.
  - Refactor stable, reusable code into the `src/` directory.
  - Follow consistent file naming conventions and document your code.

- **Reporting Issues**:
  - Use GitHub Issues to report bugs, request features, or ask questions.

## Contact

For any questions or issues, please contact [samdeet_khan@brown.edu](mailto:samdeet_khan@brown.edu) or reach out via Slack. Thanks!
