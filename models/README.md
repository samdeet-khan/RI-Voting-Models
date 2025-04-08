# Models Folder

This folder is reserved for storing trained model artifacts (e.g., `.pkl` files) produced by the project's training scripts.

## How to Generate and Save Models

Run the training scripts from the command line, specifying the model path in the `models/` folder. For example:

- **Random Forest for Same-Day Registration:**

  ```bash
  python -m src.models.train_random_forest --csv data/raw/survey_data.csv --policy sdr --model models/rf_sdr.pkl

- **Random Forest for Ranked-Choice Voting:**

  ```bash
  python -m src.models.train_random_forest --csv data/raw/survey_data.csv --policy rcv --model models/rf_rcv.pkl
