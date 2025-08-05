
# Finding Donors for CharityML - Udacity Machine Learning Project

**This project was submitted on September 2, 2024**

This repository contains my complete solution to the **Finding Donors for CharityML** project from the Udacity Introduction to Machine Learning Nanodegree program using TensorFlow/PyTorch.

## Project Overview

In this supervised learning project, I employed several machine learning algorithms to accurately model individuals' income using data collected from the 1994 U.S. Census. The goal was to construct a model that accurately predicts whether an individual makes more than $50,000 annually, which can help CharityML (a fictitious charity organization) identify people most likely to donate to their cause.

## Key Features

- **Data Exploration and Analysis**: Comprehensive analysis of the U.S. Census dataset
- **Data Preprocessing**: Feature transformation, normalization, and encoding techniques
- **Model Comparison**: Implementation and evaluation of multiple supervised learning algorithms
- **Model Optimization**: Fine-tuning of the best performing model
- **Performance Analysis**: Detailed evaluation using various metrics and visualizations

## Dataset

The project uses the **Adult Census Income** dataset from the UCI Machine Learning Repository, which contains demographic information about individuals and their income brackets. The dataset includes features such as:

- Age, education, marital status, occupation
- Work hours per week, capital gains/losses
- Native country, race, gender
- **Target variable**: Income (≤50K or >50K)

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24.4-orange?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-purple?style=flat-square&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-yellow?style=flat-square&logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.5-green?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-lightblue?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)

### Core Libraries:
- **Python 3.8**: Programming language
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## Project Structure

```
finding_donors/
├── finding_donors_solution.ipynb    # Complete solution notebook
├── census.csv                       # Dataset
├── visuals.py                      # Visualization helper functions
├── environment.yml                 # Conda environment specification
├── README.md                       # This file
└── project_description.md          # Original project description
```

## Setup and Installation

### Option 1: Using Conda (Recommended)

1. **Clone or download this repository**
2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```
3. **Activate the environment:**
   ```bash
   conda activate finding_donors
   ```
4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook finding_donors_solution.ipynb
   ```

### Option 2: Manual Setup

1. **Create a new conda environment:**
   ```bash
   conda create -n finding_donors python=3.8 -y
   conda activate finding_donors
   ```
2. **Install required packages:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter ipython
   ```

## Key Algorithms Implemented

- **Naive Bayes (Gaussian)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Gradient Boosting**
- **Logistic Regression**

## Results and Performance

The final optimized model achieved:
- **Accuracy**: High accuracy on test set
- **Precision/Recall**: Balanced performance for both income classes
- **F1-Score**: Optimized for the specific use case
- **Feature Importance**: Analysis of most predictive features

## Key Learning Outcomes

✅ **Data Preprocessing**: Learned to identify and apply appropriate preprocessing techniques  
✅ **Benchmark Establishment**: Created baseline models for comparison  
✅ **Algorithm Selection**: Understood when and where different supervised learning algorithms excel  
✅ **Model Evaluation**: Investigated model adequacy using various metrics  
✅ **Feature Engineering**: Explored feature importance and selection techniques  

## Tags and Keywords

`machine-learning` `supervised-learning` `classification` `python` `scikit-learn` `data-science` `census-data` `income-prediction` `udacity` `nanodegree` `jupyter-notebook` `pandas` `numpy` `matplotlib` `seaborn` `feature-engineering` `model-optimization` `cross-validation` `charity-ml`

## Usage

1. **Activate the conda environment**: `conda activate finding_donors`
2. **Open the solution notebook**: `jupyter notebook finding_donors_solution.ipynb`
3. **Run all cells** to see the complete analysis and results
4. **Modify parameters** and experiment with different approaches

## Project Highlights

- **Comprehensive EDA**: Thorough exploration of the census dataset
- **Multiple Algorithm Comparison**: Systematic evaluation of various ML algorithms
- **Hyperparameter Tuning**: Grid search and optimization techniques
- **Visualization**: Rich visual analysis of data and model performance
- **Real-world Application**: Practical insights for charity fundraising strategies

## About This Solution

This notebook represents my complete solution to the Udacity Machine Learning Nanodegree project. It demonstrates proficiency in:

- Data analysis and preprocessing
- Machine learning algorithm implementation
- Model evaluation and optimization
- Statistical analysis and interpretation
- Professional documentation and presentation

## Dataset Details

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. 

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

## License

This project is part of the Udacity Machine Learning Nanodegree program. The code is available for educational purposes.

---

**Note**: This is a complete, working solution that successfully passed all project requirements and rubric criteria for the Udacity Introduction to Machine Learning Nanodegree program.
