# Health Insurance Candidate Application Decision Process

This repository contains two python modules for two use cases in the `src` folder, each exploring different aspects of a health insurance decision-making process using machine learning techniques. The scripts focus on the prediction of application outcomes and decision-making processes, providing a practical demonstration of machine learning models in healthcare.

## Modules

### Use Case 1: Health Insurance Candidate Application Decision Process
This module explores the decision-making process for health insurance candidate applications. It implements machine learning models to predict whether a candidate is eligible for health insurance coverage.

#### Contents
- **Data Preprocessing**: Steps to clean and prepare the data, including handling missing values and removing collinear features.
- **Model Training**: Implementation of models like K-Nearest Neighbors, RandomForest, and XGBoost.
- **Model Evaluation**: Evaluation of models using accuracy, classification report, and confusion matrix.




###  Use Case 2: Advanced Feature Engineering and Model Evaluation

This Python module builds on the first use case by further exploring feature engineering and model evaluation for health insurance candidate applications. It uses multiple machine learning models to analyze candidate data and predict outcomes.

#### Contents
- **Data Preprocessing**: 
  - Merging multiple datasets
  - Handling missing values
  - Creating engineered features like BMI ranges
- **Model Training**: Implementing the following models:
  - RandomForest
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Model Evaluation**: Evaluating models using accuracy, confusion matrices, and classification reports.

## Getting Started

### Prerequisites
- Python 3.9
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and any additional libraries used in the scripts.

To install the required libraries, you can run:

```bash
pip install -r requirements.txt
