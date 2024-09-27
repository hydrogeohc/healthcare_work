
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, List


# Function to load data
# This function reads the base and medication data from CSV files.
def load_data(base_data_path: str, meds_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_data = pd.read_csv(base_data_path, index_col=0)
    meds_data = pd.read_csv(meds_data_path, index_col=0)
    return base_data, meds_data


# Function to visualize data
# This function creates visualizations of age distribution and prescription distribution by year.
def visualize_data(base_data: pd.DataFrame, meds_data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(base_data['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=meds_data, x='rxStartYear')
    plt.title('Prescription Distribution by Year')
    plt.xticks(rotation=45)
    plt.show()


# Function to preprocess data
# This function merges the base and meds data, handles missing values, and removes duplicates.
def preprocess_data(base_data: pd.DataFrame, meds_data: pd.DataFrame) -> pd.DataFrame:
    merged_data = pd.merge(meds_data, base_data, on='id')
    # Remove duplicate rows based on 'id'
    merged_data = merged_data.drop_duplicates(subset='id')
    # Fill missing numerical values with the median
    for col in merged_data.select_dtypes(include=['float64', 'int64']).columns:
        merged_data[col].fillna(merged_data[col].median(), inplace=True)
    return merged_data


# Function to train models
# This function trains RandomForest, Logistic Regression, and SVM models on the training data.
def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestClassifier, LogisticRegression, SVC]:
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, max_iter=1000)
    
    # Train each model
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    return rf_model, lr_model, svm_model


# Function to evaluate models
# This function evaluates the trained models using accuracy, confusion matrix, and classification report.
def evaluate_models(models: List, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    for model in models:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        print(f"Model: {model.__class__.__name__}, Accuracy: {accuracy}")
        print(classification_report(y_test, predictions))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.show()


# Example run (adjust with appropriate paths)
if __name__ == "__main__":
    base_data, meds_data = load_data('base_data.csv', 'meds_data.csv')
    visualize_data(base_data, meds_data)
    merged_data = preprocess_data(base_data, meds_data)
    X = merged_data.drop(columns=['target'])
    y = merged_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model, lr_model, svm_model = train_models(X_train, y_train)
    evaluate_models([rf_model, lr_model, svm_model], X_test, y_test)
