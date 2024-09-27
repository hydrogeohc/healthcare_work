
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, List


# Function to load and clean data
# This function reads the CSV file, removes unnecessary columns, handles missing values for 'PatientID',
# and replaces categorical 'one' and 'zero' with numeric values.
def load_and_clean_data(path: str) -> pd.DataFrame:
    case_data = pd.read_csv(path)
    # Drop the first column
    case_data = case_data.iloc[:, 1:]
    # Remove rows where 'PatientID' is missing
    case_data = case_data[pd.notnull(case_data['PatientID'])]
    # Replace 'one' and 'zero' with 1 and 0
    case_data = case_data.replace({'one': 1, 'zero': 0})
    return case_data


# Function to preprocess data
# This function handles missing values, removes collinear features, and categorizes BMI into ranges.
def preprocess_data(case_data: pd.DataFrame) -> pd.DataFrame:
    # Fill missing 'HbA1c' values with 0
    case_data['HbA1c'] = case_data['HbA1c'].fillna(0)
    # Drop rows with at least 6 non-null values
    case_data_test = case_data.dropna(thresh=6)
    # Drop 'Label' and 'PatientID' columns
    featuresok = case_data_test.drop(['Label', 'PatientID'], axis=1)
    # Remove collinear features
    featuresfok1 = remove_collinear_features(featuresok, 0.5)
    # Fill missing BMI values with 0
    featuresfok1['BMI'] = featuresfok1['BMI'].fillna(0)
    # Create BMI ranges
    bins = [0, 19, 25, 35, 40]
    featuresfok1['BMIRange'] = np.digitize(featuresfok1['BMI'], bins)
    return featuresfok1


# Function to remove collinear features
# This function removes columns with a correlation higher than the given threshold.
def remove_collinear_features(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr_matrix = x.corr()
    drop_cols = []
    # Identify columns with high correlation
    for i in range(len(corr_matrix.columns) - 1):
        for j in range(i):
            if abs(corr_matrix.iloc[j, i]) >= threshold:
                drop_cols.append(corr_matrix.columns[i])
    # Drop highly correlated columns
    x = x.drop(columns=drop_cols)
    return x


# Function to train models
# This function trains KNN, RandomForest, and XGBoost models on the training data.
def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[KNeighborsClassifier, RandomForestClassifier, xgb.XGBClassifier]:
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    # Train RandomForest
    RF = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    RF.fit(X_train, y_train)
    # Train XGBoost
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    gbm.fit(X_train, y_train)
    return knn, RF, gbm


# Function to evaluate models
# This function evaluates the trained models using accuracy, classification report, and confusion matrix.
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
    case_data = load_and_clean_data('your_data_path.csv')
    featuresfok1 = preprocess_data(case_data)
    targets = case_data['Label']
    train_features, test_features, train_labels, test_labels = train_test_split(featuresfok1, targets, test_size=0.3, random_state=42)
    knn, RF, gbm = train_models(train_features, train_labels)
    evaluate_models([knn, RF, gbm], test_features, test_labels)
