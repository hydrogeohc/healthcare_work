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

## Running the Project in Docker

You can run this project inside a Docker container to ensure consistency across different environments.

### 1. Build the Docker Image

Before building, ensure that the `Dockerfile` and `requirements.txt` are in place. Then run the following command to build the Docker image:

```bash
docker build -t healthcare_work_image .
```

### 2. Run the Docker Container

Once the image is built, run the container using:

```bash
docker run -d -p 8080:8080 --name healthcare_work_container healthcare_work_image
```

This will start the container and expose it on port 8080.

### 3. Verify the Container

You can verify that the container is running by using the following command:

```bash
docker ps
```

### 4. Access the Application

Once the container is running, you can access the application (or API, if relevant) in your browser at:

```
http://localhost:8080
```

Modify the port if needed depending on your application's configuration.

### 5. Stop the Docker Container

To stop the container, run:

```bash
docker stop healthcare_work_container
```

### Additional Notes

- Make sure that your environment is set up with the correct `requirements.txt` file that lists all the necessary Python dependencies for the project.
- If the project uses multiple services (e.g., a database), consider setting up `docker-compose.yml` for multi-container setups.
