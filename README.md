# Credit Card Fraud Detection

## Dataset
The dataset used for this project is a highly imbalanced dataset containing transactions made by credit cards in September 2013 by European cardholders. It includes 284,807 transactions, of which 492 are fraudulent.

You can download the dataset from Kaggle's Credit Card Fraud Detection Dataset. ( https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud )

## Models Applied
The project explores various machine learning algorithms to detect fraudulent transactions:

- **Logistic Regression**: A linear model used for binary classification, predicting the probability of fraud based on transaction features.

- **Decision Tree**: A non-linear model that partitions the data into subsets based on feature values, creating a tree-like structure to predict fraud.

- **Random Forest**: An ensemble method that combines multiple decision trees to improve predictive accuracy and control overfitting.

- **Support Vector Machine (SVM)**: A linear model that finds the optimal hyperplane to separate fraudulent and non-fraudulent transactions.

- **XGBoost**: An efficient implementation of gradient boosting that builds additive models in a forward stage-wise manner.

- **Feedforward Neural Network (FNN)**: A deep learning model with multiple layers that captures complex patterns in the data.

## MLflow Integration
MLflow is utilized to manage the machine learning lifecycle, including:

- **Experiment Tracking**: Logging metrics, parameters, and models for each experiment to facilitate comparison and reproducibility.

- **Model Management**: Storing and organizing trained models for easy retrieval and deployment.

- **Visualization**: Providing a user interface to visualize metrics and compare model performance across different runs.

By integrating MLflow, the project ensures systematic tracking of experiments and efficient management of models, enhancing collaboration and scalability.
