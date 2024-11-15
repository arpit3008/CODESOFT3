# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# Optional: Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Function to display plots inline (useful for Jupyter notebooks)
def display_plots():
    plt.tight_layout()
    plt.show()

# Load dataset
try:
    data_path = r"C:\Users\user\Downloads\creditcard.csv\creditcard.csv"  # Update with your actual path
    print("Loading dataset..")
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print(f"Error: The file at path '{data_path}' was not found.")
    exit()

# Dataset overview
print("Dataset Overview:")
print(data.head(), "\n")

print("Dataset Information:")
print(data.info(), "\n")

# Check for missing values
print("Checking for missing values...")
missing_values = data.isnull().sum()
print(missing_values, "\n")

if missing_values.sum() == 0:
    print("No missing values detected.\n")
else:
    print("Missing values detected. Please handle them before proceeding.\n")

# Basic statistics
print("Class Distribution (Fraud vs. Not Fraud):")
print(data['Class'].value_counts(), "\n")

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.xticks([0,1], ['Not Fraud', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Count')
display_plots()

# Preprocessing
print("Preprocessing the data...")
X = data.drop('Class', axis=1)
y = data['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling completed.\n")

# Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split completed.\n")

# Initialize and train the model
print("Initializing and training the Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,        # Reduced number for faster training
    max_depth=20,            # Set to a reasonable depth
    random_state=42,
    n_jobs=-1                # Utilize all CPU cores
)
model.fit(X_train, y_train)
print("Model training completed.\n")

# Predictions
print("Making predictions on the test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Predictions completed.\n")

# Model Evaluation
print("Evaluating the model...\n")

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix, "\n")

# Plot Confusion Matrix
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
display_plots()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC Score: {roc_auc:.4f}\n")

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
display_plots()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='green')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
display_plots()

# Display FPR, TPR, and Threshold values
fpr_vs_threshold = pd.DataFrame({
    'Threshold': thresholds,
    'False Positive Rate': fpr,
    'True Positive Rate': tpr
})
print("FPR vs Threshold:")
print(fpr_vs_threshold.head(), "\n")

# Additional Visualizations

# 1. Feature Importance
print("Plotting Feature Importances...\n")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
display_plots()

# 2. Correlation Heatmap
print("Plotting Correlation Heatmap...\n")
plt.figure(figsize=(12,10))
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Heatmap")
display_plots()

# 3. Distribution of Transaction Amounts
print("Plotting Distribution of Transaction Amounts...\n")
plt.figure(figsize=(8,6))
sns.histplot(data['Amount'], bins=50, kde=True, color='purple')
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
display_plots()

print("All tasks completed successfully!")
