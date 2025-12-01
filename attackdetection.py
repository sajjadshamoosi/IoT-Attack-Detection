"""
IoT Attack Detection on RT_IOT2022.csv using RandomForest

This program:
1. Loads a real IoT network traffic dataset from Kaggle (RT_IOT2022.csv).
2. Converts the original Attack_type field into a binary label: Normal vs Attack.
3. Preprocesses the data (drops index column, encodes categorical features, removes NaNs).
4. Trains a RandomForest classifier to distinguish normal traffic from attacks.
5. Evaluates the model with accuracy, classification report, and confusion matrix.
6. Prints how many samples are predicted as Normal vs Attack.

This is a simpler, classical ML approach compared to the CNN + XGBoost model in the paper.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------- 1. Load dataset ----------

# Path to the CSV file.
# IMPORTANT: RT_IOT2022.csv must be in the same folder as this script.
CSV_PATH = "RT_IOT2022.csv"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)  # Read the CSV into a pandas DataFrame
print(f"Shape of raw data: {df.shape}")  # (rows, columns)


# ---------- 2. Create binary label: Normal vs Attack ----------

# In the raw dataset, "Attack_type" has many specific labels
# (e.g., DDoS, DoS, Mirai, etc., plus some normal traffic types).
# We decide which values are considered NORMAL traffic.
# Everything else is treated as an ATTACK.
NORMAL_LABELS = ["Thing_Speak", "Wipro_bulb", "MQTT_Publish"]

# Create a new column "Label":
# - "Normal" if Attack_type is in NORMAL_LABELS
# - "Attack" otherwise
df["Label"] = df["Attack_type"].apply(
    lambda x: "Normal" if x in NORMAL_LABELS else "Attack"
)

print("\nLabel distribution:")
# Shows how many Normal vs Attack rows exist
print(df["Label"].value_counts())


# ---------- 3. Basic cleaning ----------

# Some Kaggle datasets include an extra index column called "Unnamed: 0".
# It does not contain useful information, so we drop it if it exists.
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Separate the target label (y) from the feature columns (X).
# - y is what we want to predict (Normal vs Attack).
# - X contains all input features used for prediction.
y = df["Label"]
X = df.drop(columns=["Label", "Attack_type"])  # we drop Attack_type to avoid leakage

# Identify which columns are categorical and which are numeric.
# Categorical columns are text-based and need encoding (e.g., proto, service).
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Number of numeric features:", len(numeric_cols))

# One-hot encode categorical columns (e.g., proto, service).
# This converts them into 0/1 columns so machine learning models can use them.
# drop_first=True avoids multicollinearity by dropping the first category.
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("Shape after one-hot encoding:", X.shape)

# Replace infinite values with NaN to avoid issues in training.
X = X.replace([np.inf, -np.inf], np.nan)

# Drop any rows that still contain NaN values after processing.
# Because the dataset is large, removing a few rows does not hurt performance.
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]
print("Shape after dropping NaNs:", X.shape)


# ---------- 4. Train / test split ----------

# Split the dataset into:
# - 80% training data
# - 20% testing data
# We use stratify=y to keep the same Attack/Normal ratio in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # ensures repeatable results
    stratify=y,        # preserves label distribution
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)


# ---------- 5. Train RandomForest model ----------

# Convert labels to binary integers for the classifier:
# - 1 = Attack
# - 0 = Normal
y_train_binary = (y_train == "Attack").astype(int)
y_test_binary = (y_test == "Attack").astype(int)

# Create the RandomForest classifier.
# - n_estimators: number of trees in the forest
# - max_depth: limit tree depth to prevent overfitting
# - class_weight="balanced": automatically adjust weights to handle class imbalance
# - n_jobs=-1: use all CPU cores
# - random_state: for reproducibility
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

print("\nTraining RandomForest model...")
# Fit the model on the training data
model.fit(X_train, y_train_binary)


# ---------- 6. Evaluation ----------

# Predict labels (0 or 1) on the test set
y_pred_binary = model.predict(X_test)

# Overall accuracy of the model on the test set
acc = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nTest Accuracy: {acc:.4f}")

# Print a detailed classification report:
# - precision, recall, F1-score for both Normal (0) and Attack (1)
print("\nClassification report (0 = Normal, 1 = Attack):")
print(classification_report(y_test_binary, y_pred_binary, target_names=["Normal", "Attack"]))

# Compute the confusion matrix:
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_test_binary, y_pred_binary)
print("Confusion matrix:\n", cm)

# Visualize the confusion matrix using matplotlib.
# This makes it easy to show in your demo and explain correct vs incorrect predictions.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
disp.plot(cmap="Blues", values_format="d")
plt.title("IoT Attack Detection - Confusion Matrix (RandomForest)")
plt.tight_layout()
plt.show()


# ---------- 7. How many rows are predicted as Normal vs Attack? ----------

# Convert binary predictions back to string labels for easier interpretation.
pred_labels = np.where(y_pred_binary == 1, "Attack", "Normal")

# Count how many of each label appear in the predictions.
unique, counts = np.unique(pred_labels, return_counts=True)
pred_counts = dict(zip(unique, counts))

print("\nPredicted label counts on test set:")
for label, count in pred_counts.items():
    print(f"{label}: {count}")


# ============================================================
# ----------- 8. Train and Evaluate XGBoost Model ------------
# ============================================================

print("\nTraining XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1
)

# Fit model
xgb_model.fit(X_train, y_train_binary)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Accuracy
xgb_acc = accuracy_score(y_test_binary, y_pred_xgb)
print(f"\nXGBoost Test Accuracy: {xgb_acc:.4f}")

# Classification report
print("\nXGBoost Classification Report (0 = Normal, 1 = Attack):")
print(classification_report(y_test_binary, y_pred_xgb, target_names=["Normal", "Attack"]))

# Confusion matrix
cm_xgb = confusion_matrix(y_test_binary, y_pred_xgb)
print("\nXGBoost Confusion Matrix:\n", cm_xgb)

# Plot confusion matrix
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=["Normal", "Attack"])
disp_xgb.plot(cmap="Greens", values_format="d")
plt.title("IoT Attack Detection - Confusion Matrix (XGBoost)")
plt.tight_layout()
plt.show()
