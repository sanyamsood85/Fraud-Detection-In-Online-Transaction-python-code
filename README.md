# Fraud-Detection-In-Online-Transaction-python-code
# ======================================================
# Research Project: Fraud Detection in Online Transactions Using Machine Learning
# Author: Sanyam Sood
# University: Chitkara University
# ======================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------
df = pd.read_csv("C:\\Users\\Ishu\\Desktop\\fraud_kaggle_style_v2.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

# ------------------------------------------------------
# 2️⃣ Basic Information
# ------------------------------------------------------
print("\nMissing Values:\n", df.isnull().sum())
print("\nFraud Distribution:\n", df['Class'].value_counts())

# ------------------------------------------------------
# 3️⃣ Data Visualization (Results and Analysis)
# ------------------------------------------------------

# Figure 1: Fraud vs Non-Fraud Count
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette='coolwarm')
plt.title("Figure 1: Fraud vs Non-Fraud Transactions", fontsize=13, fontweight='bold')
plt.xlabel("Class (0 = Legitimate, 1 = Fraudulent)")
plt.ylabel("Number of Transactions")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Figure1_Fraud_vs_NonFraud.png", dpi=300)
plt.show()

# Figure 2: Transaction Amount Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Amount'], bins=50, kde=True, color='steelblue')
plt.title("Figure 2: Transaction Amount Distribution", fontsize=13, fontweight='bold')
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Figure2_Amount_Distribution.png", dpi=300)
plt.show()

# Figure 3: Correlation Heatmap
corr_features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','Amount','Class']
plt.figure(figsize=(10,8))
sns.heatmap(df[corr_features].corr(), cmap='coolwarm', annot=False)
plt.title("Figure 3: Correlation Heatmap of Selected Features", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Figure3_Correlation_Heatmap.png", dpi=300)
plt.show()

# Figure 4: Boxplot - Amount vs Class
plt.figure(figsize=(6,4))
sns.boxplot(x='Class', y='Amount', data=df, palette='coolwarm')
plt.title("Figure 4: Amount Comparison Between Legitimate and Fraudulent Transactions", fontsize=13, fontweight='bold')
plt.xlabel("Class (0 = Legitimate, 1 = Fraudulent)")
plt.ylabel("Transaction Amount ($)")
plt.tight_layout()
plt.savefig("Figure4_Amount_vs_Class.png", dpi=300)
plt.show()

# Figure 5: Scatter Plot - Time vs Amount
plt.figure(figsize=(8,5))
plt.scatter(df['Time'], df['Amount'], c=df['Class'], cmap='coolwarm', alpha=0.6)
plt.title("Figure 5: Transaction Time vs Amount (Color-coded by Class)", fontsize=13, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amount ($)")
plt.tight_layout()
plt.savefig("Figure5_Time_vs_Amount.png", dpi=300)
plt.show()

# ------------------------------------------------------
# 4️⃣ Data Preprocessing
# ------------------------------------------------------
X = df.drop(columns=['Class'])
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Data Split Completed:")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ------------------------------------------------------
# 5️⃣ Model Training (Without XGBoost)
# ------------------------------------------------------

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ------------------------------------------------------
# 6️⃣ Model Evaluation
# ------------------------------------------------------
models = {
    "Logistic Regression": (y_test, y_pred_lr),
    "Random Forest": (y_test, y_pred_rf)
}

results = []
for model_name, (y_true, y_pred) in models.items():
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    results.append([model_name, acc, prec, rec, f1, auc])
    print(f"\n===== {model_name} =====")
    print(classification_report(y_true, y_pred))

# Create DataFrame of results
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
print("\nModel Comparison:\n", results_df)

# ------------------------------------------------------
# 7️⃣ Visualization of Model Performance
# ------------------------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=results_df.melt(id_vars="Model"), x="Model", y="value", hue="variable", palette="viridis")
plt.title("Figure 6: Model Performance Comparison", fontsize=13, fontweight='bold')
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Metrics")
plt.legend(title="Metrics", loc='upper right')
plt.tight_layout()
plt.savefig("Figure6_Model_Performance.png", dpi=300)
plt.show()

# ------------------------------------------------------
# 8️⃣ ROC-AUC Visualization
# ------------------------------------------------------
from sklearn.metrics import RocCurveDisplay

plt.figure(figsize=(7,5))
for model, label in zip([log_reg, rf], ['Logistic Regression', 'Random Forest']):
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=label)

plt.title("Figure 7: ROC Curves for Different Models", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Figure7_ROC_Curves.png", dpi=300)
plt.show()

print("\n✅ All graphs and model results generated successfully.")
print("Figures saved: Figure1–Figure7_*.png")
