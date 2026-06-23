"""
Heart Disease Prediction Using Machine Learning Classification Models
=====================================================================
Models: Logistic Regression, Decision Tree, Random Forest, Neural Network (MLP)
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & UNDERSTANDING
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  HEART DISEASE PREDICTION — ML PROJECT")
print("=" * 65)

df = pd.read_csv("heart.csv")

FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 (1=True)",
    "restecg":  "Resting ECG Results (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1=Yes)",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak Exercise ST",
    "ca":       "Major Vessels Colored (0-3)",
    "thal":     "Thal: 1=Normal, 2=Fixed, 3=Reversible",
    "target":   "Heart Disease (1=Yes, 0=No)",
}

print("\n[1] DATASET OVERVIEW")
print(f"    Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Target dist : {df['target'].value_counts().to_dict()}")
print(f"    Missing vals: {df.isnull().sum().sum()}")
print("\n    First 5 rows:")
print(df.head().to_string())
print("\n    Descriptive Statistics:")
print(df.describe().round(2).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] DATA CLEANING & PREPROCESSING")

# Handle missing values (replace '?' style if present)
df.replace("?", np.nan, inplace=True)
if df.isnull().sum().sum() > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("    Missing values filled with median.")
else:
    print("    No missing values found.")

# Encode categorical columns (cp, thal already numeric but treat as categorical)
categorical_cols = ["cp", "thal", "restecg", "slope"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"    After encoding — shape: {df_encoded.shape}")

# Features / Target split
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Train-Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train size  : {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA) — saved as eda_plots.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] EXPLORATORY DATA ANALYSIS → eda_plots.png")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Heart Disease — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 3a. Target distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = df["target"].value_counts()
bars = ax0.bar(["No Disease (0)", "Disease (1)"], counts.values,
               color=["#3CB371", "#E05C5C"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(val), ha="center", va="bottom", fontweight="bold")
ax0.set_title("Target Distribution")
ax0.set_ylabel("Count")

# 3b. Age distribution by target
ax1 = fig.add_subplot(gs[0, 1])
for label, color in zip([0, 1], ["#3CB371", "#E05C5C"]):
    ax1.hist(df[df["target"] == label]["age"], bins=15, alpha=0.65,
             color=color, label=f"{'Disease' if label else 'No Disease'}", edgecolor="white")
ax1.set_title("Age Distribution by Target")
ax1.set_xlabel("Age"); ax1.set_ylabel("Frequency")
ax1.legend(fontsize=9)

# 3c. Sex vs Target
ax2 = fig.add_subplot(gs[0, 2])
sex_target = df.groupby(["sex", "target"]).size().unstack(fill_value=0)
sex_target.plot(kind="bar", ax=ax2, color=["#3CB371", "#E05C5C"],
                edgecolor="white", width=0.6)
ax2.set_xticklabels(["Female", "Male"], rotation=0)
ax2.set_title("Sex vs Heart Disease")
ax2.set_ylabel("Count")
ax2.legend(["No Disease", "Disease"], fontsize=9)

# 3d. Correlation heatmap
ax3 = fig.add_subplot(gs[1, :2])
corr = df[["age","trestbps","chol","thalach","oldpeak","target"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax3,
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax3.set_title("Correlation Heatmap (Key Features)")

# 3e. Chest pain type vs target
ax4 = fig.add_subplot(gs[1, 2])
cp_target = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_target.plot(kind="bar", ax=ax4, color=["#3CB371", "#E05C5C"],
               edgecolor="white", width=0.65)
ax4.set_title("Chest Pain Type vs Disease")
ax4.set_xticklabels([f"CP {i}" for i in range(4)], rotation=0, fontsize=8)
ax4.set_ylabel("Count")
ax4.legend(["No Disease", "Disease"], fontsize=9)

# 3f. Max Heart Rate boxplot
ax5 = fig.add_subplot(gs[2, 0])
df.boxplot(column="thalach", by="target", ax=ax5,
           boxprops=dict(color="#555"), medianprops=dict(color="red", linewidth=2))
ax5.set_title("Max Heart Rate by Target")
ax5.set_xlabel("Target (0=No, 1=Yes)"); ax5.set_ylabel("thalach")
plt.sca(ax5); plt.title("Max Heart Rate by Target"); plt.suptitle("")

# 3g. Oldpeak (ST depression)
ax6 = fig.add_subplot(gs[2, 1])
df.boxplot(column="oldpeak", by="target", ax=ax6,
           boxprops=dict(color="#555"), medianprops=dict(color="red", linewidth=2))
ax6.set_title("ST Depression (oldpeak) by Target")
ax6.set_xlabel("Target (0=No, 1=Yes)"); ax6.set_ylabel("oldpeak")
plt.sca(ax6); plt.title("ST Depression by Target"); plt.suptitle("")

# 3h. Cholesterol distribution
ax7 = fig.add_subplot(gs[2, 2])
for label, color in zip([0, 1], ["#3CB371", "#E05C5C"]):
    ax7.hist(df[df["target"] == label]["chol"], bins=15, alpha=0.65,
             color=color, label=f"{'Disease' if label else 'No Disease'}", edgecolor="white")
ax7.set_title("Cholesterol Distribution")
ax7.set_xlabel("Cholesterol"); ax7.set_ylabel("Frequency")
ax7.legend(fontsize=9)

plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved eda_plots.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDING & TRAINING
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] MODEL BUILDING & TRAINING")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Neural Network (MLP)":MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                         random_state=RANDOM_STATE, early_stopping=True),
}

# Use scaled data for LR & MLP, unscaled for tree-based
scaled_models  = {"Logistic Regression", "Neural Network (MLP)"}
results        = {}

for name, model in models.items():
    Xtr = X_train_sc if name in scaled_models else X_train
    Xte = X_test_sc  if name in scaled_models else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "CV-Acc":    cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy").mean(),
    }
    print(f"    ✓ {name:30s} Acc={results[name]['Accuracy']:.3f}  AUC={results[name]['ROC-AUC']:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5] HYPERPARAMETER TUNING — Random Forest (GridSearchCV)")

param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [None, 5, 10],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
y_prob_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

print(f"    Best Params : {grid_search.best_params_}")
print(f"    Tuned RF AUC: {roc_auc_score(y_test, y_prob_rf_tuned):.3f}")

results["Random Forest (Tuned)"] = {
    "model":     best_rf,
    "y_pred":    y_pred_rf_tuned,
    "y_prob":    y_prob_rf_tuned,
    "Accuracy":  accuracy_score(y_test, y_pred_rf_tuned),
    "Precision": precision_score(y_test, y_pred_rf_tuned),
    "Recall":    recall_score(y_test, y_pred_rf_tuned),
    "F1-Score":  f1_score(y_test, y_pred_rf_tuned),
    "ROC-AUC":   roc_auc_score(y_test, y_prob_rf_tuned),
    "CV-Acc":    results["Random Forest"]["CV-Acc"],
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL EVALUATION & VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] MODEL EVALUATION → evaluation_plots.png")

# Summary table
metrics_df = pd.DataFrame({
    name: {k: v for k, v in info.items() if k not in ["model","y_pred","y_prob"]}
    for name, info in results.items()
}).T.round(4)

print("\n    ── Performance Summary ──")
print(metrics_df.to_string())

# ── Plot ──
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Heart Disease Prediction — Model Evaluation", fontsize=15, fontweight="bold")

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
names  = list(results.keys())

# 6a. Accuracy bar
ax = axes[0, 0]
accs = [results[n]["Accuracy"] for n in names]
bars = ax.barh(names, accs, color=COLORS, edgecolor="white")
ax.set_xlim(0.5, 1.0)
ax.set_title("Accuracy Comparison")
ax.set_xlabel("Accuracy")
for bar, val in zip(bars, accs):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

# 6b. ROC-AUC bar
ax = axes[0, 1]
aucs = [results[n]["ROC-AUC"] for n in names]
bars = ax.barh(names, aucs, color=COLORS, edgecolor="white")
ax.set_xlim(0.5, 1.0)
ax.set_title("ROC-AUC Comparison")
ax.set_xlabel("ROC-AUC")
for bar, val in zip(bars, aucs):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

# 6c. All metrics grouped bar
ax = axes[0, 2]
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metric_cols))
width = 0.15
for i, (name, col) in enumerate(zip(names, COLORS)):
    vals = [results[name][m] for m in metric_cols]
    ax.bar(x + i * width, vals, width, label=name, color=col, alpha=0.88)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metric_cols, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0.4, 1.05)
ax.set_title("All Metrics Comparison")
ax.legend(fontsize=6, loc="lower right")
ax.set_ylabel("Score")

# 6d. ROC Curves
ax = axes[1, 0]
for name, col in zip(names, COLORS):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
    auc_val = results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, label=f"{name} ({auc_val:.2f})", color=col, linewidth=1.8)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_title("ROC Curves")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# 6e. Confusion Matrix — Best model (highest AUC)
best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
ax = axes[1, 1]
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            linewidths=1, linecolor="white")
ax.set_title(f"Confusion Matrix\n({best_name})")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# 6f. Feature Importance (best RF)
ax = axes[1, 2]
rf_model = results["Random Forest (Tuned)"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top10 = importances.nlargest(10).sort_values()
colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
bars = ax.barh(top10.index, top10.values, color=colors_imp)
ax.set_title("Top 10 Feature Importances\n(Tuned Random Forest)")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved evaluation_plots.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. CLASSIFICATION REPORTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] CLASSIFICATION REPORTS")
for name, info in results.items():
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, info["y_pred"],
                                target_names=["No Disease", "Disease"]))

# ──────────────────────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"\n  Best Model  : {best_name}")
print(f"  Accuracy    : {results[best_name]['Accuracy']:.4f}")
print(f"  Precision   : {results[best_name]['Precision']:.4f}")
print(f"  Recall      : {results[best_name]['Recall']:.4f}")
print(f"  F1-Score    : {results[best_name]['F1-Score']:.4f}")
print(f"  ROC-AUC     : {results[best_name]['ROC-AUC']:.4f}")
print("\n  Output Files:")
print("   • eda_plots.png        — EDA visualizations")
print("   • evaluation_plots.png — Model evaluation & comparisons")
print("\n  Done!")
