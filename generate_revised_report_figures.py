from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_outputs"
cv = pd.read_csv(OUT / "cross_validation_results.csv")
test = pd.read_csv(OUT / "independent_test_results.csv")

metrics = ["ROC-AUC Mean", "PR-AUC Mean", "Recall Mean", "Precision Mean", "F1-Score Mean"]
labels = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8), sharey=True)
colors = ["#2c7fb8" if model == "XGBoost" else "#9ecae1" for model in cv["Model"]]
for ax, metric, label in zip(axes, metrics, labels):
    ax.barh(cv["Model"], cv[metric], color=colors)
    ax.set_title(label)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=.2)
fig.suptitle("Five-Fold Stratified Cross-Validation (Training Set Only)", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "cross_validation_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)

selected = test.loc[test["Model"] == "XGBoost"].iloc[0]
fig, ax = plt.subplots(figsize=(10, 4.8)); ax.axis("off")
items = [("Accuracy", selected["Accuracy"]), ("Precision", selected["Precision"]),
         ("Recall", selected["Recall (Sensitivity)"]), ("F1-Score", selected["F1-Score"]),
         ("ROC-AUC", selected["ROC-AUC"]), ("PR-AUC", selected["PR-AUC"])]
for i, (label, value) in enumerate(items):
    x = .08 + (i % 3) * .31; y = .70 - (i // 3) * .42
    ax.add_patch(plt.Rectangle((x, y), .25, .28, facecolor="#e8f4fb", edgecolor="#2c7fb8", linewidth=1.2))
    ax.text(x+.125, y+.19, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x+.125, y+.09, f"{value:.2%}", ha="center", va="center", fontsize=16, color="#174a6e")
ax.text(.5, .03, "Independent test set: n=1,022 (50 stroke-positive observations); threshold=0.50",
        ha="center", fontsize=10)
ax.set_title("Final Weighted XGBoost Evaluation", fontsize=15, fontweight="bold", pad=15)
fig.savefig(OUT / "final_model_performance_dashboard.png", dpi=200, bbox_inches="tight"); plt.close(fig)
