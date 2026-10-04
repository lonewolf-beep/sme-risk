"""Step 2: train the model and save everything the app needs.

Input : data/clean.csv
Output: model/model.joblib         trained gradient boosting model
        model/features.json        feature order
        model/metrics.json         AUC etc.
        model/test_sample.csv      scored hold-out sample used by the app
"""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/clean.csv")
X, y = df.drop(columns="default"), df["default"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Keep training fast: sklearn's GradientBoosting is single-threaded.
if len(Xtr) > 200_000:
    Xtr, _, ytr, _ = train_test_split(
        Xtr, ytr, train_size=200_000, random_state=SEED, stratify=ytr
    )

print(f"Training on {len(Xtr):,} rows...")
model = GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=SEED,
)
model.fit(Xtr, ytr)

proba = model.predict_proba(Xte)[:, 1]
metrics = {
    "auc": round(float(roc_auc_score(yte, proba)), 4),
    "pr_auc": round(float(average_precision_score(yte, proba)), 4),
    "base_default_rate": round(float(y.mean()), 4),
    "n_train": int(len(Xtr)),
    "n_test": int(len(Xte)),
}
print(metrics)

joblib.dump(model, "model/model.joblib", compress=3)
json.dump(list(X.columns), open("model/features.json", "w"))
json.dump(metrics, open("model/metrics.json", "w"), indent=2)

# Small scored sample for the app (keeps the deployed repo light)
sample = Xte.copy()
sample["default"] = yte.values
sample["pred_risk"] = proba
sample = sample.sample(n=min(5000, len(sample)), random_state=SEED)
sample.reset_index(drop=True).to_csv("model/test_sample.csv", index=False)
print("Saved model/ files. Done.")
