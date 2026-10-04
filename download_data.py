"""Downloads SBAnational.csv from Kaggle into data/ (run once)."""
import os
import shutil

import kagglehub

path = kagglehub.dataset_download(
    "mirbektoktogaraev/should-this-loan-be-approved-or-denied"
)
print("Downloaded to:", path, os.listdir(path))

os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(path, "SBAnational.csv"), "data/SBAnational.csv")
print("Copied to data/SBAnational.csv")
