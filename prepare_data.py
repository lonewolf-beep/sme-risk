"""Step 1: clean SBAnational.csv into a model-ready file.

Input : data/SBAnational.csv  (Kaggle: "SBA National Loan Approval")
Output: data/clean.csv        (numeric features + 'default' target)
"""
import numpy as np
import pandas as pd

RAW = "data/SBAnational.csv"
OUT = "data/clean.csv"


def money(s: pd.Series) -> pd.Series:
    """'$50,000.00 ' -> 50000.0"""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce"
    )


df = pd.read_csv(RAW, low_memory=False, encoding="latin-1")
print("Raw rows:", len(df))

# Target: 1 = charged off (default), 0 = paid in full. Drop unlabeled rows.
df = df[df["MIS_Status"].isin(["CHGOFF", "P I F"])].copy()
df["default"] = (df["MIS_Status"] == "CHGOFF").astype(int)

# Dollar columns arrive as text
for c in ["DisbursementGross", "GrAppv", "SBA_Appv"]:
    df[c] = money(df[c])

# ApprovalFY has a few junk values like '1976A'
df["ApprovalFY"] = pd.to_numeric(
    df["ApprovalFY"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
)

# Loans approved after 2010 are mostly still active, so their outcome is not
# known yet (right-censoring). Keep only loans with a resolved history.
df = df[df["ApprovalFY"].between(1990, 2010)]

# ---- Features (everything below is known AT APPROVAL: no leakage) ----
feat = pd.DataFrame(index=df.index)
feat["Term"] = pd.to_numeric(df["Term"], errors="coerce")               # months
feat["NoEmp"] = pd.to_numeric(df["NoEmp"], errors="coerce")
feat["NewBusiness"] = (pd.to_numeric(df["NewExist"], errors="coerce") == 2).astype(int)
feat["CreateJob"] = pd.to_numeric(df["CreateJob"], errors="coerce")
feat["RetainedJob"] = pd.to_numeric(df["RetainedJob"], errors="coerce")
feat["IsFranchise"] = (pd.to_numeric(df["FranchiseCode"], errors="coerce") > 1).astype(int)
feat["Urban"] = (pd.to_numeric(df["UrbanRural"], errors="coerce") == 1).astype(int)
feat["RevolvingLine"] = (df["RevLineCr"] == "Y").astype(int)
feat["LowDoc"] = (df["LowDoc"] == "Y").astype(int)
feat["LoanAmount"] = df["GrAppv"]
feat["SBA_GuaranteePct"] = (df["SBA_Appv"] / df["GrAppv"]).clip(0, 1)
feat["IndustrySector"] = (
    pd.to_numeric(df["NAICS"], errors="coerce").floordiv(10000)
)  # first 2 digits of NAICS (0 = unknown)
# Economic scenario instead of a calendar year. A loan is "recession exposed"
# if a U.S. recession (NBER: 1990-91, 2001, 2007-09) occurs in the year it was
# approved or the following two years. This is a scenario flag, not something a
# bank knows at approval time.
RECESSION_YEARS = {1990, 1991, 2001, 2007, 2008, 2009}
fy = df["ApprovalFY"]
exposed = sum((fy + k).isin(RECESSION_YEARS) for k in range(3)) > 0
feat["RecessionExposed"] = exposed.astype(int)
feat["default"] = df["default"]

feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
feat["IndustrySector"] = feat["IndustrySector"].astype(int)

print("Clean rows:", len(feat))
print("Default rate: {:.1%}".format(feat["default"].mean()))
feat.to_csv(OUT, index=False)
print("Saved", OUT)
