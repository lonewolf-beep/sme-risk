# SME Credit Risk Evaluator

**An explainable machine-learning tool that estimates how likely a small-business loan is to default, and shows *why*.**

**Live demo:** https://sme-risk.streamlit.app/
---

## 1. What this project does:

When a bank lends money to a small business, some borrowers never pay it back. That is called a **default**, and the bank loses money.

This project learns from about 900,000 real past loans and uses what it learned to score a new loan application:

> "Loans that looked like this one defaulted about **32%** of the time."

It then helps a lender decide, and explains the score in plain language, so a person can trust it or challenge it.

**Example.** A bank receives an application for a $150,000, 7-year loan to a new restaurant with 5 employees and a 75% government guarantee. You enter those facts into the app. It returns a risk score, an approve/reject suggestion under the lender's chosen policy, and the main reasons behind the score.

> This is decision support, not an autopilot. A human credit officer makes the final call.

---

## 2. Why does it matter?

A lender has to balance two things:

| If the lender is... | Result |
|---|---|
| Too strict | Safe, but turns away good businesses and earns less |
| Too relaxed | Approves many loans, but more of them go bad |

The app makes this trade-off visible with one slider (see "Using the app"). Regulators and auditors also require lenders to justify credit decisions, so a black-box score is not enough. This project explains every score.

---

## 3. How it works

```
Past loans (SBA data, ~900k rows, each labelled "repaid" or "defaulted")
        |
        v   prepare_data.py   -> clean the data, build 13 features
        v   train.py          -> train a gradient boosting model, test it on unseen loans
        v
Saved model  --->  app.py (Streamlit)  --->  risk score + decision + reasons
```

1. **Learn from history.** Each past loan has facts (size, length, industry, ...) and a known outcome. The model finds which combinations of facts tend to go with defaults.
2. **Score a new loan.** The model outputs a probability of default, from 0% to 100%.
3. **Apply a policy.** The lender sets a threshold, for example "approve only if risk is below 20%".
4. **Explain.** The app shows which facts pushed the risk up or down, in plain English and with a SHAP chart.

### The model
A **gradient boosting classifier** (scikit-learn): 150 small decision trees that each fix the mistakes of the ones before. Their combined vote becomes the default probability. It was trained on a stratified 80/20 split, so 20% of loans were held back and never seen during training.

### What goes in (13 facts about the loan, not the person)

| Input | Meaning |
|---|---|
| Term | Loan length in months |
| LoanAmount | Dollar amount approved |
| SBA_GuaranteePct | Share of the loan the government guarantees (0 to 1) |
| NoEmp | Number of employees |
| CreateJob / RetainedJob | Jobs the borrower promises to create / keep |
| NewBusiness | New business (under 2 years old) |
| IsFranchise | Part of a franchise |
| Urban | Urban location |
| RevolvingLine | Reusable credit line (like a credit card) instead of a fixed loan |
| LowDoc | SBA fast-track, low-paperwork program |
| IndustrySector | 2-digit NAICS industry code |
| RecessionExposed | Economic scenario: a recession hits in the loan's first years |

### What comes out
- **Default risk** (a percentage)
- **Decision** (approve or reject, relative to the lender's threshold)
- **Reasons**: the top factors that raise and lower the risk

---

## 4. Using the app

| Tab | What it does |
|---|---|
| **Sidebar slider** | Sets the lending policy: approve only if risk is below this value |
| **Portfolio impact** | Shows what that policy would have done on past loans: approval rate, default rate of approved loans, and share of bad loans avoided |
| **Applicant explainer** | Pick a real past loan and edit its details to see the risk change (what-if) |
| **New application** | Enter every field for a brand-new applicant. Shows the risk in normal times versus a recession (a stress test) |
| **Batch scoring** | Upload a CSV of many applications and get a score for each, plus a downloadable result |
| **How to read this** | Plain-English glossary and limitations, inside the app |

### Optional: AI-written explanation (Groq)
The app can also write a short credit memo or answer questions such as "What would lower this risk?". The model's own numbers (risk and SHAP contributions) are the source of truth, and the language model only turns them into prose. It is grounded on those facts, limited per session, and falls back to the rule-based explanation if the API is unavailable.


---

## 5. Key terms

- **Default / charge-off:** the borrower stopped repaying and the lender wrote the loss off. This is what the model predicts.
- **SBA (U.S. Small Business Administration):** a government agency that guarantees part of certain small-business loans.
- **SBA guarantee:** a promise to cover part of the bank's loss if the borrower defaults. With a 75% guarantee on $100,000, the bank risks about $25,000. The borrower still owes the full debt.
- **Threshold:** the lender's cut-off. Lower is stricter.
- **AUC:** the chance the model scores a random defaulted loan as riskier than a random repaid one. 0.5 is guessing and 1.0 is perfect. It is measured on loans the model never saw.
- **SHAP:** a method that splits the gap between the average risk and this loan's risk across the input facts, so each one's push is visible.
- **Economic scenario (recession flag):** whether a recession hits during the loan's first three years. A bank cannot know this at approval, so it is a what-if stress test and not a fact about the applicant.

---

## 6. Data

- **Source:** "Should This Loan be Approved or Denied?" (U.S. Small Business Administration data on Kaggle), about 899,000 loans. Kaggle: `mirbektoktogaraev/should-this-loan-be-approved-or-denied`.
- **Reference:** Li, Mickel and Taylor (2018), *Journal of Statistics Education*.
- **Used here:** loans approved 1990 to 2010. Later loans are excluded because many had not finished repaying, so their outcomes were unknown.
- **Target:** charged off (default) versus paid in full.
- **No leakage:** only facts known at approval are used as inputs.

## 7. Results

- Hold-out AUC: 0.96
- Effect of the recession flag:

| Threshold | Approval rate | Default rate of approved loans | Bad loans avoided |
|---|---|---|---|
| | | | |

---

## 8. Limitations (please read)

- **Old data.** Training loans date from 1990 to 2010. Lending today differs in loan sizes, rates and industries, so the score is a *risk ranking*, not a forecast for a 2026 loan.
- **Patterns, not causes.** The model finds associations. Changing one input does not guarantee the real risk changes the same way.
- **Missing information.** Real underwriting also uses credit history, cash flow and collateral. This public dataset has none of them.
- **Dollar amounts are not inflation-adjusted.**
- **Inflation is not modelled.** 1990 to 2010 had no high-inflation period to learn from.
- **Fairness is untested.** A real lender would audit the model for bias before use.
- **Demo only.** Not a real lending decision tool.

## 9. Roadmap

- Retrain on the newer SBA 7(a) loan files (FY2010 onward) to make the model more modern.
- Redefine the target as "default within 5 years of approval" so recent loans can be used fairly.
- Out-of-time validation: train on earlier loans, test on later ones.
- Optional LLM-written credit memo built on top of the SHAP results.

---

## 10. Run it yourself

Requires Python 3.12. Commands below are for Windows PowerShell.

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt kagglehub
python download_data.py      # downloads SBAnational.csv into data/
python prepare_data.py       # cleans data and builds features
python train.py              # trains the model, saves model/
streamlit run app.py         # opens the app at http://localhost:8501
```

### Project structure

```
sme-risk/
├── download_data.py    downloads the dataset from Kaggle
├── prepare_data.py     cleaning and feature engineering
├── train.py            model training and evaluation
├── app.py              the Streamlit app
├── explain_text.py     turns SHAP output into plain English
├── llm_explain.py      optional Groq-written explanation
├── requirements.txt    dependencies
├── data/               raw and cleaned data (not committed)
└── model/              trained model, metrics, test sample
```

## 11. Tech stack

Python, pandas, scikit-learn (gradient boosting), SHAP (explainability), Streamlit (web app), matplotlib.
