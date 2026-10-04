"""Streamlit app. Run with:  streamlit run app.py"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from explain_text import SECTORS, describe_value, plain_reasons
from llm_explain import DEFAULT_MODEL, ask_groq, build_facts, build_messages

st.set_page_config(page_title="SME Credit Risk Evaluator", layout="wide")



@st.cache_resource
def load():
    model = joblib.load("model/model.joblib")
    feats = json.load(open("model/features.json"))
    metrics = json.load(open("model/metrics.json"))
    sample = pd.read_csv("model/test_sample.csv")
    # Explain in probability units, using a small background set
    explainer = shap.TreeExplainer(
        model,
        data=sample[feats].sample(200, random_state=0),
        model_output="probability",
        feature_perturbation="interventional",
    )
    return model, feats, metrics, sample, explainer


model, feats, metrics, sample, explainer = load()

st.title("SME Credit Risk Evaluator")
st.caption(
    "Gradient boosting model trained on historical SBA-guaranteed small business "
    "loans. Every decision is explained with SHAP."
)

# ---------------- Sidebar: the policy lever ----------------
st.sidebar.header("Lending policy")
cut = st.sidebar.slider(
    "Approve if predicted default risk is below", 0.02, 0.60, 0.20, 0.01,
    format="%.2f",
)
st.sidebar.markdown(
    f"**Model quality (hold-out):** AUC {metrics['auc']:.3f}  \n"
    f"**Portfolio default rate:** {metrics['base_default_rate']:.1%}"
)
st.sidebar.caption("AUC: 0.5 = coin flip, 1.0 = perfect. It measures how well the "
                   "model ranks risky loans above safe ones.")


MAX_AI_CALLS = 8  # per browser session, to protect the free API quota


def groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def ai_explanation(key, risk, base, up, down, X1):
    """Optional AI-written explanation, grounded in the model's own numbers."""
    st.markdown("**AI explanation (optional)**")
    api_key = groq_key()
    if not api_key:
        st.caption("AI explanation is off. Add GROQ_API_KEY to the Streamlit "
                   "secrets to enable it.")
        return
    q = st.text_input("Ask about this decision, or leave blank for a short memo",
                      key=f"{key}_q",
                      placeholder="e.g. What would lower this risk?")
    used = st.session_state.get("ai_calls", 0)
    if st.button("Explain with AI", key=f"{key}_ai"):
        if used >= MAX_AI_CALLS:
            st.warning("AI explanation limit reached for this session.")
            return
        st.session_state["ai_calls"] = used + 1
        try:
            inputs = {f: describe_value(f, v)
                      for f, v in zip(feats, X1.iloc[0].values)}
            facts = build_facts(risk, base, cut, up, down, inputs)
            msgs = build_messages(facts, q)
            model_name = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
            with st.spinner("Writing explanation..."):
                text = ask_groq(msgs, api_key, model_name)
            st.markdown(text)
            st.caption("Written by an AI from the numbers above. It can still "
                       "make mistakes, so check it against the figures.")
        except Exception as e:
            detail = str(e).replace(str(api_key), "[key hidden]")[:300]
            st.warning(f"AI explanation unavailable ({type(e).__name__}): "
                       f"{detail}. The standard explanation above still applies.")


def score_and_explain(X1: pd.DataFrame, key: str = "x"):
    """Show risk, decision, plain-English reasons and the SHAP chart."""
    X1 = X1[feats].astype(float)
    risk = float(model.predict_proba(X1)[0, 1])
    approved = risk < cut
    st.metric("Predicted default risk", f"{risk:.1%}",
              "APPROVE" if approved else "REJECT",
              delta_color="normal" if approved else "inverse")

    sv = explainer(X1)
    summary, up, down = plain_reasons(
        feats, sv.values[0], X1.iloc[0].values,
        float(sv.base_values[0]), risk, cut)

    st.markdown("**Why this decision?**")
    st.markdown(summary)
    c_up, c_down = st.columns(2)
    with c_up:
        st.markdown("**Raises the risk**")
        for _, pts, text in up or [(None, 0, "nothing significant")]:
            st.markdown(f"- {text}" + (f" (+{pts:.1f} pts)" if pts else ""))
    with c_down:
        st.markdown("**Lowers the risk**")
        for _, pts, text in down or [(None, 0, "nothing significant")]:
            st.markdown(f"- {text}" + (f" ({pts:.1f} pts)" if pts else ""))
    st.caption("Points = percentage points of default probability. These are "
               "patterns the model learned from past loans, not proven causes.")

    ai_explanation(key, risk, float(sv.base_values[0]), up, down, X1)

    with st.expander("Technical view: SHAP waterfall chart", expanded=False):
        shap.plots.waterfall(sv[0], show=False, max_display=10)
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)
        st.caption("Starts at the average risk (bottom), then each bar adds or "
                   "removes risk. Red pushes risk up, blue pushes it down.")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Portfolio impact", "Applicant explainer", "New application",
     "Batch scoring", "How to read this"]
)

# ---------------- Tab 1: approval vs default trade-off ----------------
with tab1:
    p, d = sample["pred_risk"].values, sample["default"].values
    grid = np.linspace(0.02, 0.60, 59)
    appr, dflt = [], []
    for t in grid:
        m = p < t
        appr.append(m.mean())
        dflt.append(d[m].mean() if m.sum() else np.nan)

    m = p < cut
    approval_rate = m.mean()
    default_rate = d[m].mean() if m.sum() else float("nan")
    base = d.mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Approval rate", f"{approval_rate:.1%}")
    c2.metric("Default rate of approved loans", f"{default_rate:.1%}",
              f"{(default_rate - base) * 100:+.1f} pts vs approve-all",
              delta_color="inverse")
    c3.metric("Bad loans avoided", f"{d[~m].sum() / max(d.sum(), 1):.0%}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, np.array(appr) * 100, label="Approval rate (%)")
    ax.plot(grid, np.array(dflt) * 100, label="Default rate of approved (%)")
    ax.axvline(cut, color="gray", ls="--", label="Current threshold")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "**Approval rate:** share of applications the policy accepts. "
        "**Default rate of approved:** of the accepted loans, how many went bad. "
        "**Bad loans avoided:** share of all loans that defaulted that the policy "
        "would have rejected. A lower threshold is stricter: fewer defaults, but "
        "fewer loans approved. Measured on a hold-out sample the model never "
        "trained on."
    )

# ---------------- Tab 2: existing applicant + what-if ----------------
with tab2:
    left, right = st.columns([1, 1.4])
    with left:
        idx = st.number_input("Applicant # (from hold-out sample)",
                              0, len(sample) - 1, 0)
        row = sample.loc[idx, feats].astype(float).copy()

        st.markdown("**What-if: edit the application**")
        row["LoanAmount"] = st.number_input(
            "Loan amount ($)", 0.0, 20_000_000.0,
            float(min(row["LoanAmount"], 20_000_000.0)), 5000.0)
        row["Term"] = st.number_input(
            "Term (months)", 0, 700, int(min(row["Term"], 700)))
        row["NoEmp"] = st.number_input(
            "Number of employees", 0, 10000, int(min(row["NoEmp"], 10000)))
        row["SBA_GuaranteePct"] = st.slider(
            "SBA guarantee share", 0.0, 1.0,
            float(min(max(row["SBA_GuaranteePct"], 0.0), 1.0)), 0.05)
        row["NewBusiness"] = int(st.checkbox("New business (<2 years)",
                                             bool(row["NewBusiness"])))

    with right:
        score_and_explain(pd.DataFrame([row]), key="t2")
        st.markdown(
            f"Original outcome in the data: "
            f"**{'Defaulted' if sample.loc[idx, 'default'] == 1 else 'Repaid'}**"
        )

# ---------------- Tab 3: brand-new application, all fields manual ----------------
with tab3:
    st.markdown("Enter a new loan application from scratch.")
    left, right = st.columns([1, 1.4])
    with left:
        a, b = st.columns(2)
        loan = a.number_input("Loan amount ($)", 1000.0, 20_000_000.0,
                              150_000.0, 5000.0, key="m_loan")
        guar = b.slider("SBA guarantee share", 0.0, 1.0, 0.75, 0.05, key="m_guar")
        term = a.number_input("Term (months)", 0, 700, 84, key="m_term")
        emp = b.number_input("Number of employees", 0, 10000, 5, key="m_emp")
        create = a.number_input("Jobs to be created", 0, 10000, 0, key="m_create")
        retain = b.number_input("Jobs to be retained", 0, 10000, 3, key="m_retain")
        sector_label = st.selectbox("Industry (NAICS sector)", list(SECTORS),
                                    index=list(SECTORS).index("44 - Retail (vehicles, home, food)"),
                                    key="m_sector")
        scenario = st.radio(
            "Economic scenario", ["Normal times", "Recession"],
            horizontal=True, key="m_scn",
            help="Scenario, not a fact about the applicant. 'Recession' means "
                 "a downturn hits during the loan's first years (learned from "
                 "the 1990-91, 2001 and 2007-09 recessions).")
        c1, c2 = st.columns(2)
        new_biz = c1.checkbox("New business (<2 years)", False, key="m_new")
        franchise = c2.checkbox("Franchise", False, key="m_fran")
        urban = c1.checkbox("Urban location", True, key="m_urban")
        revolving = c2.checkbox("Revolving line of credit", False, key="m_rev")
        lowdoc = c1.checkbox("LowDoc (fast-track) loan", False, key="m_low")

    X_new = pd.DataFrame([{
        "Term": term, "NoEmp": emp, "NewBusiness": int(new_biz),
        "CreateJob": create, "RetainedJob": retain, "IsFranchise": int(franchise),
        "Urban": int(urban), "RevolvingLine": int(revolving), "LowDoc": int(lowdoc),
        "LoanAmount": loan, "SBA_GuaranteePct": guar,
        "IndustrySector": SECTORS[sector_label],
        "RecessionExposed": int(scenario == "Recession"),
    }])
    with right:
        r_norm = float(model.predict_proba(
            X_new.assign(RecessionExposed=0)[feats].astype(float))[0, 1])
        r_rec = float(model.predict_proba(
            X_new.assign(RecessionExposed=1)[feats].astype(float))[0, 1])
        st.markdown("**Stress test: same applicant, different economy**")
        s1, s2 = st.columns(2)
        s1.metric("Risk in normal times", f"{r_norm:.1%}")
        s2.metric("Risk in a recession", f"{r_rec:.1%}",
                  f"{(r_rec - r_norm) * 100:+.1f} pts", delta_color="inverse")
        score_and_explain(X_new, key="t3")

# ---------------- Tab 4: batch scoring from CSV ----------------
with tab4:
    st.markdown(
        "Upload a CSV of applications to score many at once. It must contain "
        "these columns: `" + "`, `".join(feats) + "`."
    )
    st.download_button(
        "Download CSV template (5 example rows)",
        sample[feats].head(5).to_csv(index=False),
        file_name="applications_template.csv", mime="text/csv",
    )
    up = st.file_uploader("Upload applications CSV", type="csv")

    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        missing = [c for c in feats if c not in raw.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            X = raw[feats].apply(pd.to_numeric, errors="coerce")
            bad = X.isna().any(axis=1)
            out = raw.copy()
            out["pred_risk"] = np.nan
            if (~bad).any():
                out.loc[~bad, "pred_risk"] = model.predict_proba(X[~bad])[:, 1]
            out["decision"] = np.where(
                bad, "INVALID ROW",
                np.where(out["pred_risk"] < cut, "APPROVE", "REJECT"),
            )

            ok = out[~bad]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Applications", f"{len(out):,}")
            c2.metric("Approved", f"{(ok['decision'] == 'APPROVE').mean():.1%}"
                      if len(ok) else "n/a")
            c3.metric("Average risk", f"{ok['pred_risk'].mean():.1%}"
                      if len(ok) else "n/a")
            c4.metric("Invalid rows", f"{int(bad.sum()):,}")

            if len(ok):
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(ok["pred_risk"], bins=30)
                ax.axvline(cut, color="gray", ls="--", label="Threshold")
                ax.set_xlabel("Predicted default risk")
                ax.set_ylabel("Applications")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            shown = out.sort_values("pred_risk", ascending=False)
            st.dataframe(shown.head(200), use_container_width=True)
            st.download_button(
                "Download scored results", out.to_csv(index=False),
                file_name="scored_applications.csv", mime="text/csv",
            )

# ---------------- Tab 5: glossary and method ----------------
with tab5:
    st.markdown("""
### What this app does
A bank lending to small businesses must decide who to approve. This model reads an
application and estimates the **probability that the loan defaults**. The lender then
picks a **threshold**: approve only if risk is below it. A stricter threshold means
fewer losses but also fewer loans made, which is the trade-off in the first tab.

### Key terms
- **Default / charge-off:** the borrower stopped repaying and the lender wrote the
  loss off. This is what the model predicts.
- **SBA (U.S. Small Business Administration):** a government agency that guarantees
  part of certain small-business loans, so the bank loses less if the borrower fails.
- **SBA guarantee share:** the portion of the loan the SBA covers.
- **Term:** how many months the borrower has to repay.
- **Revolving line of credit:** a reusable credit limit (like a credit card), as
  opposed to a one-time fixed loan.
- **LowDoc:** an SBA fast-track program with less paperwork.
- **NAICS sector:** a standard code for the borrower's industry.
- **Economic scenario (recession flag):** whether a U.S. recession hits during
  the loan's first three years. A bank cannot know this at approval, so it is used
  as a what-if stress test rather than a fact about the applicant.
- **Jobs created / retained:** employment the borrower commits to; SBA programs
  aim to support jobs.

### How to read the explanation
The model is a gradient-boosted tree ensemble. To explain one decision I use
**SHAP**: it splits the difference between the average risk and this applicant's
risk across the input features, so you can see which facts pushed the risk up or
down and by how many percentage points.

### Model quality
**AUC** is the chance that the model scores a random defaulted loan as riskier than
a random repaid loan. 0.5 is guessing; 1.0 is perfect. It is measured on loans the
model never saw during training.

### Limitations (read before trusting any number)
- Data covers U.S. SBA loans approved 1990-2010, so it may not reflect today.
- Inflation is not modelled: 1990-2010 had no high-inflation period to learn from.
- The model finds patterns, not causes. Changing one input does not guarantee the
  real-world risk would change the same way.
- Real underwriting also uses credit history, cash flow and collateral, which this
  public dataset does not include.
- A real lender would also test the model for unfair bias before using it.
- This is a portfolio demo, not a lending decision tool.
""")

st.divider()
st.caption(
    "Demo project on public historical SBA data. Not a real lending decision tool."
)
