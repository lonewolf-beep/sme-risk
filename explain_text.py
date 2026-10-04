"""Plain-English helpers for the app (no streamlit / shap imports)."""

SECTORS = {
    "11 - Agriculture": 11, "21 - Mining & oil/gas": 21, "22 - Utilities": 22,
    "23 - Construction": 23, "31 - Manufacturing (food, textiles)": 31,
    "32 - Manufacturing (wood, chemicals)": 32, "33 - Manufacturing (metal, machinery)": 33,
    "42 - Wholesale trade": 42, "44 - Retail (vehicles, home, food)": 44,
    "45 - Retail (general, online)": 45, "48 - Transportation": 48,
    "49 - Warehousing & postal": 49, "51 - Information": 51,
    "52 - Finance & insurance": 52, "53 - Real estate": 53,
    "54 - Professional services": 54, "56 - Admin & support services": 56,
    "61 - Education": 61, "62 - Health care": 62, "71 - Arts & recreation": 71,
    "72 - Accommodation & food": 72, "81 - Other services": 81,
    "0 - Unknown": 0,
}
_CODE_TO_SECTOR = {v: k.split(" - ", 1)[1] for k, v in SECTORS.items()}


def describe_value(feat: str, v: float) -> str:
    """Turn a feature value into a phrase a non-technical reader understands."""
    if feat == "Term":
        return f"loan term of {int(v)} months"
    if feat == "NoEmp":
        return f"{int(v)} employees"
    if feat == "NewBusiness":
        return "new business (under 2 years old)" if v >= 0.5 else "established business (2+ years old)"
    if feat == "CreateJob":
        return f"{int(v)} new jobs promised"
    if feat == "RetainedJob":
        return f"{int(v)} existing jobs to be kept"
    if feat == "IsFranchise":
        return "franchise business" if v >= 0.5 else "not a franchise"
    if feat == "Urban":
        return "urban location" if v >= 0.5 else "rural or unclassified location"
    if feat == "RevolvingLine":
        return ("revolving credit line (reusable limit, like a credit card)"
                if v >= 0.5 else "one-time fixed loan (not a revolving line)")
    if feat == "LowDoc":
        return "fast-track low-paperwork loan" if v >= 0.5 else "full-documentation loan"
    if feat == "LoanAmount":
        return f"loan amount of ${v:,.0f}"
    if feat == "SBA_GuaranteePct":
        return f"{v:.0%} of the loan guaranteed by the SBA"
    if feat == "IndustrySector":
        return f"industry: {_CODE_TO_SECTOR.get(int(v), 'sector code ' + str(int(v)))}"
    if feat == "RecessionExposed":
        return ("recession during the loan's first years" if v >= 0.5
                else "normal economic conditions")
    return f"{feat} = {v}"


def plain_reasons(feats, contribs, values, base, risk, cut, top=3, min_pts=0.5):
    """Build a plain-English summary of a prediction.

    contribs: per-feature SHAP values in probability units (same order as feats)
    values  : per-feature input values
    """
    rows = [(f, c * 100, describe_value(f, v))
            for f, c, v in zip(feats, contribs, values)]
    up = sorted([r for r in rows if r[1] >= min_pts], key=lambda r: -r[1])[:top]
    down = sorted([r for r in rows if r[1] <= -min_pts], key=lambda r: r[1])[:top]

    diff = (risk - base) * 100
    direction = "higher" if diff >= 0 else "lower"
    decision = "APPROVE" if risk < cut else "REJECT"
    summary = (
        f"On average, about {base * 100:.0f} in 100 loans in this data defaulted. "
        f"This application is estimated at **{risk * 100:.0f} in 100**, "
        f"which is {abs(diff):.0f} points {direction} than average. "
        f"At the current policy (approve below {cut:.0%} risk) the decision is "
        f"**{decision}**."
    )
    return summary, up, down
