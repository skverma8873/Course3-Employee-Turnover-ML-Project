# Analysis: Correlation Heatmap

## What is this chart?
A correlation heatmap is a colour-coded grid that shows how strongly every pair of variables in the dataset is related to each other.

- **Dark blue** = strong positive relationship (both values rise together)
- **Dark red** = strong negative relationship (one rises, the other falls)
- **Near-white** = no meaningful relationship

Values range from −1.0 (perfect inverse relationship) to +1.0 (perfect direct relationship).

---

## Key Findings — the `left` column (who leaves)

| Variable | Correlation with `left` | Plain English |
|---|---|---|
| `satisfaction_level` | **−0.39** | Strongest predictor. Lower satisfaction = more likely to leave. |
| `time_spend_company` | **+0.14** | Slightly longer-tenured employees leave a bit more (loyalty can wear thin). |
| `Work_accident` | **−0.15** | Employees who had accidents are slightly *less* likely to leave (possibly due to recovery or compensation). |
| `average_montly_hours` | **+0.07** | Very minor — overworked employees leave marginally more, but effect is weak on its own. |
| `number_project` | **+0.02** | Virtually no direct linear link (the real pattern is non-linear — see project_count_bar). |

---

## Other Notable Relationship

- `number_project` vs `average_montly_hours` = **+0.42** — employees with more projects work significantly longer hours. Workload pressure is real and measurable.

---

## What the Numbers Mean

**−0.39** is a moderate-to-strong negative correlation. In practical terms:
- If you rank all employees from least to most satisfied, those near the bottom are substantially more likely to appear in the "left" group.
- No other single variable approaches this predictive strength.

**Why negative?** The `left` column is coded 1 (left) or 0 (stayed). Satisfaction is 0–1 (higher = happier). So low satisfaction (small number) paired with high departure rate (1) creates a negative correlation.

---

## HR Decision-Making Implications

- **Satisfaction surveys are the single most important early-warning tool** in this dataset — more powerful than hours worked, tenure, or performance scores alone.
- A drop in team satisfaction scores should immediately trigger individual conversations, not wait for the next annual review.
- The `time_spend_company` positive correlation (+0.14) suggests retention risk is not just a new-hire problem — mid-tenure employees (3–5 years) who have not been promoted or recognised become restless.
- Combining low satisfaction + high tenure + many projects creates compounding risk — use the full model (not just this heatmap) to catch those cases.
