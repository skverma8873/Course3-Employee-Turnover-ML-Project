# Analysis: Confusion Matrix — Logistic Regression

## What is a Confusion Matrix?
A confusion matrix compares what the model predicted against what actually happened, for every employee in the test set (3,000 employees).

| | Model predicted: Stay | Model predicted: Leave |
|---|---|---|
| **Actually Stayed** | True Negative (TN) ✓ | False Positive (FP) ✗ |
| **Actually Left** | False Negative (FN) ✗ | True Positive (TP) ✓ |

- **TN (True Negative):** Correctly identified a stayer — no action needed, no wasted effort.
- **TP (True Positive):** Correctly identified a leaver — HR can intervene in time.
- **FP (False Positive):** Flagged as leaver but they stayed — unnecessary HR intervention.
- **FN (False Negative):** Flagged as stayer but they left — **the costly miss**.

---

## Results for Logistic Regression

**Test set: 3,000 employees (2,286 stayed, 714 left)**

| | Predicted Stay | Predicted Leave |
|---|---|---|
| **Actually Stayed** | TN = **1,833** | FP = **453** |
| **Actually Left** | FN = **199** | TP = **515** |

---

## Interpreting the Numbers

**Missed Departures (FN = 199):**
199 employees who actually left were predicted as safe. The model gave HR no warning for these individuals. They resigned unexpectedly (from HR's perspective), taking their skills and knowledge with them. At an average replacement cost of 6–9 months' salary per employee, this represents significant avoidable cost.

**False Alarms (FP = 453):**
453 employees who stayed were incorrectly flagged as flight risks. HR would have invested retention effort (conversations, salary reviews, adjustments) on employees who were not actually leaving. This wastes HR capacity and budget.

**Correct Catches (TP = 515):**
515 real leavers were correctly identified — HR had a genuine opportunity to intervene with these employees.

---

## Derived Metrics

| Metric | Value | Meaning |
|---|---|---|
| **Recall (Left)** | 515 / (515+199) = **72.1%** | Caught 7 in 10 real leavers |
| **Precision (Left)** | 515 / (515+453) = **53.2%** | Only half of flagged employees were real risks |
| **Overall Accuracy** | (1833+515) / 3000 = **78.3%** | Correct on 78% of all predictions |

---

## HR Decision-Making Implications

- **Recall of 72.1%** means roughly **1 in 3 real leavers is completely missed**. For a company trying to minimise undetected departures, this is a significant gap.
- **Precision of 53.2%** means **every other HR flag is a false alarm** — an unsustainable ratio that wastes time and risks annoying employees who have no intention of leaving.
- This model is **not recommended for production use** in this pipeline. It is included for comparison to demonstrate why more sophisticated models are necessary.
- The Random Forest model reduces missed departures from 199 to just 15 — a 92% improvement in catching real flight risks.
