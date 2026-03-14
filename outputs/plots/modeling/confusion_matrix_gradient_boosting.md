# Analysis: Confusion Matrix — Gradient Boosting

## What is a Confusion Matrix?
A confusion matrix compares what the model predicted against what actually happened, for every employee in the test set (3,000 employees).

| | Model predicted: Stay | Model predicted: Leave |
|---|---|---|
| **Actually Stayed** | True Negative (TN) ✓ | False Positive (FP) ✗ |
| **Actually Left** | False Negative (FN) ✗ | True Positive (TP) ✓ |

- **TN:** Correctly identified a stayer — no wasted effort.
- **TP:** Correctly identified a leaver — HR can intervene in time.
- **FP:** Flagged as leaver but stayed — unnecessary intervention.
- **FN:** Flagged as stayer but actually left — **the costly miss**.

---

## Results for Gradient Boosting

**Test set: 3,000 employees (2,286 stayed, 714 left)**

| | Predicted Stay | Predicted Leave |
|---|---|---|
| **Actually Stayed** | TN = **2,230** | FP = **56** |
| **Actually Left** | FN = **49** | TP = **665** |

---

## Interpreting the Numbers

**Missed Departures (FN = 49):**
49 employees who actually left were not flagged. HR had no warning for these individuals. This is significantly better than Logistic Regression (199 misses) but weaker than Random Forest (15 misses).

**False Alarms (FP = 56):**
56 employees were incorrectly flagged. A low false alarm rate — HR effort invested in these cases would largely be wasted, but the ratio is manageable.

**Total Wrong = 105 out of 3,000 = 96.5% overall accuracy.**

---

## Derived Metrics

| Metric | Value | Meaning |
|---|---|---|
| **Recall (Left)** | 665 / (665+49) = **93.1%** | Catches 93 in every 100 real leavers |
| **Precision (Left)** | 665 / (665+56) = **92.2%** | 92% of flagged employees are genuine risks |
| **Overall Accuracy** | (2230+665) / 3000 = **96.5%** | Correct on 96.5% of all predictions |

---

## Three-Model Comparison

| Metric | Logistic Regression | Gradient Boosting | Random Forest |
|---|---|---|---|
| Missed leavers (FN) | 199 | **49** | 15 |
| False alarms (FP) | 453 | **56** | 14 |
| Recall (Left) | 72.1% | **93.1%** | 97.9% |
| Precision (Left) | 53.2% | **92.2%** | 98.0% |
| Overall Accuracy | 78.3% | **96.5%** | 99.0% |

---

## HR Decision-Making Implications

- Gradient Boosting is a strong model — it catches 93% of real leavers and generates relatively few false alarms.
- In a deployment where Random Forest is unavailable or needs to be retrained, Gradient Boosting is an excellent fallback.
- The 49 missed leavers (vs Random Forest's 15) represent the practical cost of using the second-best model: approximately 34 additional unexpected resignations per 3,000 employees in a prediction cycle.
- Both tree-based models (Random Forest and Gradient Boosting) dramatically outperform Logistic Regression, confirming that employee turnover is driven by non-linear combinations of factors, not simple additive effects.
