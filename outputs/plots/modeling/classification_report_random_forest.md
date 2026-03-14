# Analysis: Classification Report — Random Forest

## What is a Classification Report?
A classification report shows the quality of predictions broken down by outcome class (Stayed vs Left) across three metrics:

- **Precision:** Of all employees the model *predicted* for this class, what fraction was correct? High precision = few false alarms.
- **Recall:** Of all employees who *actually belong* to this class, what fraction did the model catch? High recall = few misses.
- **F1-Score:** The harmonic mean of precision and recall — balances both into a single score. Closer to 1.0 is better.

**Averages:**
- **Macro avg:** Simple average across both classes, treating them equally regardless of size.
- **Weighted avg:** Average weighted by class size.

---

## Results for Random Forest

| Group | Precision | Recall | F1-Score |
|---|---|---|---|
| **Stayed** | 0.993 | 0.994 | 0.994 |
| **Left** | **0.980** | **0.979** | **0.980** |
| Macro avg | 0.987 | 0.986 | 0.987 |
| Weighted avg | 0.990 | 0.990 | 0.990 |

---

## Interpreting Each Number

**For the "Left" group (the critical one):**

**Precision = 0.980**
Of every 100 employees the model flags as potential leavers, 98 genuinely are at risk. Only 2 are false alarms. HR can act on nearly every alert with confidence.

**Recall = 0.979**
Of every 100 employees who actually resigned, the model caught 97.9 of them in advance. Only about 2 slipped through undetected. This is an exceptional result.

**F1-Score = 0.980**
Both precision and recall are nearly equal and both are near 1.0. The model is not trading off one for the other — it achieves excellence on both dimensions simultaneously.

**For the "Stayed" group:**
Precision = 0.993, Recall = 0.994. Nearly perfect identification of employees who will stay. Very few stayers are wasted in false positives.

**Balance across classes:**
Macro avg F1 = 0.987 — the model performs equally well on both the majority class (stayed) and minority class (left). There is no class bias, which confirms that the SMOTE upsampling applied during preprocessing worked correctly.

---

## Three-Model F1 Comparison (for "Left")

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.532 | 0.721 | 0.612 |
| Gradient Boosting | 0.922 | 0.931 | 0.927 |
| **Random Forest** | **0.980** | **0.979** | **0.980** |

---

## HR Decision-Making Implications

- **This is the selected production model.** These are the metrics that govern the quality of risk scores in the retention report.
- With Recall = 97.9%, HR will receive advance warning for **all but ~2%** of future leavers — assuming the workforce characteristics remain similar to the training data.
- With Precision = 98.0%, almost every retention conversation initiated on the basis of a model alert will be with an employee who genuinely needs it.
- The model can realistically be described to leadership as: *"We will catch 98 out of every 100 employees who are about to leave, and we will rarely waste time on employees who are not at risk."*
- **Re-run frequency recommendation:** Run the model monthly on all active employees. Re-train quarterly as new data accumulates to keep the model calibrated to any shifts in workforce behaviour.
