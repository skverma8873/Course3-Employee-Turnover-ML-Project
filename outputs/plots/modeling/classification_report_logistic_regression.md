# Analysis: Classification Report — Logistic Regression

## What is a Classification Report?
A classification report shows the quality of predictions broken down by outcome class (Stayed vs Left) across three metrics:

- **Precision:** Of all employees the model *predicted* for this class, what fraction was correct? High precision = few false alarms.
- **Recall:** Of all employees who *actually belong* to this class, what fraction did the model catch? High recall = few misses.
- **F1-Score:** The harmonic mean of precision and recall — balances both into a single score. Closer to 1.0 is better.

**Averages:**
- **Macro avg:** Simple average across both classes, treating them equally regardless of size.
- **Weighted avg:** Average weighted by class size (since there are more "Stayed" employees than "Left").

---

## Results for Logistic Regression

| Group | Precision | Recall | F1-Score |
|---|---|---|---|
| **Stayed** | 0.902 | 0.802 | 0.849 |
| **Left** | **0.532** | **0.721** | **0.612** |
| Macro avg | 0.717 | 0.762 | 0.731 |
| Weighted avg | 0.818 | 0.784 | 0.796 |

---

## Interpreting Each Number

**For the "Left" group (the critical one):**

**Precision = 0.532**
Of every 100 employees the model flagged as potential leavers, only 53 actually left. The other 47 were false alarms — employees who were going nowhere. HR would be wasting nearly half its retention effort on the wrong people.

**Recall = 0.721**
Of every 100 employees who actually resigned, the model caught 72 in advance. The remaining 28 left without any warning from the system. Roughly **1 in 4 real departures is completely undetected**.

**F1-Score = 0.612**
A balanced score below 0.70 is considered poor for a business-critical detection task. This model's combined performance on the turnover detection job is unreliable.

**For the "Stayed" group:**
The model performs reasonably well at identifying stayers (F1 = 0.849), but that is the easier task — the majority class. The challenge is detecting the minority class (leavers), which is where Logistic Regression falls short.

---

## Why These Numbers Matter for HR

In employee retention, the two failure modes have very different costs:

| Failure | Metric | Logistic Regression | Business Cost |
|---|---|---|---|
| Missed leaver | False Negative (Recall gap) | 28% of leavers missed | Unplanned vacancy, replacement cost (6–9× monthly salary), knowledge loss |
| False alarm | False Positive (Precision gap) | 47% of flags are noise | Wasted HR time, unnecessary compensation spend, risk of making employees feel surveilled |

At Logistic Regression's recall of 72.1%, a company with 300 annual departures would **miss ~84 of them entirely** — paying full replacement costs with zero forewarning.

---

## Conclusion

Logistic Regression is included in this analysis for educational comparison. Its F1-Score of 0.612 for the "Left" class makes it unsuitable as a production retention tool. The Random Forest model (F1 = 0.980) is the correct choice for this use case.
