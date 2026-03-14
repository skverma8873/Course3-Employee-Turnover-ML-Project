# Analysis: All Distributions

## What is this chart?
Three histograms stacked vertically — one each for satisfaction level, last evaluation score, and average monthly hours. Each bar shows how many employees fall within a particular value range. The shape of the distribution (not just the average) reveals hidden workforce segments.

---

## Distribution 1 — Satisfaction Level (0 to 1)

**Shape: Bimodal (two humps)**

- Large spike near **~0.10** — a cluster of deeply unhappy employees
- Separate peak around **~0.75** — the majority of reasonably happy employees

**What this means:**
The workforce is NOT uniformly unhappy. There is a distinct sub-group of employees who are extremely dissatisfied. These are the imminent flight risks. A company-wide average of ~0.61 hides this split entirely — reporting the average would give a false sense of security.

**HR Implication:**
- Do not manage by average. Identify and segment the low-satisfaction tail (below 0.30).
- This group is small but disproportionately likely to leave.
- Targeted engagement, 1:1 conversations, and workload/recognition reviews for this group are urgent.

---

## Distribution 2 — Last Evaluation Score (0 to 1)

**Shape: Bimodal (two humps)**

- Peak near **~0.50** — average performers
- Peak near **~0.85** — high performers

**What this means:**
Portobello Tech has two distinct performance tiers. The high-performer peak is concerning because high performers are also the employees most likely to be poached by competitors (see clustering analysis). If the company loses its top-evaluated employees, institutional knowledge and productivity will suffer disproportionately.

**HR Implication:**
- High-evaluation employees (0.75+) need proactive retention: competitive compensation, clear promotion paths, and meaningful work.
- Average performers (0.45–0.60) who are also dissatisfied represent a different risk — disengagement rather than poaching.

---

## Distribution 3 — Average Monthly Hours (96 to 310)

**Shape: Bimodal (two humps)**

- Hump around **~150 hours/month** — normal workload (~37 hours/week)
- Second hump around **~250 hours/month** — severely overworked (~62 hours/week)

**What this means:**
A significant group of employees is working 60+ hour weeks consistently. This is not a minor outlier — it is a structural problem. Sustained overwork causes burnout, health decline, and departure. The 250-hour group is a ticking clock.

**HR Implication:**
- Workload redistribution is not optional — it is a retention intervention.
- Project assignment reviews for anyone consistently logging 200+ hours/month.
- Consider whether the 250-hour group is being compensated appropriately for the extra load, and whether their role expectations are realistic.

---

## The Common Thread

All three distributions share the same pattern: **two distinct employee populations exist within the same company.** Standard HR metrics (averages, company-wide surveys) will always miss the at-risk sub-population. The ML model in this pipeline is specifically designed to find and flag that hidden group before they resign.
