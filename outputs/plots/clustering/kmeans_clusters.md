# Analysis: K-Means Clusters (Employee Departure Archetypes)

## What is this chart?
A scatter plot showing **only the employees who already left** the company. Each dot represents one employee, positioned by:

- **X-axis** — Satisfaction Level (0 = completely dissatisfied, 1 = fully satisfied)
- **Y-axis** — Last Evaluation Score (0 = poor performer, 1 = excellent performer)

The K-Means algorithm (k=3) automatically grouped these ex-employees into 3 clusters based on their position in this space. Each cluster is coloured differently; the large **X** marks the cluster centroid (average position).

---

## The 3 Clusters: Who Are These People?

---

### Cluster 0 — "Disengaged Low Performers"
**Centroid: Satisfaction ~0.40, Evaluation ~0.52**
**Colour: Red**

**Profile:**
- Moderate satisfaction — not deeply unhappy, but not content either
- Below-average to average evaluation scores
- Neither happy nor particularly high-performing

**Why they left:**
These employees had mediocre performance records and limited prospects for advancement. They likely left because a better opportunity appeared elsewhere, or because they realised they were unlikely to be promoted and chose to cut their losses. Some may have been managed out informally.

**HR Opportunity Missed:**
Better performance coaching, clearer role expectations, and earlier conversations about career development could have retained a portion of this group. Many of them were not irreparably disengaged — they drifted away from lack of direction.

**Retention Strategy (for current employees at risk of this profile):**
- Performance improvement plans with genuine support (not punitive)
- Skills training and clear progression criteria
- Regular 1:1s focused on professional development, not just task review

---

### Cluster 1 — "Poached High Performers"
**Centroid: Satisfaction ~0.80, Evaluation ~0.91**
**Colour: Green**

**Profile:**
- High satisfaction — genuinely liked working at Portobello Tech
- Very high performance scores — among the best in the company
- Left despite being happy

**Why they left:**
This is the most damaging cluster for the business. These employees were both excellent and content — yet they still left. The only credible explanation is that an external offer was simply too attractive to refuse: significantly higher compensation, a senior title, a more exciting role, equity, or a prestigious employer brand.

**Business Impact:**
Losing your happiest and highest-performing employees is a double blow. These individuals take institutional knowledge, client relationships, and team morale with them. They are also the hardest to replace.

**Retention Strategy (for current employees at risk of this profile):**
- Competitive market salary benchmarking — at least annually
- Long-term incentive structures (bonus, equity, profit-sharing)
- Clear, fast-tracked promotion pathways for top performers
- "Stay bonuses" or project ownership to increase switching cost

---

### Cluster 2 — "Burned-Out Stars"
**Centroid: Satisfaction ~0.12, Evaluation ~0.87**
**Colour: Blue**

**Profile:**
- Extremely low satisfaction — among the most miserable employees in the dataset
- Very high performance scores — the company's top contributors
- Classic burnout profile

**Why they left:**
These employees were delivering excellent work while being profoundly unhappy. The combination is unsustainable — over time, the cognitive dissonance of producing great results for an employer they resent becomes intolerable. They left angry, exhausted, and likely feeling unrecognised and undercompensated relative to what they contributed.

**Business Impact:**
This is the most alarming cluster. These employees did not just leave — they left with grievances. They are most likely to write negative employer reviews, warn colleagues away, and take key knowledge and relationships with them. Their departure often triggers a cascade — other team members notice and start questioning their own situation.

**Retention Strategy (for current employees at risk of this profile):**
- Immediate workload cap and project redistribution
- Explicit recognition and public acknowledgement of contributions
- Compensation review — are they being paid commensurate with their output?
- Promotion or role expansion that gives them ownership and status
- Weekly senior manager check-ins until satisfaction recovers

---

## Why 3 Clusters?

K-Means with k=3 was chosen based on the natural structure of the data. These three archetypes represent meaningfully distinct departure reasons — not just statistical artefacts. Each requires a completely different HR response, which is why a single "retention programme" applied uniformly across all at-risk employees would be ineffective.

---

## HR Decision-Making Implications

| Cluster | Detection Signal | Urgency | Primary Lever |
|---|---|---|---|
| Disengaged Low Performers | Average/low evaluation + flat career | Medium | Coaching, clarity, development |
| Poached High Performers | High evaluation + satisfaction starting to tick down | High | Compensation, fast-track promotion |
| Burned-Out Stars | High evaluation + very low satisfaction | Critical | Immediate workload relief + recognition |

The burned-out stars should always be the first priority — they have the highest value and the lowest tolerance for continued inaction.
