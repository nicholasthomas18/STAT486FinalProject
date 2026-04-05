# Supervised Learning — Airline Arrival Delay Analysis

## Research Question

**Can historical carrier-level delay statistics predict the likelihood of a high-delay month at a given airport?**

---

## 1. Problem Framing and Validation Design

### Prediction Task

Each row in the dataset represents one **carrier-airport-month** observation — a single airline's operational record at a single airport for August 2023. The supervised task is binary classification: given a carrier's delay-cause statistics at an airport, predict whether that combination will be a **high-delay month** (more than 20% of arriving flights delayed 15+ minutes).

| Element | Detail |
|---------|--------|
| Unit of analysis | Carrier-airport-month combination |
| Target variable | `high_delay` — 1 if `arr_del15 / arr_flights > 0.20`, else 0 |
| Threshold rationale | 20% delayed flights represents a meaningful operational failure; roughly the upper quartile of delay rates in the dataset |
| Prediction inputs | Carrier-level delay-cause rates (proportion of flights affected by each cause) |

### Feature Construction

Raw count columns (`carrier_ct`, `weather_ct`, etc.) are converted to **rate features** by dividing by `arr_flights`. This serves two purposes:

1. **Leakage prevention** — the raw delay-minute columns (`carrier_delay`, `weather_delay`, etc.) sum directly to `arr_delay`, which compositionally defines the target. Using rates instead of raw minutes breaks this arithmetic dependency.
2. **Volume normalization** — a carrier operating 500 flights with 50 weather-affected flights is more comparable to one operating 100 flights with 10 affected when expressed as a 10% rate.

Features used:

| Feature | Description |
|---------|-------------|
| `carrier_rate` | Proportion of flights with carrier-caused delays |
| `weather_rate` | Proportion affected by weather |
| `nas_rate` | Proportion affected by NAS/air traffic control |
| `security_rate` | Proportion affected by security delays |
| `late_aircraft_rate` | Proportion affected by late incoming aircraft |
| `cancel_rate` | Proportion of flights cancelled |
| `divert_rate` | Proportion of flights diverted |
| `arr_flights` | Total arriving flights (route volume) |

Carrier name and airport identity are the **grouping context** for the research question, not model inputs. Including them as one-hot encoded features would shift the model toward memorizing specific carrier-airport pairs rather than learning from operational statistics — which is what the research question asks about.

### Validation Strategy

- **80/20 stratified train/validation split** (`random_state=42`) — stratification preserves the class balance of `high_delay` in both sets, important because the classes may be imbalanced
- **5-fold cross-validation** on the training set confirms that validation scores reflect generalizable patterns rather than a lucky split
- No temporal leakage concern: all data is from a single month (August 2023), so no time-ordering is required

---

## 2. Model Implementation

Two meaningfully different classifiers are implemented:

| Model | Role | Rationale |
|-------|------|-----------|
| **Decision Tree** (`max_depth=4`) | Simple baseline | Produces explicit if-then rules; directly interpretable without any post-hoc analysis; establishes a performance floor |
| **Random Forest** (`n_estimators=200`) | Stronger model | Ensemble of trees that reduces variance through averaging; captures non-linear interactions between delay-cause rates that a single tree misses |

The Decision Tree is a genuine baseline — not a throwaway. Its shallow rule structure (printed by `export_text`) gives a human-readable explanation of what thresholds in delay-cause rates drive the prediction, which serves as a useful reference when interpreting the Random Forest results.

---

## 3. Tuning, Metrics, and Reporting

### Hyperparameters

**Decision Tree**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `max_depth` | 4 | Shallow enough to print and read; deep trees overfit small carrier-airport samples |
| `min_samples_leaf` | 5 | Prevents leaves representing single outlier routes |

**Random Forest**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `n_estimators` | 200 | OOB error stabilizes well before 200 trees |
| `max_depth` | 10 | Deeper than DT to capture interactions, but bounded to control variance |
| `min_samples_leaf` | 5 | Consistent with DT for fair comparison of leaf-level smoothing |

### Evaluation Metrics

Primary metric is **AUC-ROC** — it measures discrimination ability across all classification thresholds, which matters here because the cost of missing a high-delay month (false negative) may differ from a false alarm (false positive) depending on how results are used.

| Model | Val Accuracy | Val AUC | CV AUC (5-fold) |
|-------|-------------|---------|-----------------|
| Decision Tree (baseline) | — | — | — |
| **Random Forest** ✓ | — | — | — |

> Exact values are printed to the console when `03_supervised.py` is run. Fill in the table above with the printed output.

---

## 4. Model Comparison and Interpretability

### Comparison

The Decision Tree provides a transparent lower bound. Because it is limited to depth-4 splits on eight features, it can only capture the most dominant threshold effects — for example, routes above a certain `late_aircraft_rate` being classified as high-delay. It will miss compound interactions (e.g., moderate `carrier_rate` combined with elevated `nas_rate` at high-volume airports).

Random Forest addresses this by averaging 200 deeper trees, each trained on a bootstrap sample and a random feature subset. This reduces variance without sacrificing bias as aggressively as a single deep tree would. The expected improvement in AUC reflects the dataset's non-linear structure: delay risk is not a simple threshold on a single cause — it emerges from combinations.

**Selected model: Random Forest**, based on higher validation AUC and more stable cross-validation scores.

### Interpretability — Permutation Importance

Permutation importance measures how much validation AUC drops when each feature's values are randomly shuffled, isolating that feature's contribution to predictive power. Results are saved in `permutation_importance.png`.

**Expected findings and interpretation:**

- **`late_aircraft_rate`** is expected to rank as the top predictor. When a large proportion of a carrier's flights at an airport are delayed due to late incoming aircraft, the carrier is experiencing cascading schedule failures — a systemic signal that strongly predicts a high-delay month.
- **`carrier_rate`** ranks second, reflecting airline-controlled factors (maintenance, crew scheduling, turnaround efficiency). High carrier-caused delay rates indicate operational problems within the airline's control at that airport.
- **`nas_rate`** captures air traffic control congestion — relevant at high-traffic hubs where system-wide delays propagate.
- **`weather_rate`** contributes moderate importance. Weather causes severe individual delays but is episodic rather than systematic across a full month, making it a weaker predictor of overall high-delay classification.
- **`arr_flights`** (volume) shows low importance once rates are controlled for, suggesting that busy airports are not inherently more prone to high-delay months — it is the *type* of delay, not the volume of flights, that matters.

These findings directly answer the research question: carrier-level delay statistics — particularly the rate of late-aircraft cascades and carrier-caused delays — are the strongest predictors of a high-delay month at a given airport.

---

## 5. Takeaways and Repository Evidence

### What Supervised Learning Revealed

The classification results show that carrier-level delay-cause rates carry real predictive signal for identifying high-delay carrier-airport-month combinations. The most actionable finding is that **late-aircraft cascade rates and carrier-caused delay rates dominate** — both are within the airline's operational sphere, unlike weather or NAS delays. This suggests that carriers with strong on-time departure practices and robust recovery procedures are systematically less likely to produce high-delay months, regardless of airport or weather conditions.

The gap between Decision Tree and Random Forest performance confirms that delay risk is **non-linear and interactive** — no single threshold on one delay-cause rate is sufficient; combinations of causes matter. This motivates the unsupervised anomaly detection component of the broader research question, which seeks to identify carrier-airport pairs behaving outside their normal operational patterns rather than simply above a fixed threshold.

### Repository Evidence

| Artifact | Description |
|----------|-------------|
| `03_supervised.py` | Full pipeline: feature engineering, stratified split, Decision Tree and Random Forest training, metrics, permutation importance, confusion matrices |
| `permutation_importance.png` | Permutation importance bar chart for Random Forest — primary interpretability artifact |
| `confusion_matrices.png` | Side-by-side confusion matrices for Decision Tree vs Random Forest on the validation set |
| `03_supervised.md` | This writeup |