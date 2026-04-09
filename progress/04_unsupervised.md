# Unsupervised Learning - Airline Operational Anomaly Detection

## Research Question

**Can unsupervised anomaly detection identify carrier-airport-month combinations that behave outside normal operational patterns, and does that add insight beyond supervised high-delay prediction?**

---

## 1. Problem Framing and Method Choice

### Why an Unsupervised Method Is Needed

The supervised model predicts whether a carrier-airport-month is likely to have a high delay rate. That is useful, but it does not answer a separate operational question:

> Is this month structurally unusual compared with normal operating behavior, even if the delay-rate label alone does not fully capture it?

This project uses **Isolation Forest** for that purpose. Isolation Forest is a strong fit because:

- the dataset is tabular and numeric after feature engineering
- anomalies are expected to be relatively rare
- we do not have anomaly labels, so a label-free method is required
- the algorithm scales well to 171K+ rows

### Unit of Analysis

Each row remains one **carrier-airport-month** observation, consistent with supervised analysis.

---

## 2. Feature Construction and Preprocessing

### Inputs to Isolation Forest

The unsupervised feature vector uses operational rates and cause composition:

| Feature | Meaning |
|---------|---------|
| `delay_rate` | Share of arrivals delayed 15+ minutes (`arr_del15 / arr_flights`) |
| `cancel_rate` | Share of arrivals cancelled |
| `divert_rate` | Share of arrivals diverted |
| `mean_delay_mins` | Average delay minutes per arrival |
| `pct_carrier_ct` | Carrier share of delay-cause counts |
| `pct_weather_ct` | Weather share of delay-cause counts |
| `pct_nas_ct` | NAS share of delay-cause counts |
| `pct_security_ct` | Security share of delay-cause counts |
| `pct_late_aircraft_ct` | Late-aircraft share of delay-cause counts |
| `arr_flights` | Monthly arriving-flight volume |

### Preprocessing Choices

- Rows with `arr_flights == 0` are removed to avoid invalid rate calculations.
- Features are standardized with `StandardScaler` so large-scale variables (like flight volume) do not dominate split behavior.
- No label is used during fitting.

---

## 3. Model Implementation and Parameter Rationale

Isolation Forest configuration in `code/04_unsupervised.ipynb`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | `200` | Enough trees for stable anomaly ranking while staying fast |
| `contamination` | `0.05` | Operational assumption that roughly 5% of months are genuinely unusual |
| `random_state` | `42` | Reproducibility |
| `n_jobs` | `-1` | Uses all CPU cores for faster fitting |

The raw decision function is converted to a normalized `anomaly_score` in `[0, 1]` where higher means more anomalous.

---

## 4. Results and Interpretation

### Core Output

On this dataset (171,426 rows after filtering), the model flags:

- **8,572 anomalies** out of 171,426 rows
- anomaly rate = **5.0%**, matching the contamination setting

### Meaningful Patterns

`top_anomalies.csv` is dominated by operationally extreme months such as:

- very high cancellation-rate months (including COVID-disruption periods)
- very high delay-rate months (near-100% delayed arrivals for specific carrier-airport-months)
- unusually large mean delay minutes for lower-volume routes

These are plausible operational outliers rather than random noise, which supports model validity.

### Supporting Visual Outputs

The notebook produces:

- `figures/unsupervised_score_distribution.png`  
  Distribution of anomaly scores and threshold split between normal vs anomalous rows.
- `figures/unsupervised_carrier_month_heatmap.png`  
  Mean anomaly score by carrier and month to identify persistent seasonal/operational irregularities.

---

## 5. Connection to Supervised Analysis

To connect unsupervised output to supervised framing, rows are grouped into anomaly-score quintiles and compared on `high_delay` prevalence (`high_delay = 1` if `arr_del15 / arr_flights > 0.20`).

Observed pattern:

- overall `high_delay` prevalence: **0.379**
- anomalous rows `high_delay` prevalence: **0.453**
- non-anomalous rows `high_delay` prevalence: **0.376**

This shows anomalous rows are materially more delay-prone than normal rows, but not perfectly equivalent to "high delay." That distinction is important: anomaly detection contributes **additional structure** (unusual behavior profiles) rather than duplicating the supervised label.

The notebook also saves:

- `outputs/anomaly_supervised_connection.csv`
- `figures/unsupervised_supervised_connection.png`

These artifacts support a concise claim that anomaly score helps contextualize supervised risk.

---

## 6. Reproducibility and Run Order

### Exact Run Order

1. `code/02_eda.ipynb`
2. `code/03_supervised.ipynb`
3. `code/04_unsupervised.ipynb`
4. `progress/03_supervised.md`
5. `progress/04_unsupervised.md`

### Reproducibility Notes

- Fixed seed: `random_state = 42`
- Deterministic inputs: `data/Airline_Delay_Cause.csv`
- Model outputs saved to `figures/` and `outputs/` for review without rerunning every notebook cell

---

## 7. Final Conclusion

Isolation Forest successfully identifies carrier-airport-month observations with unusually abnormal operational profiles, especially cancellation-heavy or extreme-delay months. This extends the supervised pipeline by adding a reliability and context layer: not just whether a month is high delay, but whether it behaves outside normal historical structure. Together, the supervised and unsupervised components provide a more complete operational view for airline-delay risk analysis.
