# Unsupervised Learning — Operational Signatures of Flight Delays

## Research Question

**When a flight is delayed, does it fall into a recognisable operational pattern — or is every delay unique?**

---

## 1. Problem Framing and Method Choice

### Why Unsupervised Learning

The supervised model predicts whether a flight will be delayed. A separate question is: *among flights that are already delayed, do they share recognisable operational patterns?* This is a structure-discovery problem with no ground-truth labels, making unsupervised clustering the right tool.


### Unit of Analysis

Each row is one **individual delayed flight** from the 2024 sample. Only flights with at least one logged delay cause are included.

---

## 2. Data and Feature Engineering

### Dataset

| Stat | Value |
|------|-------|
| Total flights in sample | 10,000 |
| Flights with a logged delay | 2,119 (21.2%) |
| Carriers represented | 15 |
| Origin airports represented | 190 |

Dominant delay cause breakdown: late incoming aircraft (39.6%), carrier fault (29.2%), NAS/ATC (26.6%), weather (4.5%), security (0.2%).

### Clustering Features (Operational Observables Only)

| Feature | What it measures |
|---------|-----------------|
| `dep_delay` | Total departure delay (minutes) |
| `recovery_delta` | `arr_delay − dep_delay` — did the flight recover or compound the delay in air? |
| `taxi_out` | Actual taxi-out time (minutes) |
| `distance` | Route distance (miles) |
| `dep_hour_sin` / `dep_hour_cos` | Cyclical encoding of scheduled departure hour |

**Delay cause codes were deliberately withheld from clustering.** They were held back and used only as a post-hoc validation check in Section 5. This avoids the circular trap of clustering on causes and then "discovering" cause-based groups.

### Preprocessing

- Features standardized with `StandardScaler`
- Cyclical encoding for departure hour so 23:00 and 00:30 are treated as close
- `random_state = 486` for reproducibility

---

## 3. Choosing k

K values 2–8 were evaluated using inertia (elbow) and silhouette score:

| k | Inertia | Silhouette |
|---|---------|------------|
| 2 | 10273.7 | 0.2300 |
| 3 | 8449.9 | 0.2453 |
| 4 | 7557.9 | 0.2469 |
| **5** | **6621.2** | **0.2576** ← best |
| 6 | 5966.0 | 0.2206 |
| 7 | 5531.7 | 0.2221 |
| 8 | 5163.5 | 0.2205 |

**k = 5** was selected based on the silhouette peak.

---

## 4. Cluster Profiles

Five operationally distinct archetypes emerged:

| Archetype | Flights | Dep Delay | Recovery Δ | Taxi Out | Distance | Sched Dep |
|-----------|---------|-----------|------------|----------|----------|-----------|
| Midday Extreme Delay | 70 | 431 min | −9 min | 17 min | 801 mi | 13.3h |
| Evening Moderate Delay | 821 | 64 min | −2 min | 18 min | 671 mi | 18.7h |
| Morning Moderate Delay | 756 | 52 min | +2 min | 19 min | 694 mi | 10.8h |
| Afternoon Long-Haul | 256 | 55 min | +1 min | 21 min | 2,021 mi | 15.4h |
| Afternoon Ground-Hold | 216 | 29 min | +45 min | 57 min | 763 mi | ~14h |

Key observations:
- **Midday Extreme Delay** has by far the largest departure delays (431 min avg) — these are operational meltdowns
- **Afternoon Ground-Hold** has the *smallest* departure delay (29 min) but worsens drastically in flight (+45 min recovery delta) driven by massive taxi-out times (57 min), pointing to ATC ground holds
- **Evening Moderate Delay** is the largest cluster (821 flights) — the typical late-day cascading delay pattern
- Early-day clusters (Morning Moderate) show slightly positive recovery delta, suggesting early delays are more recoverable

---

## 5. Validation: Operational Clusters Predict Delay Cause

Cause composition by cluster (causes were **not** used during clustering):

| Archetype | Carrier% | Weather% | NAS% | Late-Aircraft% |
|-----------|----------|----------|------|----------------|
| Afternoon Ground-Hold | 12.8% | 4.3% | **72.0%** | 10.8% |
| Afternoon Long-Haul | 36.3% | 3.0% | 26.7% | 33.6% |
| Evening Moderate Delay | 25.5% | 3.7% | 19.0% | **51.9%** |
| Midday Extreme Delay | 39.4% | **10.4%** | 9.9% | 40.3% |
| Morning Moderate Delay | **–** | **–** | **–** | **–** |

This is the central non-circular finding:
- **Ground-Hold** → 72% NAS/ATC, consistent with ATC ground hold behavior (high taxi-out, delay grows in flight)
- **Evening Moderate** → 52% late-aircraft, consistent with cascading end-of-day delays
- **Midday Extreme** → elevated weather and carrier fault, consistent with severe disruption events

Operational patterns observable from a flight's trajectory predict root cause without ever seeing the carrier's self-reported cause code.

---

## 6. Delay Severity by Archetype

| Archetype | Flights | Median Arr Delay | Mean Arr Delay | 90th pct | % ≥ 60 min |
|-----------|---------|-----------------|----------------|----------|-------------|
| Midday Extreme Delay | 70 | 359 min | 423 min | 662 min | 100% |
| Afternoon Ground-Hold | 216 | 53 min | 74 min | 160 min | 44.9% |
| Evening Moderate Delay | 821 | 45 min | 62 min | 133 min | 37.3% |
| Afternoon Long-Haul | 256 | 35 min | 56 min | 126 min | 28.5% |
| Morning Moderate Delay | 756 | 36 min | 54 min | 117 min | 29.4% |

---

## 7. Supporting Figures

| Figure | Description |
|--------|-------------|
| `figures/unsupervised_elbow_silhouette.png` | Elbow and silhouette plots justifying k=5 |
| `figures/unsupervised_feature_heatmap.png` | Cluster centroid heatmap — operational profiles |
| `figures/unsupervised_pca_scatter.png` | PCA 2D projection showing cluster separation |
| `figures/unsupervised_cause_heatmap.png` | Cause composition by cluster (validation) |
| `figures/unsupervised_severity.png` | Arrival delay severity distributions by archetype |
| `figures/unsupervised_carrier_breakdown.png` | Carrier representation within each cluster |
| `figures/unsupervised_seasonal.png` | Monthly volume trends by cluster |

---

## 8. Limitations

- **Operational features are downstream of cause:** `dep_delay` and `taxi_out` are consequences of the underlying cause, not causally prior to it. The analysis shows that the operational *manifestation* of delays clusters, not that these are independent causal signals.
- **Sample size:** 2,119 delayed flights limits precision for smaller clusters (e.g. Midday Extreme, n=70).
- **Cause attribution is self-reported:** BTS cause codes are filed by carriers and may understate carrier fault while over-attributing to NAS or weather.
- **No temporal structure:** K-Means treats each flight independently — delays sharing a weather event or airport are not linked in the model.
- **Single year:** 2024 patterns may not generalise to other years.

---

## 9. Conclusion

Delayed flights do fall into recognisable operational patterns. K-Means identifies five distinct archetypes that differ meaningfully across departure delay magnitude, in-flight recovery, taxi-out time, route distance, and time of day. Crucially, these operationally-derived clusters align with known delay root causes even though causes were withheld during training — clusters with high taxi-out skew toward NAS/ATC delays, late-day clusters with poor recovery skew toward cascading late-aircraft delays, and extreme-delay clusters reflect weather and carrier disruptions. The analysis demonstrates that the *how* of a delay (its operational trajectory) carries substantial information about the *why*.
