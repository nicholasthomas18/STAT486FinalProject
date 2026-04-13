# Deliverable 3: Supervised Modeling

## 1. Problem Context and Research Question

This project analyzes 9,836 non-cancelled U.S. domestic flights from 2024 BTS data. The core question is: **At the moment of departure, can we identify which flights will arrive 15+ minutes late — and for those flagged flights, how late will they be?** This is framed as a two-stage decision pipeline: a classifier that screens all departing flights for delay risk, followed by a regressor that quantifies expected arrival delay for flagged flights.

---

## 2. Supervised Models Implemented

### Feature Set and Leakage Design

Both stages share the same feature set: pre-departure schedule information (scheduled departure/arrival hour, month, day of month, day of week, weekend indicator, scheduled elapsed time, distance, carrier, origin, destination) plus **`dep_delay`** — the departure delay in minutes, which is observable at gate pushback before wheels-off.

Including `dep_delay` is not leakage: it is known to gate agents, crew, and passengers before the flight departs, and represents the most operationally realistic trigger for delay alerts. All truly post-departure columns (taxi_out, wheels_off/on, taxi_in, actual_elapsed_time, air_time, arr_time) and all five delay attribution columns (carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay — computed only after arrival) are excluded.

All preprocessing (StandardScaler on numeric features, OrdinalEncoder on categoricals) is wrapped inside a `sklearn.Pipeline` so that transformations fit only on training folds, never on held-out data.

### Two-Stage Architecture

| Stage | Model | Task | Training Data | Key Hyperparameters Explored | Validation | Final CV Metrics |
|---|---|---|---|---|---|---|
| **Stage 1** | Random Forest | Classification: will arr_delay ≥ 15 min? | All 9,836 flights | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10, None}, `min_samples_split` ∈ {2, 10}; `class_weight='balanced'` | StratifiedKFold k=5, GridSearchCV (ROC-AUC) | Accuracy=0.920, F1=0.807, ROC-AUC=0.921 |
| **Stage 2** | XGBoost Regressor | Regression: how many minutes late? | 2,119 delayed flights only | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {4, 6}, `lr` ∈ {0.05, 0.1}, `subsample` ∈ {0.8, 1.0}, `colsample_bytree` ∈ {0.8, 1.0} | KFold k=5, GridSearchCV (neg RMSE) | RMSE=17.9 min, MAE=12.3 min, R²=0.935 |

Best hyperparameters — Stage 1: `max_depth=10, min_samples_split=10, n_estimators=200`; Stage 2: `colsample_bytree=1.0, learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8`.

Metrics were selected to match each task: F1 and ROC-AUC for classification, given class imbalance (only 21.5% of flights delayed ≥15 min, so accuracy alone is misleading); RMSE, MAE, and R² for regression, where RMSE penalizes large misses and R² measures proportion of variance explained.

---

## 3. Model Comparison and Selection

**Stage 1 — Random Forest Classifier:** With CV ROC-AUC of 0.921 and F1 of 0.807, the classifier performs strongly. Adding `dep_delay` as a feature transformed this from a difficult pure-schedule problem (prior AUC ~0.64) into a well-defined one: a flight already departing 30 minutes late is highly likely to arrive late, and the model captures this signal precisely. The modest gap between train AUC (0.988) and test AUC (0.921) indicates mild overfitting that could be further reduced with smaller `max_depth`, but test performance is strong enough to proceed.

**Stage 2 — XGBoost Regressor:** Restricting to the 2,119 delayed-only flights and adding `dep_delay` raised R² from 0.045 to **0.935** — a dramatic improvement. The departure delay has a 0.975 correlation with arrival delay in the delayed subset, providing strong signal. RMSE of 17.9 minutes means the model's arrival delay predictions are typically within about 18 minutes, a level of precision actionable for airline operations and passenger notifications. The near-identical train R² (0.957) and test R² (0.935) confirm minimal overfitting.

**Why the two-stage framing works:** Predicting exact delay in minutes across all flights is hard because most flights are on time — the target is dominated by near-zero values with a long sparse tail of extreme delays. By first filtering to at-risk flights (Stage 1) and then regressing only within that population (Stage 2), we eliminate the structural noise of on-time flights and let the regressor focus on variance that `dep_delay` and route features can genuinely explain.

**Challenge addressed:** The previous purely pre-departure regression (R² = 0.045) suffered from an under-specified feature set — weather, ATC, and mechanical events dominate delay magnitude but aren't observable pre-departure. `dep_delay` serves as an efficient proxy: by the time a flight departs late, those upstream disruptions are already encoded in the departure time.

---

## 4. Explainability and Interpretability

Gini feature importances were extracted from the Stage 1 Random Forest, and SHAP (SHapley Additive exPlanations) values were computed for the Stage 2 XGBoost regressor on a 500-flight sample of the delayed subset. Both are shown below.

![Feature Importance](feature_importance.png)

**Stage 1 — Random Forest (classification):** `dep_delay` dominates Gini importance by a large margin. This confirms the model is primarily learning a threshold rule — flights with meaningful positive departure delay are almost certain to arrive late. Secondary features including `distance`, `dep_hour`, and `crs_elapsed_time` capture the residual cases: long-haul flights with small departure delays can still accumulate delay en route, and late-day departures face higher ATC congestion.

**Stage 2 — XGBoost SHAP (regression):** `dep_delay` similarly leads SHAP values, which makes physical sense — arrival delay is largely departure delay plus or minus in-flight recovery. `crs_elapsed_time` and `distance` contribute meaningfully, reflecting that longer flights have more opportunity to recover lost time in the air. `dep_hour` and `month` retain importance even in the delayed subset, consistent with afternoon and summer flights facing heavier congestion that amplifies initial delays.

Both models agree on the feature hierarchy, providing consistent and interpretable findings across the pipeline.

---

## 5. Final Takeaways

The two-stage supervised pipeline answers the research question directly and with strong empirical support. At the moment of departure, Stage 1 correctly classifies 92.0% of flights by delay status (ROC-AUC = 0.921, F1 = 0.807), and Stage 2 predicts arrival delay for flagged flights with R² = 0.935 and MAE of 12.3 minutes. Together, the pipeline enables a realistic decision workflow: flag at-risk departures, then provide a quantified delay estimate useful for crew coordination, gate rebooking, and passenger notification.

The central modeling insight is that **departure delay is a near-sufficient statistic for arrival delay** — it efficiently encodes the upstream disruptions (weather, maintenance, ATC) that are the true delay drivers but not directly observable from pre-departure schedule data alone. Route characteristics (distance, elapsed time, departure hour) then modulate how much of that departure delay is recovered or amplified in flight.

Future work should explore whether real-time weather forecasts at origin/destination can improve Stage 1 recall on flights that depart on time but still arrive late (the hardest cases), and whether historical on-time rate per route could replace some of the route-level signal currently proxied by raw distance.

---

*Code:* `03_supervised.py` | *Interpretability artifact:* `feature_importance.png`
*Data path (local):* `C:\Users\erica\stat486proj\code\non_cancelled_flights.csv`
