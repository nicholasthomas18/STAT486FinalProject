# Deliverable 3: Supervised Modeling

## 1. Problem Context and Research Question

This project analyzes a sample of 9,836 non-cancelled U.S. domestic flights from 2024 BTS data to answer: **Can we predict whether (and by how much) a flight will be delayed at arrival, using only information available before departure?** Two complementary supervised tasks are pursued — binary classification (delayed ≥15 min or not) and continuous regression (minutes of arrival delay).

---

## 2. Supervised Models Implemented

All features are constructed solely from **pre-departure schedule information**: scheduled departure/arrival hour (extracted from HHMM integer), month, day of month, day of week, a weekend indicator, scheduled elapsed time, distance, carrier code, origin airport, and destination airport. Post-departure columns (actual departure time, taxi times, wheels-off/on, actual elapsed time, air time) and all five delay attribution columns (carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay) were explicitly excluded to prevent data leakage — those values are only known after the event occurs.

Preprocessing (StandardScaler on numeric features, OrdinalEncoder on categoricals with `handle_unknown='use_encoded_value'`) was wrapped in a `sklearn.Pipeline` so all transformations occur strictly within each cross-validation fold. The 99th-percentile cap (217 minutes) was applied to the regression target to reduce the influence of extreme outliers (e.g., multi-hour diversions) that are essentially unpredictable from schedule data alone.

| Model | Task | Key Hyperparameters Explored | Validation Setup | Final CV Metrics |
|---|---|---|---|---|
| **Random Forest** | Classification (delayed ≥15 min) | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10, None}, `min_samples_split` ∈ {2, 10}; `class_weight='balanced'` | StratifiedKFold k=5, GridSearchCV (ROC-AUC) | Accuracy=0.713, F1=0.371, ROC-AUC=0.643 |
| **XGBoost Regressor** | Regression (arrival delay, minutes) | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {4, 6}, `lr` ∈ {0.05, 0.1}, `subsample` ∈ {0.8, 1.0}, `colsample_bytree` ∈ {0.8, 1.0} | KFold k=5, GridSearchCV (neg RMSE) | RMSE=40.0 min, MAE=24.7 min, R²=0.045 |

Best hyperparameters selected by GridSearchCV: Random Forest — `max_depth=10, min_samples_split=10, n_estimators=200`; XGBoost — `colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=100, subsample=1.0`.

Metrics were chosen to match task type: F1 and ROC-AUC for classification (appropriate given class imbalance — only 21.5% of flights delayed ≥15 min, so accuracy alone is misleading); RMSE, MAE, and R² for regression (RMSE penalizes large misses, MAE is interpretable in minutes, R² indicates proportion of variance explained).

---

## 3. Model Comparison and Selection

**Classification (Random Forest):** With a CV ROC-AUC of 0.643 and F1 of 0.371, the classifier captures meaningful signal — it performs better than random guessing (AUC > 0.5) and correctly identifies a meaningful share of delayed flights. The relatively low F1 reflects the inherent class imbalance (only ~21.5% delayed) even with `class_weight='balanced'`. A notable gap between training AUC (0.907) and test AUC (0.643) indicates moderate overfitting — the model is learning some training-specific patterns. Regularizing further (smaller `max_depth`, larger `min_samples_split`) could narrow this gap at the cost of some discrimination power.

**Regression (XGBoost):** CV R² of 0.045 and RMSE of ~40 minutes reveal that predicting the *exact* delay in minutes from schedule features alone is very difficult. This is expected: the dominant drivers of delay magnitude — weather events, ATC ground stops, and mechanical issues — are largely absent from pre-departure schedule data. The model does better than the naive mean-prediction baseline (R² > 0), but the gap is small. The train R² of 0.119 vs test 0.045 suggests mild overfitting as well, consistent with limited signal in the feature set.

**Key insight:** Classification is substantially more tractable than regression for this feature set. Identifying *whether* a flight will be significantly delayed is easier than predicting *by how much*, because the binary threshold captures systematic patterns (e.g., certain routes, times of day, and carriers are chronically more prone to delay) whereas exact delay magnitude is dominated by stochastic events not measurable from schedules. This motivates future work adding weather and NAS congestion features.

**Challenges:** Class imbalance for classification, high variance in delay magnitude for regression, and the fundamental limitation that the most predictive delay drivers (weather, maintenance) are not available as pre-departure features.

---

## 4. Explainability and Interpretability

SHAP (SHapley Additive exPlanations) values were computed for the XGBoost regressor, and Gini-based feature importances were extracted from the Random Forest, both shown below.

![Feature Importance](feature_importance.png)

**SHAP interpretation (XGBoost — regression):** `month` emerges as the top SHAP feature, reflecting strong seasonal patterns in U.S. air travel — summer and holiday months (June–August, November–December) see systematically higher delays due to increased traffic volume and more frequent weather disruptions. `distance` and `crs_elapsed_time` are also highly ranked, consistent with longer flights having greater exposure to cumulative schedule slippage. `dep_hour` contributes meaningfully, capturing the well-known "afternoon/evening delay cascade" where delays compound across aircraft rotations throughout the day.

**Random Forest interpretation (classification):** `distance` tops the Gini importance ranking, with `crs_elapsed_time` close behind — confirming that route characteristics drive delay risk. `dep_hour` and `month` also feature prominently. Both models converge on the same set of influential features, increasing confidence that these are genuine signals rather than model-specific artifacts.

---

## 5. Final Takeaways

The supervised modeling analysis confirms that **pre-departure schedule features carry real but limited predictive signal for flight delays**. The Random Forest classifier achieves a ROC-AUC of 0.643, meaningfully better than random, and successfully identifies systematic patterns linking route length, time of day, and season to delay risk. The XGBoost regressor's low R² (0.045) demonstrates that predicting exact delay magnitude from schedules alone is extremely hard — the residual variance is dominated by events (weather, ATC) not observable pre-departure.

These results directly answer the research question: yes, we *can* predict delay likelihood from pre-departure features with moderate accuracy, but predicting magnitude requires richer real-time inputs. The consistent importance of `month`, `distance`, `dep_hour`, and `crs_elapsed_time` across both models points toward the most actionable features for a production delay alert system. The next modeling step should incorporate historical on-time performance rates per route/carrier as engineered features, and explore integrating weather forecast data at origin/destination, which prior literature identifies as the single strongest delay predictor.