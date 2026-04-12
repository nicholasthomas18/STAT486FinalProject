"""
Flight Delay Prediction - Supervised Modeling
Data: C:\\Users\\erica\\stat486proj\\code\\non_cancelled_flights.csv (10,000 rows)

Two tasks:
  1. Classification  - Random Forest: will arrival delay be >= 15 min?
  2. Regression      - XGBoost: how many minutes will the flight be delayed?

Leakage guard: only pre-departure schedule features used.
All preprocessing inside sklearn Pipeline / cross-validation folds.
"""

import pandas as pd
import numpy as np
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     cross_validate, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
DATA_PATH = 'C:/Users/erica/stat486proj/figures/non_cancelled_flights.csv'
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

df = df.dropna(subset=['arr_delay'])
print(f"After dropping missing arr_delay: {len(df)} rows")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (pre-departure only)
# ─────────────────────────────────────────────
df['dep_hour']   = df['crs_dep_time'] // 100
df['arr_hour']   = df['crs_arr_time'] // 100
df['is_weekend'] = df['day_of_week'].isin([6,7]).astype(int)

# EXCLUDED (post-departure / leaky):
#   dep_time, dep_delay, taxi_out, wheels_off, wheels_on, taxi_in,
#   actual_elapsed_time, air_time, arr_time,
#   carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay

FEATURES = [
    'month', 'day_of_month', 'day_of_week', 'dep_hour', 'arr_hour',
    'is_weekend', 'crs_elapsed_time', 'distance',
    'op_unique_carrier', 'origin', 'dest'
]

cat_cols = ['op_unique_carrier', 'origin', 'dest']
num_cols = [c for c in FEATURES if c not in cat_cols]

X = df[FEATURES].copy()
y_clf = (df['arr_delay'] >= 15).astype(int)

p99 = df['arr_delay'].quantile(0.99)
y_reg = df['arr_delay'].clip(upper=p99)

print(f"\nClass balance (delayed>=15): {y_clf.mean():.1%} positive ({y_clf.sum()} flights)")
print(f"Delay stats (clipped at {p99:.0f} min):\n{y_reg.describe()}")

# ─────────────────────────────────────────────
# 3. SHARED PREPROCESSOR
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1), cat_cols)
], remainder='drop')

# ─────────────────────────────────────────────
# 4. MODEL 1 - RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 1: Random Forest Classifier")
print("="*60)

rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])

param_grid_rf = {
    'clf__n_estimators':     [100, 200],
    'clf__max_depth':        [5, 10, None],
    'clf__min_samples_split': [2, 10],
}

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=cv_strat,
                     scoring='roc_auc', n_jobs=-1, refit=True, verbose=1)
gs_rf.fit(X, y_clf)

best_rf = gs_rf.best_estimator_
print(f"Best RF params: {gs_rf.best_params_}")

rf_cv = cross_validate(best_rf, X, y_clf, cv=cv_strat,
                       scoring=['accuracy','f1','roc_auc'], return_train_score=True)

rf_metrics = {
    'accuracy':      rf_cv['test_accuracy'].mean(),
    'accuracy_std':  rf_cv['test_accuracy'].std(),
    'f1':            rf_cv['test_f1'].mean(),
    'f1_std':        rf_cv['test_f1'].std(),
    'roc_auc':       rf_cv['test_roc_auc'].mean(),
    'roc_auc_std':   rf_cv['test_roc_auc'].std(),
    'train_roc_auc': rf_cv['train_roc_auc'].mean(),
}
print(f"CV Accuracy  : {rf_metrics['accuracy']:.3f} +/- {rf_metrics['accuracy_std']:.3f}")
print(f"CV F1        : {rf_metrics['f1']:.3f} +/- {rf_metrics['f1_std']:.3f}")
print(f"CV ROC-AUC   : {rf_metrics['roc_auc']:.3f} +/- {rf_metrics['roc_auc_std']:.3f}")
print(f"Train ROC-AUC: {rf_metrics['train_roc_auc']:.3f}  (vs test {rf_metrics['roc_auc']:.3f})")

# ─────────────────────────────────────────────
# 5. MODEL 2 - XGBOOST REGRESSOR
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2: XGBoost Regressor")
print("="*60)

xgb_pipe = Pipeline([
    ('pre', preprocessor),
    ('reg', xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1))
])

param_grid_xgb = {
    'reg__n_estimators':     [100, 200],
    'reg__max_depth':        [4, 6],
    'reg__learning_rate':    [0.05, 0.1],
    'reg__subsample':        [0.8, 1.0],
    'reg__colsample_bytree': [0.8, 1.0],
}

cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

gs_xgb = GridSearchCV(xgb_pipe, param_grid_xgb, cv=cv_kfold,
                      scoring='neg_root_mean_squared_error',
                      n_jobs=-1, refit=True, verbose=1)
gs_xgb.fit(X, y_reg)

best_xgb = gs_xgb.best_estimator_
print(f"Best XGB params: {gs_xgb.best_params_}")

xgb_cv = cross_validate(best_xgb, X, y_reg, cv=cv_kfold,
                        scoring=['neg_root_mean_squared_error',
                                 'neg_mean_absolute_error', 'r2'],
                        return_train_score=True)

xgb_metrics = {
    'rmse':      -xgb_cv['test_neg_root_mean_squared_error'].mean(),
    'rmse_std':   xgb_cv['test_neg_root_mean_squared_error'].std(),
    'mae':       -xgb_cv['test_neg_mean_absolute_error'].mean(),
    'r2':         xgb_cv['test_r2'].mean(),
    'r2_std':     xgb_cv['test_r2'].std(),
    'train_r2':   xgb_cv['train_r2'].mean(),
}
print(f"CV RMSE  : {xgb_metrics['rmse']:.2f} +/- {xgb_metrics['rmse_std']:.2f} min")
print(f"CV MAE   : {xgb_metrics['mae']:.2f} min")
print(f"CV R2    : {xgb_metrics['r2']:.3f} +/- {xgb_metrics['r2_std']:.3f}")
print(f"Train R2 : {xgb_metrics['train_r2']:.3f}  (vs test {xgb_metrics['r2']:.3f})")

# ─────────────────────────────────────────────
# 6. EXPLAINABILITY
# ─────────────────────────────────────────────
print("\nComputing SHAP values on 500-row sample...")
best_xgb.fit(X, y_reg)
best_rf.fit(X, y_clf)

X_sample = X.sample(500, random_state=42)
X_transformed = best_xgb['pre'].transform(X_sample)
feature_names_out = num_cols + cat_cols

explainer  = shap.TreeExplainer(best_xgb['reg'])
shap_vals  = explainer.shap_values(X_transformed)
mean_shap  = np.abs(shap_vals).mean(axis=0)
shap_order = np.argsort(mean_shap)[::-1]

rf_imp     = best_rf['clf'].feature_importances_
rf_order   = np.argsort(rf_imp)[::-1]

# ─────────────────────────────────────────────
# 7. FIGURE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')
BG = '#1a1d2e'

n = len(feature_names_out)

axes[0].barh(
    [feature_names_out[i] for i in shap_order[::-1]],
    mean_shap[shap_order[::-1]],
    color=plt.cm.Blues(np.linspace(0.4, 0.9, n))
)
axes[0].set_facecolor(BG)
axes[0].tick_params(colors='white', labelsize=9)
axes[0].set_xlabel('Mean |SHAP value| (minutes)', color='white', fontsize=10)
axes[0].set_title('XGBoost SHAP Feature Importance\n(Delay Regression)', color='white', fontweight='bold')
for sp in axes[0].spines.values(): sp.set_edgecolor('#444')

axes[1].barh(
    [feature_names_out[i] for i in rf_order[::-1]],
    rf_imp[rf_order[::-1]],
    color=plt.cm.Greens(np.linspace(0.4, 0.9, n))
)
axes[1].set_facecolor(BG)
axes[1].tick_params(colors='white', labelsize=9)
axes[1].set_xlabel('Gini Importance', color='white', fontsize=10)
axes[1].set_title('Random Forest Feature Importance\n(Delay Classification)', color='white', fontweight='bold')
for sp in axes[1].spines.values(): sp.set_edgecolor('#444')

plt.tight_layout(pad=2)
plt.savefig('/home/claude/feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("Saved feature_importance.png")

# ─────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)
print(f"""
Random Forest Classifier
  Best params  : {gs_rf.best_params_}
  CV Accuracy  : {rf_metrics['accuracy']:.3f} +/- {rf_metrics['accuracy_std']:.3f}
  CV F1        : {rf_metrics['f1']:.3f} +/- {rf_metrics['f1_std']:.3f}
  CV ROC-AUC   : {rf_metrics['roc_auc']:.3f} +/- {rf_metrics['roc_auc_std']:.3f}
  Train ROC-AUC: {rf_metrics['train_roc_auc']:.3f}

XGBoost Regressor
  Best params  : {gs_xgb.best_params_}
  CV RMSE      : {xgb_metrics['rmse']:.2f} min
  CV MAE       : {xgb_metrics['mae']:.2f} min
  CV R2        : {xgb_metrics['r2']:.3f} +/- {xgb_metrics['r2_std']:.3f}
  Train R2     : {xgb_metrics['train_r2']:.3f}

Top SHAP feature (XGB): {feature_names_out[shap_order[0]]}
Top Gini feature  (RF): {feature_names_out[rf_order[0]]}
""")

# Save for markdown
all_metrics = {
    **rf_metrics,
    **{f'xgb_{k}': v for k, v in xgb_metrics.items()},
    'top_shap_feat': feature_names_out[shap_order[0]],
    'top_rf_feat':   feature_names_out[rf_order[0]],
    'rf_best':  str(gs_rf.best_params_),
    'xgb_best': str(gs_xgb.best_params_),
    'p99_cap':  float(p99),
}
with open('/home/claude/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("Saved metrics.json")
