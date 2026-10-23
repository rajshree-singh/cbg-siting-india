"""Step 4: first ML model — what separates OPERATIONAL from PLANNED (not started) CBG plants?
Label: 1 = operational, 0 = planned. Under-construction plants are excluded (outcome not yet known).
Validation: 5-fold CV grouped by state (the test state is never seen in training), and plain stratified 5-fold CV.
Models: logistic regression (baseline), random forest, XGBoost.
Explanation: SHAP on XGBoost.
Sensitivity runs: (a) without feedstock category 'OTHERS'; (b) only plants with livestock data (6 states);
(c) without Reliance (RIL) plants."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

m = pd.read_csv('cbg_plants_features_v3.csv')
m = m[m.lat.notna() & (m.state != 'Unknown')].copy()

# ---------- features ----------
m['log_dist_tapoff'] = np.log1p(m.dist_tapoff_km)
m['within_75km'] = (m.dist_tapoff_km <= 75).astype(int)
m['ga_residual_tpd'] = m.residualCapacityTpd
m['ga_intake_tpd'] = m.intakeCapacityTpd
m['log_res_total'] = np.log1p(m.res_total_t)
m['log_res_rice'] = np.log1p(m.res_rice_t)
m['log_res_sugarcane'] = np.log1p(m.res_sugarcane_t)
m['log_dung'] = np.log1p(m.dung_t_per_year)
m['plants_in_district'] = m.groupby(['state', 'district']).plantId.transform('size')
m['plants_in_GA'] = m.groupby('gaId').plantId.transform('size')
m['tpd_in_GA'] = m.groupby('gaId').capacityTpd.transform('sum')
m['ga_headroom_ratio'] = m.ga_residual_tpd / m.tpd_in_GA.replace(0, np.nan)   # buyer capacity vs competing supply
FEED = ['AGR RES', 'PRESS MUD', 'MIXED', 'MUNICIPAL WASTE', 'CATTLE DUNG', 'SPENT WASH', 'OTHERS']
for f in FEED:
    m['feed_' + f.replace(' ', '_').lower()] = (m.feedstockCategory == f).astype(int)
NUM = ['capacityTpd', 'log_dist_tapoff', 'within_75km', 'ga_residual_tpd', 'ga_intake_tpd', 'ga_headroom_ratio',
       'log_res_total', 'log_res_rice', 'log_res_sugarcane', 'log_dung', 'plants_in_district', 'plants_in_GA']
FEATS = NUM + ['feed_' + f.replace(' ', '_').lower() for f in FEED]
m.to_csv('cbg_model_table_v1.csv', index=False)

def models():
    return {
        'Logistic regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight='balanced')),
        'Random forest': make_pipeline(SimpleImputer(strategy='median'),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                              class_weight='balanced', random_state=42)),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric='logloss', random_state=42),
    }

def evaluate(d, feats, label):
    X, y, g = d[feats], d.y, d.state
    out = {}
    for cvname, cv, groups in [('grouped-by-state', StratifiedGroupKFold(5, shuffle=True, random_state=42), g),
                               ('random', StratifiedKFold(5, shuffle=True, random_state=42), None)]:
        for name, mdl in models().items():
            if name == 'XGBoost':
                mdl.set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            p = cross_val_predict(mdl, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]
            out[(cvname, name)] = dict(AUC=round(roc_auc_score(y, p), 3),
                                       BalAcc=round(balanced_accuracy_score(y, p > 0.5), 3),
                                       F1=round(f1_score(y, p > 0.5), 3))
    r = pd.DataFrame(out).T
    r.index.names = ['CV', 'model']
    print(f'\n=== {label}: n={len(d)} (operational {int(d.y.sum())}, planned {int((d.y == 0).sum())}) ===')
    print(r.to_string())
    return r

train = m[m.status.isin(['operational', 'planned'])].copy()
train['y'] = (train.status == 'operational').astype(int)

res_main = evaluate(train, FEATS, 'MAIN — all features')
feats_no_feed = [f for f in FEATS if not f.startswith('feed_')]
res_nofeed = evaluate(train, feats_no_feed, 'WITHOUT feedstock type (location factors only)')
t2 = train[train.feedstockCategory != 'OTHERS']
res_noothers = evaluate(t2, [f for f in FEATS if f != 'feed_others'], 'SENSITIVITY — drop OTHERS feedstock')
t3 = train[train.log_dung.notna()]
res_lv = evaluate(t3, FEATS, 'SENSITIVITY — only 6 states with livestock data')

# ---------- simple MCDA-style baseline (equal-weight score, like traditional siting) ----------
def mcda(d):
    s = pd.DataFrame({
        'near_pipe': -d.log_dist_tapoff,
        'buyer': d.ga_residual_tpd,
        'residue': d.log_res_total,
        'dung': d.log_dung.fillna(d.log_dung.median()),
    })
    s = (s - s.mean()) / s.std()
    return s.mean(axis=1)
print('\nMCDA-style equal-weight score AUC (no learning):', round(roc_auc_score(train.y, mcda(train)), 3))

# ---------- final XGBoost on all training rows + SHAP ----------
import shap
X, y = train[FEATS], train.y
xgb = models()['XGBoost'].set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum()).fit(X, y)
sv = shap.TreeExplainer(xgb).shap_values(X)
imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False)
print('\nSHAP mean |value| (XGBoost, all features):'); print(imp.round(3).to_string())
imp.to_csv('shap_importance_v1.csv', header=['mean_abs_shap'])

# direction of effect: correlation between feature value and its SHAP value
dirn = {f: np.corrcoef(X[f].fillna(X[f].median()), sv[:, i])[0, 1] if X[f].nunique() > 1 else np.nan
        for i, f in enumerate(FEATS)}
print('\nDirection (+ = higher value pushes towards OPERATIONAL):')
print(pd.Series(dirn).reindex(imp.index).round(2).to_string())

# location-only model SHAP too
xgb2 = models()['XGBoost'].set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum()).fit(train[feats_no_feed], y)
sv2 = shap.TreeExplainer(xgb2).shap_values(train[feats_no_feed])
imp2 = pd.Series(np.abs(sv2).mean(0), index=feats_no_feed).sort_values(ascending=False)
print('\nSHAP (location-only model):'); print(imp2.round(3).to_string())

# plots
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(); shap.summary_plot(sv, X, show=False, max_display=12); plt.tight_layout(); plt.savefig('fig_shap_summary_v1.png', dpi=200); plt.close()

pd.concat({'main': res_main, 'location_only': res_nofeed, 'no_OTHERS': res_noothers, 'livestock_states': res_lv}) \
    .to_csv('model_results_v1.csv')

# ---------- sensitivity: without Reliance (RIL) plants ----------
train['owner'] = train.name.str.extract(r'^P-([A-Z0-9]+)-')[0]
noril = train[train.owner != 'RIL']
res_noril = evaluate(noril, FEATS, 'SENSITIVITY — without Reliance (RIL) plants')
xg = models()['XGBoost'].set_params(scale_pos_weight=(noril.y == 0).sum() / (noril.y == 1).sum()).fit(noril[FEATS], noril.y)
s3 = shap.TreeExplainer(xg).shap_values(noril[FEATS])
imp3 = pd.Series(np.abs(s3).mean(0), index=FEATS).sort_values(ascending=False)
print('\nSHAP without RIL:'); print(imp3.head(10).round(3).to_string())
imp3.to_csv('shap_importance_noRIL_v1.csv', header=['mean_abs_shap'])
pd.concat({'main': res_main, 'location_only': res_nofeed, 'no_OTHERS': res_noothers, 'livestock_states': res_lv, 'no_RIL': res_noril}) \
    .to_csv('model_results_v1.csv')
