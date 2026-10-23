"""Step 6: ML model v2 = v1 + developer markers + 20-km catchment feedstock + nearby competition.
Same task and validation as v1 (operational vs planned; 5-fold CV grouped by state)."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, warnings, shap, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

m = pd.read_csv('cbg_model_table_v2.csv')
FEED = [c for c in m.columns if c.startswith('feed_')]
SITE = ['log_dist_tapoff', 'ga_residual_tpd', 'ga_intake_tpd', 'ga_headroom_ratio',
        'log_c20_res_total_t', 'log_c20_res_rice_t', 'log_c20_res_wheat_t', 'log_c20_res_sugarcane_t', 'log_c20_dung_t_per_year',
        'comp_tpd_20km', 'comp_tpd_75km']
PROJECT = ['capacityTpd'] + FEED
DEV = ['dev_n_plants', 'dev_repeat', 'dev_psu', 'dev_large_corp']
SETS = {'v2 full (site + project + developer)': SITE + PROJECT + DEV,
        'v2 without developer': SITE + PROJECT,
        'v2 site only': SITE}
m.to_csv('cbg_model_table_v2.csv', index=False)
train = m[m.status.isin(['operational', 'planned'])].copy()
train['y'] = (train.status == 'operational').astype(int)

def mk(name, y):
    if name == 'Logistic regression':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
    if name == 'Random forest':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(500, min_samples_leaf=3, class_weight='balanced', random_state=42))
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         eval_metric='logloss', random_state=42, scale_pos_weight=(y == 0).sum() / (y == 1).sum())

def cv_eval(d, feats):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42); r = {}
    for name in ['Logistic regression', 'Random forest', 'XGBoost']:
        p = cross_val_predict(mk(name, d.y), d[feats], d.y, cv=cv, groups=d.state, method='predict_proba')[:, 1]
        r[name] = (round(roc_auc_score(d.y, p), 3), round(balanced_accuracy_score(d.y, p > 0.5), 3))
    return r

rows = []
for label, feats in SETS.items():
    for subset, d in [('all', train), ('without RIL', train[train.owner != 'RIL'])]:
        for model, (auc, ba) in cv_eval(d, feats).items():
            rows.append(dict(feature_set=label, data=subset, model=model, AUC=auc, BalAcc=ba, n=len(d)))
# MCDA-style baseline on the same site factors (equal weights)
z = lambda s: ((s - s.mean()) / s.std()).fillna(0)   # missing -> average
mc = (z(-train.log_dist_tapoff) + z(train.ga_residual_tpd) + z(train.log_c20_res_total_t) +
      z(train.log_c20_dung_t_per_year.fillna(train.log_c20_dung_t_per_year.median())) + z(-train.comp_tpd_20km)) / 5
rows.append(dict(feature_set='MCDA equal-weight (site factors)', data='all', model='no learning', AUC=round(roc_auc_score(train.y, mc), 3), BalAcc=np.nan, n=len(train)))
res = pd.DataFrame(rows); print(res.to_string(index=False)); res.to_csv('model_results_v2.csv', index=False)

# final models + SHAP
FULL = SETS['v2 full (site + project + developer)']
xgb = mk('XGBoost', train.y).fit(train[FULL], train.y)
sv = shap.TreeExplainer(xgb).shap_values(train[FULL])
imp = pd.Series(np.abs(sv).mean(0), index=FULL).sort_values(ascending=False)
dirn = pd.Series({f: np.corrcoef(train[f].fillna(train[f].median()), sv[:, i])[0, 1] if train[f].nunique() > 1 else np.nan
                  for i, f in enumerate(FULL)})
out = pd.DataFrame({'mean_abs_shap': imp, 'direction_corr': dirn.reindex(imp.index)}).round(3)
print('\nSHAP v2 full:'); print(out.head(15).to_string()); out.to_csv('shap_importance_v2.csv')
plt.figure(); shap.summary_plot(sv, train[FULL], show=False, max_display=14); plt.tight_layout(); plt.savefig('fig_shap_summary_v2.png', dpi=200); plt.close()

nr = train[train.owner != 'RIL']
xgb_nr = mk('XGBoost', nr.y).fit(nr[FULL], nr.y)
sv_nr = shap.TreeExplainer(xgb_nr).shap_values(nr[FULL])
print('\nSHAP v2 full, without RIL (top 10):'); print(pd.Series(np.abs(sv_nr).mean(0), index=FULL).sort_values(ascending=False).head(10).round(3).to_string())

# save final model (all features) for the siting step
xgb.save_model('xgb_v2_full.json')
json.dump({'FULL': FULL, 'SITE': SITE, 'PROJECT': PROJECT, 'DEV': DEV}, open('features_v2.json', 'w'))

