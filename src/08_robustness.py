"""Step 8: robustness of the model v2 results.
(a) Uncertainty: 5-fold CV grouped by state repeated with 20 random fold assignments, and 95% bootstrap
    confidence intervals (2,000 resamples) for the AUC of each model and for the difference to the MCDA score.
(b) Parameter sensitivity: steps 01-06 rerun with low and high residue-to-product ratios and dung rates
    (config.RPR_SCENARIOS, config.DUNG_SCENARIOS); results written to outputs/sensitivity/.
(c) Plant capacity: share of operational plants by capacity class (Wilson 95% intervals) and SHAP dependence."""
import os, sys, json, subprocess, warnings
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import numpy as np, pandas as pd, shap
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

N_SEEDS, N_BOOT = 20, 2000
F = json.load(open('features_v2.json')); FULL, SITE, PROJECT, DEV = F['FULL'], F['SITE'], F['PROJECT'], F['DEV']
m = pd.read_csv('cbg_model_table_v2.csv')
tr = m[m.status.isin(['operational', 'planned'])].reset_index(drop=True)
y = (tr.status == 'operational').astype(int).values

def mk(name):
    if name == 'Random forest':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(500, min_samples_leaf=3, class_weight='balanced', random_state=42))
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         eval_metric='logloss', random_state=42, scale_pos_weight=(y == 0).sum() / (y == 1).sum())

def oof(name, feats, seed):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(mk(name), tr[feats], y, cv=cv, groups=tr.state, method='predict_proba')[:, 1]

z = lambda s: ((s - s.mean()) / s.std()).fillna(0)
mcda = ((z(-tr.log_dist_tapoff) + z(tr.ga_residual_tpd) + z(tr.log_c20_res_total_t) + z(tr.log_c20_dung_t_per_year) + z(-tr.comp_tpd_20km)) / 5).values

# ---------- (a) repeated grouped CV + bootstrap ----------
SETS = {'All factors': FULL, 'Without developer': SITE + PROJECT, 'Location only': SITE}
rng = np.random.default_rng(0)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
def boot_ci(p, ref=None):
    vals = []
    for b in boot_idx:
        if y[b].min() == y[b].max(): continue
        a = roc_auc_score(y[b], p[b]); vals.append(a - roc_auc_score(y[b], ref[b]) if ref is not None else a)
    return np.percentile(vals, [2.5, 97.5])
rows = []
for label, feats in SETS.items():
    for name in ['Random forest', 'XGBoost']:
        aucs, p0 = [], None
        for s in range(N_SEEDS):
            p = oof(name, feats, s); aucs.append(roc_auc_score(y, p))
            if s == 0: p0 = p
        lo, hi = boot_ci(p0); dlo, dhi = boot_ci(p0, mcda)
        rows.append(dict(feature_set=label, model=name, AUC_mean_20cv=np.mean(aucs), AUC_sd_20cv=np.std(aucs),
                         AUC_seed0=roc_auc_score(y, p0), CI95_low=lo, CI95_high=hi,
                         diff_vs_MCDA=roc_auc_score(y, p0) - roc_auc_score(y, mcda), diff_CI95_low=dlo, diff_CI95_high=dhi))
lo, hi = boot_ci(mcda)
rows.append(dict(feature_set='MCDA equal-weight (location factors)', model='no learning', AUC_mean_20cv=roc_auc_score(y, mcda),
                 AUC_sd_20cv=0.0, AUC_seed0=roc_auc_score(y, mcda), CI95_low=lo, CI95_high=hi))
unc = pd.DataFrame(rows).round(3); unc.to_csv('robustness_auc_uncertainty.csv', index=False)
print(unc.to_string(index=False))

# ---------- (b) parameter sensitivity ----------
SCEN = [('rpr_low', 'low', 'base'), ('rpr_high', 'high', 'base'), ('dung_low', 'base', 'low'), ('dung_high', 'base', 'high')]
base_res = pd.read_csv('model_results_v2.csv'); base_shap = pd.read_csv('shap_importance_v2.csv', index_col=0)
sens = []
def summarise(tag, res, shp):
    g = lambda fs: res[(res.feature_set == fs) & (res.data == 'all') & (res.model == 'XGBoost')].AUC.iloc[0]
    rf = lambda fs: res[(res.feature_set == fs) & (res.data == 'all') & (res.model == 'Random forest')].AUC.iloc[0]
    mc = res[res.model == 'no learning'].AUC.iloc[0]
    rho = spearmanr(base_shap.mean_abs_shap, shp.reindex(base_shap.index).mean_abs_shap).correlation
    sens.append(dict(scenario=tag, AUC_XGB_all=g('v2 full (site + project + developer)'), AUC_RF_all=rf('v2 full (site + project + developer)'),
                     AUC_XGB_location_only=g('v2 site only'), AUC_MCDA=mc, SHAP_rank_corr_vs_base=rho,
                     top5_SHAP=', '.join(shp.mean_abs_shap.sort_values(ascending=False).index[:5])))
summarise('base', base_res, base_shap)
for tag, rpr, dung in SCEN:
    out = OUT / 'sensitivity' / tag
    env = dict(os.environ, CBG_RPR=rpr, CBG_DUNG=dung, CBG_OUT=str(out))
    for s in ['01_distance.py', '02_crop_residue.py', '03_livestock.py', '04_ml_model_v1.py', '05_catchment_developer.py', '06_ml_model_v2.py']:
        subprocess.run([sys.executable, os.path.join(_HERE, s)], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summarise(tag, pd.read_csv(out / 'model_results_v2.csv'), pd.read_csv(out / 'shap_importance_v2.csv', index_col=0))
sens = pd.DataFrame(sens); sens.to_csv('robustness_parameter_sensitivity.csv', index=False)
print(sens.round(3).to_string(index=False))

# ---------- (c) capacity ----------
bins = [0, 2, 5, 10, 15, 20, 1000]; labels = ['≤2', '2–5', '5–10', '10–15', '15–20', '>20']
tr['cap_class'] = pd.cut(tr.capacityTpd, bins, labels=labels)
def wilson(k, n, zc=1.96):
    if n == 0: return np.nan, np.nan, np.nan
    p = k / n; d = 1 + zc ** 2 / n; c = (p + zc ** 2 / (2 * n)) / d; h = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h
cap = []
for lab in labels:
    s = tr[tr.cap_class == lab]; k = (s.status == 'operational').sum(); p, lo, hi = wilson(k, len(s))
    k2 = ((s.status == 'operational') & (s.owner != 'RIL')).sum(); n2 = (s.owner != 'RIL').sum(); p2, _, _ = wilson(k2, n2)
    cap.append(dict(capacity_class_tpd=lab, n=len(s), operational=k, share=p, ci_low=lo, ci_high=hi, n_without_RIL=n2, share_without_RIL=p2))
cap = pd.DataFrame(cap); cap.to_csv('robustness_capacity_classes.csv', index=False); print(cap.round(3).to_string(index=False))
c = cap[cap.n > 0]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(range(len(c)), c.share, yerr=[c.share - c.ci_low, c.ci_high - c.share], fmt='o-', color='#1f4e79', capsize=4, label='All plants')
ax.plot(range(len(c)), c.share_without_RIL, 's--', color='#c55a11', label='Excluding Reliance plants')
ax.set_xticks(range(len(c))); ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(c.capacity_class_tpd, c.n)]); ax.set_ylim(0, 1)
ax.set_xlabel('Registered capacity (tonnes per day)'); ax.set_ylabel('Share operational (95% CI)'); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('fig_capacity_share_operational.png', dpi=300); plt.close()

xgb = mk('XGBoost').fit(tr[FULL], y)
sv = shap.TreeExplainer(xgb).shap_values(tr[FULL]); j = FULL.index('capacityTpd')
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(tr.capacityTpd, sv[:, j], s=10, alpha=0.6, c=np.where(tr.owner == 'RIL', '#c55a11', '#1f4e79'))
ax.axhline(0, color='grey', lw=0.8); ax.set_xscale('log')
ax.set_xlabel('Registered capacity (tonnes per day, log scale)'); ax.set_ylabel('SHAP value of capacity (log-odds)')
ax.scatter([], [], c='#1f4e79', s=10, label='Other developers'); ax.scatter([], [], c='#c55a11', s=10, label='Reliance'); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('fig_capacity_shap_dependence.png', dpi=300); plt.close()
