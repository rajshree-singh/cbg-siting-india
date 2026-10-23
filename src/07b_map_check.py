"""Out-of-state validation of the siting map: rebuild cell scores with models that never saw the state being scored
(train on other states, score that state's cells), then test whether operational plants sit in higher-scoring
cells than planned plants (AUC of nearest-cell score)."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
F = json.load(open('features_v2.json')); FULL = F['FULL']
plants = pd.read_csv('cbg_model_table_v2.csv'); cells = pd.read_csv('siting_grid_cells_v1.csv')
tr = plants[plants.status.isin(['operational', 'planned'])].copy(); tr['y'] = (tr.status == 'operational').astype(int)
# nearest cell for each plant (lat/lon -> rough km)
def nn(alat, alon, blat, blon):
    out = np.empty(len(alat), int); d = np.empty(len(alat))
    for i, (a, b) in enumerate(zip(alat, alon)):
        dd = np.hypot((blat - a) * 111, (blon - b) * 111 * np.cos(np.radians(a))); out[i] = dd.argmin(); d[i] = dd.min()
    return out, d
j, d = nn(tr.lat.values, tr.lng.values, cells.lat.values, cells.lon.values)
tr['cell'] = j; tr['cell_d'] = d
design = {f: 0 for f in FULL if f.startswith('feed_') or f.startswith('dev_')}; design.update(capacityTpd=10, feed_mixed=1, dev_n_plants=1)
X = pd.DataFrame({f: (cells[f] if f in cells else 0) for f in FULL}); [X.__setitem__(k, v) for k, v in design.items()]
oof = np.full(len(tr), np.nan)
for trn, tst in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, tr.y, tr.state):
    a = tr.iloc[trn]
    mdl = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, scale_pos_weight=(a.y == 0).sum() / (a.y == 1).sum()).fit(a[FULL], a.y)
    oof[tst] = mdl.predict_proba(X.iloc[tr.cell.values[tst]][FULL])[:, 1]
ok = tr.cell_d <= 15
print('plants within 15 km of a candidate cell:', ok.sum(), '/', len(tr))
print('OUT-OF-STATE map check AUC (operational vs planned, reference design):', round(roc_auc_score(tr.y[ok], oof[ok]), 3))
print('median out-of-state cell score: operational', round(np.median(oof[ok & (tr.y == 1)]), 3), ' planned', round(np.median(oof[ok & (tr.y == 0)]), 3))
