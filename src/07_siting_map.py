"""Step 7: siting map. Score candidate locations for a NEW CBG plant.
Candidates: centres of a 15-km grid over India, kept if within 75 km of a tap-off IP/SV (75-km connection radius used by the GOBARdhan portal and the DPI scheme).
Each cell gets the same SITE features as plants (distance to nearest tap-off, city-gas area capacity,
20-km catchment residue/dung, registered competition within 20/75 km). Project and developer features are held
fixed at a reference design, so differences between cells come from LOCATION only.
Reference design A (main): 10 TPD, MIXED feedstock, first-time independent developer.
Design B (sensitivity): 20 TPD, crop-residue feedstock, first-time independent developer.
Each tap-off point is then ranked by the best cell within 25 km of it."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, geopandas as gpd, json, re, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import geo_match as G
from catchment_developer import catchment, P, EQ

F = json.load(open('features_v2.json')); FULL = F['FULL']
xgb = XGBClassifier(); xgb.load_model('xgb_v2_full.json')
plants = pd.read_csv('cbg_model_table_v2.csv', dtype={'gaId': str})
tap = pd.read_csv(IPSV)
tap = tap[tap.tapOffAvailable == True].dropna(subset=['lat', 'lng']).reset_index(drop=True)
ga = pd.read_csv(GA, dtype={'gaId': str})
ga['gaId'] = ga.gaId.str.strip(); plants['gaId'] = plants.gaId.str.strip()
ga = ga.merge(plants.groupby('gaId').capacityTpd.sum().rename('tpd_in_GA'), on='gaId', how='left').fillna({'tpd_in_GA': 0})

tapg = gpd.GeoDataFrame(tap, geometry=gpd.points_from_xy(tap.lng, tap.lat), crs=4326).to_crs(EQ)
india = P.dissolve().geometry.iloc[0]
xmin, ymin, xmax, ymax = india.bounds
S = 15000
xs, ys = np.meshgrid(np.arange(xmin, xmax, S) + S / 2, np.arange(ymin, ymax, S) + S / 2)
cells = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=EQ)
cells = cells[cells.within(india)].reset_index(drop=True)
txy = np.c_[tapg.geometry.x, tapg.geometry.y]
cxy = np.c_[cells.geometry.x, cells.geometry.y]
def nearest(a, b):
    d = np.empty(len(a)); j = np.empty(len(a), int)
    for i in range(0, len(a), 2000):
        D = np.sqrt(((a[i:i + 2000, None, :] - b[None, :, :]) ** 2).sum(-1))
        j[i:i + 2000] = D.argmin(1); d[i:i + 2000] = D.min(1)
    return d / 1000, j
cells['dist_tapoff_km'], cells['near_tap'] = nearest(cxy, txy)
cells = cells[cells.dist_tapoff_km <= 75].reset_index(drop=True)
cells['pid'] = np.arange(len(cells))
print('candidate cells within 75 km of a tap-off:', len(cells))

# district (2011 polygon) of each cell -> state
cells = gpd.sjoin(cells, P[['censuscode', 'DISTRICT', 'ST_NM', 'ks', 'geometry']], how='left', predicate='within').drop(columns='index_right')
cells = cells.drop_duplicates('pid').reset_index(drop=True)

# city-gas area: districts named in the GA name (same state); otherwise GA of nearest registered plant in same state
hits = []
for _, r in ga.iterrows():
    nm = re.sub(r'[^a-z]', '', str(r.gaName).lower()); ks = G.STATE_ALIAS.get(G.norm(r.stateName), G.norm(r.stateName))
    for _, d in P[P.ks == ks].iterrows():
        kd = G.norm(d.DISTRICT)
        if len(kd) >= 4 and kd in nm: hits.append((d.censuscode, r.gaId))
d2ga = pd.DataFrame(hits, columns=['censuscode', 'gaId']).merge(ga, on='gaId')
d2ga = d2ga.groupby('censuscode').agg(ga_residual_tpd=('residualCapacityTpd', 'mean'), ga_intake_tpd=('intakeCapacityTpd', 'mean'), tpd_in_GA=('tpd_in_GA', 'mean'))
cells = cells.merge(d2ga, left_on='censuscode', right_index=True, how='left')
pg = gpd.GeoDataFrame(plants.drop(columns=['residualCapacityTpd', 'intakeCapacityTpd', 'tpd_in_GA'], errors='ignore'), geometry=gpd.points_from_xy(plants.lng, plants.lat), crs=4326).to_crs(EQ).merge(
    ga[['gaId', 'residualCapacityTpd', 'intakeCapacityTpd', 'tpd_in_GA']], on='gaId', how='left')
pxy = np.c_[pg.geometry.x, pg.geometry.y]
dp, jp = nearest(np.c_[cells.geometry.x, cells.geometry.y], pxy)
miss = cells.ga_residual_tpd.isna() & (dp <= 100)
cells.loc[miss, 'ga_residual_tpd'] = pg.residualCapacityTpd.values[jp[miss]]
cells.loc[miss, 'ga_intake_tpd'] = pg.intakeCapacityTpd.values[jp[miss]]
cells.loc[miss, 'tpd_in_GA'] = pg.tpd_in_GA.values[jp[miss]]
cells['ga_source'] = np.where(~miss & cells.ga_residual_tpd.notna(), 'district-name', np.where(miss, 'nearest-plant', 'none'))
cells['ga_headroom_ratio'] = cells.ga_residual_tpd / cells.tpd_in_GA.replace(0, np.nan)
print('GA assignment:', cells.ga_source.value_counts().to_dict())

# catchment feedstock + competition
cols = ['res_total_t', 'res_rice_t', 'res_wheat_t', 'res_sugarcane_t', 'dung_t_per_year']
c20 = catchment(cells[['pid', 'geometry']], 20, cols).add_prefix('c20_')
cells = cells.merge(c20, left_on='pid', right_index=True, how='left')
for c in c20.columns: cells['log_' + c] = np.log1p(cells[c])
cxy = np.c_[cells.geometry.x, cells.geometry.y]; cap = plants.capacityTpd.values
c20n = np.zeros(len(cells)); c75n = np.zeros(len(cells))
for i in range(0, len(cells), 2000):
    D = np.sqrt(((cxy[i:i + 2000, None, :] - pxy[None, :, :]) ** 2).sum(-1)) / 1000
    c20n[i:i + 2000] = ((D <= 20) * cap).sum(1); c75n[i:i + 2000] = ((D <= 75) * cap).sum(1)
cells['comp_tpd_20km'] = c20n; cells['comp_tpd_75km'] = c75n
cells['log_dist_tapoff'] = np.log1p(cells.dist_tapoff_km)

def score(design):
    X = pd.DataFrame(index=cells.index)
    for f in FULL:
        X[f] = cells[f] if f in cells else 0
    for k, v in design.items(): X[k] = v
    return xgb.predict_proba(X[FULL])[:, 1]
base = {f: 0 for f in FULL if f.startswith('feed_') or f.startswith('dev_')}
A_ = dict(base, capacityTpd=10, feed_mixed=1, dev_n_plants=1)
B_ = dict(base, capacityTpd=20, feed_agr_res=1, dev_n_plants=1)
cells['score_A'] = score(A_); cells['score_B'] = score(B_)
rho = spearmanr(cells.score_A, cells.score_B).correlation
print('rank agreement design A vs B (Spearman):', round(rho, 2))
print('score_A spread: p10', round(cells.score_A.quantile(.1), 2), 'median', round(cells.score_A.median(), 2), 'p90', round(cells.score_A.quantile(.9), 2))

# in-sample comparison: cell scores near operational vs planned plants
pc = np.c_[pg.geometry.x, pg.geometry.y]; dd, jj = nearest(pc, cxy)
pg['cell_score_A'] = np.where(dd <= 15, cells.score_A.values[jj], np.nan)
print('median cell score near operational vs planned:', pg.groupby('status').cell_score_A.median().round(3).to_dict())

# rank tap-off points by best cell within 25 km
tx = np.c_[tapg.geometry.x, tapg.geometry.y]; best = np.full(len(tapg), np.nan); bestB = best.copy(); res20 = best.copy()
for i in range(len(tapg)):
    dd_ = np.sqrt(((cxy - tx[i]) ** 2).sum(1)) / 1000; k = dd_ <= 25
    if k.any():
        j = np.where(k)[0][cells.score_A.values[k].argmax()]
        best[i] = cells.score_A.values[j]; bestB[i] = cells.score_B.values[k].max(); res20[i] = cells.c20_res_total_t.values[j]
tapg['site_score_A'] = best; tapg['site_score_B'] = bestB; tapg['c20_residue_best_cell_t'] = res20
tapg['rank_A'] = tapg.site_score_A.rank(ascending=False, method='min')
out = tapg.drop(columns='geometry').sort_values('rank_A')
out[['rank_A', 'code', 'name', 'state', 'pipelineName', 'gridPipelineOperator', 'site_score_A', 'site_score_B', 'c20_residue_best_cell_t', 'lat', 'lng']] \
    .to_csv('tapoff_points_ranked_v1.csv', index=False)
print('\nTop 15 tap-off points (design A):')
print(out[['rank_A', 'code', 'state', 'site_score_A', 'site_score_B']].head(15).round(3).to_string(index=False))
print('\nTop-50 by state:', out.head(50).state.value_counts().to_dict())
cells.drop(columns='geometry').assign(lon=cells.to_crs(4326).geometry.x, lat=cells.to_crs(4326).geometry.y) \
    .to_csv('siting_grid_cells_v1.csv', index=False)

# map
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 10))
P.boundary.plot(ax=ax, color='#d0d0d0', linewidth=0.2)
sc = ax.scatter(cells.geometry.x, cells.geometry.y, c=cells.score_A, s=6, cmap='viridis', marker='s', linewidths=0)
top = tapg.sort_values('rank_A').head(25)
ax.scatter(top.geometry.x, top.geometry.y, s=40, facecolors='none', edgecolors='red', linewidths=1.2, label='Top 25 tap-off points')
ax.set_axis_off(); ax.legend(loc='lower left', frameon=False)
cb = plt.colorbar(sc, ax=ax, shrink=0.5); cb.set_label('Relative site score (model probability, exploratory)\nreference plant: 10 TPD, mixed feedstock, new developer')
ax.set_title('CBG siting score v1 — cells within 75 km of a pipeline tap-off\n(out-of-state check AUC 0.59: location adds only modest signal)')
plt.tight_layout(); plt.savefig('fig_siting_map_v1.png', dpi=200); plt.close()
