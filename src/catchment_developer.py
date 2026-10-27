"""Step 5 (module): (a) developer marker, (b) feedstock within 20 km of each plant, (c) nearby competition.
Catchment method (area-weighting): each 2011 district polygon gets a residue / dung total; the amount inside a
20-km circle = district total x (area of district inside circle / district area). Assumes feedstock is spread evenly
inside a district.
Crop residue per 2011 polygon: district SHARES of state residue from 2013-14..2015-16 (district map close to 2011),
multiplied by recent STATE totals (avg 2022-23..2024-25). Telangana is counted inside 2011 'Andhra Pradesh'.
Boundaries: datameet/maps Census 2011 districts (CC BY 2.5 IN)."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, geopandas as gpd, glob
import geo_match as G
G.g['ks'] = G.g.ST_NM.map(G.norm).replace(G.STATE_ALIAS); G.g['kd'] = G.g.DISTRICT.map(G.norm)
EQ = 'EPSG:7755'  # India NSF LCC, metres
poly = G.g[['censuscode', 'DISTRICT', 'ST_NM', 'ks', 'geometry']].to_crs(EQ)
poly['area_km2'] = poly.area / 1e6

# ---------- crop residue onto 2011 polygons ----------
u = pd.read_csv(CROP)
u = u[u.crop_name.isin(RPR)].copy()
u['res_t'] = np.where(u.crop_name.isin(AREA_BASED), u.area, u.production) * u.crop_name.map(RPR)
u['ks'] = u.state_name.map(G.norm).replace(G.STATE_ALIAS)
early = u[u.year.isin(['2013-2014', '2014-2015', '2015-2016'])]
e = early.groupby(['ks', 'state_name', 'district_name', 'crop_name', 'year']).res_t.sum().groupby(['ks', 'state_name', 'district_name', 'crop_name']).mean().reset_index()
names = G.match_table(e[['state_name', 'district_name']].drop_duplicates(), 'state_name', 'district_name')
e = e.merge(names, on=['state_name', 'district_name'])
print('early-year district matches:', names.geo_match.value_counts().to_dict())
e = e[e.censuscode.notna()]
share = e.groupby(['ks', 'censuscode', 'crop_name']).res_t.sum()
share = share / share.groupby(['ks', 'crop_name']).transform('sum')
recent = u[u.year.isin(['2022-2023', '2023-2024', '2024-2025'])].groupby(['ks', 'crop_name', 'year']).res_t.sum().groupby(['ks', 'crop_name']).mean()
pr = (share.reset_index(name='share').merge(recent.reset_index(name='state_res'), on=['ks', 'crop_name']))
pr['res_t'] = pr.share * pr.state_res
crop_poly = pr.pivot_table(index='censuscode', columns='crop_name', values='res_t', aggfunc='sum').fillna(0)
crop_poly.columns = ['res_' + c.lower() + '_t' for c in crop_poly.columns]
crop_poly['res_total_t'] = crop_poly.sum(axis=1)
print('residue check: national total (Mt) polygons', round(crop_poly.res_total_t.sum() / 1e6, 1),
      ' vs recent data', round(recent.sum() / 1e6, 1))

# ---------- dung onto 2011 polygons (6 states) ----------
lv = pd.read_csv('district_livestock_dung_2019.csv')
lvm = G.match_table(lv, 'state_name', 'district_name')
print('livestock matches:', lvm.geo_match.value_counts().to_dict(), lvm[lvm.geo_match == 'none'][['state_name', 'district_name']].values.tolist())
dung_poly = lvm[lvm.censuscode.notna()].groupby('censuscode')[['cattle', 'buffalo', 'dung_t_per_year']].sum()
LV_STATES = set(lvm.state_name.map(G.norm))

P = poly.set_index('censuscode').join(crop_poly).join(dung_poly)
P.loc[~P.ks.isin(LV_STATES), 'dung_t_per_year'] = np.nan   # unknown outside the 6 states
P = P.reset_index()
P.to_file('district_feedstock_2011poly.gpkg', driver='GPKG')

def catchment(points_gdf, radius_km, cols):
    b = points_gdf[['pid', 'geometry']].copy(); b['geometry'] = b.buffer(radius_km * 1000)
    ov = gpd.overlay(b, P[['censuscode', 'area_km2', 'geometry'] + cols], how='intersection', keep_geom_type=True)
    ov['frac'] = (ov.area / 1e6) / ov.area_km2
    out = {}
    for c in cols:
        ov[c + '_w'] = ov[c] * ov.frac
        out[c] = ov.groupby('pid')[c + '_w'].sum(min_count=1)
    return pd.DataFrame(out)

if __name__ == '__main__':
    m = pd.read_csv('cbg_model_table_v1.csv')
    pts = gpd.GeoDataFrame(m[['plantId']].rename(columns={'plantId': 'pid'}), geometry=gpd.points_from_xy(m.lng, m.lat), crs=4326).to_crs(EQ)
    cols = ['res_total_t', 'res_rice_t', 'res_wheat_t', 'res_sugarcane_t', 'dung_t_per_year']
    c20 = catchment(pts, 20, cols).add_prefix('c20_')
    m = m.merge(c20, left_on='plantId', right_index=True, how='left')
    for c in c20.columns: m['log_' + c] = np.log1p(m[c])

    # competition: registered capacity of OTHER plants within 20 / 75 km
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000
    np.fill_diagonal(D, np.inf)
    cap = m.capacityTpd.values
    m['comp_tpd_20km'] = ((D <= 20) * cap[None, :]).sum(1)
    m['comp_tpd_75km'] = ((D <= 75) * cap[None, :]).sum(1)

    # developer marker (from plant-name prefix, e.g. "P-RIL-...")
    m['owner'] = m.name.str.extract(r'^P-([A-Za-z0-9]+)-')[0].str.upper()
    m['dev_n_plants'] = m.groupby('owner').plantId.transform('size').fillna(1)
    m['dev_repeat'] = (m.dev_n_plants >= 3).astype(int)
    # public-sector oil and gas companies, their subsidiaries and joint ventures
    # HPCL1, HRGEL: HPCL / HPCL Renewable & Green Energy; IGRPL: IOC GPS Renewables (IndianOil JV); BGBPL: Bharat GPS Bioenergy (BPCL JV)
    PSU = ['IOCL', 'BPCL', 'HPCL', 'GAIL', 'ONGC', 'NTPC', 'OIL', 'IGL', 'MNGL', 'NTPCVVN', 'GGL', 'BGL',
           'HPCL1', 'HRGEL', 'IGRPL', 'BGBPL']
    nm = m.name.str.upper()
    m['dev_psu'] = (m.owner.isin(PSU) | nm.str.contains(r'INDIAN OIL|BHARAT PETROLEUM|HINDUSTAN PETROLEUM|\bGAIL\b|\bNTPC\b|\bONGC\b|\bIOCL\b|\bBPCL\b|\bHPCL\b')).astype(int)
    m['dev_large_corp'] = (m.owner.isin(['RIL', 'ADANI', 'TOTAL', 'EVERENVIRO', 'GPS']) |
                           nm.str.contains(r'RELIANCE|ADANI|TOTAL ENERGIES|EVERENVIRO|VERBIO|TORRENT|GREEN GROWTH|ESSAR|\bJSW\b')).astype(int)
    m['dev_listed_or_psu'] = ((m.dev_psu + m.dev_large_corp) > 0).astype(int)
    print('\nowner parsed:', m.owner.notna().mean().round(3), ' repeat devs (>=3 plants):', m.dev_repeat.sum(),
          ' PSU plants:', m.dev_psu.sum(), ' large corp plants:', m.dev_large_corp.sum())
    print(m.groupby('status')[['c20_res_total_t', 'c20_res_rice_t', 'c20_dung_t_per_year', 'comp_tpd_20km', 'dev_repeat', 'dev_psu', 'dev_large_corp']].median(numeric_only=True).round(0).to_string())
    print(m.groupby('status')[['dev_repeat', 'dev_psu', 'dev_large_corp']].mean().round(2).to_string())
    m.to_csv('cbg_model_table_v2.csv', index=False)
