"""Step 2: district crop residue (gross) from UPAG APY data, joined to CBG plants by state+district name.
Gross residue = 3-year average production (2022-23 to 2024-25) x residue-to-product ratio (RPR)."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np, re, difflib
RPR={'Rice':1.5,'Wheat':1.5,'Maize':2.0,'Sugarcane':0.4,'Bajra':2.0,'Jowar':1.7,'Cotton':3.8}   # sugarcane: trash and tops per tonne of cane
YEARS=['2022-2023','2023-2024','2024-2025']
u=pd.read_csv(CROP)
u=u[u.year.isin(YEARS)&u.crop_name.isin(RPR)]
# sum seasons within a year, then average over years
y=u.groupby(['state_name','district_name','crop_name','year']).production.sum().reset_index()
avg=y.groupby(['state_name','district_name','crop_name']).production.mean().unstack(fill_value=0)
for c in RPR:
    if c not in avg: avg[c]=0
res=pd.DataFrame({f'res_{c.lower()}_t':avg[c]*RPR[c] for c in RPR})
res['res_total_t']=res.sum(axis=1)
res=res.reset_index()
res.to_csv('district_crop_residue_2022_25.csv',index=False)
def norm(s): return re.sub(r'[^a-z]','',str(s).lower().replace('district',''))
res['k_state']=res.state_name.map(norm); res['k_dist']=res.district_name.map(norm)
p=pd.read_csv('cbg_plants_with_distances.csv')
ALIAS={'gurgaon':'gurugram','belgaum':'belagavi','mangaloredakshinakannada':'dakshinakannada','aurangabad':'chhatrapatisambhajinagar',
       'ahmednagar':'ahilyanagar','lakhimpurkheri':'kheri','midnapurwest':'paschimmedinipur','vavtharad':'banaskantha'}  # renamed/split districts
p['k_state']=p.state.map(norm); p['k_dist']=p.district.map(norm).replace(ALIAS)
lookup={}
for s,g in res.groupby('k_state'): lookup[s]=list(g.k_dist)
def match(r):
    c=lookup.get(r.k_state,[])
    if r.k_dist in c: return r.k_dist,'exact'
    m=difflib.get_close_matches(r.k_dist,c,n=1,cutoff=0.8)
    return (m[0],'fuzzy') if m else (None,'none')
mm=p.apply(match,axis=1,result_type='expand'); p['k_dist2']=mm[0]; p['district_match']=mm[1]
res2=res.rename(columns={'k_dist':'k_dist2'}).drop(columns=['state_name','district_name'])
out=p.merge(res2,on=['k_state','k_dist2'],how='left').drop(columns=['k_state','k_dist','k_dist2'])
out.to_csv('cbg_plants_with_distances_residue.csv',index=False)
print(out.district_match.value_counts())
print('unmatched examples:'); print(out[out.district_match=='none'][['state','district']].drop_duplicates().head(25).to_string())
