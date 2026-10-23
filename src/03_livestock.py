"""Step 3: district cattle + buffalo (20th Livestock Census 2019) -> gross dung, joined to plants.
Gross wet dung: cattle 10 kg/day, buffalo 15 kg/day; no collectability factor applied."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, re, glob, difflib
frames=[]
for f in [f for f in glob.glob(str(LIVESTOCK_DIR/'livestock_census_2019_*.csv')) if 'Maharashtra' not in f]:
    d=pd.read_csv(f); frames.append(d[['state_name','district_name','cattle','buffalo']])
lc=pd.concat(frames).drop_duplicates()
# Maharashtra: Table 3.11 of the state 20th Livestock Census report (district cattle & buffalo), extracted to CSV
mh=pd.read_csv(LIVESTOCK_DIR/'livestock_census_2019_Maharashtra_table3_11.csv')
print('MH districts',len(mh),'cattle sum',mh.cattle.sum(),'(report total 13992304)')
lc=pd.concat([lc,mh])
lc['dung_t_per_year']=(lc.cattle*10+lc.buffalo*15)*365/1000
lc.to_csv('district_livestock_dung_2019.csv',index=False)
def norm(s): return re.sub(r'[^a-z]','',str(s).lower().replace('district',''))
ALIAS={'gurgaon':'gurugram','nuh':'mewat','ahmadabad':'ahmadabad','ahmedabad':'ahmadabad','banaskantha':'banaskantha','vavtharad':'banaskantha',
 'aurangabad':'aurangabad','chhatrapatisambhajinagar':'aurangabad','ahilyanagar':'ahmednagar','ahmadnagar':'ahmednagar','lakhimpurkheri':'kheri',
 'prayagraj':'allahabad','ayodhya':'faizabad','bhadohi':'santravidasnagar','narmadapuram':'hoshangabad','khandwa':'eastnimar','sasnagar':'sasnagar',
 'mohali':'sasnagar','sahibzadaajitsinghnagar':'sasnagar','malerkotla':'sangrur','nawanshahr':'nawanshahrsbsnagar','shahidbhagatsinghnagar':'nawanshahrsbsnagar',
 'srimuktsarsahib':'muktsar','ferozepur':'firozepur','dharashiv':'osmanabad','kushinagar':'kushinagar','sabarkantha':'sabarkantha','panchmahal':'panchmahals'}
lc['k_state']=lc.state_name.map(norm); lc['k_dist']=lc.district_name.map(norm).replace(ALIAS)
p=pd.read_csv('cbg_plants_with_distances_residue.csv')
p['k_state']=p.state.map(norm); p['k_dist']=p.district.map(norm).replace(ALIAS)
look={s:list(g.k_dist) for s,g in lc.groupby('k_state')}
def match(r):
    c=look.get(r.k_state)
    if c is None: return None,'state_not_covered'
    if r.k_dist in c: return r.k_dist,'exact'
    m=difflib.get_close_matches(r.k_dist,c,n=1,cutoff=0.75)
    return (m[0],'fuzzy') if m else (None,'none')
mm=p.apply(match,axis=1,result_type='expand'); p['k2']=mm[0]; p['livestock_match']=mm[1]
lc2=lc.groupby(['k_state','k_dist'])[['cattle','buffalo','dung_t_per_year']].sum().reset_index().rename(columns={'k_dist':'k2'})
out=p.merge(lc2,on=['k_state','k2'],how='left').drop(columns=['k_state','k_dist','k2'])
out.to_csv('cbg_plants_features_v3.csv',index=False)
print(out.livestock_match.value_counts())
print(out[out.livestock_match.isin(['none','fuzzy'])][['state','district','livestock_match']].drop_duplicates().to_string())
