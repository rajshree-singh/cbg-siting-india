"""Step 1: straight-line distance from each CBG plant to the nearest pipeline IP/SV with a tap-off,
city-gas geographical-area capacity, and comparison by plant status."""
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu, kruskal
p=pd.read_csv(PLANTS,dtype={'gaId':str})
c=pd.read_csv(IPSV)
g=pd.read_csv(GA,dtype={'gaId':str})
def hav(lat1,lon1,lat2,lon2):
    R=6371.0; la1,lo1,la2,lo2=map(np.radians,[lat1,lon1,lat2,lon2])
    a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
ok=p.lat.notna()&p.lng.notna()
tap=c[c.tapOffAvailable==True].dropna(subset=['lat','lng'])
allp=c.dropna(subset=['lat','lng'])
def nearest(df,pts):
    out=np.full(len(df),np.nan); code=np.array([None]*len(df),dtype=object)
    for i,(la,lo) in enumerate(zip(df.lat,df.lng)):
        if np.isnan(la): continue
        d=hav(la,lo,pts.lat.values,pts.lng.values); j=d.argmin(); out[i]=d[j]; code[i]=pts.code.values[j]
    return out,code
p['dist_tapoff_km'],p['nearest_tapoff_code']=nearest(p,tap)
p['dist_any_IPSV_km'],_=nearest(p,allp)
p['within_75km_tapoff']=p.dist_tapoff_km<=75
g['gaId']=g.gaId.str.strip()
p['gaId']=p.gaId.str.strip()
m=p.merge(g[['gaId','residualCapacityTpd','intakeCapacityTpd','category']],on='gaId',how='left')
m.to_csv('cbg_plants_with_distances.csv',index=False)
order=['operational','under_construction','planned']
print('GA match',m.residualCapacityTpd.notna().sum(),'/',len(m))
s=m.groupby('status').agg(n=('plantId','size'),median_km=('dist_tapoff_km','median'),mean_km=('dist_tapoff_km','mean'),pct_within75=('within_75km_tapoff','mean'),median_tpd=('capacityTpd','median'),median_residGA=('residualCapacityTpd','median')).reindex(order)
print(s.round(2).to_string())
a=m[m.status=='operational'].dist_tapoff_km.dropna(); b=m[m.status=='planned'].dist_tapoff_km.dropna()
print('MWU operational vs planned dist p=',mannwhitneyu(a,b).pvalue)
print('Kruskal 3 groups p=',kruskal(*[m[m.status==k].dist_tapoff_km.dropna() for k in order]).pvalue)
for col in ['capacityTpd','residualCapacityTpd']:
    a=m[m.status=='operational'][col].dropna(); b=m[m.status=='planned'][col].dropna(); print(col,'MWU p=',mannwhitneyu(a,b).pvalue)
m['band']=pd.cut(m.dist_tapoff_km,[0,10,25,50,75,1e4],labels=['0-10','10-25','25-50','50-75','>75'])
t=pd.crosstab(m.band,m.status,normalize='index').reindex(columns=order); t['n']=m.band.value_counts(); print((t).round(2).to_string())
print(pd.crosstab(m.feedstockCategory,m.status).reindex(columns=order).to_string())
pb=m[m.state.isin(['Punjab','Haryana'])]
print('PB+HR',pb.groupby('status').dist_tapoff_km.agg(['size','median']).reindex(order).round(1).to_string())
print('overall median',round(m.dist_tapoff_km.median(),1),' within75',round(m.within_75km_tapoff.mean(),3))
