"""Cross-checks the GOBARdhan locator-feed snapshot (data/raw/gobardhan/gobardhan_*_<date>.csv) against the
'Registered CBG Plants by State/UT' table downloaded from the same portal: plant names, coordinates and status."""
import numpy as np, pandas as pd
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import PLANTS, IPSV, GA, RAW

reg = pd.read_csv(RAW / 'gobardhan' / 'Registered_CBG_Plants_by_State_UT.csv')
feed = pd.read_csv(PLANTS)
reg['key'] = reg['Plant Name'].str.strip().str.upper()
feed['key'] = feed['name'].str.replace(r'^P-[A-Za-z0-9]+-', '', regex=True).str.strip().str.upper()
m = feed.merge(reg, on='key', how='outer', indicator=True)
both = m[m._merge == 'both']
dist_km = np.hypot(both.lat - pd.to_numeric(both.Latitude, errors='coerce'),
                   both.lng - pd.to_numeric(both.Longitude, errors='coerce')) * 111
status_map = {'operational': 'Commissioned', 'under_construction': 'Under Construction', 'planned': 'Yet to Start'}
print('plants: feed', len(feed), '| registered table', len(reg))
print('name matches:', m._merge.value_counts().to_dict())
print('coordinates within 1 km:', int((dist_km <= 1).sum()), 'of', len(both))
print('status agreement:', round((both.status.map(status_map) == both['Plant Status']).mean(), 4))
print('status counts (feed):', feed.status.value_counts().to_dict())
print('status counts (registered):', reg['Plant Status'].value_counts().to_dict())
print('pipeline IP/SV points:', len(pd.read_csv(IPSV)), '| city-gas areas:', len(pd.read_csv(GA)))
