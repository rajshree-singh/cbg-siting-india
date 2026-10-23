"""Downloads the public GOBARdhan CBG locator feed (Ministry of Petroleum and Natural Gas) and saves it as three
dated CSVs: plants, pipeline IP/SV points and city-gas geographical-area capacities.
The analysis uses the snapshot of 22 October 2026 included in data/raw/gobardhan/."""
import json, datetime, urllib.request
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
import pandas as pd
from config import RAW
URL = 'https://gobardhan.gov.in/api/v1/public/locator'
d = json.load(urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60))
day = datetime.date.today().isoformat(); out = RAW / 'gobardhan'; out.mkdir(parents=True, exist_ok=True)
def save(rows, name):
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, list)).any(): df[c] = df[c].map(lambda v: '|'.join(map(str, v)) if isinstance(v, list) else v)
    df.to_csv(out / f'{name}_{day}.csv', index=False); print(name, len(df))
save(d['plants'], 'gobardhan_cbg_plants')
save(d['clusters'], 'gobardhan_pipeline_IP_SV_points')
save(d['gaResidualCapacities'], 'gobardhan_CGD_GA_residual_capacity')
print('feed generatedAt:', d.get('generatedAt'), ' clusterRadiusKm:', d.get('clusterRadiusKm'))
