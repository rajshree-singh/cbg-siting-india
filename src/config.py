"""Shared paths and settings."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
EXT = ROOT / 'data' / 'external'
OUT = ROOT / 'outputs'
SNAPSHOT = os.environ.get('GOBARDHAN_SNAPSHOT', '2026-10-22')   # GOBARdhan portal snapshot date
PLANTS = RAW / 'gobardhan' / f'gobardhan_cbg_plants_{SNAPSHOT}.csv'
IPSV = RAW / 'gobardhan' / f'gobardhan_pipeline_IP_SV_points_{SNAPSHOT}.csv'
GA = RAW / 'gobardhan' / f'gobardhan_CGD_GA_residual_capacity_{SNAPSHOT}.csv'
CROP = RAW / 'crop' / 'cropwise-area-production-yield-upag.csv'
LIVESTOCK_DIR = RAW / 'livestock'
DISTRICTS_2011 = EXT / 'datameet_districts_2011' / '2011_Dist.shp'
# Residue factors and wet dung rates; 'base' is used for all reported results,
# 'low' / 'high' are the sensitivity scenarios of 08_robustness.py (selected with CBG_RPR / CBG_DUNG).
# Residue-to-product ratios (t residue per t product), except crops in AREA_BASED (t residue per ha).
# Base values: Hiloidhari et al. (2014), Table 1 (sugarcane = bagasse 0.33 + tops and leaves 0.05).
# Low values for rice, wheat and maize: pooled estimates of Kabir et al. (2026).
RPR_SCENARIOS = {
    'base': {'Rice': 1.5, 'Wheat': 1.5, 'Maize': 2.0, 'Sugarcane': 0.38, 'Bajra': 2.0, 'Jowar': 1.7, 'Cotton': 3.8},
    'low':  {'Rice': 1.34, 'Wheat': 1.27, 'Maize': 1.39, 'Sugarcane': 0.3, 'Bajra': 1.5, 'Jowar': 1.4, 'Cotton': 3.0},
    'high': {'Rice': 1.8, 'Wheat': 1.7, 'Maize': 3.0, 'Sugarcane': 0.5, 'Bajra': 2.5, 'Jowar': 2.0, 'Cotton': 4.5},
}
AREA_BASED = {'Cotton'}
DUNG_SCENARIOS = {'base': (10, 15), 'low': (5, 12), 'high': (15, 18)}   # kg/head/day: (cattle, buffalo)
RPR = RPR_SCENARIOS[os.environ.get('CBG_RPR', 'base')]
DUNG_CATTLE_KG, DUNG_BUFFALO_KG = DUNG_SCENARIOS[os.environ.get('CBG_DUNG', 'base')]
OUT = Path(os.environ.get('CBG_OUT', str(OUT)))
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)   # intermediate files and results are written to outputs/
