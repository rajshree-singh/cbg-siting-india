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
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)   # intermediate files and results are written to outputs/
