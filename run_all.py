"""Runs the full analysis in order."""
import subprocess, sys, pathlib
src = pathlib.Path(__file__).parent / 'src'
for s in ['00_fetch_boundaries.py', '00_check_snapshot.py', '01_distance.py', '02_crop_residue.py', '03_livestock.py', '04_ml_model_v1.py',
          '05_catchment_developer.py', '06_ml_model_v2.py', '07_siting_map.py', '07b_map_check.py', '08_robustness.py']:
    print(f'\n===== {s} =====', flush=True)
    subprocess.run([sys.executable, str(src / s)], check=True)
