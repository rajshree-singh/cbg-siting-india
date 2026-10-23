"""Step 5: developer markers, 20-km catchment feedstock and nearby competition (runs catchment_developer.py)."""
import runpy
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
runpy.run_module("catchment_developer", run_name="__main__")
