"""Downloads the Census-2011 district boundaries (datameet/maps, CC BY 2.5 IN) into data/external/ (requires git)."""
import subprocess, shutil, tempfile
import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import EXT
dest = EXT / 'datameet_districts_2011'
if (dest / '2011_Dist.shp').exists():
    print('already present:', dest); raise SystemExit
tmp = tempfile.mkdtemp()
subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', 'https://github.com/datameet/maps.git', tmp], check=True)
subprocess.run(['git', '-C', tmp, 'sparse-checkout', 'set', 'Districts/Census_2011'], check=True)
dest.mkdir(parents=True, exist_ok=True)
for f in os.listdir(os.path.join(tmp, 'Districts', 'Census_2011')):
    shutil.copy(os.path.join(tmp, 'Districts', 'Census_2011', f), dest / f)
shutil.rmtree(tmp, ignore_errors=True)
print('saved to', dest)
