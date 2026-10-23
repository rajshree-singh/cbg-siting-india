# cbg-siting-india

Machine-learning analysis of what distinguishes operational from not-yet-started Compressed Biogas (CBG) plants in India,
and a gas-grid siting score for new plants.

Dataset and results: Zenodo, https://doi.org/10.5281/zenodo.23588147

## Overview
Of the 776 CBG plants registered on the GOBARdhan portal (Ministry of Petroleum and Natural Gas) on 22 October 2026,
208 were operational, 241 under construction and 327 not yet started. The analysis relates plant status to:

- **location** — distance to the nearest gas-pipeline IP/SV with a tap-off, spare capacity of the city-gas
  geographical area, crop residue and dung within 20 km, and registered competing capacity nearby;
- **project design** — capacity and feedstock category;
- **developer** — number of plants registered by the same developer, public-sector oil and gas companies, large corporates.

Models (logistic regression, random forest, XGBoost) are evaluated with 5-fold cross-validation grouped by state and
explained with SHAP. A 15-km grid within 75 km of a tap-off is then scored for a reference plant, and the 701 tap-off
points are ranked.

## Results
| Model (5-fold CV grouped by state) | AUC, random forest / XGBoost |
|---|---|
| All factors | 0.86 / 0.86 (95% CI 0.82–0.89) |
| All factors, excluding Reliance plants | 0.84 / 0.85 |
| All factors except developer | 0.80 / 0.82 |
| Location factors only | 0.58 / 0.56 |
| Equal-weight multi-criteria score of location factors | 0.58 |

AUC values are means over 20 random fold assignments; confidence intervals are bootstrap 95% intervals.
Feedstock category, plant capacity and developer account for most of the predictive power; location factors alone
perform no better than an equal-weight multi-criteria score. Results are stable under low and high residue-to-product
ratios and dung rates (AUC 0.86–0.87; SHAP rank correlation with the base case ≥ 0.99). The siting score separates
operational from planned plants with an out-of-state AUC of 0.59.

## Repository structure
```
data/raw/gobardhan/   GOBARdhan portal snapshot, 22 Oct 2026: plants, pipeline IP/SV points, city-gas area capacities,
                      and the portal's "Registered CBG Plants by State/UT" table
data/raw/crop/        District crop area, production and yield (UPAG / DES)
data/raw/livestock/   20th Livestock Census 2019, district cattle and buffalo (six states)
data/external/        Census-2011 district boundaries (downloaded by src/00_fetch_boundaries.py)
src/                  Analysis scripts, run in numerical order
outputs/              Created by the scripts
```

| Script | Purpose |
|---|---|
| `00_fetch_boundaries.py` | Download Census-2011 district boundaries (datameet/maps) |
| `00_check_snapshot.py` | Cross-check the portal snapshot against the Registered CBG Plants table |
| `00_fetch_gobardhan.py` | Download a new snapshot of the GOBARdhan locator feed (optional) |
| `01_distance.py` | Distance to nearest tap-off; city-gas area capacity |
| `02_crop_residue.py` | District crop residue |
| `03_livestock.py` | District cattle, buffalo and dung |
| `04_ml_model_v1.py` | Model v1 with district-level features |
| `05_catchment_developer.py` | 20-km catchment feedstock, competition, developer markers |
| `06_ml_model_v2.py` | Model v2 and SHAP |
| `07_siting_map.py` | Grid scoring, ranked tap-off points, map |
| `07b_map_check.py` | Out-of-state validation of the siting score |
| `08_robustness.py` | Repeated cross-validation, bootstrap intervals, parameter sensitivity, capacity analysis |

## Reproduce
```
pip install -r requirements.txt
python run_all.py
```
Steps 00–07 run in a few minutes on a laptop; step 08 reruns the analysis four times and takes about 10 minutes. Requires git for the boundary download.

## Parameters
| Parameter | Value |
|---|---|
| Residue-to-product ratio | rice 1.5, wheat 1.5, maize 2.0, sugarcane 0.4, bajra 2.0, jowar 1.7, cotton 3.8 (low and high scenarios in `src/config.py`) |
| Wet dung | cattle 10 kg/day, buffalo 15 kg/day (low 5/12, high 15/18) |
| Catchment radius | 20 km, area-weighted from district totals |
| Connection radius | 75 km from a tap-off |
| Siting reference plant | 10 TPD, mixed feedstock, first-time developer (sensitivity: 20 TPD, crop residue) |

Distances are straight-line. "Operational" refers to commissioning status on the portal. District boundaries are used
for analysis only and do not represent official international boundaries.

## Data sources
- GOBARdhan CBG portal, Ministry of Petroleum and Natural Gas, Government of India — https://gobardhan.gov.in (accessed 22 Oct 2026)
- District-wise crop area, production and yield, UPAG / Directorate of Economics & Statistics, Ministry of Agriculture & Farmers Welfare — India Data Portal
- 20th Livestock Census 2019, Department of Animal Husbandry & Dairying — data.gov.in; Maharashtra: Animal Husbandry Department, Government of Maharashtra, 20th Livestock Census 2019 report, Table 3.11
- Census-2011 district boundaries — datameet/maps, CC BY 2.5 India

## Licence
Code: MIT. Data produced by the scripts: CC BY 4.0. Source data remain under the terms of their providers (see `LICENSE-DATA.md`).

## Citation
See `CITATION.cff`. Dataset: Singh, R. (2026). *Compressed Biogas (CBG) plants in India: plant-level features, model results and gas-grid siting scores* (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23588147
