"""Postprocess recorded current weights and subsequent MH decisions; no sampling."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
cfg=json.loads((OUT/'config.json').read_text());raw=ROOT/cfg['raw_directory'];rows=[]
for case in cfg['cases']:
    current=[];accept=[]
    for i in range(cfg['chains']):
        a=np.load(raw/(case['id']+f'_independence_chain{i}.npz'));burn=cfg['burn_in']
        current.extend(a['log_weight'][burn:-1]);accept.extend(a['accepted'][burn:])
    current=np.array(current);accept=np.array(accept);bands=[]
    for label,mask in [('weight>=0.5',current>=np.log(.5)),('weight<0.01',current<np.log(.01))]:
        bands.append(dict(band=label,current_states=int(mask.sum()),acceptance=float(accept[mask].mean()) if mask.any() else None))
    rows.append(dict(case=case['id'],bands=bands))
(OUT/'weight_residence_diagnostics.json').write_text(json.dumps(rows,indent=2))
