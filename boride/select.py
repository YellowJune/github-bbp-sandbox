from __future__ import annotations
import json,shutil
from pathlib import Path
import numpy as np

ROOT=Path('artifacts');OUT=ROOT/'select';OUT.mkdir(parents=True,exist_ok=True)
res={}
for e in ['chgnet','mace']:
    p=ROOT/e/'result.json'
    if p.exists():res[e]=json.loads(p.read_text())

summary={'candidate':'ScTi5Cu6B12','engines':res}
advance=False
if len(res)==2:
    both_pass=all(res[e].get('engine_pass',False) for e in res)
    d={}
    dh={}
    for e in res:
        s0=res[e]['pressures']['0.00000000']['selected']
        s4=res[e]['pressures']['0.04053000']['selected']
        d[e]=(s0['mean_CuB_nearest4_A'],s4['mean_CuB_nearest4_A'])
        dh[e]=(s0['decomposition_deltaH_eV_atom'],s4['decomposition_deltaH_eV_atom'])
    agree=max(abs(d['chgnet'][i]-d['mace'][i]) for i in [0,1])<=.10
    summary.update({'both_engine_pass':both_pass,'CuB_distances_A':d,'decomposition_deltaH_eV_atom':dh,'max_engine_distance_disagreement_A':max(abs(d['chgnet'][i]-d['mace'][i]) for i in [0,1]),'geometry_engine_agreement':agree})
    advance=bool(both_pass and agree)
    if advance:
        for e in ['chgnet','mace']:
            for p in ['0.00000000','0.04053000']:
                src=ROOT/e/f'{e}_relaxed_{p}GPa.cif'
                if src.exists():shutil.copy2(src,OUT/f'{e}_relaxed_{p}GPa.cif')
summary['advance_to_QE']=advance
(OUT/'selection.json').write_text(json.dumps(summary,indent=2));(OUT/'ADVANCE').write_text('1' if advance else '0');print(json.dumps(summary,indent=2))
