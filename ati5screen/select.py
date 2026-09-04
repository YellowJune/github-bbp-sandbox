from __future__ import annotations
import json,glob
from pathlib import Path

rows=[]
for f in glob.glob('artifacts/*/result.json'):
    try:
        r=json.load(open(f));r['_file']=f;rows.append(r)
    except Exception: pass

eligible=[r for r in rows if r.get('both_pressures_survive') and r.get('ED80_geometry_both')]
def score(r):
    sel=list(r['selected'].values())
    worst=max(x['mean_CuN4_A'] for x in sel)
    high=1 if r.get('ED97_geometry_both') else 0
    return (-high,worst)

out=Path('artifacts/select');out.mkdir(parents=True,exist_ok=True)
if eligible:
    win=sorted(eligible,key=score)[0]
    A=win['A']; advance='1'
    for p in ['0.00000000','0.04053000']:
        src=Path(f'artifacts/{A}/relaxed_{p}GPa.cif')
        if src.exists():(out/f'relaxed_{p}GPa.cif').write_bytes(src.read_bytes())
else:
    win=None;A='NONE';advance='0'
summary={'all_candidates':[{'candidate':r.get('candidate'),'both_pressures_survive':r.get('both_pressures_survive'),'ED80_geometry_both':r.get('ED80_geometry_both'),'ED97_geometry_both':r.get('ED97_geometry_both')} for r in rows],
         'winner':A,'advance_to_QE':advance=='1','winner_result':win}
(out/'selection.json').write_text(json.dumps(summary,indent=2));(out/'WINNER').write_text(A);(out/'ADVANCE').write_text(advance)
print(json.dumps(summary,indent=2))
