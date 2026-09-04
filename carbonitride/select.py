from __future__ import annotations
import glob,json,shutil
from pathlib import Path

rows=[]
for f in glob.glob('artifacts/*/result.json'):
    try:
        r=json.load(open(f));r['_file']=f;rows.append(r)
    except Exception:pass

eligible=[r for r in rows if r.get('both_pressures_survive') and r.get('compact_plane_both')]
def score(r):
    sel=list(r['selected'].values())
    # Prefer smaller worst Cu-ligand distance, then lower enthalpy proxy.
    worst=max(x['mean_Cu_ligand4_A'] for x in sel)
    h=sum(x['enthalpy_proxy_eV_atom'] for x in sel)/len(sel)
    return (worst,h)

out=Path('artifacts/select');out.mkdir(parents=True,exist_ok=True)
if eligible:
    win=min(eligible,key=score);name=win['candidate'];adv='1'
    for p in ['0.00000000','0.04053000']:
        src=Path(f'artifacts/{name}/relaxed_{p}GPa.cif')
        if src.exists():shutil.copy2(src,out/f'relaxed_{p}GPa.cif')
else:
    win=None;name='NONE';adv='0'
summary={'candidates':[{'candidate':r.get('candidate'),'both_pressures_survive':r.get('both_pressures_survive'),'compact_plane_both':r.get('compact_plane_both')} for r in rows],
         'winner':name,'advance_to_QE':adv=='1','winner_result':win}
(out/'selection.json').write_text(json.dumps(summary,indent=2));(out/'WINNER').write_text(name);(out/'ADVANCE').write_text(adv)
print(json.dumps(summary,indent=2))
