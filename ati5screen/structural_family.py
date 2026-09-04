from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208
STARTS=[('c1',3.45,2.55),('c2',3.60,2.70),('c3',3.75,2.90),('c4',3.90,3.10)]


def build(A,a,c):
    sy=[];fr=[];ai=0
    for y in range(2):
        for x in range(3):
            sy.append(A if ai==0 else 'Ti');fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Cu');fr.append([x/3,y/2,0])
            sy.append('N');fr.append([(x+.5)/3,y/2,0])
            sy.append('N');fr.append([x/3,(y+.5)/2,0])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)


def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    per=[];coord=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        n=sorted(float(d[i,j]) for j,t in enumerate(sy) if t=='N' and i!=j)
        per.append(float(np.mean(n[:4])) if len(n)>=4 else float('nan'))
        coord.append(sum(x<2.25 for x in n))
    nn=[float(d[i,j]) for i,s in enumerate(sy) for j,t in enumerate(sy) if i<j and s=='N' and t=='N']
    return {
        'mean_CuN4_A':float(np.nanmean(per)),'max_CuN4_A':float(np.nanmax(per)),
        'min_CuN4_A':float(np.nanmin(per)),'min_CuN_coord_lt2p25':int(min(coord)),
        'mean_CuN_coord_lt2p25':float(np.mean(coord)),'min_NN_A':float(min(nn)) if nn else float('nan')
    }


def relax(A,label,a,c,p,model,out):
    at=AseAtomsAdaptor.get_atoms(build(A,a,c));v0=float(at.get_volume())
    at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(out/f'{label}_{p:.8f}.log')).run(fmax=.045,steps=460)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.array(at.get_stress(voigt=True))*160.21766208
    m=metrics(at);epa=float(at.get_potential_energy()/len(at));hpa=epa+p*(at.get_volume()/len(at))*GPA_TO_EV_A3
    rec={'A':A,'start':label,'a0_A':a,'c0_A':c,'pressure_GPa':p,'pressure_atm':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':float(hpa),
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3]))}
    rec.update(m)
    rec['gross_pass']=bool(fmax<.075 and .70<rec['volume_ratio']<1.32 and m['min_NN_A']>1.28 and abs(rec['hydrostatic_pressure_GPa_from_stress']-p)<.40)
    rec['ED80_geometry_gate']=bool(m['mean_CuN4_A']<=1.82 and m['max_CuN4_A']<=1.87 and m['min_CuN_coord_lt2p25']>=3)
    rec['ED97_geometry_gate']=bool(m['mean_CuN4_A']<=1.80 and m['max_CuN4_A']<=1.85 and m['min_CuN_coord_lt2p25']>=3)
    cif=out/f'{label}_{p:.8f}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--A',choices=['B','Al','Sc'],required=True);a=ap.parse_args()
    out=Path(f'artifacts/{a.A}');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[];selected={}
    for p in [0.0,400*ATM_TO_GPA]:
        rs=[relax(a.A,*st,p,model,out) for st in STARTS];rows.extend(rs)
        viable=[r for r in rs if r['gross_pass']]
        if viable:
            best=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom']);selected[f'{p:.8f}']=best
            Structure.from_file(best['cif']).to(filename=str(out/f'relaxed_{p:.8f}GPa.cif'))
    both=len(selected)==2
    ed80=both and all(r['ED80_geometry_gate'] for r in selected.values())
    ed97=both and all(r['ED97_geometry_gate'] for r in selected.values())
    result={'candidate':f'{a.A}Ti5Cu6N12','A':a.A,'formal_average_Cu_valence':13/6,'formal_hole_doping':1/6,
            'all_runs':rows,'selected':selected,'both_pressures_survive':both,'ED80_geometry_both':ed80,'ED97_geometry_both':ed97,
            'advance_to_QE':bool(ed80)}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
