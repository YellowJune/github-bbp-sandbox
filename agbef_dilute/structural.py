from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208
OUT=Path('artifacts/dilute'); OUT.mkdir(parents=True,exist_ok=True)
STARTS=[('compact',3.90,6.8,.22),('central',4.02,7.3,.22),('expanded',4.14,7.8,.21)]

def build(a,c,dz):
    # 6x2 AgF2 sheet + all-fluoride spacer; one remote Al3+ among 12 Be2+ sites.
    # Formula AlBe11Ag12F48; formal Ag valence = 23/12 = +1.9167 (1/12 e-doped).
    sy=[]; fr=[]; ai=0
    for y in range(2):
        for x in range(6):
            sy.append('Al' if ai==0 else 'Be'); fr.append([(x+.5)/6,(y+.5)/2,.5]); ai+=1
            sy.append('Ag'); fr.append([x/6,y/2,0])
            sy.append('F'); fr.append([(x+.5)/6,y/2,0])
            sy.append('F'); fr.append([x/6,(y+.5)/2,0])
            sy.append('F'); fr.append([(x+.5)/6,(y+.5)/2,.5-dz])
            sy.append('F'); fr.append([(x+.5)/6,(y+.5)/2,.5+dz])
    return Structure(Lattice.orthorhombic(6*a,2*a,c),sy,fr)

def micvec(at,i,j): return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag=[i for i,s in enumerate(sy) if s=='Ag']; fs=[i for i,s in enumerate(sy) if s=='F']
    n4=[]; coords=[]; ang=[]; bridge=0
    for i in ag:
        vals=sorted(float(d[i,j]) for j in fs); n4.append(float(np.mean(vals[:4]))); coords.append(sum(x<2.55 for x in vals))
    for f in fs:
        close=sorted((float(d[f,i]),i) for i in ag)
        close=[x for x in close if x[0]<2.55]
        if len(close)>=2:
            i,j=close[0][1],close[1][1]; v1=micvec(at,f,i); v2=micvec(at,f,j)
            den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                ang.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1))))); bridge+=1
    ff=[float(d[i,j]) for ii,i in enumerate(fs) for j in fs[ii+1:]]
    return {'mean_Ag_F_nearest4_A':float(np.mean(n4)),'max_Ag_F_nearest4_A':float(np.max(n4)),
            'min_Ag_F_coord_lt2p55':int(min(coords)),'n_bridging_F':int(bridge),
            'mean_Ag_F_Ag_angle_deg':float(np.mean(ang)) if ang else float('nan'),
            'p10_Ag_F_Ag_angle_deg':float(np.quantile(ang,.10)) if ang else float('nan'),
            'min_F_F_A':float(min(ff))}

def relax(label,a,c,dz,p,model):
    s=build(a,c,dz); at=AseAtomsAdaptor.get_atoms(s); v0=float(at.get_volume()); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model); flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    tag=f'AlBe11Ag12F48_{label}_{p:.8f}GPa'; FIRE(flt,logfile=str(OUT/f'{tag}.log')).run(fmax=.05,steps=600)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max()); m=metrics(at)
    epa=float(at.get_potential_energy()/len(at)); vpa=float(at.get_volume()/len(at))
    rec={'tag':tag,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),
         'natoms':len(at),'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':epa+p*vpa*GPA_TO_EV_A3,
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()]}
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_pass']=bool(fmax<.085 and .60<rec['volume_ratio']<1.45 and m['min_F_F_A']>1.15)
    rec['plane_pass']=bool(m['mean_Ag_F_nearest4_A']<2.18 and m['max_Ag_F_nearest4_A']<2.30 and
                           m['min_Ag_F_coord_lt2p55']>=4 and m['n_bridging_F']>=20 and
                           m['mean_Ag_F_Ag_angle_deg']>=165 and m['p10_Ag_F_Ag_angle_deg']>=155)
    cif=OUT/f'{tag}.cif'; AseAtomsAdaptor.get_structure(at).to(filename=str(cif)); rec['cif']=str(cif)
    return rec

def main():
    model=CHGNet.load(); rows=[]; selected={}
    for p in [0.0,400*ATM_TO_GPA]:
        rs=[relax(*st,p,model) for st in STARTS]; rows+=rs
        viable=[r for r in rs if r['gross_pass']]
        if viable: selected[f'{p:.8f}']=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom'])
    advance=bool(len(selected)==2 and all(r['gross_pass'] and r['plane_pass'] for r in selected.values()))
    out={'candidate':'AlBe11Ag12F48','formal_Ag_valence':23/12,'electron_doping_per_Ag':1/12,
         'all_runs':rows,'selected':selected,'advance':advance}
    (OUT/'result.json').write_text(json.dumps(out,indent=2)); (OUT/'ADVANCE').write_text('1' if advance else '0')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
