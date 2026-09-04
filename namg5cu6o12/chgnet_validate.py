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

def build(name,a0):
    # 3x2 infinite-layer supercell: A at z=1/2, CuO2 at z=0.
    c0=3.20
    sy=[]; fr=[]; tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Mg','Cu','O','O']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],[ix/3,iy/2,0],[(ix+.5)/3,iy/2,0],[ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Ox',ix,iy),('Oy',ix,iy)]
    if name=='NaMg5Cu6O12':
        sy[tags.index(('A',0,0))]='Na'
    elif name=='Mg6Cu6O12':
        pass
    else:
        raise ValueError(name)
    s=Structure(Lattice.orthorhombic(3*a0,2*a0,c0),sy,fr)
    if any(x.specie.symbol=='H' for x in s): raise RuntimeError('hydrogen forbidden')
    return s

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    cuo=[]; cucu=[]; oo=[]
    for i,a in enumerate(sy):
        for j,b in enumerate(sy):
            if i>=j: continue
            if {a,b}=={'Cu','O'}: cuo.append(d[i,j])
            if a==b=='Cu': cucu.append(d[i,j])
            if a==b=='O': oo.append(d[i,j])
    # Cu first-shell O coordination.
    coord=[]
    for i,a in enumerate(sy):
        if a!='Cu': continue
        coord.append(sum(1 for j,b in enumerate(sy) if b=='O' and d[i,j]<2.25))
    return {'min_CuO_A':float(min(cuo)),'median_first_CuO_A':float(np.median(sorted(cuo)[:24])),
            'min_CuCu_A':float(min(cucu)),'min_OO_A':float(min(oo)),
            'mean_Cu_O_coord_lt2p25':float(np.mean(coord))}

def run(name,a0,p,out,model):
    at=AseAtomsAdaptor.get_atoms(build(name,a0));v0=at.get_volume();m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(filt,logfile=str(out/f'{name}_a{a0:.2f}_p{p:.5f}.log')).run(fmax=.055,steps=260)
    forces=at.get_forces(); stress=at.get_stress(voigt=True)*160.21766208
    m=metrics(at)
    rec={'name':name,'initial_a_A':a0,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'natoms':len(at),
         'max_force_eV_A':float(np.linalg.norm(forces,axis=1).max()),
         'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in at.cell.angles()],
         'stress_GPa_voigt':[float(x) for x in stress]}
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.09 and .75<rec['volume_ratio']<1.25 and 1.65<m['min_CuO_A']<2.15 and m['mean_Cu_O_coord_lt2p25']>=3.5 and m['min_OO_A']>1.6)
    rec['bandwidth_300K_005t_geometry_pass']=bool(m['median_first_CuO_A']<=1.8622)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_a{a0:.2f}_p{p:.5f}_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['Mg6Cu6O12','NaMg5Cu6O12'],required=True);ap.add_argument('--a0',type=float,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[run(a.name,a.a0,0.0,out,model),run(a.name,a.a0,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
