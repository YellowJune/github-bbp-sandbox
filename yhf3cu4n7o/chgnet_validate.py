from __future__ import annotations
import argparse, json, math
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

def build(name):
    if name=='CaCuO2':
        a,c=3.86,3.20; sy=[]; fr=[]
        for ix in range(2):
            for iy in range(2):
                sy += ['Ca','Cu','O','O']
                fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
        return Structure(Lattice.tetragonal(2*a,c),sy,fr)
    if name=='YHf3Cu4N7O':
        a,c=3.82,3.28; sy=[]; fr=[]; tags=[]
        for ix in range(2):
            for iy in range(2):
                sy += ['Hf','Cu','N','N']
                fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
                tags += [('A',ix,iy),('Cu',ix,iy),('Lx',ix,iy),('Ly',ix,iy)]
        # Local charge compensation: one Y3+ and nearest one O2- in an otherwise Hf4+/N3- cell.
        sy[tags.index(('A',0,0))]='Y'
        sy[tags.index(('Lx',0,0))]='O'
        return Structure(Lattice.tetragonal(2*a,c),sy,fr)
    if name=='Cu3N':
        a=3.82
        return Structure(Lattice.cubic(a),['N','Cu','Cu','Cu'],[[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
    raise ValueError(name)

def bond_metrics(atoms):
    sy=atoms.get_chemical_symbols(); d=atoms.get_all_distances(mic=True)
    def mind(a,b):
        vals=[d[i,j] for i,s in enumerate(sy) for j,t in enumerate(sy) if i!=j and ((s==a and t==b) or (s==b and t==a))]
        return float(min(vals)) if vals else float('nan')
    # Cu coordination counts inside a broad first-shell cutoff.
    cuN=[];cuO=[]
    for i,s in enumerate(sy):
        if s!='Cu': continue
        cuN.append(sum(1 for j,t in enumerate(sy) if t=='N' and d[i,j]<2.35))
        cuO.append(sum(1 for j,t in enumerate(sy) if t=='O' and d[i,j]<2.35))
    return {'min_Cu_N_A':mind('Cu','N'),'min_Cu_O_A':mind('Cu','O'),'min_N_N_A':mind('N','N'),'mean_Cu_N_coord_lt2p35':float(np.mean(cuN)) if cuN else float('nan'),'mean_Cu_O_coord_lt2p35':float(np.mean(cuO)) if cuO else float('nan')}

def run_one(name,pressure_gpa,out):
    s=build(name); atoms=AseAtomsAdaptor.get_atoms(s); init=atoms.copy(); model=CHGNet.load(); atoms.calc=CHGNetCalculator(model=model)
    initV=float(atoms.get_volume()); initM=bond_metrics(atoms)
    filt=FrechetCellFilter(atoms,scalar_pressure=pressure_gpa*GPA_TO_EV_A3)
    opt=FIRE(filt,logfile=str(out/f'{name}_{pressure_gpa:.5f}.log'))
    opt.run(fmax=.055,steps=260)
    f=atoms.get_forces(); stress=atoms.get_stress(voigt=True)
    rec={'name':name,'pressure_GPa_target':pressure_gpa,'pressure_atm_target':pressure_gpa/ATM_TO_GPA,
         'natoms':len(atoms),'formula':atoms.get_chemical_formula(),'max_force_eV_A':float(np.linalg.norm(f,axis=1).max()),
         'energy_eV_atom':float(atoms.get_potential_energy()/len(atoms)),'volume_ratio':float(atoms.get_volume()/initV),
         'cell_lengths_A':[float(x) for x in atoms.cell.lengths()],'cell_angles_deg':[float(x) for x in atoms.cell.angles()],
         'stress_GPa_voigt':[float(x*160.21766208) for x in stress]}
    rec.update({'initial_'+k:v for k,v in initM.items()});rec.update({'final_'+k:v for k,v in bond_metrics(atoms).items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.08 and .78<rec['volume_ratio']<1.22 and (not np.isfinite(rec['final_min_N_N_A']) or rec['final_min_N_N_A']>1.35) and (not np.isfinite(rec['final_min_Cu_N_A']) or rec['final_min_Cu_N_A']<2.20))
    AseAtomsAdaptor.get_structure(atoms).to(filename=str(out/f'{name}_{pressure_gpa:.5f}GPa_relaxed.cif'))
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['CaCuO2','Cu3N','YHf3Cu4N7O'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[run_one(a.name,0.0,out),run_one(a.name,400*ATM_TO_GPA,out)]
    (out/'result.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
