from __future__ import annotations
import argparse, json
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
        a,c=3.86,3.20;sy=[];fr=[]
        for ix in range(2):
            for iy in range(2):
                sy += ['Ca','Cu','O','O']
                fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
        return Structure(Lattice.tetragonal(2*a,c),sy,fr)
    if name=='Cu3N':
        a=3.82
        return Structure(Lattice.cubic(a),['N','Cu','Cu','Cu'],[[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
    if name=='ScTiCu2N3O':
        a,c=3.72,3.00
        # 2x1 infinite-layer cell: Sc3+ + Ti4+ + 2Cu2+ + 3N3- + O2- = 0.
        sy=['Sc','Cu','O','N','Ti','Cu','N','N']
        fr=[[.25,.5,.5],[0,0,0],[.25,0,0],[0,.5,0],
            [.75,.5,.5],[.5,0,0],[.75,0,0],[.5,.5,0]]
        return Structure(Lattice.orthorhombic(2*a,a,c),sy,fr)
    raise ValueError(name)

def metrics(atoms):
    sy=atoms.get_chemical_symbols();d=atoms.get_all_distances(mic=True)
    def vals(a,b):
        return [d[i,j] for i,s in enumerate(sy) for j,t in enumerate(sy) if i!=j and ((s==a and t==b) or (s==b and t==a))]
    def mn(a,b):
        v=vals(a,b);return float(min(v)) if v else float('nan')
    cuN=[];cuO=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        cuN.append(sum(t=='N' and d[i,j]<2.30 for j,t in enumerate(sy)))
        cuO.append(sum(t=='O' and d[i,j]<2.30 for j,t in enumerate(sy)))
    return {'min_Cu_N_A':mn('Cu','N'),'min_Cu_O_A':mn('Cu','O'),'min_N_N_A':mn('N','N'),
            'min_Sc_N_A':mn('Sc','N'),'min_Ti_N_A':mn('Ti','N'),
            'mean_Cu_N_coord_lt2p30':float(np.mean(cuN)) if cuN else float('nan'),
            'mean_Cu_O_coord_lt2p30':float(np.mean(cuO)) if cuO else float('nan')}

def run(name,p,out,model):
    s=build(name);at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    fil=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    opt=FIRE(fil,logfile=str(out/f'{name}_{p:.5f}.log'));opt.run(fmax=.05,steps=300)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());m1=metrics(at)
    rec={'name':name,'target_GPa':p,'target_atm':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),
         'natoms':len(at),'max_force_eV_A':fmax,'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in at.cell.angles()],
         'stress_GPa_voigt':[float(x*160.21766208) for x in at.get_stress(voigt=True)]}
    rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in m1.items()})
    rec['gross_structure_pass']=bool(fmax<.08 and .75<rec['volume_ratio']<1.25 and (not np.isfinite(m1['min_N_N_A']) or m1['min_N_N_A']>1.35))
    if name=='ScTiCu2N3O':
        rec['pairing_geometry_pass_300scale']=bool(np.isfinite(m1['min_Cu_N_A']) and m1['min_Cu_N_A']<=1.93 and m1['mean_Cu_N_coord_lt2p30']>=2.5)
        rec['pairing_geometry_high_margin']=bool(np.isfinite(m1['min_Cu_N_A']) and m1['min_Cu_N_A']<=1.91)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_{p:.5f}GPa_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['CaCuO2','Cu3N','ScTiCu2N3O'],required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[run(a.name,0,out,model),run(a.name,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
