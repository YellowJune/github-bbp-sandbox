from __future__ import annotations
import argparse,json,urllib.request
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208
PARENT='https://raw.githubusercontent.com/aimat-lab/3DSC/8d69ca9c94b83d387378549225d3b7c3af85ca42/superconductors_3D/data/source/MP/raw/cifs/mp-22601.cif'

def build(name,out):
    fn=out/'parent.cif'
    if not fn.exists(): urllib.request.urlretrieve(PARENT,fn)
    s=Structure.from_file(fn)
    if name=='Ag1223':
        for i,site in enumerate(list(s)):
            if site.specie.symbol=='Hg': s.replace(i,'Ag')
    elif name!='Hg1223': raise ValueError(name)
    if 'H' in s.composition: raise RuntimeError('hydrogen forbidden')
    return s

def metrics(atoms):
    sy=atoms.get_chemical_symbols();d=atoms.get_all_distances(mic=True)
    def pairs(a,b):
        v=[]
        for i,x in enumerate(sy):
            for j,y in enumerate(sy):
                if i<j and ({x,y}=={a,b}): v.append(d[i,j])
        return v
    rec={}
    for a,b,k in [('Cu','O','Cu_O'),('Hg','O','Hg_O'),('Ag','O','Ag_O'),('Ca','O','Ca_O'),('Ba','O','Ba_O')]:
        v=pairs(a,b);rec['min_'+k+'_A']=float(min(v)) if v else float('nan')
    return rec

def run(name,p,out,model):
    s=build(name,out);at=AseAtomsAdaptor.get_atoms(s);v0=at.get_volume();m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    opt=FIRE(filt,logfile=str(out/f'{name}_{p:.5f}.log'))
    opt.run(fmax=.06,steps=220)
    f=at.get_forces();stress=at.get_stress(voigt=True)*160.21766208
    rec={'name':name,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'natoms':len(at),
         'max_force_eV_A':float(np.linalg.norm(f,axis=1).max()),
         'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'stress_GPa_voigt':[float(x) for x in stress]}
    rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in metrics(at).items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.10 and .80<rec['volume_ratio']<1.20 and rec['final_min_Cu_O_A']<2.25)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_{p:.5f}GPa_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['Hg1223','Ag1223'],required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[run(a.name,0.0,out,model),run(a.name,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
