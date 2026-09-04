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


def build(name,a0=3.80,c0=3.30):
    sy=[]; fr=[]; tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Hf','Cu','N','N']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],
                   [ix/3,iy/2,0],
                   [(ix+.5)/3,iy/2,0],
                   [ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Nx',ix,iy),('Ny',ix,iy)]
    ai=tags.index(('A',0,0))
    if name=='Hf6Cu6N12': pass
    elif name=='ScHf5Cu6N12': sy[ai]='Sc'
    elif name=='YHf5Cu6N12': sy[ai]='Y'
    else: raise ValueError(name)
    s=Structure(Lattice.orthorhombic(3*a0,2*a0,c0),sy,fr)
    if any(x.specie.symbol=='H' for x in s): raise RuntimeError('hydrogen forbidden')
    return s


def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cu=[i for i,s in enumerate(sy) if s=='Cu']; nn=[i for i,s in enumerate(sy) if s=='N']
    cun=[]; nnd=[]; cucu=[]
    first=[]; coord=[]
    for i in cu:
        ds=sorted(float(d[i,j]) for j in nn if i!=j)
        first.extend(ds[:4]); coord.append(sum(x<2.35 for x in ds))
    for i in range(len(sy)):
        for j in range(i+1,len(sy)):
            if {sy[i],sy[j]}=={'Cu','N'}: cun.append(float(d[i,j]))
            if sy[i]==sy[j]=='N': nnd.append(float(d[i,j]))
            if sy[i]==sy[j]=='Cu': cucu.append(float(d[i,j]))
    return {
      'mean_first4_CuN_A':float(np.mean(first)),
      'std_first4_CuN_A':float(np.std(first)),
      'min_CuN_A':float(min(cun)),
      'max_first4_CuN_A':float(max(first)),
      'mean_Cu_N_coord_lt2p35':float(np.mean(coord)),
      'min_NN_A':float(min(nnd)),
      'min_CuCu_A':float(min(cucu)),
    }


def run(name,a0,p_gpa,out,model):
    at=AseAtomsAdaptor.get_atoms(build(name,a0)); v0=at.get_volume(); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
    FIRE(filt,logfile=str(out/f'{name}_p{p_gpa:.5f}.log')).run(fmax=.060,steps=320)
    forces=at.get_forces(); stress=at.get_stress(voigt=True)*160.21766208
    m=metrics(at); lengths=at.cell.lengths(); angles=at.cell.angles()
    rec={
      'name':name,'initial_a_A':a0,'pressure_GPa_target':p_gpa,
      'pressure_atm_target':p_gpa/ATM_TO_GPA,'formula':at.get_chemical_formula(),
      'natoms':len(at),'max_force_eV_A':float(np.linalg.norm(forces,axis=1).max()),
      'energy_eV_atom':float(at.get_potential_energy()/len(at)),
      'volume_ratio':float(at.get_volume()/v0),
      'cell_lengths_A':[float(x) for x in lengths],
      'cell_angles_deg':[float(x) for x in angles],
      'stress_GPa_voigt':[float(x) for x in stress],
    }
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(
      rec['max_force_eV_A']<.10 and .70<rec['volume_ratio']<1.30 and
      1.65<m['min_CuN_A']<2.25 and m['max_first4_CuN_A']<2.35 and
      m['mean_Cu_N_coord_lt2p35']>=3.8 and m['min_NN_A']>1.30 and
      min(angles)>75 and max(angles)<105)
    rec['short_CuN_target_pass']=bool(m['mean_first4_CuN_A']<=1.93)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_p{p_gpa:.5f}_relaxed.cif'))
    return rec


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--name',choices=['Hf6Cu6N12','ScHf5Cu6N12','YHf5Cu6N12'],required=True)
    ap.add_argument('--a0',type=float,default=3.80); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); model=CHGNet.load()
    rows=[run(a.name,a.a0,0.0,out,model),run(a.name,a.a0,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
