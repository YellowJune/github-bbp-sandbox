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
A0=3.933714; C0=9.827602
BASE=[('Ba',(.5,.5,.695597)),('Ba',(.5,.5,.304403)),('Cu',(0,0,.5)),('Hg',(0,0,0)),('O',(0,0,.794971)),('O',(0,0,.205029)),('O',(0,.5,.5)),('O',(.5,0,.5))]

def build(name,scale=1.0):
    sy=[];fr=[];tags=[]
    for ix in range(3):
      for iy in range(2):
        for j,(s,p) in enumerate(BASE):
          sy.append(s);fr.append(((p[0]+ix)/3,(p[1]+iy)/2,p[2]));tags.append((ix,iy,j))
    if name in ('full_tln','k_doped'):
      for ix in range(3):
       for iy in range(2):
        sy[tags.index((ix,iy,3))]='Tl'
        sy[tags.index((ix,iy,4))]='N'
    if name=='k_doped':
      sy[tags.index((0,0,0))]='K'
    elif name not in ('parent_hg','full_tln'):
      raise ValueError(name)
    st=Structure(Lattice.orthorhombic(3*A0*scale,2*A0*scale,C0*scale),sy,fr)
    if any(x.specie.symbol=='H' for x in st):raise RuntimeError('H forbidden')
    return st

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    cu=[i for i,x in enumerate(sy) if x=='Cu'];lig=[i for i,x in enumerate(sy) if x in ('O','N')]
    first=[];coord=[]
    for i in cu:
      p=sorted([float(d[i,j]) for j in lig])[:4];first+=p;coord.append(sum(x<2.30 for x in p))
    ll=[float(d[i,j]) for ii,i in enumerate(lig) for j in lig[ii+1:]]
    res=[]
    for i,e in enumerate(sy):
      if e not in ('Hg','Tl'):continue
      for j,l in enumerate(sy):
        if l in ('O','N'):res.append((float(d[i,j]),e,l))
    return {'median_CuLig_A':float(np.median(first)),'p90_CuLig_A':float(np.quantile(first,.9)),'min_CuLig_A':float(min(first)),'mean_Cu_coord':float(np.mean(coord)),'min_liglig_A':float(min(ll)),'median_TlN_A':float(np.median([x[0] for x in res if x[1]=='Tl' and x[2]=='N'])) if any(x[1]=='Tl' and x[2]=='N' for x in res) else None}

def run(name,scale,p,out,model):
    at=AseAtomsAdaptor.get_atoms(build(name,scale));v0=at.get_volume();m0=metrics(at);at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3);FIRE(filt,logfile=str(out/f'{name}_{scale:.2f}_{p:.5f}.log')).run(fmax=.055,steps=300)
    f=at.get_forces();m=metrics(at);stress=at.get_stress(voigt=True)*160.21766208
    r={'name':name,'start_scale':scale,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),'max_force_eV_A':float(np.linalg.norm(f,axis=1).max()),'energy_eV_atom':float(at.get_potential_energy()/len(at)),'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],'stress_GPa_voigt':[float(x) for x in stress]}
    r.update({'initial_'+k:v for k,v in m0.items()});r.update({'final_'+k:v for k,v in m.items()})
    r['gross_structure_pass']=bool(r['max_force_eV_A']<.09 and .7<r['volume_ratio']<1.3 and m['mean_Cu_coord']>=3.5 and 1.6<m['min_CuLig_A']<2.2 and m['p90_CuLig_A']<2.2 and m['min_liglig_A']>1.45)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_{scale:.2f}_{p:.5f}.cif'))
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['parent_hg','full_tln','k_doped'],required=True);ap.add_argument('--scale',type=float,default=1.0);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[run(a.name,a.scale,0,out,model),run(a.name,a.scale,400*ATM_TO_GPA,out,model)];(out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
