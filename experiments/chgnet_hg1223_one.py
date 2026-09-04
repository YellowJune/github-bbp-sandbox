#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325; GPA_TO_EV_A3=1/160.21766208
A=3.892682;C=16.248444
SY=['Ba','Ba','Ca','Ca','Cu','Cu','Cu','Hg','O','O','O','O','O','O','O','O']
FR=[(.5,.5,.816083),(.5,.5,.183917),(.5,.5,.599071),(.5,.5,.400929),(0,0,.691918),(0,0,.308082),(0,0,.5),(0,0,0),(.5,0,.5),(0,.5,.5),(0,0,.876271),(0,0,.123729),(.5,0,.303695),(.5,0,.696305),(0,.5,.303695),(0,.5,.696305)]

def structure(name):
    s=list(SY)
    if name!='parent':
        cat,loc=name.split('N_',1)
        if cat=='La': s[0]='La'
        elif cat in ('Bi','Tl'): s[7]=cat
        else: raise ValueError(cat)
        s[{'apical':10,'outer':13,'center':8}[loc]]='N'
    return Structure(Lattice.tetragonal(A,C),s,FR)

def geom(atoms):
    d=atoms.get_all_distances(mic=True); n=len(atoms); vals=d[np.triu_indices(n,1)]
    sy=atoms.get_chemical_symbols(); cu_l=[]
    for i,x in enumerate(sy):
        if x!='Cu':continue
        for j,y in enumerate(sy):
            if y in ('O','N') and i!=j:cu_l.append(d[i,j])
    return {'min_pair_A':float(vals.min()),'min_Cu_ligand_A':float(min(cu_l)),'mean_4short_Cu_ligand_A':float(np.mean(sorted(cu_l)[:12]))}

def relax(st,model,p_gpa,out,tag):
    at=AseAtomsAdaptor.get_atoms(st); v0=at.get_volume();at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
    opt=FIRE(filt,logfile=str(out/f'{tag}.log'));opt.run(fmax=.08,steps=180)
    f=at.get_forces(); rec={'pressure_GPa':p_gpa,'pressure_atm':p_gpa/ATM_TO_GPA,'max_force_eV_A':float(np.max(np.linalg.norm(f,axis=1))),'energy_eV_atom':float(at.get_potential_energy()/len(at)),'volume_ratio':float(at.get_volume()/v0),'natoms':len(at)};rec.update(geom(at));rec['structural_pass']=bool(rec['max_force_eV_A']<.10 and .85<rec['volume_ratio']<1.15 and rec['min_pair_A']>1.35)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{tag}.cif'));return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--only',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    st=structure(a.only); assert all(x.specie.symbol!='H' for x in st)
    model=CHGNet.load();rows=[]
    for pg in [0.0,400*ATM_TO_GPA]:
        try:r=relax(st,model,pg,out,f'{a.only}_{pg:.5f}GPa');r['name']=a.only
        except Exception as e:r={'name':a.only,'pressure_GPa':pg,'error':repr(e),'structural_pass':False}
        rows.append(r);print(json.dumps(r,default=float))
    keys=sorted({k for r in rows for k in r});
    with open(out/'relax.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2,default=float))
if __name__=='__main__':main()
