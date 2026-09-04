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
A0=3.933714
C0=9.827602
BASE=[('Ba',(.5,.5,.695597)),('Ba',(.5,.5,.304403)),('Cu',(0,0,.5)),('Hg',(0,0,0)),('O',(0,0,.794971)),('O',(0,0,.205029)),('O',(0,.5,.5)),('O',(.5,0,.5))]

def build(name,scale):
    sy=[];fr=[];tags=[]
    for ix in range(2):
        for j,(s,p) in enumerate(BASE):
            sy.append(s);fr.append(((p[0]+ix)/2,p[1],p[2]));tags.append((ix,j))
    if name=='Ba4Cu2Hg2O8':
        pass
    elif name=='Ba4Cu2HgTlNO7':
        sy[tags.index((0,3))]='Tl'   # one Hg -> Tl
        sy[tags.index((0,4))]='N'    # paired apical O -> N
    else:
        raise ValueError(name)
    lat=Lattice.orthorhombic(2*A0*scale,A0*scale,C0*scale)
    st=Structure(lat,sy,fr)
    if any(x.specie.symbol=='H' for x in st): raise RuntimeError('hydrogen forbidden')
    return st

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cu=[i for i,x in enumerate(sy) if x=='Cu']; lig=[i for i,x in enumerate(sy) if x in ('O','N')]
    first=[];coord=[];by={'O':[],'N':[]}
    for i in cu:
        p=sorted([(float(d[i,j]),sy[j]) for j in lig],key=lambda z:z[0])[:6]
        first += [x[0] for x in p[:4]]
        coord.append(sum(x[0]<2.30 for x in p))
        for dist,e in p:
            if dist<3.4: by[e].append(dist)
    # reservoir-ligand distances
    res=[]
    for i,e in enumerate(sy):
        if e not in ('Hg','Tl'): continue
        for j,l in enumerate(sy):
            if l in ('O','N'): res.append((float(d[i,j]),e,l))
    res=sorted(res,key=lambda z:z[0])
    ll=[]
    for i in lig:
        for j in lig:
            if i<j: ll.append(float(d[i,j]))
    return {
        'median_first_CuLig_A':float(np.median(first)),
        'p90_first_CuLig_A':float(np.quantile(first,.9)),
        'min_first_CuLig_A':float(min(first)),
        'mean_Cu_ligand_coord_lt2p30':float(np.mean(coord)),
        'median_near_CuO_A':float(np.median(by['O'])) if by['O'] else None,
        'median_near_CuN_A':float(np.median(by['N'])) if by['N'] else None,
        'min_reservoir_ligand_A':float(res[0][0]),
        'min_TlN_A':float(min([x[0] for x in res if x[1]=='Tl' and x[2]=='N'],default=np.nan)),
        'min_HgO_A':float(min([x[0] for x in res if x[1]=='Hg' and x[2]=='O'],default=np.nan)),
        'min_ligand_ligand_A':float(min(ll)),
    }

def run(name,scale,p,out,model):
    at=AseAtomsAdaptor.get_atoms(build(name,scale));v0=at.get_volume();m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(filt,logfile=str(out/f'{name}_{scale:.3f}_p{p:.5f}.log')).run(fmax=.05,steps=320)
    f=at.get_forces(); stress=at.get_stress(voigt=True)*160.21766208; m=metrics(at)
    rec={'name':name,'start_scale':scale,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'max_force_eV_A':float(np.linalg.norm(f,axis=1).max()),
         'energy_eV_atom':float(at.get_potential_energy()/len(at)),'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],'cell_angles_deg':[float(x) for x in at.cell.angles()],
         'stress_GPa_voigt':[float(x) for x in stress]}
    rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.085 and .7<rec['volume_ratio']<1.3 and m['mean_Cu_ligand_coord_lt2p30']>=4 and 1.65<m['min_first_CuLig_A']<2.2 and m['min_ligand_ligand_A']>1.5)
    rec['reservoir_geometry_retained']=bool((name!='Ba4Cu2HgTlNO7') or (np.isfinite(m['min_TlN_A']) and 1.6<m['min_TlN_A']<2.8))
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{name}_{scale:.3f}_p{p:.5f}_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['Ba4Cu2Hg2O8','Ba4Cu2HgTlNO7'],required=True);ap.add_argument('--scale',type=float,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[run(a.name,a.scale,0.0,out,model),run(a.name,a.scale,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
