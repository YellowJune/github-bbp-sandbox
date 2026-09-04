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

A_ORDER=[(0,0),(1,1),(2,0),(0,1),(1,0),(2,1)]
N_ORDER=[('Nx',0,0),('Ny',1,1),('Nx',2,0),('Ny',0,1),('Nx',1,0),('Ny',2,1),
         ('Ny',0,0),('Nx',1,1),('Ny',2,0),('Nx',0,1),('Ny',1,0),('Nx',2,1)]

def build(k,a0=3.72,c0=3.20):
    if k not in (4,5,6): raise ValueError(k)
    nsc=k-1
    sy=[];fr=[];tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Mg','Cu','O','O']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],[ix/3,iy/2,0],[(ix+.5)/3,iy/2,0],[ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Nx',ix,iy),('Ny',ix,iy)]
    for ix,iy in A_ORDER[:nsc]: sy[tags.index(('A',ix,iy))]='Sc'
    for tag in N_ORDER[:k]: sy[tags.index(tag)]='N'
    s=Structure(Lattice.orthorhombic(3*a0,2*a0,c0),sy,fr)
    assert sum(x.specie.symbol=='Sc' for x in s)==nsc
    assert sum(x.specie.symbol=='Mg' for x in s)==7-k
    assert sum(x.specie.symbol=='N' for x in s)==k
    assert sum(x.specie.symbol=='O' for x in s)==12-k
    return s

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cu=[i for i,s in enumerate(sy) if s=='Cu']; lig=[i for i,s in enumerate(sy) if s in {'O','N'}]
    first=[]; firstN=[]; firstO=[]; coord=[]
    for i in cu:
        ds=sorted([(float(d[i,j]),sy[j]) for j in lig],key=lambda x:x[0])[:4]
        first.extend(x[0] for x in ds); firstN.extend(x[0] for x in ds if x[1]=='N'); firstO.extend(x[0] for x in ds if x[1]=='O')
        coord.append(sum(float(d[i,j])<2.35 for j in lig))
    anion=[]; cucu=[]
    for i in range(len(sy)):
        for j in range(i+1,len(sy)):
            if sy[i] in {'O','N'} and sy[j] in {'O','N'}: anion.append(float(d[i,j]))
            if sy[i]==sy[j]=='Cu': cucu.append(float(d[i,j]))
    return {
      'mean_first4_CuLig_A':float(np.mean(first)),'std_first4_CuLig_A':float(np.std(first)),
      'max_first4_CuLig_A':float(max(first)),'min_first4_CuLig_A':float(min(first)),
      'mean_first_CuN_A':float(np.mean(firstN)) if firstN else np.nan,
      'mean_first_CuO_A':float(np.mean(firstO)) if firstO else np.nan,
      'mean_Cu_lig_coord_lt2p35':float(np.mean(coord)),
      'min_anion_anion_A':float(min(anion)),'min_CuCu_A':float(min(cucu))}

def run(k,p,out,model):
    at=AseAtomsAdaptor.get_atoms(build(k)); v0=at.get_volume(); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(filt,logfile=str(out/f'k{k}_p{p:.5f}.log')).run(fmax=.060,steps=320)
    m=metrics(at); forces=at.get_forces(); ang=at.cell.angles()
    rec={'k_N':k,'name':f'Sc{k-1}Mg{7-k}Cu6N{k}O{12-k}','pressure_GPa_target':p,
         'pressure_atm_target':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),'natoms':len(at),
         'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'max_force_eV_A':float(np.linalg.norm(forces,axis=1).max()),
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in ang]}
    rec.update({'initial_'+x:v for x,v in m0.items()}); rec.update({'final_'+x:v for x,v in m.items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.10 and .70<rec['volume_ratio']<1.30 and
       1.60<m['min_first4_CuLig_A'] and m['max_first4_CuLig_A']<2.35 and
       m['mean_Cu_lig_coord_lt2p35']>=3.8 and m['min_anion_anion_A']>1.25 and min(ang)>75 and max(ang)<105)
    rec['electronic_geometry_target_pass']=bool(m['mean_first4_CuLig_A']<=1.89)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'k{k}_p{p:.5f}_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--k',type=int,choices=[4,5,6],required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[run(a.k,0.,out,model),run(a.k,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
