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

# Exact robust ligand pattern from Stage 38 bond map:
# bond indices (1,2,7,8) = Ny(0,0), Nx(1,0), Ny(0,1), Nx(1,1)
N_TAGS={('Ny',0,0),('Nx',1,0),('Ny',0,1),('Nx',1,1)}
A_PATTERNS={
 'dispersed':{(0,0),(1,1),(2,0)},
 'N_aligned':{(0,0),(0,1),(1,0)},
 'N_away':{(2,0),(2,1),(1,1)},
}

def build(ordering,a0=3.715,c0=3.20):
    sc=A_PATTERNS[ordering]
    sy=[]; fr=[]; tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Mg','Cu','O','O']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],
                   [ix/3,iy/2,0],
                   [(ix+.5)/3,iy/2,0],
                   [ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Nx',ix,iy),('Ny',ix,iy)]
    for p in sc: sy[tags.index(('A',*p))]='Sc'
    for tag in N_TAGS: sy[tags.index(tag)]='N'
    s=Structure(Lattice.orthorhombic(3*a0,2*a0,c0),sy,fr)
    assert s.composition.reduced_formula in {'Sc3Mg3Cu6N4O8','Mg3Sc3Cu6N4O8'}
    return s,tags

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cu=[i for i,x in enumerate(sy) if x=='Cu']; lig=[i for i,x in enumerate(sy) if x in {'O','N'}]
    first=[]; nfirst=[]; ofirst=[]; coord=[]
    for i in cu:
        ds=sorted([(float(d[i,j]),sy[j]) for j in lig],key=lambda x:x[0])[:4]
        first += [x[0] for x in ds]
        nfirst += [x[0] for x in ds if x[1]=='N']
        ofirst += [x[0] for x in ds if x[1]=='O']
        coord.append(sum(float(d[i,j])<2.35 for j in lig))
    an=[]
    for i in lig:
        for j in lig:
            if j>i: an.append(float(d[i,j]))
    return {
      'mean_first4_CuLig_A':float(np.mean(first)),
      'std_first4_CuLig_A':float(np.std(first)),
      'min_first4_CuLig_A':float(min(first)),
      'max_first4_CuLig_A':float(max(first)),
      'mean_first_CuN_A':float(np.mean(nfirst)) if nfirst else np.nan,
      'mean_first_CuO_A':float(np.mean(ofirst)) if ofirst else np.nan,
      'mean_Cu_lig_coord_lt2p35':float(np.mean(coord)),
      'min_anion_anion_A':float(min(an))}

def run(ordering,p,out,model):
    s,_=build(ordering); at=AseAtomsAdaptor.get_atoms(s); v0=at.get_volume(); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    filt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(filt,logfile=str(out/f'{ordering}_{p:.5f}GPa.log')).run(fmax=.050,steps=420)
    m=metrics(at); f=np.linalg.norm(at.get_forces(),axis=1); ang=at.cell.angles()
    rec={'ordering':ordering,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'max_force_eV_A':float(f.max()),'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in ang]}
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.10 and .70<rec['volume_ratio']<1.30 and
      m['min_first4_CuLig_A']>1.55 and m['max_first4_CuLig_A']<2.35 and
      m['mean_Cu_lig_coord_lt2p35']>=3.8 and m['min_anion_anion_A']>1.20 and min(ang)>75 and max(ang)<105)
    # Pre-registered Stage 31/38 target. Geometry gate only, not Tc.
    rec['strict_geometry_target_pass']=bool(m['mean_first4_CuLig_A']<=1.89)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'{ordering}_{p:.5f}GPa_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ordering',choices=list(A_PATTERNS),required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); model=CHGNet.load()
    rows=[run(a.ordering,0.0,out,model),run(a.ordering,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
