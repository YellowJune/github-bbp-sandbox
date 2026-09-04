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

# Stage 40/41 Pareto motif.
# N bonds (1,3,4,5,10) = Ny(0,0), Ny(1,0), Nx(2,0), Ny(2,0), Nx(2,1)
N_TAGS={('Ny',0,0),('Ny',1,0),('Nx',2,0),('Ny',2,0),('Nx',2,1)}
# Formal-charge Ewald optimum was invariant across all 12 (a,c) stress geometries.
SC_SITES={(0,0),(1,0),(2,0),(2,1)}

def build(a0=3.70,c0=3.20):
    sy=[];fr=[];tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Mg','Cu','O','O']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],
                   [ix/3,iy/2,0],
                   [(ix+.5)/3,iy/2,0],
                   [ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Nx',ix,iy),('Ny',ix,iy)]
    for p in SC_SITES: sy[tags.index(('A',*p))]='Sc'
    for t in N_TAGS: sy[tags.index(t)]='N'
    s=Structure(Lattice.orthorhombic(3*a0,2*a0,c0),sy,fr)
    assert sum(x.specie.symbol=='Sc' for x in s)==4
    assert sum(x.specie.symbol=='Mg' for x in s)==2
    assert sum(x.specie.symbol=='Cu' for x in s)==6
    assert sum(x.specie.symbol=='N' for x in s)==5
    assert sum(x.specie.symbol=='O' for x in s)==7
    return s

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cu=[i for i,s in enumerate(sy) if s=='Cu']; lig=[i for i,s in enumerate(sy) if s in {'O','N'}]
    first=[];firstN=[];firstO=[];coord=[]
    for i in cu:
        ds=sorted([(float(d[i,j]),sy[j]) for j in lig],key=lambda x:x[0])[:4]
        first += [z[0] for z in ds]
        firstN += [z[0] for z in ds if z[1]=='N']
        firstO += [z[0] for z in ds if z[1]=='O']
        coord.append(sum(float(d[i,j])<2.35 for j in lig))
    an=[]
    for q,i in enumerate(lig):
        for j in lig[q+1:]: an.append(float(d[i,j]))
    return {
      'mean_first4_CuLig_A':float(np.mean(first)),
      'std_first4_CuLig_A':float(np.std(first)),
      'min_first4_CuLig_A':float(min(first)),
      'max_first4_CuLig_A':float(max(first)),
      'mean_first_CuN_A':float(np.mean(firstN)) if firstN else np.nan,
      'mean_first_CuO_A':float(np.mean(firstO)) if firstO else np.nan,
      'mean_Cu_lig_coord_lt2p35':float(np.mean(coord)),
      'min_anion_anion_A':float(min(an))}

def relax(s,p,model,log,fmax=.05,steps=420):
    at=AseAtomsAdaptor.get_atoms(s); at.calc=CHGNetCalculator(model=model)
    FIRE(FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3),logfile=str(log)).run(fmax=fmax,steps=steps)
    return at

def Hpa(at,p):
    return float(at.get_potential_energy()/len(at)+p*GPA_TO_EV_A3*at.get_volume()/len(at))

def binary(A,B,proto,a):
    if proto=='rocksalt': return Structure.from_spacegroup('Fm-3m',Lattice.cubic(a),[A,B],[[0,0,0],[.5,.5,.5]])
    if proto=='zincblende': return Structure.from_spacegroup('F-43m',Lattice.cubic(a),[A,B],[[0,0,0],[.25,.25,.25]])
    raise ValueError(proto)

def binary_mix(p,out,model):
    specs={
      'ScN':(('Sc','N'),[('rocksalt',4.50)]),
      'MgO':(('Mg','O'),[('rocksalt',4.21)]),
      'CuO':(('Cu','O'),[('rocksalt',4.30),('zincblende',4.30)]),
      'CuN':(('Cu','N'),[('rocksalt',4.05),('zincblende',4.05)]),
    }
    best={}; detail={}
    for name,(AB,ps) in specs.items():
        vals=[]
        for proto,a in ps:
            at=relax(binary(*AB,proto,a),p,model,out/f'{name}_{proto}_{p:.5f}.log',fmax=.045,steps=260)
            vals.append({'prototype':proto,'enthalpy_eV_atom':Hpa(at,p),'max_force_eV_A':float(np.linalg.norm(at.get_forces(),axis=1).max())})
        detail[name]=vals; best[name]=min(x['enthalpy_eV_atom'] for x in vals)
    # 4 ScN + 2 MgO + 5 CuO + 1 CuN = Sc4Mg2Cu6N5O7, 24 atoms total.
    mix=(8*best['ScN']+4*best['MgO']+10*best['CuO']+2*best['CuN'])/24
    return mix,best,detail

def run(a0,p,out,model):
    s=build(a0=a0); at0=AseAtomsAdaptor.get_atoms(s); v0=at0.get_volume(); m0=metrics(at0)
    at=relax(s,p,model,out/f'cand_a{a0:.2f}_p{p:.5f}.log')
    m=metrics(at); f=np.linalg.norm(at.get_forces(),axis=1); ang=at.cell.angles(); h=Hpa(at,p)
    mix,best,detail=binary_mix(p,out,model)
    rec={'candidate':'Sc4Mg2Cu6N5O7','a0_A':a0,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'enthalpy_eV_atom':h,'binary_mix_enthalpy_eV_atom':mix,
         'deltaH_vs_binary_mix_eV_atom':float(h-mix),'binary_best_eV_atom':best,'binary_details':detail,
         'max_force_eV_A':float(f.max()),'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],'cell_angles_deg':[float(x) for x in ang]}
    rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(rec['max_force_eV_A']<.10 and .70<rec['volume_ratio']<1.30 and
        m['min_first4_CuLig_A']>1.55 and m['max_first4_CuLig_A']<2.35 and
        m['mean_Cu_lig_coord_lt2p35']>=3.8 and m['min_anion_anion_A']>1.20 and min(ang)>75 and max(ang)<105)
    rec['strict_geometry_target_pass']=bool(m['mean_first4_CuLig_A']<=1.89)
    rec['conservative_binary_mix_gate']=bool(rec['deltaH_vs_binary_mix_eV_atom']<=.10)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'cand_a{a0:.2f}_p{p:.5f}_relaxed.cif'))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--a0',type=float,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[run(a.a0,0.,out,model),run(a.a0,400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
