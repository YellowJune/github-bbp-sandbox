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
OUT=Path('artifacts/multivalent')
OUT.mkdir(parents=True,exist_ok=True)
META={
 'Al':3,'Ga':3,'Sc':3,
 'Ti':4,'Zr':4,'Hf':4,
}
# six spacer-center sites in the 3x2 supercell; subsets preserve total spacer neutrality:
# 4 M3+ + 12 F- = 0 or 3 M4+ + 12 F- = 0.
PATTERNS_3={
 'spread':[0,2,3,5],
 'stagger':[0,1,4,5],
}
PATTERNS_4={
 'spread':[0,2,4],
 'stagger':[1,3,5],
}
STARTS=[('central',4.05,7.8,.21),('expanded',4.16,8.4,.20)]

def centers():
    return [((x+.5)/3,(y+.5)/2,.5) for y in range(2) for x in range(3)]

def build(M,valence,pattern,a,c,dz):
    sy=[]; fr=[]
    # AgF2 magnetic sheet: 6 Ag + 12 bridging in-plane F.
    for y in range(2):
        for x in range(3):
            sy.append('Ag'); fr.append([x/3,y/2,0])
            sy.append('F');  fr.append([(x+.5)/3,y/2,0])
            sy.append('F');  fr.append([x/3,(y+.5)/2,0])
    # 12 spacer F, two around each plaquette center.
    for y in range(2):
        for x in range(3):
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5+dz])
    pats=PATTERNS_3 if valence==3 else PATTERNS_4
    for idx in pats[pattern]:
        sy.append(M); fr.append(list(centers()[idx]))
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def micvec(at,i,j):
    return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)

def metrics(at,M):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag=[i for i,s in enumerate(sy) if s=='Ag']; fs=[i for i,s in enumerate(sy) if s=='F']; ms=[i for i,s in enumerate(sy) if s==M]
    n4=[]; coords=[]; angles=[]; bridge=0
    for i in ag:
        vals=sorted(float(d[i,j]) for j in fs)
        n4.append(float(np.mean(vals[:4])))
        coords.append(sum(x<2.55 for x in vals))
    for f in fs:
        close=sorted((float(d[f,i]),i) for i in ag)
        close=[x for x in close if x[0]<2.55]
        if len(close)>=2:
            i,j=close[0][1],close[1][1]
            v1=micvec(at,f,i); v2=micvec(at,f,j); den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))))
                bridge+=1
    pairs=[]
    for i in range(len(at)):
        for j in range(i+1,len(at)):
            pairs.append(float(d[i,j]))
    magsep=min((float(d[i,j]) for i in ag for j in ms),default=float('nan'))
    return {
      'mean_Ag_F_nearest4_A':float(np.mean(n4)),
      'max_Ag_F_nearest4_A':float(np.max(n4)),
      'min_Ag_F_coord_lt2p55':int(min(coords)),
      'n_bridging_F':int(bridge),
      'mean_Ag_F_Ag_angle_deg':float(np.mean(angles)) if angles else float('nan'),
      'p10_Ag_F_Ag_angle_deg':float(np.quantile(angles,.10)) if angles else float('nan'),
      'min_any_pair_A':float(min(pairs)),
      'min_M_Ag_A':magsep,
    }

def relax_one(M,valence,pattern,label,a,c,dz,atm,model):
    s=build(M,valence,pattern,a,c,dz); at=AseAtomsAdaptor.get_atoms(s); v0=float(at.get_volume())
    at.calc=CHGNetCalculator(model=model)
    p=atm*ATM_TO_GPA
    flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    tag=f'{M}_v{valence}_{pattern}_{label}_{atm}atm'
    FIRE(flt,logfile=str(OUT/f'{tag}.log')).run(fmax=.05,steps=650)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max()); m=metrics(at,M)
    epa=float(at.get_potential_energy()/len(at)); vpa=float(at.get_volume()/len(at))
    st=np.asarray(at.get_stress(voigt=True))*160.21766208
    rec={'M':M,'valence':valence,'tag':tag,'pressure_atm':atm,'natoms':len(at),
         'formula':at.get_chemical_formula(),'max_force_eV_A':fmax,'energy_eV_atom':epa,
         'enthalpy_proxy_eV_atom':epa+p*vpa*GPA_TO_EV_A3,'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'hydrostatic_GPa_from_stress':float(-np.mean(st[:3]))}
    rec.update(m)
    rec['gross_pass']=bool(fmax<.085 and .60<rec['volume_ratio']<1.45 and m['min_any_pair_A']>1.15)
    rec['plane_pass']=bool(m['mean_Ag_F_nearest4_A']<2.18 and m['max_Ag_F_nearest4_A']<2.30 and
                           m['min_Ag_F_coord_lt2p55']>=4 and m['n_bridging_F']>=10 and
                           m['mean_Ag_F_Ag_angle_deg']>=165 and m['p10_Ag_F_Ag_angle_deg']>=155)
    cif=OUT/f'{tag}.cif'; AseAtomsAdaptor.get_structure(at).to(filename=str(cif)); rec['cif']=str(cif)
    return rec

def main(M):
    valence=META[M]; model=CHGNet.load(); rows=[]; selected={}
    pats=list((PATTERNS_3 if valence==3 else PATTERNS_4).keys())
    for atm in [0,400]:
        rr=[]
        for pattern in pats:
            for label,a,c,dz in STARTS:
                r=relax_one(M,valence,pattern,label,a,c,dz,atm,model); rr.append(r); rows.append(r)
        viable=[r for r in rr if r['gross_pass']]
        if viable:
            selected[str(atm)]=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom'])
    advance=bool(len(selected)==2 and all(x['plane_pass'] for x in selected.values()))
    formula=f'{M}{12//valence}Ag6F24'
    result={'candidate':formula,'spacer_formal_unit':f'{M}F{valence}','formal_Ag_valence':2.0,
            'all_runs':rows,'selected':selected,'advance':advance}
    md=OUT/M; md.mkdir(parents=True,exist_ok=True)
    (md/'result.json').write_text(json.dumps(result,indent=2)); (md/'ADVANCE').write_text('1' if advance else '0')
    # Copy selected CIFs into candidate subdir for follow-up QE.
    for atm,r in selected.items():
        src=Path(r['cif']); (md/f'selected_{atm}atm.cif').write_text(src.read_text())
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--M',required=True,choices=sorted(META)); a=ap.parse_args(); main(a.M)
