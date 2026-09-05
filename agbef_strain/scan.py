from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import UnitCellFilter
from ase.optimize import FIRE

GPA_PER_EV_A3=160.21766208
STRAINS=[0.0,-0.02,-0.04,-0.05,-0.06]
A0=4.10
C0=7.80
DZ=0.21


def build(kind,a):
    sy=[];fr=[];ai=0
    dopant={'Parent':None,'Li':'Li','Al':'Al'}[kind]
    for y in range(2):
        for x in range(3):
            if ai==0 and dopant is not None: A=dopant
            else: A='Be'
            sy.append(A);fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Ag');fr.append([x/3,y/2,0])
            sy.append('F');fr.append([(x+.5)/3,y/2,0])
            sy.append('F');fr.append([x/3,(y+.5)/2,0])
            sy.append('F');fr.append([(x+.5)/3,(y+.5)/2,.5-DZ])
            sy.append('F');fr.append([(x+.5)/3,(y+.5)/2,.5+DZ])
    return Structure(Lattice.orthorhombic(3*a,2*a,C0),sy,fr)


def micvec(at,i,j): return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)


def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag=[i for i,s in enumerate(sy) if s=='Ag']; fs=[i for i,s in enumerate(sy) if s=='F']
    nearest=[];coord=[];angles=[];bridge=0
    for i in ag:
        vals=sorted(float(d[i,j]) for j in fs); nearest.append(float(np.mean(vals[:4])));coord.append(sum(v<2.55 for v in vals))
    for f in fs:
        close=sorted((float(d[f,i]),i) for i in ag);close=[x for x in close if x[0]<2.55]
        if len(close)>=2:
            v1=micvec(at,f,close[0][1]);v2=micvec(at,f,close[1][1]);den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))));bridge+=1
    dm=float(np.mean(nearest)); am=float(np.mean(angles)) if angles else float('nan')
    jproxy=206.2*(2.07/dm)**9.02*(max(math.cos(math.radians(180-am))**2,1e-4)**1.07) if np.isfinite(am) else float('nan')
    allpairs=[float(d[i,j]) for i in range(len(at)) for j in range(i+1,len(at))]
    return {'mean_Ag_F_nearest4_A':dm,'max_Ag_F_nearest4_A':float(max(nearest)),
            'min_Ag_F_coord_lt2p55':int(min(coord)),'n_bridging_F':int(bridge),
            'mean_Ag_F_Ag_angle_deg':am,'p10_Ag_F_Ag_angle_deg':float(np.quantile(angles,.10)) if angles else float('nan'),
            'min_any_pair_A':float(min(allpairs)),'controlfit_DFTU_J_proxy_meV':float(jproxy)}


def formal_valence(kind):
    # F24 plus six A sites: Parent=Be6, LiBe5, AlBe5.
    apos={'Parent':12,'Li':11,'Al':13}[kind]
    return (24-apos)/6


def relax_one(kind,strain,model,out):
    a=A0*(1+strain); s=build(kind,a); at=AseAtomsAdaptor.get_atoms(s); v0=float(at.get_volume())
    at.calc=CHGNetCalculator(model=model)
    # Epitaxial constraint: freeze xx,yy and shear; relax atoms plus only c-axis strain.
    flt=UnitCellFilter(at,mask=[0,0,1,0,0,0])
    tag=f'{kind}_strain_{strain:+.3f}'
    FIRE(flt,logfile=str(out/f'{tag}.log')).run(fmax=.05,steps=600)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max()); m=metrics(at)
    stress=np.asarray(at.get_stress(voigt=True),float)*GPA_PER_EV_A3
    epa=float(at.get_potential_energy()/len(at))
    rec={'kind':kind,'strain':strain,'a_fixed_A':a,'formula':at.get_chemical_formula(),'formal_Ag_valence':formal_valence(kind),
         'carrier_per_Ag_signed_hole_positive':formal_valence(kind)-2.0,'max_force_eV_A':fmax,'energy_eV_atom':epa,
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'stress_xx_GPa':float(stress[0]),'stress_yy_GPa':float(stress[1]),'stress_zz_GPa':float(stress[2])}
    rec.update(m)
    rec['plane_pass']=bool(fmax<.085 and m['mean_Ag_F_nearest4_A']<2.15 and m['max_Ag_F_nearest4_A']<2.25 and
                           m['min_Ag_F_coord_lt2p55']>=4 and m['mean_Ag_F_Ag_angle_deg']>=170 and m['p10_Ag_F_Ag_angle_deg']>=165 and m['min_any_pair_A']>1.10)
    rec['geometry_384meV_gate']=bool(rec['plane_pass'] and m['controlfit_DFTU_J_proxy_meV']>=384.0)
    cif=out/f'{tag}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['Parent','Li','Al'],required=True);a=ap.parse_args()
    out=Path(f'artifacts/strain/{a.kind}');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[]
    for eps in STRAINS: rows.append(relax_one(a.kind,eps,model,out))
    e0=next(r['energy_eV_atom'] for r in rows if r['strain']==0.0)
    for r in rows:r['delta_energy_from_unstrained_eV_atom']=r['energy_eV_atom']-e0
    survivors=[r for r in rows if r['geometry_384meV_gate']]
    result={'kind':a.kind,'candidate_formulas':{'Parent':'Be6Ag6F24','Li':'LiBe5Ag6F24','Al':'AlBe5Ag6F24'},
            'rows':rows,'geometry_gate_survivors':survivors,'advance':bool(survivors)}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'ADVANCE').write_text('1' if survivors else '0')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
