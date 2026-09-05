from __future__ import annotations
import argparse,json,math
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
ALLOWED={'Be','Mg','Ca','Sr','Zn'}
STARTS=[('compact',3.92,7.0,.22),('central',4.06,7.7,.21),('expanded',4.20,8.5,.20)]

def build(M,a,c,dz):
    sy=[];fr=[]
    for y in range(2):
        for x in range(3):
            sy.append(M); fr.append([(x+.5)/3,(y+.5)/2,.5])
            sy.append('Ag'); fr.append([x/3,y/2,0])
            sy.append('F'); fr.append([(x+.5)/3,y/2,0])
            sy.append('F'); fr.append([x/3,(y+.5)/2,0])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5+dz])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def micvec(at,i,j): return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    ag=[i for i,s in enumerate(sy) if s=='Ag']; fs=[i for i,s in enumerate(sy) if s=='F']
    n4=[]; coord=[]; ang=[]; bridge=0
    for i in ag:
        vals=sorted(float(d[i,j]) for j in fs); n4.append(float(np.mean(vals[:4])));coord.append(sum(x<2.55 for x in vals))
    for f in fs:
        close=sorted((float(d[f,i]),i) for i in ag);close=[x for x in close if x[0]<2.55]
        if len(close)>=2:
            v1=micvec(at,f,close[0][1]);v2=micvec(at,f,close[1][1]);den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                ang.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))));bridge+=1
    pairs=[float(d[i,j]) for i in range(len(at)) for j in range(i+1,len(at))]
    dm=float(np.mean(n4)); am=float(np.mean(ang)) if ang else float('nan')
    jproxy=206.2*(2.07/dm)**9.02*(max(math.cos(math.radians(180-am))**2,1e-4)**1.07) if np.isfinite(am) else float('nan')
    return {'mean_Ag_F_nearest4_A':dm,'max_Ag_F_nearest4_A':float(np.max(n4)),'min_Ag_F_coord_lt2p55':int(min(coord)),
            'n_bridging_F':int(bridge),'mean_Ag_F_Ag_angle_deg':am,'p10_Ag_F_Ag_angle_deg':float(np.quantile(ang,.10)) if ang else float('nan'),
            'min_any_pair_A':float(min(pairs)),'geometry_J_proxy_meV':jproxy}

def relax(M,label,a,c,dz,atm,model,out):
    s=build(M,a,c,dz);at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());p=atm*ATM_TO_GPA
    at.calc=CHGNetCalculator(model=model);flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    tag=f'{M}_{label}_{atm}atm';FIRE(flt,logfile=str(out/f'{tag}.log')).run(fmax=.05,steps=550)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());m=metrics(at);epa=float(at.get_potential_energy()/len(at));vpa=float(at.get_volume()/len(at))
    st=np.asarray(at.get_stress(voigt=True))*160.21766208
    cif=out/f'{tag}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif))
    r={'M':M,'tag':tag,'pressure_atm':atm,'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':epa+p*vpa*GPA_TO_EV_A3,
       'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
       'hydrostatic_GPa_from_stress':float(-np.mean(st[:3])),'cif':str(cif),**m}
    r['gross_pass']=bool(fmax<.085 and .60<r['volume_ratio']<1.45 and m['min_any_pair_A']>1.10)
    r['plane_pass']=bool(m['mean_Ag_F_nearest4_A']<2.16 and m['max_Ag_F_nearest4_A']<2.28 and m['min_Ag_F_coord_lt2p55']>=4 and m['n_bridging_F']>=10 and m['mean_Ag_F_Ag_angle_deg']>=168 and m['p10_Ag_F_Ag_angle_deg']>=160)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--M',required=True);z=ap.parse_args();M=z.M
    if M not in ALLOWED: raise ValueError(M)
    out=Path(f'artifacts/spacer/{M}');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[];sel={}
    for atm in [0,400]:
        rr=[relax(M,*st,atm,model,out) for st in STARTS];rows+=rr;ok=[r for r in rr if r['gross_pass']]
        if ok: sel[str(atm)]=min(ok,key=lambda r:r['enthalpy_proxy_eV_atom'])
    advance=bool(len(sel)==2 and all(r['plane_pass'] for r in sel.values()))
    result={'candidate':f'{M}6Ag6F24','formal_Ag_valence':2.0,'all_runs':rows,'selected':sel,'advance':advance}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'ADVANCE').write_text('1' if advance else '0');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
