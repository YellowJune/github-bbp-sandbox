from __future__ import annotations
import glob,json
from pathlib import Path
import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208
ROOT=Path('/tmp/agf2parent/data')
OUT=Path('artifacts/miii');OUT.mkdir(parents=True,exist_ok=True)
TARGET=Composition({'Rb':1,'Cs':1,'K':1,'Mg':2,'Ag':1,'F':9})


def same_reduced(c1,c2):
    r1,_=c1.get_reduced_composition_and_factor();r2,_=c2.get_reduced_composition_and_factor()
    els=set(r1.as_dict())|set(r2.as_dict())
    return all(abs(r1[e]-r2[e])<1e-8 for e in els)


def micvec(at,i,j): return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)


def geom(s:Structure):
    at=AseAtomsAdaptor.get_atoms(s);sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    ag=[i for i,x in enumerate(sy) if x=='Ag'];fs=[i for i,x in enumerate(sy) if x=='F']
    agrows=[];angles=[]
    for i in ag:
        vf=sorted(float(d[i,j]) for j in fs)
        nearest4=float(np.mean(vf[:4])) if len(vf)>=4 else float('nan')
        local=[]
        for f in fs:
            aa=sorted((float(d[f,j]),j) for j in ag)
            if len(aa)>=2 and aa[0][0]<2.35 and aa[1][0]<2.35 and (aa[0][1]==i or aa[1][1]==i):
                j=aa[1][1] if aa[0][1]==i else aa[0][1]
                v1=micvec(at,f,i);v2=micvec(at,f,j);den=np.linalg.norm(v1)*np.linalg.norm(v2)
                if den>1e-12:
                    local.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))))
        agrows.append({'Ag_index':i,'nearest4_AgF_A':nearest4,'mean_bridge_angle_deg':float(np.mean(local)) if local else float('nan'),'n_bridge_angles':len(local)})
        angles+=local
    valid=[r for r in agrows if np.isfinite(r['mean_bridge_angle_deg'])]
    flat=max(valid,key=lambda r:r['mean_bridge_angle_deg']) if valid else {'Ag_index':-1,'nearest4_AgF_A':float('nan'),'mean_bridge_angle_deg':float('nan'),'n_bridge_angles':0}
    return {'Ag_sites':agrows,'flattest_Ag':flat,'overall_mean_bridge_angle_deg':float(np.mean(angles)) if angles else float('nan')}


def find_parent():
    hits=[]
    for f in glob.glob(str(ROOT/'**'/'*.vasp'),recursive=True):
        try:s=Structure.from_file(f)
        except Exception:continue
        if same_reduced(s.composition,TARGET):
            g=geom(s);hits.append((g['flattest_Ag']['mean_bridge_angle_deg'],f,s,g))
    if not hits: raise RuntimeError('No reduced-composition RbCsKMg2AgF9 VASP found')
    hits.sort(key=lambda x:(-999 if not np.isfinite(x[0]) else x[0]),reverse=True)
    ang,f,s,g=hits[0]
    (OUT/'parent_selection.json').write_text(json.dumps({'matches':[{'file':x[1],'flat_angle':x[0],'formula':x[2].composition.reduced_formula} for x in hits],'selected':f,'selected_geometry':g},indent=2))
    return s


def transform(parent,variant):
    s=parent.copy()
    if variant=='parent':return s
    mg=[i for i,x in enumerate(s) if x.specie.symbol=='Mg']
    if len(mg)%2:raise RuntimeError('Mg count must be even')
    mg=sorted(mg,key=lambda i:tuple(np.round(s[i].frac_coords[[2,1,0]],8)))
    for k,i in enumerate(mg):s[i]='Li' if k%2==0 else 'Al'
    return s


def relax(s,variant,p_atm,model):
    p=p_atm*ATM_TO_GPA;at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());g0=geom(AseAtomsAdaptor.get_structure(at))
    at.calc=CHGNetCalculator(model=model);flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    tag=f'{variant}_{int(p_atm)}atm';FIRE(flt,logfile=str(OUT/f'{tag}.log')).run(fmax=.045,steps=500)
    st=AseAtomsAdaptor.get_structure(at);g=geom(st);forces=np.linalg.norm(at.get_forces(),axis=1);stress=np.asarray(at.get_stress(voigt=True))*160.21766208
    rec={'variant':variant,'pressure_atm':p_atm,'pressure_GPa':p,'formula':st.composition.reduced_formula,'natoms':len(st),'max_force_eV_A':float(forces.max()),'energy_eV_atom':float(at.get_potential_energy()/len(at)),'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],'cell_angles_deg':[float(x) for x in at.cell.angles()],'hydrostatic_GPa_from_stress':float(-np.mean(stress[:3])),'initial_geometry':g0,'final_geometry':g}
    flat=g['flattest_Ag'];rec['gross_pass']=bool(rec['max_force_eV_A']<.08 and .72<rec['volume_ratio']<1.30)
    rec['flat_short_target']=bool(rec['gross_pass'] and np.isfinite(flat['mean_bridge_angle_deg']) and flat['mean_bridge_angle_deg']>=172 and flat['nearest4_AgF_A']<=2.03)
    cif=OUT/f'{tag}.cif';st.to(filename=str(cif));rec['cif']=str(cif)
    return rec


def main():
    parent=find_parent();model=CHGNet.load();rows=[]
    for v in ['parent','LiAl']:
        for p in [0,400]:rows.append(relax(transform(parent,v),v,p,model))
    out={'generated_candidate':'RbCsKLiAlAgF9','rows':rows,'LiAl_both_pressures_flat_short':all(r['flat_short_target'] for r in rows if r['variant']=='LiAl')}
    (OUT/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
