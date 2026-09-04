from __future__ import annotations
import argparse,json,glob
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208


def load_parent(root:str)->Structure:
    fs=glob.glob(str(Path(root)/'**'/'Cs2K.vasp'),recursive=True)
    if not fs: raise FileNotFoundError('Cs2K.vasp not found in downloaded dataset')
    return Structure.from_file(fs[0])


def transform(s:Structure,variant:str)->Structure:
    s=s.copy()
    mg=[i for i,x in enumerate(s) if x.specie.symbol=='Mg']
    if variant=='parent': return s
    if not mg: raise RuntimeError('parent contains no Mg sites')
    if variant=='Be2':
        for i in mg: s[i]='Be'
        return s
    pair={'LiAl':('Li','Al'),'LiGa':('Li','Ga'),'LiSc':('Li','Sc')}[variant]
    if len(mg)%2: raise RuntimeError(f'{variant} requires even Mg count, got {len(mg)}')
    # Spatially deterministic split, preserving exact average +2 charge.
    mg=sorted(mg,key=lambda i:tuple(np.round(s[i].frac_coords[[2,1,0]],8)))
    for k,i in enumerate(mg): s[i]=pair[k%2]
    return s


def micvec(at,i,j):
    return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)


def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag=[i for i,x in enumerate(sy) if x=='Ag']; ff=[i for i,x in enumerate(sy) if x=='F']
    nearestF=[]; coord=[]; nearestAg=[]; angles=[]; bridges=0
    for i in ag:
        vf=sorted(float(d[i,j]) for j in ff)
        if len(vf)>=4: nearestF.append(np.mean(vf[:4]))
        coord.append(sum(x<2.35 for x in vf))
        va=sorted(float(d[i,j]) for j in ag if j!=i)
        if va: nearestAg.append(np.mean(va[:min(4,len(va))]))
    for f in ff:
        aa=sorted((float(d[f,i]),i) for i in ag)
        close=[x for x in aa if x[0]<2.35]
        if len(close)>=2:
            i,j=close[0][1],close[1][1]
            v1=micvec(at,f,i);v2=micvec(at,f,j);den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))));bridges+=1
    ffd=[float(d[i,j]) for a,i in enumerate(ff) for j in ff[a+1:]]
    return {
        'n_Ag':len(ag),'n_F':len(ff),
        'mean_AgF_nearest4_A':float(np.mean(nearestF)) if nearestF else float('nan'),
        'max_AgF_nearest4_A':float(np.max(nearestF)) if nearestF else float('nan'),
        'min_AgF_coord_lt2p35':int(min(coord)) if coord else -1,
        'mean_AgAg_nearest_A':float(np.mean(nearestAg)) if nearestAg else float('nan'),
        'n_AgFAg_bridges':int(bridges),
        'mean_AgFAg_angle_deg':float(np.mean(angles)) if angles else float('nan'),
        'p10_AgFAg_angle_deg':float(np.quantile(angles,.10)) if angles else float('nan'),
        'min_FF_A':float(min(ffd)) if ffd else float('nan')
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--variant',choices=['parent','Be2','LiAl','LiGa','LiSc'],required=True);ap.add_argument('--pressure-atm',type=float,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    p_gpa=a.pressure_atm*ATM_TO_GPA; out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    s=transform(load_parent(a.root),a.variant); initial_formula=s.composition.reduced_formula
    # Variant cells get a second, slightly compressed start; parent uses the raw DFT cell only.
    starts=[('raw',1.0)] if a.variant=='parent' else [('raw',1.0),('minus2pct',.98)]
    model=CHGNet.load(); rows=[]
    for label,scale in starts:
        ss=s.copy(); ss.scale_lattice(ss.volume*scale**3)
        at=AseAtomsAdaptor.get_atoms(ss);v0=float(at.get_volume());m0=metrics(at)
        at.calc=CHGNetCalculator(model=model);flt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
        FIRE(flt,logfile=str(out/f'{label}.log')).run(fmax=.045,steps=500)
        m=metrics(at);fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.asarray(at.get_stress(voigt=True))*160.21766208
        epa=float(at.get_potential_energy()/len(at));vpa=float(at.get_volume()/len(at))
        rec={'variant':a.variant,'start':label,'pressure_atm':a.pressure_atm,'pressure_GPa':p_gpa,'initial_formula':initial_formula,'relaxed_formula':at.get_chemical_formula(),'natoms':len(at),'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':epa+p_gpa*vpa*GPA_TO_EV_A3,'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],'cell_angles_deg':[float(x) for x in at.cell.angles()],'hydrostatic_GPa_from_stress':float(-np.mean(stress[:3]))}
        rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in m.items()})
        rec['gross_pass']=bool(fmax<.080 and .72<rec['volume_ratio']<1.30 and m['min_FF_A']>1.20)
        rec['flat_short_AgF2_target']=bool(rec['gross_pass'] and m['min_AgF_coord_lt2p35']>=4 and m['mean_AgAg_nearest_A']<=4.05 and m['mean_AgAg_nearest_A']>=3.90 and m['mean_AgFAg_angle_deg']>=172 and m['p10_AgFAg_angle_deg']>=165)
        cif=out/f'{label}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif);rows.append(rec)
    viable=[r for r in rows if r['gross_pass']]
    best=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom']) if viable else min(rows,key=lambda r:r['energy_eV_atom'])
    Path(out/'result.json').write_text(json.dumps({'variant':a.variant,'pressure_atm':a.pressure_atm,'rows':rows,'selected':best},indent=2));print(json.dumps({'variant':a.variant,'pressure_atm':a.pressure_atm,'rows':rows,'selected':best},indent=2))

if __name__=='__main__':main()
