from __future__ import annotations
import json, math
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
OUT=Path('artifacts/struct'); OUT.mkdir(parents=True,exist_ok=True)
STARTS=[('compact',3.88,6.6,.22),('central',4.02,7.2,.22),('expanded',4.16,7.8,.21)]


def build(kind:str,a:float,c:float,dz:float)->Structure:
    # 3x2 generated layered prototype:
    # flat AgF2 plane (12 shared in-plane F) + A6F12 spacer layer.
    # A6 = LiBe5 (hole-doped) or AlBe5 (electron-doped).
    sy=[]; fr=[]; ai=0
    for y in range(2):
        for x in range(3):
            sy.append('Li' if (kind=='Li' and ai==0) else ('Al' if (kind=='Al' and ai==0) else 'Be'))
            fr.append([(x+.5)/3,(y+.5)/2,.5]); ai+=1
            sy.append('Ag'); fr.append([x/3,y/2,0])
            # two shared in-plane fluorines per Ag site
            sy.append('F'); fr.append([(x+.5)/3,y/2,0])
            sy.append('F'); fr.append([x/3,(y+.5)/2,0])
            # two spacer fluorines around the A-site
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5+dz])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)


def micvec(at,i,j):
    return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)


def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag_indices=[i for i,s in enumerate(sy) if s=='Ag']
    f_indices=[i for i,s in enumerate(sy) if s=='F']
    nearest4=[]; coords=[]; bridge_angles=[]; bridge_count=0
    for i in ag_indices:
        vals=sorted(float(d[i,j]) for j in f_indices)
        if len(vals)>=4: nearest4.append(float(np.mean(vals[:4])))
        coords.append(sum(x<2.55 for x in vals))
    # A bridging F is defined by having at least two Ag neighbors inside 2.55 A.
    for f in f_indices:
        ags=sorted([(float(d[f,i]),i) for i in ag_indices])
        close=[x for x in ags if x[0]<2.55]
        if len(close)>=2:
            i,j=close[0][1],close[1][1]
            v1=micvec(at,f,i); v2=micvec(at,f,j)
            den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                cs=float(np.clip(np.dot(v1,v2)/den,-1,1))
                bridge_angles.append(float(np.degrees(np.arccos(cs)))); bridge_count+=1
    ff=[float(d[i,j]) for i in f_indices for j in f_indices if i<j]
    return {
        'mean_Ag_F_nearest4_A':float(np.mean(nearest4)) if nearest4 else float('nan'),
        'max_Ag_F_nearest4_A':float(np.max(nearest4)) if nearest4 else float('nan'),
        'min_Ag_F_coord_lt2p55':int(min(coords)) if coords else -1,
        'mean_Ag_F_coord_lt2p55':float(np.mean(coords)) if coords else float('nan'),
        'n_bridging_F':int(bridge_count),
        'mean_Ag_F_Ag_angle_deg':float(np.mean(bridge_angles)) if bridge_angles else float('nan'),
        'p10_Ag_F_Ag_angle_deg':float(np.quantile(bridge_angles,.10)) if bridge_angles else float('nan'),
        'min_F_F_A':float(min(ff)) if ff else float('nan'),
    }


def relax(kind,label,a,c,dz,p_gpa,model):
    s=build(kind,a,c,dz); at=AseAtomsAdaptor.get_atoms(s); v0=float(at.get_volume()); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
    tag=f'{kind}_{label}_{p_gpa:.8f}GPa'
    FIRE(flt,logfile=str(OUT/f'{tag}.log')).run(fmax=.050,steps=500)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max())
    stress=np.asarray(at.get_stress(voigt=True))*160.21766208
    m=metrics(at); epa=float(at.get_potential_energy()/len(at)); vpa=float(at.get_volume()/len(at))
    rec={
        'kind':kind,'tag':tag,'pressure_GPa_target':p_gpa,'pressure_atm_target':p_gpa/ATM_TO_GPA,
        'formula':at.get_chemical_formula(),'natoms':len(at),'max_force_eV_A':fmax,
        'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':epa+p_gpa*vpa*GPA_TO_EV_A3,
        'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
        'cell_angles_deg':[float(x) for x in at.cell.angles()],
        'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3])),
        'stress_GPa_voigt':[float(x) for x in stress],
    }
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(fmax<.085 and .60<rec['volume_ratio']<1.45 and m['min_F_F_A']>1.15)
    rec['agf2_plane_survival']=bool(
        np.isfinite(m['mean_Ag_F_nearest4_A']) and m['mean_Ag_F_nearest4_A']<2.25 and
        m['max_Ag_F_nearest4_A']<2.40 and m['min_Ag_F_coord_lt2p55']>=4 and
        m['n_bridging_F']>=10 and np.isfinite(m['mean_Ag_F_Ag_angle_deg']) and
        m['mean_Ag_F_Ag_angle_deg']>=155.0 and m['p10_Ag_F_Ag_angle_deg']>=145.0
    )
    cif=OUT/f'{tag}.cif'; AseAtomsAdaptor.get_structure(at).to(filename=str(cif)); rec['cif']=str(cif)
    return rec


def main():
    model=CHGNet.load(); allrows=[]; selected={}
    for kind in ['Li','Al']:
        selected[kind]={}
        for p in [0.0,400*ATM_TO_GPA]:
            rows=[]
            for label,a,c,dz in STARTS:
                r=relax(kind,label,a,c,dz,p,model); rows.append(r); allrows.append(r)
            viable=[r for r in rows if r['gross_structure_pass']]
            if viable:
                best=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom'])
                selected[kind][f'{p:.8f}']=best
                Structure.from_file(best['cif']).to(filename=str(OUT/f'relaxed_{kind}_{p:.8f}GPa.cif'))
    advance={}
    for kind in ['Li','Al']:
        rows=list(selected[kind].values())
        advance[kind]=bool(len(rows)==2 and all(r['gross_structure_pass'] and r['agf2_plane_survival'] for r in rows))
    out={
        'generated_candidates':{
            'Li':'LiBe5Ag6F24','Al':'AlBe5Ag6F24'
        },
        'formal_Ag_valence':{'Li':13/6,'Al':11/6},
        'all_runs':allrows,'selected_by_candidate_pressure':selected,'advance_to_QE':advance
    }
    (OUT/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
    for kind in ['Li','Al']:
        (OUT/f'ADVANCE_{kind}').write_text('1' if advance[kind] else '0')

if __name__=='__main__': main()
