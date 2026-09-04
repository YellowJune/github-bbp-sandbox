from __future__ import annotations
import json
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
STARTS=[('compact',3.60,2.65),('central',3.75,2.90),('expanded',3.90,3.15)]


def build_candidate(a:float,c:float)->Structure:
    sy=[]; fr=[]; ai=0
    # 3x2 infinite-layer topology. One Sc3+ among five Ti4+ spacers fixes
    # average Cu valence +2.1667; all 12 in-plane bridges are N.
    for y in range(2):
        for x in range(3):
            sy.append('Sc' if ai==0 else 'Ti'); fr.append([(x+.5)/3,(y+.5)/2,.5]); ai+=1
            sy.append('Cu'); fr.append([x/3,y/2,0])
            sy.append('N'); fr.append([(x+.5)/3,y/2,0])
            sy.append('N'); fr.append([x/3,(y+.5)/2,0])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)


def build_control(name:str)->Structure:
    if name=='CaCuO2':
        a,c=3.86,3.20
        return Structure(Lattice.tetragonal(a,c),['Ca','Cu','O','O'],
                         [[.5,.5,.5],[0,0,0],[.5,0,0],[0,.5,0]])
    if name=='Cu3N':
        a=3.82
        return Structure(Lattice.cubic(a),['N','Cu','Cu','Cu'],
                         [[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
    raise ValueError(name)


def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    cun4=[]; cun_coord=[]; cuo4=[]
    for i,s in enumerate(sy):
        if s!='Cu': continue
        n=sorted(float(d[i,j]) for j,t in enumerate(sy) if t=='N' and i!=j)
        o=sorted(float(d[i,j]) for j,t in enumerate(sy) if t=='O' and i!=j)
        if len(n)>=4: cun4.append(float(np.mean(n[:4])))
        cun_coord.append(sum(x<2.30 for x in n))
        if len(o)>=4: cuo4.append(float(np.mean(o[:4])))
    def pairmin(a,b):
        vals=[float(d[i,j]) for i,s in enumerate(sy) for j,t in enumerate(sy)
              if i!=j and ((s==a and t==b) or (s==b and t==a))]
        return min(vals) if vals else float('nan')
    return {
        'min_Cu_N_A':pairmin('Cu','N'),
        'min_Cu_O_A':pairmin('Cu','O'),
        'min_N_N_A':pairmin('N','N'),
        'mean_Cu_N_nearest4_A':float(np.mean(cun4)) if cun4 else float('nan'),
        'max_perCu_Cu_N_nearest4_A':float(np.max(cun4)) if cun4 else float('nan'),
        'min_perCu_Cu_N_nearest4_A':float(np.min(cun4)) if cun4 else float('nan'),
        'mean_Cu_N_coord_lt2p30':float(np.mean(cun_coord)) if cun_coord else float('nan'),
        'min_Cu_N_coord_lt2p30':int(min(cun_coord)) if cun_coord else -1,
        'mean_Cu_O_nearest4_A':float(np.mean(cuo4)) if cuo4 else float('nan'),
    }


def relax_structure(s:Structure,p_gpa:float,model,tag:str,steps:int=420):
    at=AseAtomsAdaptor.get_atoms(s); v0=float(at.get_volume()); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(OUT/f'{tag}.log')).run(fmax=.045,steps=steps)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max())
    stress=np.array(at.get_stress(voigt=True))*160.21766208
    m=metrics(at); epa=float(at.get_potential_energy()/len(at))
    hpa=epa+p_gpa*float(at.get_volume()/len(at))*GPA_TO_EV_A3
    rec={
        'tag':tag,'pressure_GPa_target':p_gpa,'pressure_atm_target':p_gpa/ATM_TO_GPA,
        'formula':at.get_chemical_formula(),'natoms':len(at),'max_force_eV_A':fmax,
        'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':hpa,'volume_ratio':float(at.get_volume()/v0),
        'cell_lengths_A':[float(x) for x in at.cell.lengths()],
        'cell_angles_deg':[float(x) for x in at.cell.angles()],
        'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3])),
        'stress_GPa_voigt':[float(x) for x in stress],
    }
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(
        fmax<.075 and .72<rec['volume_ratio']<1.30 and
        (not np.isfinite(m['min_N_N_A']) or m['min_N_N_A']>1.30) and
        abs(rec['hydrostatic_pressure_GPa_from_stress']-p_gpa)<.35
    )
    rec['pairing_geometry_pass_300scale']=bool(
        np.isfinite(m['mean_Cu_N_nearest4_A']) and m['mean_Cu_N_nearest4_A']<=1.93 and
        m['max_perCu_Cu_N_nearest4_A']<=1.98 and m['min_Cu_N_coord_lt2p30']>=3
    )
    rec['pairing_geometry_high_margin']=bool(
        np.isfinite(m['mean_Cu_N_nearest4_A']) and m['mean_Cu_N_nearest4_A']<=1.91 and
        m['min_Cu_N_coord_lt2p30']>=3
    )
    cif=OUT/f'{tag}.cif'; AseAtomsAdaptor.get_structure(at).to(filename=str(cif)); rec['cif']=str(cif)
    return rec


def main():
    model=CHGNet.load()
    controls=[]
    for name in ['CaCuO2','Cu3N']:
        r=relax_structure(build_control(name),0.0,model,f'control_{name}',steps=300)
        controls.append(r)
    control_sane=bool(all(r['gross_structure_pass'] for r in controls))

    allrows=[]; selected={}
    for p in [0.0,400*ATM_TO_GPA]:
        rows=[]
        for label,a,c in STARTS:
            rows.append(relax_structure(build_candidate(a,c),p,model,f'{label}_{p:.8f}GPa'))
        allrows.extend(rows)
        viable=[r for r in rows if r['gross_structure_pass']]
        if viable:
            best=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom'])
            selected[f'{p:.8f}']=best
            Structure.from_file(best['cif']).to(filename=str(OUT/f'relaxed_{p:.8f}GPa.cif'))

    both_pressures=len(selected)==2
    geom_both=both_pressures and all(r['pairing_geometry_pass_300scale'] for r in selected.values())
    high_margin_any=both_pressures and any(r['pairing_geometry_high_margin'] for r in selected.values())
    decision=bool(control_sane and both_pressures and geom_both and high_margin_any)
    out={
        'candidate':'ScTi5Cu6N12','formal_average_Cu_valence':13/6,
        'formal_hole_doping_vs_Cu2':1/6,'controls':controls,'all_candidate_runs':allrows,
        'selected_by_pressure':selected,'control_sane':control_sane,
        'preregistered_gate':{
            'both_pressures_gross_structure_pass':both_pressures,
            'both_pressures_mean_CuN_nearest4_le_1p93A':geom_both,
            'at_least_one_pressure_mean_CuN_nearest4_le_1p91A':high_margin_any,
        },
        'advance_to_QE':decision,
    }
    (OUT/'result.json').write_text(json.dumps(out,indent=2));
    (OUT/'ADVANCE').write_text('1' if decision else '0')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
