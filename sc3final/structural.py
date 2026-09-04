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

def build():
    a,c=3.75,2.85
    sy=[]; fr=[]
    # Exhaustive 1320-order search winner: Sc spacer sites 1,4,5; O bonds x10,x11.
    sc={1,4,5}; O={2,8}; ai=0; li=0
    for y in range(2):
        for x in range(3):
            sy.append('Sc' if ai in sc else 'Ti'); fr.append([(x+.5)/3,(y+.5)/2,.5]); ai+=1
            sy.append('Cu'); fr.append([x/3,y/2,0])
            sy.append('O' if li in O else 'N'); fr.append([(x+.5)/3,y/2,0]); li+=1
            sy.append('O' if li in O else 'N'); fr.append([x/3,(y+.5)/2,0]); li+=1
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def metrics(at):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    def mn(a,b):
        v=[d[i,j] for i,s in enumerate(sy) for j,t in enumerate(sy)
           if i!=j and ((s==a and t==b) or (s==b and t==a))]
        return float(min(v)) if v else float('nan')
    cun=[]; cuo=[]
    for i,s in enumerate(sy):
        if s!='Cu': continue
        cun.append(sum(1 for j,t in enumerate(sy) if t=='N' and d[i,j]<2.30))
        cuo.append(sum(1 for j,t in enumerate(sy) if t=='O' and d[i,j]<2.30))
    return {'min_Cu_N_A':mn('Cu','N'),'min_Cu_O_A':mn('Cu','O'),'min_N_N_A':mn('N','N'),
            'mean_Cu_N_coord_lt2p30':float(np.mean(cun)),
            'mean_Cu_O_coord_lt2p30':float(np.mean(cuo))}

def relax(p_gpa,model):
    at=AseAtomsAdaptor.get_atoms(build()); v0=float(at.get_volume()); m0=metrics(at)
    at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p_gpa*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(OUT/f'relax_{p_gpa:.8f}.log')).run(fmax=.045,steps=420)
    m=metrics(at); fmax=float(np.linalg.norm(at.get_forces(),axis=1).max())
    stress=np.array(at.get_stress(voigt=True))*160.21766208
    rec={'pressure_GPa_target':p_gpa,'pressure_atm_target':p_gpa/ATM_TO_GPA,
         'formula':at.get_chemical_formula(),'natoms':len(at),'max_force_eV_A':fmax,
         'energy_eV_atom':float(at.get_potential_energy()/len(at)),
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in at.cell.angles()],
         'stress_GPa_voigt':[float(x) for x in stress]}
    rec.update({'initial_'+k:v for k,v in m0.items()}); rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_structure_pass']=bool(fmax<.075 and .75<rec['volume_ratio']<1.25 and m['min_N_N_A']>1.35)
    rec['pairing_geometry_pass_300scale']=bool(m['min_Cu_N_A']<=1.93 and m['mean_Cu_N_coord_lt2p30']>=3.0)
    rec['pairing_geometry_high_margin']=bool(m['min_Cu_N_A']<=1.91)
    cif=OUT/f'relaxed_{p_gpa:.8f}GPa.cif'
    AseAtomsAdaptor.get_structure(at).to(filename=str(cif)); rec['cif']=str(cif)
    return rec

def main():
    model=CHGNet.load()
    rows=[relax(0.0,model),relax(400*ATM_TO_GPA,model)]
    # QE is allowed only if both pressures preserve a converged lattice and at least one
    # pressure retains the pre-registered 300-K geometry threshold.
    decision=bool(all(r['gross_structure_pass'] for r in rows) and any(r['pairing_geometry_pass_300scale'] for r in rows))
    out={'candidate':'Sc3Ti3Cu6N10O2','rows':rows,'advance_to_QE':decision}
    (OUT/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
    (OUT/'ADVANCE').write_text('1' if decision else '0')
if __name__=='__main__': main()
