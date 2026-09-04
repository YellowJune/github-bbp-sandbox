from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
ATM_TO_GPA=.000101325; GPA_TO_EV_A3=1/160.21766208

def build():
    a,c=3.75,2.85;sy=[];fr=[]
    # 3x2 plane. Sc spacer indices 1,4,5; O ligand bonds x10,x11 from exhaustive 1320-order search.
    sc={1,4,5};O={2,8};ai=0;li=0
    for y in range(2):
        for x in range(3):
            sy.append('Sc' if ai in sc else 'Ti');fr.append([((x+.5)/3),((y+.5)/2),.5]);ai+=1
            sy.append('Cu');fr.append([x/3,y/2,0])
            sy.append('O' if li in O else 'N');fr.append([((x+.5)/3),y/2,0]);li+=1
            sy.append('O' if li in O else 'N');fr.append([x/3,((y+.5)/2),0]);li+=1
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    def mn(a,b):
        v=[d[i,j] for i,s in enumerate(sy) for j,t in enumerate(sy) if i!=j and ((s==a and t==b) or (s==b and t==a))]
        return float(min(v)) if v else float('nan')
    cun=[];cuo=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        cun.append(sum(t=='N' and d[i,j]<2.30 for j,t in enumerate(sy)))
        cuo.append(sum(t=='O' and d[i,j]<2.30 for j,t in enumerate(sy)))
    return {'min_Cu_N_A':mn('Cu','N'),'min_Cu_O_A':mn('Cu','O'),'min_N_N_A':mn('N','N'),
            'mean_Cu_N_coord_lt2p30':float(np.mean(cun)),'mean_Cu_O_coord_lt2p30':float(np.mean(cuo))}

def one(p,out,model):
    at=AseAtomsAdaptor.get_atoms(build());v0=at.get_volume();m0=metrics(at);at.calc=CHGNetCalculator(model=model)
    fil=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3);FIRE(fil,logfile=str(out/f'relax_{p:.5f}.log')).run(fmax=.05,steps=340)
    m=metrics(at);fmax=float(np.linalg.norm(at.get_forces(),axis=1).max())
    r={'pressure_GPa':p,'pressure_atm':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),'natoms':len(at),'max_force_eV_A':fmax,
       'energy_eV_atom':float(at.get_potential_energy()/len(at)),'volume_ratio':float(at.get_volume()/v0),
       'cell_lengths_A':[float(x) for x in at.cell.lengths()],'cell_angles_deg':[float(x) for x in at.cell.angles()],
       'stress_GPa_voigt':[float(x*160.21766208) for x in at.get_stress(voigt=True)]}
    r.update({'initial_'+k:v for k,v in m0.items()});r.update({'final_'+k:v for k,v in m.items()})
    r['gross_structure_pass']=bool(fmax<.08 and .75<r['volume_ratio']<1.25 and m['min_N_N_A']>1.35)
    r['pairing_geometry_pass_300scale']=bool(m['min_Cu_N_A']<=1.93 and m['mean_Cu_N_coord_lt2p30']>=3.0)
    r['pairing_geometry_high_margin']=bool(m['min_Cu_N_A']<=1.91)
    AseAtomsAdaptor.get_structure(at).to(filename=str(out/f'relaxed_{p:.5f}GPa.cif'))
    return r

def main():
    out=Path('artifacts/sc3ti3cu6n10o2');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load()
    rows=[one(0,out,model),one(400*ATM_TO_GPA,out,model)]
    (out/'result.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
