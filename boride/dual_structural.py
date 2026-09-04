from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208
STARTS=[('s1',3.20,2.30),('s2',3.35,2.50),('s3',3.50,2.70),('s4',3.65,2.90),('s5',3.80,3.10)]


def candidate(a,c):
    sy=[];fr=[];ai=0
    for y in range(2):
        for x in range(3):
            sy.append('Sc' if ai==0 else 'Ti');fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Cu');fr.append([x/3,y/2,0])
            sy.append('B');fr.append([(x+.5)/3,y/2,0])
            sy.append('B');fr.append([x/3,(y+.5)/2,0])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)


def diboride(metal,a,c):
    return Structure(Lattice.hexagonal(a,c),[metal,'B','B'],[[0,0,0],[1/3,2/3,.5],[2/3,1/3,.5]])

def product(name):
    if name=='ScB2':return diboride('Sc',3.1482,3.5148)
    if name=='TiB2':return diboride('Ti',3.030,3.230)
    if name=='Cu':
        a=3.61
        return Structure(Lattice.cubic(a),['Cu']*4,[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    raise ValueError(name)


def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    cub4=[];coord=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        bs=sorted(float(d[i,j]) for j,t in enumerate(sy) if t=='B' and i!=j)
        if len(bs)>=4:cub4.append(float(np.mean(bs[:4])))
        coord.append(sum(x<2.35 for x in bs))
    bb=[float(d[i,j]) for i,s in enumerate(sy) for j,t in enumerate(sy) if i<j and s=='B' and t=='B']
    return {
        'mean_CuB_nearest4_A':float(np.mean(cub4)) if cub4 else float('nan'),
        'max_CuB_nearest4_A':float(np.max(cub4)) if cub4 else float('nan'),
        'min_CuB_coord_lt2p35':int(min(coord)) if coord else -1,
        'min_BB_A':float(min(bb)) if bb else float('nan'),
    }


def load_calc(engine):
    if engine=='chgnet':
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(model=CHGNet.load())
    if engine=='mace':
        from mace.calculators import mace_mp
        return mace_mp(model='small',dispersion=False,default_dtype='float32',device='cpu')
    raise ValueError(engine)


def relax(s,p,calc,log,steps=420):
    at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());at.calc=calc
    flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(log)).run(fmax=.045,steps=steps)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.array(at.get_stress(voigt=True))*160.21766208
    epa=float(at.get_potential_energy()/len(at));hpa=epa+p*(at.get_volume()/len(at))*GPA_TO_EV_A3
    rec={'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':float(hpa),
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in at.cell.angles()],
         'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3]))}
    return at,rec


def formula_H(at,p,atoms_per_formula):
    epa=float(at.get_potential_energy()/len(at));hpa=epa+p*(at.get_volume()/len(at))*GPA_TO_EV_A3
    return hpa*atoms_per_formula


def run_engine(engine,out):
    calc=load_calc(engine); result={'engine':engine,'controls':{},'pressures':{}}
    # control validation at zero pressure
    for name,refa in [('ScB2',3.1482),('TiB2',3.030),('Cu',3.61)]:
        at,rec=relax(product(name),0.0,calc,out/f'{engine}_control_{name}.log',steps=260)
        rec['relaxed_a_A']=float(at.cell.lengths()[0]);rec['reference_a_A']=refa
        rec['relative_a_error']=abs(rec['relaxed_a_A']-refa)/refa
        result['controls'][name]=rec
    result['control_sane']=bool(all(v['max_force_eV_A']<.08 and v['relative_a_error']<.10 for v in result['controls'].values()))

    for p in [0.0,400*ATM_TO_GPA]:
        # exact reaction products ScB2 + 5 TiB2 + 6 Cu
        products={}
        for name,mult,apf in [('ScB2',1,3),('TiB2',5,3),('Cu',6,1)]:
            at,rec=relax(product(name),p,calc,out/f'{engine}_prod_{name}_{p:.8f}.log',steps=260)
            products[name]={'mult':mult,'record':rec,'H_formula_eV':formula_H(at,p,apf)}
        rows=[]
        for lab,a,c in STARTS:
            at,rec=relax(candidate(a,c),p,calc,out/f'{engine}_{lab}_{p:.8f}.log')
            rec.update({'start':lab,'pressure_GPa':p,'pressure_atm':p/ATM_TO_GPA,'formula':at.get_chemical_formula()})
            rec.update(metrics(at))
            rec['gross_pass']=bool(rec['max_force_eV_A']<.075 and .58<rec['volume_ratio']<1.50 and
                                   rec['min_BB_A']>1.15 and rec['min_CuB_coord_lt2p35']>=3 and
                                   abs(rec['hydrostatic_pressure_GPa_from_stress']-p)<.55)
            rec['geometry_300scale']=bool(rec['mean_CuB_nearest4_A']<=1.90 and rec['max_CuB_nearest4_A']<=2.00)
            rec['geometry_high_margin']=bool(rec['mean_CuB_nearest4_A']<=1.85 and rec['max_CuB_nearest4_A']<=1.95)
            cif=out/f'{engine}_{lab}_{p:.8f}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
            rows.append(rec)
        viable=[r for r in rows if r['gross_pass']]
        selected=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom']) if viable else None
        dh=None
        if selected is not None:
            Hcand=selected['enthalpy_proxy_eV_atom']*24
            Hprod=sum(v['mult']*v['H_formula_eV'] for v in products.values())
            dh=(Hcand-Hprod)/24;selected['decomposition_deltaH_eV_atom']=float(dh)
            Structure.from_file(selected['cif']).to(filename=str(out/f'{engine}_relaxed_{p:.8f}GPa.cif'))
        result['pressures'][f'{p:.8f}']={'runs':rows,'selected':selected,'products':products,'decomposition_deltaH_eV_atom':dh}
    sels=[v['selected'] for v in result['pressures'].values()];both=all(x is not None for x in sels)
    result['both_pressures_survive']=bool(both)
    result['geometry_300scale_both']=bool(both and all(x['geometry_300scale'] for x in sels))
    result['high_margin_any']=bool(both and any(x['geometry_high_margin'] for x in sels))
    result['decomposition_screen_both']=bool(both and all(x.get('decomposition_deltaH_eV_atom',9)<=.15 for x in sels))
    result['engine_pass']=bool(result['control_sane'] and result['both_pressures_survive'] and result['geometry_300scale_both'] and result['decomposition_screen_both'])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--engine',choices=['chgnet','mace'],required=True);a=ap.parse_args()
    out=Path(f'artifacts/{a.engine}');out.mkdir(parents=True,exist_ok=True)
    result=run_engine(a.engine,out);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
