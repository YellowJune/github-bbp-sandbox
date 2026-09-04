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
STARTS=[('s1',3.30,2.40),('s2',3.45,2.60),('s3',3.60,2.80),('s4',3.75,3.00)]

CAND={
 'Nb3W3Cu6C10N2':{
   'A':['W','W','Nb','W','Nb','Nb'],
   'C':{'x00','x01','x10','x11','y00','y01','y10','y11','y20','y21'},
   'reaction':{'NbC':3,'WC':3,'Cu3N':2,'Cgraph':4},
 },
 'NbW5Cu6C12':{
   'A':['W','W','W','W','W','Nb'],
   'C':{'x00','x01','x10','x11','x20','x21','y00','y01','y10','y11','y20','y21'},
   'reaction':{'NbC':1,'WC':5,'Cu_fcc':6,'Cgraph':6},
 },
}

def build_candidate(name,a,c):
    cfg=CAND[name];sy=[];fr=[];ai=0
    for y in range(2):
        for x in range(3):
            sy.append(cfg['A'][ai]);fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Cu');fr.append([x/3,y/2,0])
            for ori,pos in [('x',[(x+.5)/3,y/2,0]),('y',[x/3,(y+.5)/2,0])]:
                lab=f'{ori}{x}{y}';sy.append('C' if lab in cfg['C'] else 'N');fr.append(pos)
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def build_product(name):
    if name=='NbC':
        a=4.47
        m=[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]]
        x=[[.5,.5,.5],[.5,0,0],[0,.5,0],[0,0,.5]]
        return Structure(Lattice.cubic(a),['Nb']*4+['C']*4,m+x)
    if name=='WC':
        return Structure(Lattice.hexagonal(2.91,2.84),['W','C'],[[0,0,0],[1/3,2/3,.5]])
    if name=='Cu3N':
        return Structure(Lattice.cubic(3.82),['N','Cu','Cu','Cu'],[[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
    if name=='Cu_fcc':
        a=3.61
        return Structure(Lattice.cubic(a),['Cu']*4,[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    if name=='Cgraph':
        return Structure(Lattice.hexagonal(2.46,6.71),['C']*4,[[0,0,0],[1/3,2/3,0],[0,0,.5],[2/3,1/3,.5]])
    raise ValueError(name)

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    cu4=[];cuc=[];cun=[];coord=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        lig=sorted((float(d[i,j]),t) for j,t in enumerate(sy) if t in {'C','N'} and i!=j)
        first=lig[:4]
        if len(first)==4:cu4.append(float(np.mean([x for x,_ in first])))
        c=[x for x,t in first if t=='C']; n=[x for x,t in first if t=='N']
        cuc.extend(c);cun.extend(n);coord.append(sum(x<2.35 for x,_ in lig))
    ll=[float(d[i,j]) for i,s in enumerate(sy) for j,t in enumerate(sy) if i<j and s in {'C','N'} and t in {'C','N'}]
    return {
      'mean_Cu_ligand4_A':float(np.mean(cu4)) if cu4 else float('nan'),
      'max_Cu_ligand4_A':float(np.max(cu4)) if cu4 else float('nan'),
      'mean_CuC_firstshell_A':float(np.mean(cuc)) if cuc else float('nan'),
      'mean_CuN_firstshell_A':float(np.mean(cun)) if cun else float('nan'),
      'min_Cu_ligand_coord_lt2p35':int(min(coord)) if coord else -1,
      'mean_Cu_ligand_coord_lt2p35':float(np.mean(coord)) if coord else float('nan'),
      'min_ligand_ligand_A':float(min(ll)) if ll else float('nan'),
    }

def relax_structure(s,p,model,log,steps=420):
    at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(log)).run(fmax=.045,steps=steps)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.array(at.get_stress(voigt=True))*160.21766208
    epa=float(at.get_potential_energy()/len(at));hpa=epa+p*(at.get_volume()/len(at))*GPA_TO_EV_A3
    return at,{'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':float(hpa),
               'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
               'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3]))}

def product_formula_energy(at,name,p):
    epa=float(at.get_potential_energy()/len(at)); vpa=float(at.get_volume()/len(at)); hpa=epa+p*vpa*GPA_TO_EV_A3
    # energy/enthalpy per chemical formula unit used in reactions
    atoms_per_formula={'NbC':2,'WC':2,'Cu3N':4,'Cu_fcc':1,'Cgraph':1}[name]
    return hpa*atoms_per_formula

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=sorted(CAND),required=True);a=ap.parse_args()
    out=Path(f'artifacts/{a.name}');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();result={'candidate':a.name,'pressures':{}}
    for p in [0.0,400*ATM_TO_GPA]:
        # relax reaction products once per pressure
        prod={}
        for pname in CAND[a.name]['reaction']:
            at,rec=relax_structure(build_product(pname),p,model,out/f'prod_{pname}_{p:.8f}.log',steps=300)
            prod[pname]={'record':rec,'H_formula_eV':product_formula_energy(at,pname,p)}
        rows=[]
        for label,aa,cc in STARTS:
            at,rec=relax_structure(build_candidate(a.name,aa,cc),p,model,out/f'{label}_{p:.8f}.log')
            rec.update({'start':label,'pressure_GPa':p,'pressure_atm':p/ATM_TO_GPA,'formula':at.get_chemical_formula()})
            rec.update(metrics(at))
            rec['gross_pass']=bool(rec['max_force_eV_A']<.075 and .60<rec['volume_ratio']<1.45 and
                                   rec['min_ligand_ligand_A']>1.15 and rec['min_Cu_ligand_coord_lt2p35']>=3 and
                                   abs(rec['hydrostatic_pressure_GPa_from_stress']-p)<.50)
            rec['plane_pass']=bool(rec['mean_Cu_ligand4_A']<=1.98 and rec['max_Cu_ligand4_A']<=2.08)
            cif=out/f'{label}_{p:.8f}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
            rows.append(rec)
        viable=[r for r in rows if r['gross_pass']]
        selected=min(viable,key=lambda r:r['enthalpy_proxy_eV_atom']) if viable else None
        delta=None
        if selected is not None:
            Hcand=selected['enthalpy_proxy_eV_atom']*24
            Hprod=sum(n*prod[pn]['H_formula_eV'] for pn,n in CAND[a.name]['reaction'].items())
            delta=(Hcand-Hprod)/24
            selected['decomposition_deltaH_eV_atom']=float(delta)
            Structure.from_file(selected['cif']).to(filename=str(out/f'relaxed_{p:.8f}GPa.cif'))
        result['pressures'][f'{p:.8f}']={'runs':rows,'selected':selected,'products':prod,'decomposition_deltaH_eV_atom':delta}
    sels=[v['selected'] for v in result['pressures'].values()]
    both=all(x is not None for x in sels)
    plane=both and all(x['plane_pass'] for x in sels)
    # CHGNet decomposition is a screening proxy; <=0.15 eV/atom survives to DFT hull.
    decomp=both and all(x.get('decomposition_deltaH_eV_atom',9)<=.15 for x in sels)
    result['both_pressures_survive']=bool(both);result['plane_both']=bool(plane);result['decomposition_screen_both']=bool(decomp)
    result['advance_to_QE']=bool(both and plane and decomp)
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'ADVANCE').write_text('1' if result['advance_to_QE'] else '0');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
