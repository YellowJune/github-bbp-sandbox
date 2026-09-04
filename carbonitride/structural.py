from __future__ import annotations
import argparse,json
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
STARTS=[('s1',3.40,2.45),('s2',3.55,2.65),('s3',3.70,2.85),('s4',3.85,3.05)]

CAND={
 # TiNb5 ordering updated after exhaustive 5544-state pairing + screened-Coulomb search:
 # Ti spacer index 1 and C bonds x20,x01,y01,y11,x21,y21.
 'TiNb5Cu6C6N6':{'A':['Nb','Ti','Nb','Nb','Nb','Nb'],'C':{'x20','x01','y01','y11','x21','y21'}},
 'Nb6Cu6C7N5':{'A':['Nb']*6,'C':{'x00','x10','x01','y01','x11','y11','y21'}},
 'Ti3Nb3Cu6C4N8':{'A':['Ti','Ti','Ti','Nb','Nb','Nb'],'C':{'x00','y00','y10','x01'}},
}

def build(name,a,c):
    cfg=CAND[name];sy=[];fr=[];ai=0
    for y in range(2):
        for x in range(3):
            sy.append(cfg['A'][ai]);fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Cu');fr.append([x/3,y/2,0])
            for ori,pos in [('x',[(x+.5)/3,y/2,0]),('y',[x/3,(y+.5)/2,0])]:
                lab=f'{ori}{x}{y}';sy.append('C' if lab in cfg['C'] else 'N');fr.append(pos)
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def metrics(at):
    sy=at.get_chemical_symbols();d=at.get_all_distances(mic=True)
    cu4=[];cuc=[];cun=[];coord=[]
    for i,s in enumerate(sy):
        if s!='Cu':continue
        lig=sorted((float(d[i,j]),t) for j,t in enumerate(sy) if t in {'C','N'} and i!=j)
        first=lig[:4];cu4.append(float(np.mean([x for x,_ in first])))
        cs=[x for x,t in first if t=='C'];ns=[x for x,t in first if t=='N']
        if cs:cuc.extend(cs)
        if ns:cun.extend(ns)
        coord.append(sum(x<2.35 for x,_ in lig))
    ll=[float(d[i,j]) for i,s in enumerate(sy) for j,t in enumerate(sy) if i<j and s in {'C','N'} and t in {'C','N'}]
    return {'mean_Cu_ligand4_A':float(np.mean(cu4)),'max_Cu_ligand4_A':float(np.max(cu4)),
            'mean_CuC_firstshell_A':float(np.mean(cuc)) if cuc else float('nan'),
            'mean_CuN_firstshell_A':float(np.mean(cun)) if cun else float('nan'),
            'min_Cu_ligand_coord_lt2p35':int(min(coord)),'mean_Cu_ligand_coord_lt2p35':float(np.mean(coord)),
            'min_ligand_ligand_A':float(min(ll)) if ll else float('nan')}

def relax(name,label,a,c,p,model,out):
    at=AseAtomsAdaptor.get_atoms(build(name,a,c));v0=float(at.get_volume());at.calc=CHGNetCalculator(model=model)
    flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    FIRE(flt,logfile=str(out/f'{label}_{p:.8f}.log')).run(fmax=.045,steps=480)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.array(at.get_stress(voigt=True))*160.21766208;m=metrics(at)
    epa=float(at.get_potential_energy()/len(at));hpa=epa+p*(at.get_volume()/len(at))*GPA_TO_EV_A3
    rec={'name':name,'start':label,'pressure_GPa':p,'pressure_atm':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),
         'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':float(hpa),'volume_ratio':float(at.get_volume()/v0),
         'cell_lengths_A':[float(x) for x in at.cell.lengths()],'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3]))}
    rec.update(m)
    rec['gross_pass']=bool(fmax<.075 and .68<rec['volume_ratio']<1.35 and m['min_ligand_ligand_A']>1.20 and m['min_Cu_ligand_coord_lt2p35']>=3 and abs(rec['hydrostatic_pressure_GPa_from_stress']-p)<.45)
    rec['compact_plane_pass']=bool(m['mean_Cu_ligand4_A']<=1.98 and m['max_Cu_ligand4_A']<=2.08)
    cif=out/f'{label}_{p:.8f}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=sorted(CAND),required=True);a=ap.parse_args();out=Path(f'artifacts/{a.name}');out.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[];sel={}
    for p in [0.0,400*ATM_TO_GPA]:
        rs=[relax(a.name,*st,p,model,out) for st in STARTS];rows.extend(rs);vi=[r for r in rs if r['gross_pass']]
        if vi:
            best=min(vi,key=lambda r:r['enthalpy_proxy_eV_atom']);sel[f'{p:.8f}']=best;Structure.from_file(best['cif']).to(filename=str(out/f'relaxed_{p:.8f}GPa.cif'))
    both=len(sel)==2;compact=both and all(r['compact_plane_pass'] for r in sel.values())
    result={'candidate':a.name,'formal_average_Cu_valence':13/6,'formal_hole_doping':1/6,'runs':rows,'selected':sel,'both_pressures_survive':both,'compact_plane_both':compact,'advance_electronic':bool(both and compact)}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
