from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208

PAIR_A0={
 'LiMg':3.78,'NaMg':3.84,'KMg':3.99,'RbMg':4.05,
 'NaZn':3.88,'KZn':4.02,'NaCa':3.96,'KCa':4.10,
}

def build(pair,a,c,registry='plaquette'):
    A=''.join([x for x in pair if x.isalpha()])
    # explicit parse for our matrix
    for aa in ['Li','Na','K','Rb']:
        if pair.startswith(aa): A=aa; B=pair[len(aa):]; break
    sy=[]; fr=[]
    # AgF2 square sheet at z=0
    for y in range(2):
      for x in range(3):
        sy += ['Ag','F','F']
        fr += [[x/3,y/2,0],[(x+.5)/3,y/2,0],[x/3,(y+.5)/2,0]]
    # one neutral ABF3 block, spatially separated from the AgF2 sheet
    for y in range(2):
      for x in range(3):
        if registry=='plaquette':
            axy=(x/3,y/2); bxy=((x+.5)/3,(y+.5)/2)
        else:
            axy=((x+.5)/3,(y+.5)/2); bxy=(x/3,y/2)
        sy += [A,B,'F','F','F']
        fr += [[axy[0],axy[1],.28],[bxy[0],bxy[1],.58],
               [bxy[0],bxy[1],.30],[(x+.5)/3,y/2,.58],[x/3,(y+.5)/2,.58]]
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def bulk_control(pair):
    for aa in ['Li','Na','K','Rb']:
        if pair.startswith(aa): A=aa; B=pair[len(aa):]; break
    a=PAIR_A0[pair]
    # cubic ABF3 reference topology
    s=Structure(Lattice.cubic(a),[A,B,'F','F','F'],[[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    at=AseAtomsAdaptor.get_atoms(s); at.calc=CHGNetCalculator(model=CHGNet.load())
    flt=FrechetCellFilter(at,scalar_pressure=0)
    FIRE(flt,logfile=None).run(fmax=.06,steps=300)
    st=AseAtomsAdaptor.get_structure(at)
    ib=[i for i,site in enumerate(st) if site.specie.symbol==B][0]
    ds=sorted(st.get_distance(ib,j) for j,site in enumerate(st) if site.specie.symbol=='F')
    b6=ds[:6]
    lens=np.array(at.cell.lengths())
    ok=bool(np.all((lens>3.2)&(lens<4.8)) and len(b6)>=3 and min(b6)>1.5 and max(b6)<2.7)
    return {'pair':pair,'control_ok':ok,'cell_A':lens.tolist(),'B_F_nearest_A':b6}

def analyze(at,pair,tag,atm,e0,v0):
    s=AseAtomsAdaptor.get_structure(at)
    ag=[i for i,z in enumerate(s) if z.specie.symbol=='Ag']
    ff=[i for i,z in enumerate(s) if z.specie.symbol=='F']
    nearest=[]; coords=[]
    for i in ag:
        ds=sorted(s.get_distance(i,j) for j in ff)
        nearest += ds[:4]; coords.append(sum(d<2.55 for d in ds))
    bridge=[]; angles=[]
    for j in ff:
        near=sorted([(s.get_distance(i,j),i) for i in ag])
        if len(near)>=2 and near[1][0]<2.55:
            bridge.append(j)
            a,b=near[0][1],near[1][1]
            try: angles.append(float(s.get_angle(a,j,b)))
            except: pass
    forces=at.get_forces(); fmax=float(np.linalg.norm(forces,axis=1).max())
    stress=np.asarray(at.get_stress(voigt=True))*160.21766208
    e=float(at.get_potential_energy()); vol=float(at.get_volume()); p=atm*ATM_TO_GPA
    h=e+p*vol*GPA_TO_EV_A3
    mean_d=float(np.mean(nearest)); max_d=float(np.max(nearest))
    am=float(np.mean(angles)) if angles else 0.; ap10=float(np.percentile(angles,10)) if angles else 0.
    plane=bool(mean_d<=2.05 and max_d<=2.18 and min(coords)>=4 and len(bridge)>=12 and am>=168 and ap10>=155)
    gross=bool(fmax<=.08 and min(coords)>=4 and min((s.get_distance(i,j) for i in range(len(s)) for j in range(i)),default=9)>1.25)
    return {'pair':pair,'tag':tag,'pressure_atm':atm,'formula':s.composition.reduced_formula,
      'natoms':len(s),'energy_eV_atom':e/len(s),'enthalpy_proxy_eV_atom':h/len(s),
      'volume_ratio':vol/v0,'max_force_eV_A':fmax,'cell_lengths_A':[float(x) for x in at.cell.lengths()],
      'hydrostatic_GPa_from_stress':float(-np.mean(stress[:3])),
      'mean_Ag_F_nearest4_A':mean_d,'max_Ag_F_nearest4_A':max_d,
      'min_Ag_F_coord_lt2p55':int(min(coords)),'n_bridging_F':len(bridge),
      'mean_Ag_F_Ag_angle_deg':am,'p10_Ag_F_Ag_angle_deg':ap10,
      'plane_pass':plane,'gross_pass':gross}

def run(pair,out):
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    ctrl=bulk_control(pair)
    model=CHGNet.load(); rec=[]; a0=PAIR_A0[pair]
    starts=[('compact','plaquette',a0,8.4),('expanded','plaquette',a0+0.16,9.2),
            ('compact','ag',a0,8.4),('expanded','ag',a0+0.16,9.2)]
    for atm in [0,400]:
      for size,reg,a,c in starts:
        tag=f'{pair}_{reg}_{size}_{atm}atm'; s=build(pair,a,c,reg); at=AseAtomsAdaptor.get_atoms(s)
        at.calc=CHGNetCalculator(model=model); e0=float(at.get_potential_energy()); v0=float(at.get_volume())
        flt=FrechetCellFilter(at,scalar_pressure=atm*ATM_TO_GPA*GPA_TO_EV_A3)
        FIRE(flt,logfile=str(out/(tag+'.log'))).run(fmax=.05,steps=600)
        r=analyze(at,pair,tag,atm,e0,v0); rec.append(r)
        AseAtomsAdaptor.get_structure(at).to(filename=str(out/(tag+'.cif')))
    sel={}
    for atm in [0,400]:
      rr=[r for r in rec if r['pressure_atm']==atm and r['gross_pass']]
      if not rr: rr=[r for r in rec if r['pressure_atm']==atm]
      best=min(rr,key=lambda r:r['enthalpy_proxy_eV_atom']); sel[str(atm)]=best
      Path(best['tag']+'.tmp') if False else None
      src=out/(best['tag']+'.cif'); (out/f'selected_{atm}atm.cif').write_bytes(src.read_bytes())
    advance=bool(ctrl['control_ok'] and all(sel[str(p)]['plane_pass'] for p in [0,400]))
    result={'pair':pair,'control':ctrl,'all_runs':rec,'selected':sel,'advance':advance}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/('ADVANCE' if advance else 'REJECT')).write_text('1\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--pair',required=True,choices=sorted(PAIR_A0)); ap.add_argument('--out',required=True)
    z=ap.parse_args(); run(z.pair,z.out)
