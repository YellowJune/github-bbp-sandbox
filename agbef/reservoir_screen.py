from __future__ import annotations
import argparse,json,math
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
STARTS=[('compact',3.88,6.6,.22),('central',4.02,7.2,.22),('expanded',4.16,7.8,.21)]
TRIVALENT={'B','Al','Ga','Sc'}


def build(X,a,c,dz):
    sy=[];fr=[]; planeF=[]; spacerF=[]; Oidx=None; Aidx=[]; ai=0
    for y in range(2):
        for x in range(3):
            Aidx.append(len(sy));sy.append(X if ai==0 else 'Be');fr.append([(x+.5)/3,(y+.5)/2,.5]);ai+=1
            sy.append('Ag');fr.append([x/3,y/2,0])
            planeF.append(len(sy));sy.append('F');fr.append([(x+.5)/3,y/2,0])
            planeF.append(len(sy));sy.append('F');fr.append([x/3,(y+.5)/2,0])
            # Charge-neutral reservoir intervention: X3+ for Be2+ and O2- for F-.
            if x==0 and y==0:
                Oidx=len(sy);sy.append('O');fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            else:
                spacerF.append(len(sy));sy.append('F');fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            spacerF.append(len(sy));sy.append('F');fr.append([(x+.5)/3,(y+.5)/2,.5+dz])
    s=Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)
    return s,planeF,spacerF,Oidx,Aidx


def micvec(at,i,j):
    return np.asarray(at.get_distance(i,j,mic=True,vector=True),float)


def metrics(at,planeF,Oidx,Aidx):
    sy=at.get_chemical_symbols(); d=at.get_all_distances(mic=True)
    ag=[i for i,s in enumerate(sy) if s=='Ag']
    nearest=[];coord=[];angles=[]
    for i in ag:
        vals=sorted(float(d[i,j]) for j in planeF)
        nearest.append(float(np.mean(vals[:4])))
        coord.append(sum(x<2.55 for x in vals))
    for f in planeF:
        ags=sorted((float(d[f,i]),i) for i in ag)
        close=[x for x in ags if x[0]<2.55]
        if len(close)>=2:
            v1=micvec(at,f,close[0][1]);v2=micvec(at,f,close[1][1])
            den=np.linalg.norm(v1)*np.linalg.norm(v2)
            if den>1e-12:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))))
    allpair=[float(d[i,j]) for i in range(len(at)) for j in range(i+1,len(at))]
    agO=min(float(d[i,Oidx]) for i in ag)
    xo=min(float(d[i,Oidx]) for i in Aidx)
    # DFT+U control-fit proxy from flat AgF2 controls; ranking only.
    dm=float(np.mean(nearest)); am=float(np.mean(angles)) if angles else float('nan')
    Jproxy=206.2*(2.07/dm)**9.02*(max(math.cos(math.radians(180-am))**2,1e-4)**1.07) if np.isfinite(am) else float('nan')
    return {
        'mean_Ag_planeF_nearest4_A':dm,'max_Ag_planeF_nearest4_A':float(np.max(nearest)),
        'min_Ag_planeF_coord_lt2p55':int(min(coord)),'mean_Ag_F_Ag_angle_deg':am,
        'p10_Ag_F_Ag_angle_deg':float(np.quantile(angles,.10)) if angles else float('nan'),
        'min_Ag_O_A':agO,'min_XorBe_O_A':xo,'min_any_pair_A':float(min(allpair)),
        'controlfit_DFTU_J_proxy_meV':Jproxy,
    }


def relax(X,label,a,c,dz,p,model,outdir):
    s,pf,sf,oidx,aidx=build(X,a,c,dz);at=AseAtomsAdaptor.get_atoms(s);v0=float(at.get_volume());m0=metrics(at,pf,oidx,aidx)
    at.calc=CHGNetCalculator(model=model);flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
    tag=f'{X}_{label}_{p:.8f}GPa';FIRE(flt,logfile=str(outdir/f'{tag}.log')).run(fmax=.050,steps=520)
    fmax=float(np.linalg.norm(at.get_forces(),axis=1).max());stress=np.asarray(at.get_stress(voigt=True))*160.21766208
    m=metrics(at,pf,oidx,aidx);epa=float(at.get_potential_energy()/len(at));vpa=float(at.get_volume()/len(at))
    rec={'X':X,'tag':tag,'pressure_GPa_target':p,'pressure_atm_target':p/ATM_TO_GPA,'formula':at.get_chemical_formula(),
         'max_force_eV_A':fmax,'energy_eV_atom':epa,'enthalpy_proxy_eV_atom':epa+p*vpa*GPA_TO_EV_A3,
         'volume_ratio':float(at.get_volume()/v0),'cell_lengths_A':[float(x) for x in at.cell.lengths()],
         'cell_angles_deg':[float(x) for x in at.cell.angles()],'hydrostatic_pressure_GPa_from_stress':float(-np.mean(stress[:3]))}
    rec.update({'initial_'+k:v for k,v in m0.items()});rec.update({'final_'+k:v for k,v in m.items()})
    rec['gross_pass']=bool(fmax<.085 and .60<rec['volume_ratio']<1.45 and m['min_any_pair_A']>1.10)
    rec['magnetic_plane_pass']=bool(m['mean_Ag_planeF_nearest4_A']<2.15 and m['max_Ag_planeF_nearest4_A']<2.25 and
                                    m['min_Ag_planeF_coord_lt2p55']>=4 and m['mean_Ag_F_Ag_angle_deg']>=168 and m['p10_Ag_F_Ag_angle_deg']>=160)
    rec['reservoir_separated']=bool(m['min_Ag_O_A']>2.35)
    cif=outdir/f'{tag}.cif';AseAtomsAdaptor.get_structure(at).to(filename=str(cif));rec['cif']=str(cif)
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--X',required=True);a=ap.parse_args();X=a.X
    if X not in TRIVALENT:raise ValueError(X)
    outdir=Path(f'artifacts/reservoir/{X}');outdir.mkdir(parents=True,exist_ok=True);model=CHGNet.load();rows=[];selected={}
    for p in [0.0,400*ATM_TO_GPA]:
        rr=[relax(X,*st,p,model,outdir) for st in STARTS];rows+=rr
        ok=[r for r in rr if r['gross_pass']]
        if ok:
            # Favor energetic plausibility but retain the structure for exact post-analysis.
            b=min(ok,key=lambda r:r['enthalpy_proxy_eV_atom']);selected[f'{p:.8f}']=b
            Structure.from_file(b['cif']).to(filename=str(outdir/f'relaxed_{X}_{p:.8f}GPa.cif'))
    both=len(selected)==2
    pass2=bool(both and all(r['gross_pass'] and r['magnetic_plane_pass'] and r['reservoir_separated'] for r in selected.values()))
    result={'candidate':f'{X}Be5Ag6F23O','formal_Ag_valence':2.0,'all_runs':rows,'selected':selected,'advance':pass2}
    (outdir/'result.json').write_text(json.dumps(result,indent=2));(outdir/'ADVANCE').write_text('1' if pass2 else '0');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
