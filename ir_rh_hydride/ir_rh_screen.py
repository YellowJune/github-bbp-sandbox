from __future__ import annotations

import argparse, glob, json, math, os, re, shutil, subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1.0/160.21766208
RY_TO_EV=13.605693122994

# MgAlFeH6 is the published carrier-doped calibration. The Rh/Ir entries are
# ordered K2PtCl6-derived counterfactuals: half-Mg donor substitution into
# parent Mg2RhH6/Mg2IrH6-like lattices. Exact-composition web search on
# 2026-09-06 found no hits for these four formulas; this is a falsification
# screen, not a novelty claim.
CANDIDATES={
    'MgAlFeH6':('Al','Fe',6.24),
    'MgAlRhH6':('Al','Rh',6.52),
    'MgGaRhH6':('Ga','Rh',6.58),
    'MgAlIrH6':('Al','Ir',6.60),
    'MgGaIrH6':('Ga','Ir',6.66),
}
MASS={'H':1.00794,'Mg':24.305,'Al':26.9815385,'Ga':69.723,'Fe':55.845,'Rh':102.90550,'Ir':192.217}


def build(name,xh=0.240):
    donor,tm,a=CANDIDATES[name]
    c=Structure.from_spacegroup(216,Lattice.cubic(a),['Mg',donor,tm,'H'],
        [[.75,.75,.75],[.25,.25,.25],[.5,.5,.5],[xh,0,0]])
    return c.get_primitive_structure(tolerance=1e-5)


def dmin(s):
    return min(float(s.get_distance(i,j)) for i in range(len(s)) for j in range(i))


def hmetrics(s,tm):
    hs=[i for i,x in enumerate(s) if x.specie.symbol=='H']
    ts=[i for i,x in enumerate(s) if x.specie.symbol==tm]
    hh=[s.get_distance(i,j) for k,i in enumerate(hs) for j in hs[:k]]
    th=[]
    for i in ts: th += sorted(s.get_distance(i,j) for j in hs)[:6]
    return {'min_H_H_A':float(min(hh)),'TM_H_mean_A':float(np.mean(th)),'TM_H_max_A':float(np.max(th))}


def relax(name,atm,out):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    donor,tm,_=CANDIDATES[name]; model=CHGNet.load(); rows=[]
    for xh in (.235,.245):
        s0=build(name,xh); at=AseAtomsAdaptor.get_atoms(s0); at.calc=CHGNetCalculator(model=model)
        filt=FrechetCellFilter(at,scalar_pressure=atm*ATM_TO_GPA*GPA_TO_EV_A3)
        FIRE(filt,logfile=str(out/f'relax_{xh:.3f}.log')).run(fmax=.045,steps=600)
        sf=AseAtomsAdaptor.get_structure(at); force=np.asarray(at.get_forces())
        stress=np.asarray(at.get_stress(voigt=True))*160.21766208
        hm=hmetrics(sf,tm); dm=dmin(sf); p=atm*ATM_TO_GPA
        fmax=float(np.linalg.norm(force,axis=1).max()); e=float(at.get_potential_energy()); v=float(at.get_volume())
        ent=e+p*v*GPA_TO_EV_A3
        ok=bool(fmax<=.08 and dm>=.75 and hm['min_H_H_A']>=.90 and 1.25<=hm['TM_H_mean_A']<=2.20 and hm['TM_H_max_A']<=2.35 and abs(float(-np.mean(stress[:3]))-p)<=.8)
        cif=out/f'{name}_{atm}atm_xh{xh:.3f}.cif'; sf.to(filename=str(cif))
        rows.append({'xh':xh,'gross_pass':ok,'fmax_eV_A':fmax,'stress_GPa':float(-np.mean(stress[:3])),'energy_eV_atom':e/len(sf),'enthalpy_proxy_eV_atom':ent/len(sf),'volume_A3':v,'dmin_A':dm,**hm,'cif':str(cif)})
    viable=[r for r in rows if r['gross_pass']]; sel=min(viable or rows,key=lambda r:r['enthalpy_proxy_eV_atom'])
    shutil.copy2(sel['cif'],out/'selected.cif')
    result={'candidate':name,'pressure_atm':atm,'runs':rows,'selected':sel,'advance_struct':bool(sel['gross_pass']),'note':'CHGNet is a structural-survival proxy only.'}
    (out/'struct.json').write_text(json.dumps(result,indent=2)); return result


def pseudo(ppdir,elem):
    e=elem.lower(); hits=[]
    for root,_,files in os.walk(ppdir):
        for fn in files:
            l=fn.lower()
            if l.endswith('.upf') and re.match(r'^'+re.escape(e)+r'[^a-z0-9]',l): hits.append(os.path.join(root,fn))
    if not hits: raise RuntimeError('missing pseudo '+elem)
    return os.path.basename(sorted(hits)[0])


def write_pw(path,s,ppdir,scratch,prefix,calc,k,nbnd=None):
    species=[]
    for x in s:
        z=x.specie.symbol
        if z not in species: species.append(z)
    pm={z:pseudo(ppdir,z) for z in species}
    L=['&CONTROL',f" calculation='{calc}',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{scratch}',"," disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(species)},',' ecutwfc=50.0, ecutrho=400.0,'," occupations='smearing', smearing='mv', degauss=.02,",' nosym=.true., noinv=.true.,']
    if nbnd is not None: L.append(f' nbnd={nbnd},')
    L += ['/','&ELECTRONS',' conv_thr=1.d-7, mixing_beta=.30, electron_maxstep=500,'," diagonalization='cg',",'/','ATOMIC_SPECIES']
    L += [f' {z} {MASS[z]:.8f} {pm[z]}' for z in species]
    L += ['ATOMIC_POSITIONS crystal']
    for x in s:
        f=x.frac_coords; L.append(f' {x.specie.symbol} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L += ['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]
    L += ['K_POINTS automatic',f' {k} {k} {k} 0 0 0']
    path.write_text('\n'.join(L)+'\n')


def runexe(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo: return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode


def lastf(p,t):
    m=re.findall(p,t,re.I); return float(m[-1]) if m else None

def lasti(p,t):
    m=re.findall(p,t,re.I); return int(float(m[-1])) if m else None

def loadnum(path):
    a=[]
    for l in Path(path).read_text(errors='ignore').splitlines():
        if not l.strip() or l.lstrip().startswith('#'): continue
        try: a.append([float(x) for x in l.split()])
        except: pass
    n=min(map(len,a)); return np.asarray([x[:n] for x in a])
def interp(a,ef,col=1):
    o=np.argsort(a[:,0]); return float(np.interp(ef,a[o,0],a[o,col]))


def electronic(name,atm,cif,ppdir,out,pw='/opt/espresso/7.5/pw.x'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); scratch=out/'scratch'; scratch.mkdir(exist_ok=True)
    s=Structure.from_file(cif); prefix=f'irrh_{name}_{atm}'
    si=out/'scf.in'; so=out/'scf.out'; write_pw(si,s,ppdir,scratch,prefix,'scf',6); src=runexe(pw,si,so)
    st=so.read_text(errors='ignore'); ef=lastf(r'the Fermi energy is\s+([-0-9.Ee+]+)\s+ev',st); ne=lastf(r'number of electrons\s*=\s*([-0-9.Ee+]+)',st); ns=lasti(r'number of Kohn-Sham states\s*=\s*([0-9]+)',st)
    minocc=int(math.ceil((ne or 0)/2)); nb=max(minocc+4,minocc+2); nb=min(nb,ns) if ns else nb
    ni=out/'nscf.in'; no=out/'nscf.out'; write_pw(ni,s,ppdir,scratch,prefix,'nscf',6,nb); nrc=runexe(pw,ni,no) if src==0 else 99
    nt=no.read_text(errors='ignore') if no.exists() else ''; en=lastf(r'the Fermi energy is\s+([-0-9.Ee+]+)\s+ev',nt); ef=en if en is not None else ef
    dos=out/'total.dos'; di=out/'dos.in'; di.write_text(f"&DOS\n prefix='{prefix}',\n outdir='{scratch}',\n fildos='{dos}',\n DeltaE=.01,\n/\n")
    drc=runexe(str(Path(pw).with_name('dos.x')),di,out/'dos.out') if nrc==0 else 99
    pi=out/'projwfc.in'; pi.write_text(f"&PROJWFC\n prefix='{prefix}',\n outdir='{scratch}',\n filpdos='{out/'pdos'}',\n DeltaE=.01,\n/\n")
    prc=runexe(str(Path(pw).with_name('projwfc.x')),pi,out/'projwfc.out') if nrc==0 else 99
    td=hd=desc=None; hfiles=[]
    if drc==0 and prc==0 and ef is not None and dos.exists():
        td=max(0.,interp(loadnum(dos),ef,1)); hfiles=[Path(x) for x in glob.glob(str(out/'pdos.pdos_atm#*(H)*'))]
        hv=[]
        for f in hfiles: hv.append(max(0.,interp(loadnum(f),ef,1)))
        if hv: hd=float(sum(hv)); desc=float(math.sqrt(td*hd))
    r={'candidate':name,'pressure_atm':atm,'scf_rc':src,'nscf_rc':nrc,'dos_rc':drc,'projwfc_rc':prc,'fermi_eV':ef,'total_DOS_EF':td,'H_DOS_EF':hd,'descriptor':desc,'metallic_screen':bool(td is not None and td>=.05),'n_H_pdos_files':len(hfiles),'symmetry_mode':'nosym+noinv','note':'Descriptor is ranking only; not EPC, lambda, phonon stability, or Tc.'}
    (out/'electronic.json').write_text(json.dumps(r,indent=2)); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate',required=True,choices=sorted(CANDIDATES)); ap.add_argument('--ppdir',required=True); ap.add_argument('--out',required=True); z=ap.parse_args()
    root=Path(z.out); allr=[]
    for atm in (0,400):
        sd=root/f'{atm}atm/struct'; sr=relax(z.candidate,atm,sd)
        er=None
        if sr['advance_struct']:
            er=electronic(z.candidate,atm,sd/'selected.cif',z.ppdir,root/f'{atm}atm/qe')
        allr.append({'pressure_atm':atm,'struct':sr,'electronic':er})
    out={'candidate':z.candidate,'results':allr,'advance_electronic':bool(all(x['electronic'] and x['electronic']['descriptor'] is not None for x in allr))}
    (root/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
