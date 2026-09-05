from __future__ import annotations
import argparse,glob,json,math,os,re,subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.ase import AseAtomsAdaptor

ATM_TO_GPA=0.000101325
GPA_TO_EV_A3=1/160.21766208


def build(a=4.10,c=7.80,dz=.21):
    sy=[]; fr=[]
    for y in range(2):
        for x in range(3):
            sy.append('Be'); fr.append([(x+.5)/3,(y+.5)/2,.5])
            sy.append('Ag'); fr.append([x/3,y/2,0])
            sy.append('F'); fr.append([(x+.5)/3,y/2,0])
            sy.append('F'); fr.append([x/3,(y+.5)/2,0])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5-dz])
            sy.append('F'); fr.append([(x+.5)/3,(y+.5)/2,.5+dz])
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)


def relax(out):
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    out=Path(out); out.mkdir(parents=True,exist_ok=True); model=CHGNet.load(); rec=[]
    for atm in [0,400]:
        s=build(); at=AseAtomsAdaptor.get_atoms(s); at.calc=CHGNetCalculator(model=model)
        p=atm*ATM_TO_GPA
        flt=FrechetCellFilter(at,scalar_pressure=p*GPA_TO_EV_A3)
        FIRE(flt,logfile=str(out/f'relax_{atm}.log')).run(fmax=.05,steps=500)
        fmax=float(np.linalg.norm(at.get_forces(),axis=1).max())
        st=np.asarray(at.get_stress(voigt=True))*160.21766208
        cif=out/f'parent_{atm}atm.cif'; AseAtomsAdaptor.get_structure(at).to(filename=str(cif))
        rec.append({'pressure_atm':atm,'max_force_eV_A':fmax,'cell_A':[float(x) for x in at.cell.lengths()],
                    'hydrostatic_GPa_from_stress':float(-np.mean(st[:3])),'cif':str(cif)})
    (out/'relax_result.json').write_text(json.dumps(rec,indent=2)); print(json.dumps(rec,indent=2))


def pseudo(ppdir,elem):
    # SSSP filenames are not guaranteed to preserve element-case prefixes
    # (e.g. Be vs be). Search recursively and case-insensitively, accepting
    # common separators after the element symbol.
    el=elem.lower()
    cand=[]
    for p in Path(ppdir).rglob('*'):
        if not p.is_file():
            continue
        b=p.name.lower()
        if not b.endswith(('.upf','.upf.gz')):
            continue
        if b.startswith(el+'.') or b.startswith(el+'_') or b.startswith(el+'-'):
            cand.append(p)
    if not cand:
        # Last-resort exact symbol prefix, still case-insensitive.
        cand=[p for p in Path(ppdir).rglob('*') if p.is_file() and p.name.lower().startswith(el) and '.upf' in p.name.lower()]
    if not cand:
        raise RuntimeError('pseudo '+elem+'; available sample='+','.join(sorted(p.name for p in Path(ppdir).rglob('*') if p.is_file())[:20]))
    p=sorted(cand,key=lambda x:(len(x.name),x.name.lower()))[0]
    # QE receives pseudo_dir, so copy nested files to its root when needed.
    root=Path(ppdir)/p.name
    if p.resolve()!=root.resolve():
        import shutil; shutil.copy2(p,root)
    return p.name


def aglabel(site):
    y=float(site.frac_coords[1]%1.0)
    return 'AgA' if min(abs(y),abs(y-1))<=abs(y-.5) else 'AgB'


def write_in(path,s,ppdir,scratch,mode):
    labels=[aglabel(site) if site.specie.symbol=='Ag' else site.specie.symbol for site in s]
    types=[]
    for x in labels:
        if x not in types: types.append(x)
    pm={'Be':pseudo(ppdir,'Be'),'F':pseudo(ppdir,'F'),'AgA':pseudo(ppdir,'Ag'),'AgB':pseudo(ppdir,'Ag')}
    mass={'Be':9.0121831,'F':18.99840316,'AgA':107.8682,'AgB':107.8682}
    L=['&CONTROL'," calculation='scf',"," prefix='parent',",f" pseudo_dir='{ppdir}',",f" outdir='{scratch}',"," disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(types)},',' ecutwfc=45.0, ecutrho=360.0,'," occupations='smearing', smearing='mv', degauss=.015,",' nspin=2,']
    for i,x in enumerate(types,1):
        m=.65 if x=='AgA' else ((.65 if mode=='FM' else -.65) if x=='AgB' else 0)
        L.append(f' starting_magnetization({i})={m},')
    L+=['/','&ELECTRONS',' conv_thr=1.d-6, mixing_beta=.25, electron_maxstep=220,','/','ATOMIC_SPECIES']
    L += [f' {x} {mass[x]:.8f} {pm[x]}' for x in types]
    L += ['HUBBARD (ortho-atomic)',' U AgA-4d 6.0',' U AgB-4d 6.0','ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labels):
        f=site.frac_coords; L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L += ['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]
    L += ['K_POINTS automatic',' 2 2 2 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')


def enum(pat,txt):
    m=re.findall(pat,txt,re.I); return float(m[-1]) if m else float('nan')


def qe(cif,atm,ppdir,pw,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); s=Structure.from_file(cif); E={}; rows=[]
    for mode in ['FM','AF_y']:
        scratch=out/f'scratch_{mode}'; scratch.mkdir(exist_ok=True)
        inp=out/f'{mode}.in'; oo=out/f'{mode}.out'; write_in(inp,s,ppdir,str(scratch),mode)
        with open(inp,'rb') as fi, open(oo,'wb') as fo: rc=subprocess.run([pw],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode
        txt=oo.read_text(errors='ignore'); eRy=enum(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt)
        rows.append({'pressure_atm':atm,'mode':mode,'scf_rc':rc,'energy_Ry':eRy,'energy_eV':eRy*13.605693122994 if np.isfinite(eRy) else None})
        if rc==0 and np.isfinite(eRy): E[mode]=eRy*13.605693122994
    if len(E)==2:
        J=(E['FM']-E['AF_y'])/3.0
        rows.append({'pressure_atm':atm,'mode':'mapped_J','J_eV':J,'J_meV':1000*J,'passes_0p30':bool(J>=.30),'passes_0p38':bool(J>=.38)})
    (out/'result.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
    if len(E)!=2: raise SystemExit(2)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('relax'); a.add_argument('--out',required=True)
    q=sub.add_parser('qe'); q.add_argument('--cif',required=True); q.add_argument('--atm',type=int,required=True); q.add_argument('--ppdir',required=True); q.add_argument('--pw',default='/opt/espresso/7.5/pw.x'); q.add_argument('--out',required=True)
    z=ap.parse_args(); relax(z.out) if z.cmd=='relax' else qe(z.cif,z.atm,z.ppdir,z.pw,z.out)
