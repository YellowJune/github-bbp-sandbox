#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

MASS={'Li':6.94,'Al':26.9815385,'Be':9.0121831,'AgA':107.8682,'AgB':107.8682,'F':18.99840316}
BASE={'Li':'Li','Al':'Al','Be':'Be','AgA':'Ag','AgB':'Ag','F':'F'}


def pmap(ppdir,meta,labels):
    md=json.load(open(meta)); out={}
    for lab in labels:
        e=BASE[lab]
        fn=md.get(e,{}).get('filename') if isinstance(md.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)):
            out[lab]=fn; continue
        fs=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:
            fs+=glob.glob(os.path.join(ppdir,pat))
        if not fs: raise RuntimeError(f'no pseudo for {e}')
        out[lab]=os.path.basename(sorted(fs)[0])
    return out


def ag_row_label(site):
    y=float(site.frac_coords[1]%1.0)
    d0=min(abs(y),abs(y-1.0)); d1=abs(y-.5)
    return 'AgA' if d0<=d1 else 'AgB'


def labels_for(s):
    out=[]
    for site in s:
        e=site.specie.symbol
        out.append(ag_row_label(site) if e=='Ag' else e)
    return out


def write_scf(path,prefix,s,labels,pm,ppdir,outdir,U,mode):
    sp=[]
    for x in labels:
        if x not in sp: sp.append(x)
    a,b,c=s.lattice.abc
    k=(max(2,min(4,round(24/a))),max(2,min(5,round(20/b))),max(3,min(7,round(28/c))))
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(sp)},',' ecutwfc=55.0, ecutrho=440.0,'," occupations='smearing', smearing='mv', degauss=.012,",' nspin=2,']
    for i,x in enumerate(sp,1):
        if x=='AgA': mag=.65
        elif x=='AgB': mag=.65 if mode=='FM' else -.65
        else: mag=0.0
        L.append(f' starting_magnetization({i})={mag:.4f},')
    L+=['/','&ELECTRONS',' conv_thr=2.d-8, mixing_beta=.18, electron_maxstep=420,','/']

    # QE 7.5 requires ATOMIC_SPECIES to be defined before the HUBBARD card.
    L+=['ATOMIC_SPECIES']+[f' {x} {MASS[x]:.8f} {pm[x]}' for x in sp]
    if U>0:
        L+=['HUBBARD (ortho-atomic)',f' U AgA-4d {U:.6f}',f' U AgB-4d {U:.6f}']

    L+=['ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labels):
        f=site.frac_coords; L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L+=['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]+['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')


def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:
        return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode


def lastnum(pat,txt):
    m=re.findall(pat,txt,re.I); return float(m[-1]) if m else np.nan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cif',required=True);ap.add_argument('--kind',required=True);ap.add_argument('--pressure-atm',type=float,required=True);ap.add_argument('--ppdir',required=True);ap.add_argument('--meta',required=True);ap.add_argument('--scratch',required=True);ap.add_argument('--out',required=True);ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x');a=ap.parse_args()
    s=Structure.from_file(a.cif); labels=labels_for(s); out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    if labels.count('AgA')!=3 or labels.count('AgB')!=3:
        raise RuntimeError(f'expected 3+3 Ag rows, got {labels.count("AgA")} + {labels.count("AgB")}')
    pm=pmap(a.ppdir,a.meta,sorted(set(labels))); rows=[]; any_fail=False
    for U in [4.0,6.0]:
        energies={}
        for mode in ['FM','AF_y']:
            tag=f'U{int(U)}_{mode}';od=scratch/tag;od.mkdir(parents=True,exist_ok=True);pref=f'{a.kind}_{int(a.pressure_atm)}_{tag}'
            inp=out/f'{tag}.in';oo=out/f'{tag}.out';write_scf(inp,pref,s,labels,pm,a.ppdir,str(od),U,mode)
            rc=run(a.pw,str(inp),str(oo)); rec={'kind':a.kind,'pressure_atm':a.pressure_atm,'U_Ag_eV':U,'mode':mode,'scf_rc':rc}
            if rc==0:
                txt=oo.read_text(errors='ignore'); eRy=lastnum(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt)
                rec.update({'energy_Ry':eRy,'energy_eV':eRy*13.605693122994,'total_magnetization':lastnum(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt),'absolute_magnetization':lastnum(r'absolute magnetization\s+=\s+([-0-9.Ee+]+)',txt)})
                if np.isfinite(rec['energy_eV']): energies[mode]=rec['energy_eV']
                else: any_fail=True
            else:
                any_fail=True
            rows.append(rec); print(json.dumps(rec,default=float))
        if 'FM' in energies and 'AF_y' in energies:
            # Six y-directed Ag-F-Ag bonds change from parallel to antiparallel.
            # For H=J sum S_i.S_j with S=1/2: Delta(sum S_i.S_j)=-3,
            # so J=(E_FM-E_AF_y)/3. Positive J means AFM.
            J=(energies['FM']-energies['AF_y'])/3.0
            rows.append({'kind':a.kind,'pressure_atm':a.pressure_atm,'U_Ag_eV':U,'mode':'mapped_J','J_y_eV_Shalf':J,'J_y_meV_Shalf':1000*J,'passes_J_0p38eV':bool(J>=.38)})
        else:
            any_fail=True
    keys=sorted({k for r in rows for k in r})
    with open(out/'exchange_mapping.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
    if any_fail:
        raise SystemExit(2)

if __name__=='__main__': main()
