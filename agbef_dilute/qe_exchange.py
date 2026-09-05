#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

MASS={'Al':26.9815385,'Be':9.0121831,'AgA':107.8682,'AgB':107.8682,'F':18.99840316}
BASE={'Al':'Al','Be':'Be','AgA':'Ag','AgB':'Ag','F':'F'}


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
    y=float(site.frac_coords[1] % 1.0)
    d0=min(abs(y),abs(y-1.0)); d1=abs(y-.5)
    return 'AgA' if d0<=d1 else 'AgB'


def labels_for(s):
    return [ag_row_label(site) if site.specie.symbol=='Ag' else site.specie.symbol for site in s]


def write_scf(path,prefix,s,labels,pm,ppdir,outdir,mode):
    sp=[]
    for x in labels:
        if x not in sp: sp.append(x)
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," disk_io='low', verbosity='high',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(sp)},',' ecutwfc=45.0, ecutrho=360.0,'," occupations='smearing', smearing='mv', degauss=.020,",' nspin=2,']
    for i,x in enumerate(sp,1):
        if x=='AgA': mag=.60
        elif x=='AgB': mag=.60 if mode=='FM' else -.60
        else: mag=0.0
        L.append(f' starting_magnetization({i})={mag:.4f},')
    L+=['/','&ELECTRONS'," conv_thr=1.d-6, mixing_mode='local-TF', mixing_beta=.25, electron_maxstep=220, diagonalization='david',",'/']
    L+=['ATOMIC_SPECIES']+[f' {x} {MASS[x]:.8f} {pm[x]}' for x in sp]
    L+=['HUBBARD (ortho-atomic)',' U AgA-4d 6.000000',' U AgB-4d 6.000000']
    L+=['ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labels):
        f=site.frac_coords; L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L+=['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]
    # The 6x2 cell is doubled along x relative to the 3x2 parent; 1x2x2 keeps a comparable k density.
    L+=['K_POINTS automatic',' 1 2 2 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')


def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:
        return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode


def lastnum(pat,txt):
    m=re.findall(pat,txt,re.I); return float(m[-1]) if m else np.nan


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cif',required=True); ap.add_argument('--pressure-atm',type=float,required=True)
    ap.add_argument('--ppdir',required=True); ap.add_argument('--meta',required=True); ap.add_argument('--scratch',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x'); a=ap.parse_args()
    s=Structure.from_file(a.cif); labels=labels_for(s); nA=labels.count('AgA'); nB=labels.count('AgB')
    if nA!=6 or nB!=6: raise RuntimeError(f'expected 6+6 Ag rows, got {nA}+{nB}')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); scratch=Path(a.scratch); scratch.mkdir(parents=True,exist_ok=True)
    pm=pmap(a.ppdir,a.meta,sorted(set(labels))); rows=[]; energies={}; any_fail=False
    for mode in ['FM','AF_y']:
        tag=f'U6_{mode}'; od=scratch/tag; od.mkdir(parents=True,exist_ok=True); pref=f'dilute_{int(a.pressure_atm)}_{tag}'
        inp=out/f'{tag}.in'; oo=out/f'{tag}.out'; write_scf(inp,pref,s,labels,pm,a.ppdir,str(od),mode)
        rc=run(a.pw,str(inp),str(oo)); rec={'candidate':'AlBe11Ag12F48','pressure_atm':a.pressure_atm,'U_Ag_eV':6.0,'mode':mode,'scf_rc':rc}
        txt=oo.read_text(errors='ignore') if oo.exists() else ''
        rec['last_total_energy_Ry']=lastnum(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt)
        rec['estimated_scf_accuracy_Ry']=lastnum(r'estimated scf accuracy\s+<\s+([-0-9.Ee+]+)\s+Ry',txt)
        if rc==0 and np.isfinite(rec['last_total_energy_Ry']):
            rec['energy_Ry']=rec['last_total_energy_Ry']; rec['energy_eV']=rec['energy_Ry']*13.605693122994
            rec['total_magnetization']=lastnum(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt)
            rec['absolute_magnetization']=lastnum(r'absolute magnetization\s+=\s+([-0-9.Ee+]+)',txt)
            energies[mode]=rec['energy_eV']
        else: any_fail=True
        rows.append(rec); print(json.dumps(rec,default=float))
    if len(energies)==2:
        # 12 y-directed bonds flip sign. For H=J sum S_i.S_j, S=1/2, total Delta is -6J.
        # In the doped metal this is an effective row-exchange scale, not a literal undoped Heisenberg J.
        Jeff=(energies['FM']-energies['AF_y'])/6.0
        rows.append({'candidate':'AlBe11Ag12F48','pressure_atm':a.pressure_atm,'U_Ag_eV':6.0,'mode':'mapped_effective_J',
                     'J_eff_eV_Shalf':Jeff,'J_eff_meV_Shalf':1000*Jeff,
                     'passes_0p30eV_prescreen':bool(Jeff>=.30),'passes_0p38eV_direct300_gate':bool(Jeff>=.38)})
    else: any_fail=True
    keys=sorted({k for r in rows for k in r});
    with open(out/'exchange_mapping.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
    if any_fail: raise SystemExit(2)

if __name__=='__main__': main()
