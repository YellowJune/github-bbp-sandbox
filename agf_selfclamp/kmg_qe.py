from __future__ import annotations
import argparse, json, math, os, re, subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

ATM_TO_GPA = 0.000101325
GPA_TO_EV_A3 = 1.0 / 160.21766208
RY_TO_EV = 13.605693122994


def pseudo(ppdir: str, elem: str) -> str:
    hits=[]
    e=elem.lower()
    for root,_,files in os.walk(ppdir):
        for fn in files:
            low=fn.lower()
            if low.endswith('.upf') and (low.startswith(e+'.') or low.startswith(e+'_') or low==e+'.upf'):
                hits.append(os.path.join(root,fn))
    if not hits:
        # fallback: SSSP names can contain element followed by non-alnum punctuation
        for root,_,files in os.walk(ppdir):
            for fn in files:
                low=fn.lower()
                if low.endswith('.upf') and re.match(r'^'+re.escape(e)+r'[^a-z0-9]',low):
                    hits.append(os.path.join(root,fn))
    if not hits:
        raise RuntimeError('pseudo '+elem)
    return os.path.basename(sorted(hits)[0])


def aglabel(site) -> str:
    y=float(site.frac_coords[1] % 1.0)
    return 'AgA' if min(abs(y),abs(y-1.0)) <= abs(y-0.5) else 'AgB'


def write_input(path: Path, s: Structure, ppdir: str, scratch: str, mode: str) -> None:
    labels=[aglabel(site) if site.specie.symbol=='Ag' else site.specie.symbol for site in s]
    types=[]
    for x in labels:
        if x not in types:
            types.append(x)
    elems={x:('Ag' if x.startswith('Ag') else x) for x in types}
    pm={x:pseudo(ppdir,elems[x]) for x in types}
    masses={'K':39.0983,'Mg':24.305,'F':18.998403163,'AgA':107.8682,'AgB':107.8682}
    L=['&CONTROL'," calculation='scf',"," prefix='kmg',",f" pseudo_dir='{ppdir}',",f" outdir='{scratch}',"," disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(types)},',' ecutwfc=45.0, ecutrho=360.0,',
       " occupations='smearing', smearing='mv', degauss=.015,",' nspin=2,']
    for i,x in enumerate(types,1):
        if x=='AgA': m=.65
        elif x=='AgB': m=.65 if mode=='FM' else -.65
        else: m=0.0
        L.append(f' starting_magnetization({i})={m},')
    L += ['/','&ELECTRONS',' conv_thr=1.d-6, mixing_beta=.25, electron_maxstep=220,','/','ATOMIC_SPECIES']
    L += [f' {x} {masses[x]:.8f} {pm[x]}' for x in types]
    L += ['HUBBARD (ortho-atomic)',' U AgA-4d 6.0',' U AgB-4d 6.0','ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labels):
        f=site.frac_coords
        L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L += ['CELL_PARAMETERS angstrom'] + [' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]
    L += ['K_POINTS automatic',' 2 2 2 0 0 0']
    path.write_text('\n'.join(L)+'\n')


def last_float(pattern: str, text: str):
    m=re.findall(pattern,text,re.I)
    return float(m[-1]) if m else float('nan')


def run(cif: str, atm: int, basin: str, ppdir: str, pw: str, out: str):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    s=Structure.from_file(cif)
    volume=float(s.volume)
    energies={}; rows=[]
    for mode in ['FM','AF_y']:
        scratch=Path('/tmp')/f'kmg_qe_{atm}_{basin}_{mode}'
        scratch.mkdir(parents=True,exist_ok=True)
        inp=out/f'{mode}.in'; oo=out/f'{mode}.out'
        write_input(inp,s,ppdir,str(scratch),mode)
        with open(inp,'rb') as fi, open(oo,'wb') as fo:
            rc=subprocess.run([pw],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode
        txt=oo.read_text(errors='ignore')
        ery=last_float(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt)
        eev=ery*RY_TO_EV if np.isfinite(ery) else None
        h=eev + atm*ATM_TO_GPA*volume*GPA_TO_EV_A3 if eev is not None else None
        row={'pressure_atm':atm,'basin':basin,'mode':mode,'scf_rc':rc,'energy_Ry':ery if np.isfinite(ery) else None,
             'energy_eV':eev,'volume_A3':volume,'enthalpy_eV':h}
        rows.append(row)
        if rc==0 and eev is not None:
            energies[mode]=eev
    if len(energies)==2:
        J=(energies['FM']-energies['AF_y'])/3.0
        rows.append({'pressure_atm':atm,'basin':basin,'mode':'mapped_J','J_eV':J,'J_meV':1000*J,
                     'passes_0p30':bool(J>=.30),'passes_0p38':bool(J>=.38)})
    (out/'result.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
    if len(energies)!=2:
        raise SystemExit(2)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--cif',required=True); ap.add_argument('--atm',type=int,required=True)
    ap.add_argument('--basin',required=True,choices=['compact','expanded'])
    ap.add_argument('--ppdir',required=True); ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x')
    ap.add_argument('--out',required=True)
    z=ap.parse_args(); run(z.cif,z.atm,z.basin,z.ppdir,z.pw,z.out)
