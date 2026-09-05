from __future__ import annotations
import json, math, os, re, subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure

RY_TO_EV=13.605693122994


def pseudo(ppdir,elem):
    e=elem.lower(); hits=[]
    for root,_,files in os.walk(ppdir):
        for fn in files:
            low=fn.lower()
            if low.endswith('.upf') and re.match(r'^'+re.escape(e)+r'[^a-z0-9]',low):
                hits.append(os.path.join(root,fn))
    if not hits: raise RuntimeError('pseudo '+elem)
    return os.path.basename(sorted(hits)[0])


def build():
    # Pbca AgF2 reference CIF reported as AgF2--Pbca-AFM.cif in literature SI:
    # a=5.500066, b=5.825495, c=5.052705 A; Ag 4b (0,0,1/2), F 8c (0.30552,0.36811,0.18355)
    lat=Lattice.orthorhombic(5.500066,5.825495,5.052705)
    return Structure.from_spacegroup('Pbca',lat,['Ag','F'],[[0,0,0.5],[0.30552,0.36811,0.18355]])


def aglabel(site):
    f=site.frac_coords%1.0
    # Pbca checkerboard partition for the 4b Ag sublattice.
    parity=(int(round(2*f[0]))+int(round(2*f[1])))%2
    return 'AgA' if parity==0 else 'AgB'


def write_in(path,s,ppdir,scratch,mode):
    labels=[aglabel(x) if x.specie.symbol=='Ag' else 'F' for x in s]
    types=[]
    for x in labels:
        if x not in types: types.append(x)
    pm={'AgA':pseudo(ppdir,'Ag'),'AgB':pseudo(ppdir,'Ag'),'F':pseudo(ppdir,'F')}
    mass={'AgA':107.8682,'AgB':107.8682,'F':18.998403163}
    L=['&CONTROL'," calculation='scf',"," prefix='agf2ctrl',",f" pseudo_dir='{ppdir}',",f" outdir='{scratch}',"," disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(types)},',' ecutwfc=45.0, ecutrho=360.0,',
       " occupations='smearing', smearing='mv', degauss=.015,",' nspin=2,']
    for i,x in enumerate(types,1):
        if x=='AgA': m=.65
        elif x=='AgB': m=.65 if mode=='FM' else -.65
        else: m=0.0
        L.append(f' starting_magnetization({i})={m},')
    L += ['/','&ELECTRONS',' conv_thr=1.d-6, mixing_beta=.25, electron_maxstep=220,','/','ATOMIC_SPECIES']
    L += [f' {x} {mass[x]:.8f} {pm[x]}' for x in types]
    L += ['HUBBARD (ortho-atomic)',' U AgA-4d 6.0',' U AgB-4d 6.0','ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labels):
        f=site.frac_coords; L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L += ['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]
    L += ['K_POINTS automatic',' 3 3 3 0 0 0']
    path.write_text('\n'.join(L)+'\n')


def last_energy(txt):
    m=re.findall(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt,re.I)
    return float(m[-1]) if m else float('nan')


def main(ppdir,out,pw='/opt/espresso/7.5/pw.x'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); s=build(); E={}; rows=[]
    s.to(filename=str(out/'AgF2_Pbca_reference.cif'))
    for mode in ['FM','AF_checker']:
        scratch=Path('/tmp')/('agf2ctrl_'+mode); scratch.mkdir(exist_ok=True)
        inp=out/(mode+'.in'); oo=out/(mode+'.out'); write_in(inp,s,ppdir,str(scratch),mode)
        with open(inp,'rb') as fi,open(oo,'wb') as fo:
            rc=subprocess.run([pw],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode
        er=last_energy(oo.read_text(errors='ignore')); ee=er*RY_TO_EV if np.isfinite(er) else None
        rows.append({'mode':mode,'scf_rc':rc,'energy_Ry':er if np.isfinite(er) else None,'energy_eV':ee})
        if rc==0 and ee is not None: E[mode]=ee
    if len(E)==2:
        # 4 Ag square-lattice Pbca cell: 8 in-plane NN bonds; spin-1/2 mapping gives DeltaE=4J.
        J=(E['FM']-E['AF_checker'])/4.0
        rows.append({'mode':'mapped_J','J_eV':J,'J_meV':1000*J,
                     'calibration_expected_meV':[25,80],
                     'calibration_pass':bool(25<=1000*J<=80)})
    (out/'result.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
    if len(E)!=2: raise SystemExit(2)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--ppdir',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x'); z=ap.parse_args(); main(z.ppdir,z.out,z.pw)
