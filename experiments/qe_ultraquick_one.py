#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace
import qe_quick_ligand_screen as q


def ultra_write(path,prefix,sy,fr,cell,pm,ppdir,outdir):
    species=[]
    for s in sy:
        if s not in species: species.append(s)
    L=[
      '&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',",
      " tstress=.true., tprnfor=.true., disk_io='low',",'/',
      '&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(species)},',
      " ecutwfc=30.0, ecutrho=240.0, occupations='smearing', smearing='mv', degauss=0.03, nspin=2,"
    ]
    for i,s in enumerate(species,1):
        L.append(f" starting_magnetization({i})={0.45 if s=='Cu' else (0.08 if s=='N' else 0.0):.2f},")
    L += ['/', '&ELECTRONS', ' conv_thr=1.0d-6, mixing_beta=0.35, electron_maxstep=120,', '/', 'ATOMIC_SPECIES']
    for s in species: L.append(f' {s} {q.MASS[s]:.8f} {pm[s]}')
    L.append('ATOMIC_POSITIONS crystal')
    for s,p in zip(sy,fr): L.append(f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}')
    L.append('CELL_PARAMETERS angstrom')
    for v in cell: L.append(' %.10f %.10f %.10f'%tuple(v))
    L += ['K_POINTS automatic',' 2 4 2 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

p=argparse.ArgumentParser(); p.add_argument('--only',required=True); p.add_argument('--ppdir',required=True); p.add_argument('--meta',required=True); p.add_argument('--output',required=True); p.add_argument('--scratch',required=True); p.add_argument('--pw',default='/opt/espresso/7.5/pw.x'); p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x'); a=p.parse_args()
allv=q.build()
if a.only not in allv: raise SystemExit(f'unknown variant {a.only}: {sorted(allv)}')
q.build=lambda:{a.only:allv[a.only]}
q.write_scf=ultra_write
q.main(SimpleNamespace(ppdir=a.ppdir,meta=a.meta,output=a.output,scratch=a.scratch,pw=a.pw,proj=a.proj))
