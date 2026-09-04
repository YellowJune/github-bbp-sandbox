#!/usr/bin/env python3
import argparse
from types import SimpleNamespace
import qe_quick_ligand_screen as q

p=argparse.ArgumentParser()
p.add_argument('--only',required=True)
p.add_argument('--ppdir',required=True)
p.add_argument('--meta',required=True)
p.add_argument('--output',required=True)
p.add_argument('--scratch',required=True)
p.add_argument('--pw',default='/opt/espresso/7.5/pw.x')
p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x')
a=p.parse_args()
allv=q.build()
if a.only not in allv:
    raise SystemExit(f'unknown variant {a.only}: {sorted(allv)}')
q.build=lambda:{a.only:allv[a.only]}
q.main(SimpleNamespace(ppdir=a.ppdir,meta=a.meta,output=a.output,scratch=a.scratch,pw=a.pw,proj=a.proj))
