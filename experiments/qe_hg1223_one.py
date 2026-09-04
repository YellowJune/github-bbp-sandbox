#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import qe_quick_ligand_screen as q

# 16-atom HgBa2Ca2Cu3O8 parent cell recovered from the prior validated structure run.
A=3.892682; C=16.248444
SY=['Ba','Ba','Ca','Ca','Cu','Cu','Cu','Hg','O','O','O','O','O','O','O','O']
FR=[(.5,.5,.816083),(.5,.5,.183917),(.5,.5,.599071),(.5,.5,.400929),(0,0,.691918),(0,0,.308082),(0,0,.5),(0,0,0),(.5,0,.5),(0,.5,.5),(0,0,.876271),(0,0,.123729),(.5,0,.303695),(.5,0,.696305),(0,.5,.303695),(0,.5,.696305)]
CELL=np.diag([A,A,C])
q.MASS['Ca']=40.078

def make_variant(name):
    s=list(SY)
    if name=='parent': return s,list(FR),CELL.copy()
    cat,loc=name.split('N_',1)
    # Formal +1 compensation: La3+ for Ba2+, Bi3+/Tl3+ for Hg2+.
    if cat=='La': s[0]='La'
    elif cat in ('Bi','Tl'): s[7]=cat
    else: raise ValueError(cat)
    # Representative ligand positions: reservoir/apical O10, outer-plane O13, central-plane O8.
    oi={'apical':10,'outer':13,'center':8}[loc]
    s[oi]='N'
    return s,list(FR),CELL.copy()

p=argparse.ArgumentParser(); p.add_argument('--only',required=True);p.add_argument('--ppdir',required=True);p.add_argument('--meta',required=True);p.add_argument('--output',required=True);p.add_argument('--scratch',required=True);p.add_argument('--pw',default='/opt/espresso/7.5/pw.x');p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=p.parse_args()
var=make_variant(a.only)
parent=make_variant('parent')
q.build=(lambda:{'parent':parent}) if a.only=='parent' else (lambda:{'parent':parent,a.only:var})
# Hg1223 has a long c axis; use a matched but cheaper k mesh for the fast screen.
_orig=q.write_scf
def _write(*args,**kwargs):
    _orig(*args,**kwargs)
    path=args[0]
    txt=Path(path).read_text().replace(' 4 8 4 0 0 0',' 5 5 2 0 0 0')
    Path(path).write_text(txt)
q.write_scf=_write
q.main(SimpleNamespace(ppdir=a.ppdir,meta=a.meta,output=a.output,scratch=a.scratch,pw=a.pw,proj=a.proj))
