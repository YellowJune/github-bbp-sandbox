#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,glob,os,re,subprocess
from pathlib import Path
from pymatgen.core import Structure,Lattice
import numpy as np

RY_TO_EV=13.605693122994
MASS={'Nb':92.90637,'W':183.84,'Ca':40.078,'CuA':63.546,'CuB':63.546,'C':12.011,'N':14.007,'O':15.999}
BASE={'Nb':'Nb','W':'W','Ca':'Ca','CuA':'Cu','CuB':'Cu','C':'C','N':'N','O':'O'}

def pmap(ppdir,meta,labs):
    md=json.load(open(meta));out={}
    for lab in labs:
        e=BASE[lab];fn=md.get(e,{}).get('filename') if isinstance(md.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)):out[lab]=fn;continue
        fs=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:fs+=glob.glob(os.path.join(ppdir,pat))
        if not fs:raise RuntimeError(f'no pseudo {e}')
        out[lab]=os.path.basename(sorted(fs)[0])
    return out

def control():
    a,c=3.86,3.20;sy=[];fr=[]
    for y in range(2):
        for x in range(3):
            sy += ['Ca','Cu','O','O']
            fr += [[(x+.5)/3,(y+.5)/2,.5],[x/3,y/2,0],[(x+.5)/3,y/2,0],[x/3,(y+.5)/2,0]]
    return Structure(Lattice.orthorhombic(3*a,2*a,c),sy,fr)

def labels(s):
    out=[]
    for site in s:
        e=site.specie.symbol
        if e!='Cu':out.append(e);continue
        f=site.frac_coords;ix=int(round((f[0]%1)*3))%3;iy=int(round((f[1]%1)*2))%2
        out.append('CuA' if (ix+iy)%2==0 else 'CuB')
    return out

def write_input(path,prefix,s,labs,pm,ppdir,outdir,mode):
    sp=[]
    for x in labs:
        if x not in sp:sp.append(x)
    a,b,c=s.lattice.abc;k=(max(2,min(4,round(18/a))),max(2,min(5,round(18/b))),max(4,min(10,round(24/c))))
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(sp)},',' ecutwfc=65.0, ecutrho=520.0,'," occupations='smearing', smearing='mv', degauss=.012,",' nspin=2,']
    for i,x in enumerate(sp,1):
        if x=='CuA':mag=.65
        elif x=='CuB':mag=.65 if mode=='FM' else -.65
        elif x in {'C','N','O'}:mag=.02
        else:mag=0.0
        L.append(f' starting_magnetization({i})={mag:.3f},')
    L+=['/','&ELECTRONS',' conv_thr=5.d-9, mixing_beta=.14, electron_maxstep=500,','/','HUBBARD (ortho-atomic)',' U CuA-3d 6.000000',' U CuB-3d 6.000000','ATOMIC_SPECIES']
    L += [f' {x} {MASS[x]:.7f} {pm[x]}' for x in sp]
    L += ['ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labs):
        f=site.frac_coords;L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L += ['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]+['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def last(pat,txt):
    m=re.findall(pat,txt,re.I);return float(m[-1]) if m else np.nan

def calc(label,s,ppdir,meta,scratch,out,pw):
    labs=labels(s);pm=pmap(ppdir,meta,sorted(set(labs)));ans={}
    for mode in ['AFM','FM']:
        od=Path(scratch)/f'{label}_{mode}';od.mkdir(parents=True,exist_ok=True)
        inp=Path(out)/f'{label}_{mode}.in';oo=Path(out)/f'{label}_{mode}.out'
        write_input(inp,f'{label}_{mode}',s,labs,pm,ppdir,str(od),mode)
        rc=run(pw,str(inp),str(oo));rec={'rc':rc}
        if rc==0:
            txt=oo.read_text(errors='ignore')
            rec.update({'energy_Ry':last(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt),
                        'total_magnetization':last(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt),
                        'absolute_magnetization':last(r'absolute magnetization\s+=\s+([-0-9.Ee+]+)',txt)})
        ans[mode]=rec
    return ans

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cif',required=True);ap.add_argument('--name',required=True);ap.add_argument('--pressure-atm',type=float,required=True);ap.add_argument('--ppdir',required=True);ap.add_argument('--meta',required=True);ap.add_argument('--scratch',required=True);ap.add_argument('--out',required=True);ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cand=Structure.from_file(a.cif);ctrl=control()
    cr=calc('control',ctrl,a.ppdir,a.meta,a.scratch,out,a.pw);rr=calc('candidate',cand,a.ppdir,a.meta,a.scratch,out,a.pw)
    res={'candidate':a.name,'pressure_atm':a.pressure_atm,'control':cr,'candidate_states':rr}
    ok=all(x.get('rc')==0 for x in [cr['AFM'],cr['FM'],rr['AFM'],rr['FM']])
    if ok:
        dc=(cr['FM']['energy_Ry']-cr['AFM']['energy_Ry'])*RY_TO_EV
        dr=(rr['FM']['energy_Ry']-rr['AFM']['energy_Ry'])*RY_TO_EV
        # both are 3x2 Cu planes with twelve nearest-neighbor bonds
        ratio=dr/dc if abs(dc)>1e-8 else np.nan
        J=.130*ratio;Teq=133.5*ratio
        res.update({'deltaE_control_FMminusAFM_eV':dc,'deltaE_candidate_FMminusAFM_eV':dr,
                    'exchange_ratio_to_CaCuO2':ratio,'J_calibrated_eV':J,
                    'Tc_scale_if_parent_efficiency_K':Teq,'passes_300K_exchange_scale':bool(np.isfinite(Teq) and Teq>=300),
                    'magnetic_state_sanity':bool(abs(cr['AFM'].get('total_magnetization',99))<2.0 and abs(rr['AFM'].get('total_magnetization',99))<2.5 and rr['FM'].get('absolute_magnetization',0)>rr['AFM'].get('absolute_magnetization',0)*.5)})
    (out/'exchange_result.json').write_text(json.dumps(res,indent=2,default=float));print(json.dumps(res,indent=2,default=float))
if __name__=='__main__':main()
