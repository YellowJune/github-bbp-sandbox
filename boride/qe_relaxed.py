#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

MASS={'Sc':44.95591,'Ti':47.867,'CuA':63.546,'CuB':63.546,'B':10.81}
BASE={'Sc':'Sc','Ti':'Ti','CuA':'Cu','CuB':'Cu','B':'B'}

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

def labels(s):
    labs=[]
    for site in s:
        e=site.specie.symbol
        if e!='Cu':labs.append(e);continue
        f=site.frac_coords;ix=int(round((f[0]%1)*3))%3;iy=int(round((f[1]%1)*2))%2
        labs.append('CuA' if (ix+iy)%2==0 else 'CuB')
    return labs

def write_scf(path,prefix,s,labs,pm,ppdir,outdir,U):
    sp=[]
    for x in labs:
        if x not in sp:sp.append(x)
    a,b,c=s.lattice.abc;k=(max(2,min(4,round(18/a))),max(2,min(5,round(18/b))),max(4,min(10,round(24/c))))
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(sp)},',' ecutwfc=60.0, ecutrho=480.0,'," occupations='smearing', smearing='mv', degauss=.016,",' nspin=2,']
    for i,x in enumerate(sp,1):
        mag=.58 if x=='CuA' else (-.58 if x=='CuB' else (.03 if x=='B' else 0.0));L.append(f' starting_magnetization({i})={mag:.3f},')
    L+=['/','&ELECTRONS',' conv_thr=1.d-8, mixing_beta=.18, electron_maxstep=420,','/']
    if U>0:L+=['HUBBARD (ortho-atomic)',f' U CuA-3d {U:.6f}',f' U CuB-3d {U:.6f}']
    L+=['ATOMIC_SPECIES']+[f' {x} {MASS[x]:.7f} {pm[x]}' for x in sp]+['ATOMIC_POSITIONS crystal']
    for site,lab in zip(s,labs):
        f=site.frac_coords;L.append(f' {lab} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}')
    L+=['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]+['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def num(pat,txt):
    m=re.findall(pat,txt,re.I);return float(m[-1]) if m else np.nan

def sumorb(prefix,atoms,kind):
    E=None;y=None;n=0
    for atom in atoms:
        for f in glob.glob(prefix+f'.pdos_atm#*({atom})_wfc#*({kind})'):
            a=np.atleast_2d(np.loadtxt(f,comments='#'));yy=a[:,1:].sum(1)
            if E is None:E=a[:,0];y=yy.copy()
            elif len(E)==len(a):y+=yy
            n+=1
    return E,y,n

def metrics(prefix,ef):
    E,cu,ncu=sumorb(prefix,['CuA','CuB'],'d');EB,b,nb=sumorb(prefix,['B'],'p')
    if E is None or b is None or not np.isfinite(ef):return {}
    tr=np.trapezoid
    def integ(y,lo,hi):
        m=(E>=ef+lo)&(E<=ef+hi);return float(tr(y[m],E[m])) if m.sum()>=2 else np.nan
    mc=(E>=ef-4)&(E<=ef+1)
    def cent(y):
        den=tr(y[mc],E[mc]);return float(tr(E[mc]*y[mc],E[mc])/den) if abs(den)>1e-12 else np.nan
    cd,bp=cent(cu),cent(b)
    return {'cu_d_pm0p10':integ(cu,-.10,.10),'B_p_pm0p10':integ(b,-.10,.10),'cu_d_pm0p25':integ(cu,-.25,.25),'B_p_pm0p25':integ(b,-.25,.25),'pd_hybrid_pm0p25':integ(np.sqrt(np.clip(cu,0,None)*np.clip(b,0,None)),-.25,.25),'pd_hybrid_pm0p50':integ(np.sqrt(np.clip(cu,0,None)*np.clip(b,0,None)),-.50,.50),'cu_d_centroid_eV':cd,'B_p_centroid_eV':bp,'pd_centroid_sep_eV':abs(cd-bp),'nCuD_channels':ncu,'nBp_channels':nb}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cif',required=True);ap.add_argument('--engine',required=True);ap.add_argument('--pressure-atm',type=float,required=True);ap.add_argument('--ppdir',required=True);ap.add_argument('--meta',required=True);ap.add_argument('--scratch',required=True);ap.add_argument('--out',required=True);ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x');ap.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=ap.parse_args()
    s=Structure.from_file(a.cif);labs=labels(s);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True);pm=pmap(a.ppdir,a.meta,sorted(set(labs)));rows=[]
    for U in [0.,6.]:
        tag=f'U{int(U)}';od=scratch/tag;od.mkdir(exist_ok=True);pref=f'boride_{a.engine}_{int(a.pressure_atm)}_{tag}'
        inp=out/f'scf_{tag}.in';oo=out/f'scf_{tag}.out';write_scf(inp,pref,s,labs,pm,a.ppdir,str(od),U)
        rc=run(a.pw,str(inp),str(oo));rec={'engine_geometry':a.engine,'pressure_atm':a.pressure_atm,'U_Cu_eV':U,'scf_rc':rc,'natoms':len(s)}
        if rc==0:
            txt=oo.read_text(errors='ignore');ef=num(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt)
            rec.update({'fermi_eV':ef,'energy_Ry':num(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt),'total_magnetization':num(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt),'absolute_magnetization':num(r'absolute magnetization\s+=\s+([-0-9.Ee+]+)',txt)})
            pi=out/f'proj_{tag}.in';po=out/f'proj_{tag}.out';fil=str(od/'pdos');pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{od}',\n filpdos='{fil}',\n DeltaE=.03, degauss=.012, ngauss=0,\n/\n")
            prc=run(a.proj,str(pi),str(po));rec['proj_rc']=prc
            if prc==0:rec.update(metrics(fil,ef))
        rows.append(rec);print(json.dumps(rec,default=float))
    keys=sorted({k for r in rows for k in r})
    with open(out/'qe_electronic.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
if __name__=='__main__':main()
