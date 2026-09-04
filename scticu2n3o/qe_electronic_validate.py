#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np
MASS={'Ca':40.078,'Sc':44.95591,'Ti':47.867,'Cu':63.546,'N':14.007,'O':15.999}

def build(name):
    if name=='CaCuO2':
        a,c=3.86,3.20;sy=[];fr=[]
        for ix in range(2):
            for iy in range(2):
                sy+=['Ca','Cu','O','O'];fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
        return sy,np.array(fr),np.diag([2*a,2*a,c]),(4,4,8)
    if name=='Cu3N':
        a=3.82;return ['N','Cu','Cu','Cu'],np.array([[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]]),np.diag([a,a,a]),(8,8,8)
    if name=='ScTiCu2N3O':
        a,c=3.72,3.00
        sy=['Sc','Cu','O','N','Ti','Cu','N','N']
        fr=np.array([[.25,.5,.5],[0,0,0],[.25,0,0],[0,.5,0],[.75,.5,.5],[.5,0,0],[.75,0,0],[.5,.5,0]])
        return sy,fr,np.diag([2*a,a,c]),(4,8,8)
    raise ValueError(name)

def pmap(ppdir,meta,elements):
    md=json.load(open(meta));out={}
    for e in elements:
        fn=md.get(e,{}).get('filename') if isinstance(md.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)):out[e]=fn;continue
        fs=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:fs+=glob.glob(os.path.join(ppdir,pat))
        if not fs:raise RuntimeError(f'no pseudo {e}')
        out[e]=os.path.basename(sorted(fs)[0])
    return out

def write(path,prefix,sy,fr,cell,k,pm,ppdir,outdir,U):
    sp=[]
    for s in sy:
        if s not in sp:sp.append(s)
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(sp)},',' ecutwfc=55.0, ecutrho=440.0,'," occupations='smearing', smearing='mv', degauss=.020,",' nspin=2,']
    for i,s in enumerate(sp,1):L.append(f" starting_magnetization({i})={.55 if s=='Cu' else (.08 if s=='N' else 0):.3f},")
    L+=['/','&ELECTRONS',' conv_thr=1.d-8, mixing_beta=.25, electron_maxstep=260,','/']
    if U>0:L+=['HUBBARD (ortho-atomic)',f' U Cu-3d {U:.6f}']
    L+=['ATOMIC_SPECIES']+[f' {s} {MASS[s]:.7f} {pm[s]}' for s in sp]+['ATOMIC_POSITIONS crystal']
    L += [f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}' for s,p in zip(sy,fr)]
    L+=['CELL_PARAMETERS angstrom']+[' %.10f %.10f %.10f'%tuple(v) for v in cell]+['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def parse_num(pattern,txt):
    m=re.findall(pattern,txt,re.I);return float(m[-1]) if m else np.nan

def orb(prefix,el,kind):
    fs=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({kind})');E=None;y=None
    for f in fs:
        a=np.atleast_2d(np.loadtxt(f,comments='#'));yy=a[:,1:].sum(1)
        if E is None:E=a[:,0];y=yy
        elif len(E)==len(a):y+=yy
    return E,y,len(fs)

def metrics(prefix,ef):
    E,cu,ncu=orb(prefix,'Cu','d');EN,n,nn=orb(prefix,'N','p');EO,o,no=orb(prefix,'O','p')
    if E is None or not np.isfinite(ef):return {}
    lig=np.zeros_like(cu)
    if n is not None and len(n)==len(E):lig+=n
    if o is not None and len(o)==len(E):lig+=o
    trap=np.trapezoid;m=(E>=ef-.6)&(E<=ef+.6);m2=(E>=ef-4)&(E<=ef+.5)
    def cent(y):
        d=trap(y[m2],E[m2]);return float(trap(E[m2]*y[m2],E[m2])/d) if abs(d)>1e-12 else np.nan
    cd,lp=cent(cu),cent(lig)
    return {'cu_d_EFwin':float(trap(cu[m],E[m])),'ligand_p_EFwin':float(trap(lig[m],E[m])),'pd_hybrid_overlap':float(trap(np.sqrt(np.clip(cu[m],0,None)*np.clip(lig[m],0,None)),E[m])),'cu_d_centroid_eV':cd,'ligand_p_centroid_eV':lp,'pd_centroid_sep_eV':abs(cd-lp),'nCuD':ncu,'nNp':nn,'nOp':no}

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['CaCuO2','Cu3N','ScTiCu2N3O'],required=True);p.add_argument('--ppdir',required=True);p.add_argument('--meta',required=True);p.add_argument('--scratch',required=True);p.add_argument('--out',required=True);p.add_argument('--pw',default='/opt/espresso/7.5/pw.x');p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    sy,fr,cell,k=build(a.name);pm=pmap(a.ppdir,a.meta,sorted(set(sy)));rows=[]
    for U in [0.,6.]:
        tag=f'U{int(U)}';od=scratch/tag;od.mkdir(exist_ok=True);pref=f'{a.name.lower()}_{tag}'
        inp=out/f'scf_{tag}.in';oo=out/f'scf_{tag}.out';write(inp,pref,sy,fr,cell,k,pm,a.ppdir,str(od),U)
        rc=run(a.pw,str(inp),str(oo));rec={'name':a.name,'U_Cu_eV':U,'scf_rc':rc}
        if rc==0:
            txt=oo.read_text(errors='ignore');ef=parse_num(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt)
            rec.update({'fermi_eV':ef,'energy_Ry':parse_num(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt),'magnetization':parse_num(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt)})
            pi=out/f'proj_{tag}.in';po=out/f'proj_{tag}.out';fil=str(od/'pdos');pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{od}',\n filpdos='{fil}',\n DeltaE=.04, degauss=.015, ngauss=0,\n/\n")
            prc=run(a.proj,str(pi),str(po));rec['proj_rc']=prc
            if prc==0:rec.update(metrics(fil,ef))
        rows.append(rec);print(json.dumps(rec,default=float))
    keys=sorted({k for r in rows for k in r})
    with open(out/'qe_electronic.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
if __name__=='__main__':main()
