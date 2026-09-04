#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, glob, json, os, re, subprocess
from pathlib import Path
import numpy as np

MASS={'Ca':40.078,'Y':88.90584,'Hf':178.49,'Cu':63.546,'N':14.007,'O':15.999}

def build(name):
    if name=='CaCuO2':
        a,c=3.86,3.20; sy=[]; fr=[]
        for ix in range(2):
            for iy in range(2):
                sy += ['Ca','Cu','O','O']
                fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
        cell=np.diag([2*a,2*a,c]); k=(4,4,8)
    elif name=='YHf3Cu4N7O':
        a,c=3.82,3.28; sy=[]; fr=[]; tags=[]
        for ix in range(2):
            for iy in range(2):
                sy += ['Hf','Cu','N','N']
                fr += [[(ix+.5)/2,(iy+.5)/2,.5],[ix/2,iy/2,0],[(ix+.5)/2,iy/2,0],[ix/2,(iy+.5)/2,0]]
                tags += [('A',ix,iy),('Cu',ix,iy),('Lx',ix,iy),('Ly',ix,iy)]
        sy[tags.index(('A',0,0))]='Y'
        sy[tags.index(('Lx',0,0))]='O'
        cell=np.diag([2*a,2*a,c]); k=(4,4,8)
    elif name=='Cu3N':
        a=3.82; sy=['N','Cu','Cu','Cu']; fr=[[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]]
        cell=np.diag([a,a,a]); k=(8,8,8)
    else: raise ValueError(name)
    return sy,np.array(fr,float),cell,k

def pmap(ppdir,meta_path,elements):
    meta=json.load(open(meta_path)); out={}
    for e in elements:
        fn=meta.get(e,{}).get('filename') if isinstance(meta.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)):
            out[e]=fn; continue
        fs=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:
            fs += glob.glob(os.path.join(ppdir,pat))
        if not fs: raise RuntimeError(f'No pseudopotential for {e}')
        out[e]=os.path.basename(sorted(fs)[0])
    return out

def write_scf(path,prefix,sy,fr,cell,kgrid,pm,ppdir,outdir,U):
    species=[]
    for s in sy:
        if s not in species: species.append(s)
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(species)},',' ecutwfc=55.0, ecutrho=440.0,'," occupations='smearing', smearing='mv', degauss=0.020,",' nspin=2,']
    for i,s in enumerate(species,1):
        L.append(f" starting_magnetization({i})={0.55 if s=='Cu' else (0.08 if s=='N' else 0.0):.3f},")
    L += ['/','&ELECTRONS',' conv_thr=1.0d-8, mixing_beta=0.25, electron_maxstep=260,','/']
    if U>0:
        L += ['HUBBARD (ortho-atomic)',f' U Cu-3d {U:.6f}']
    L += ['ATOMIC_SPECIES']
    for s in species: L.append(f' {s} {MASS[s]:.8f} {pm[s]}')
    L += ['ATOMIC_POSITIONS crystal']
    for s,p in zip(sy,fr): L.append(f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}')
    L += ['CELL_PARAMETERS angstrom']
    for v in cell: L.append(' %.10f %.10f %.10f'%tuple(v))
    L += ['K_POINTS automatic',f' {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi, open(out,'wb') as fo:
        return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def fermi(txt):
    m=re.findall(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt,re.I)
    if m:return float(m[-1])
    m=re.findall(r'highest occupied level.*?([-0-9.Ee+]+)',txt,re.I)
    return float(m[-1]) if m else np.nan

def energy(txt):
    m=re.findall(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt)
    return float(m[-1])*13.605693122994 if m else np.nan

def magnetization(txt):
    m=re.findall(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt,re.I)
    return float(m[-1]) if m else np.nan

def stress_kbar(txt):
    # final total stress line, kbar
    m=re.findall(r'total\s+stress\s+\(Ry/bohr\*\*3\)\s+=\s+[-0-9.Ee+]+\s+\(kbar\)\s+([-0-9.Ee+]+)',txt,re.I)
    return float(m[-1]) if m else np.nan

def read_orb(prefix,el,orb):
    files=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({orb})')
    E=None; y=None
    for f in files:
        a=np.atleast_2d(np.loadtxt(f,comments='#'))
        yy=a[:,1:].sum(1)
        if E is None: E=a[:,0]; y=yy
        elif len(E)==len(a): y+=yy
    return E,y,len(files)

def trap(y,x): return float(np.trapezoid(y,x))

def metrics(prefix,ef):
    E,cu,ncu=read_orb(prefix,'Cu','d')
    EN,n,nn=read_orb(prefix,'N','p')
    EO,o,no=read_orb(prefix,'O','p')
    if E is None or not np.isfinite(ef): return {}
    lig=np.zeros_like(cu)
    if n is not None and len(n)==len(E): lig+=n
    if o is not None and len(o)==len(E): lig+=o
    m=(E>=ef-.60)&(E<=ef+.60)
    m2=(E>=ef-4.0)&(E<=ef+.5)
    def centroid(y):
        den=trap(y[m2],E[m2]); return trap(E[m2]*y[m2],E[m2])/den if abs(den)>1e-12 else np.nan
    cd=centroid(cu); lp=centroid(lig)
    return {
      'cu_d_EFwin':trap(cu[m],E[m]),'ligand_p_EFwin':trap(lig[m],E[m]),
      'pd_hybrid_overlap':trap(np.sqrt(np.clip(cu[m],0,None)*np.clip(lig[m],0,None)),E[m]),
      'cu_d_centroid_eV':cd,'ligand_p_centroid_eV':lp,'pd_centroid_sep_eV':abs(cd-lp),
      'n_Cu_d_projectors':ncu,'n_N_p_projectors':nn,'n_O_p_projectors':no}

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['CaCuO2','Cu3N','YHf3Cu4N7O']);p.add_argument('--ppdir',required=True);p.add_argument('--meta',required=True);p.add_argument('--scratch',required=True);p.add_argument('--out',required=True);p.add_argument('--pw',default='/opt/espresso/7.5/pw.x');p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    sy,fr,cell,k=build(a.name);pm=pmap(a.ppdir,a.meta,sorted(set(sy)))
    rows=[]
    for U in [0.0,6.0]:
        tag=f'U{int(U)}'; od=scratch/tag;od.mkdir(exist_ok=True);pref=f'{a.name.lower()}_{tag}'
        inp=out/f'scf_{tag}.in';oo=out/f'scf_{tag}.out';write_scf(inp,pref,sy,fr,cell,k,pm,a.ppdir,str(od),U)
        rc=run(a.pw,str(inp),str(oo));rec={'name':a.name,'U_Cu_eV':U,'scf_rc':rc,'natoms':len(sy)}
        if rc==0:
            txt=oo.read_text(errors='ignore');ef=fermi(txt)
            rec.update({'fermi_eV':ef,'energy_eV_cell':energy(txt),'magnetization':magnetization(txt),'stress_kbar_scalar':stress_kbar(txt)})
            pi=out/f'proj_{tag}.in';po=out/f'proj_{tag}.out';fil=str(od/'pdos')
            pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{od}',\n filpdos='{fil}',\n DeltaE=0.04, degauss=0.015, ngauss=0,\n/\n")
            prc=run(a.proj,str(pi),str(po));rec['proj_rc']=prc
            if prc==0: rec.update(metrics(fil,ef))
        rows.append(rec);print(json.dumps(rec,default=float))
    keys=sorted({k for r in rows for k in r})
    with open(out/'qe_electronic.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
if __name__=='__main__':main()
