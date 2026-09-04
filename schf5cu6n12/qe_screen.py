#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np

MASS={'Hf':178.49,'Sc':44.955908,'Y':88.90584,'Cu':63.546,'N':14.007}
A0=3.80; C0=3.30

def build(name):
    sy=[]; fr=[]; tags=[]
    for ix in range(3):
        for iy in range(2):
            sy += ['Hf','Cu','N','N']
            fr += [[(ix+.5)/3,(iy+.5)/2,.5],[ix/3,iy/2,0],[(ix+.5)/3,iy/2,0],[ix/3,(iy+.5)/2,0]]
            tags += [('A',ix,iy),('Cu',ix,iy),('Nx',ix,iy),('Ny',ix,iy)]
    ai=tags.index(('A',0,0))
    if name=='Hf6Cu6N12': pass
    elif name=='ScHf5Cu6N12': sy[ai]='Sc'
    elif name=='YHf5Cu6N12': sy[ai]='Y'
    else: raise ValueError(name)
    cell=np.diag([3*A0,2*A0,C0])
    return sy,fr,cell

def pmap(ppdir,meta_path,elements):
    meta=json.load(open(meta_path)); out={}
    for e in elements:
        fn=meta.get(e,{}).get('filename') if isinstance(meta.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)): out[e]=fn; continue
        cand=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:
            cand += glob.glob(os.path.join(ppdir,pat))
        if not cand: raise RuntimeError(f'no pseudo for {e}')
        out[e]=os.path.basename(sorted(cand)[0])
    return out

def write_scf(path,prefix,sy,fr,cell,pm,ppdir,outdir):
    species=[]
    for s in sy:
        if s not in species: species.append(s)
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/',
       '&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(species)},',' ecutwfc=50.0, ecutrho=500.0,'," occupations='smearing', smearing='mv', degauss=0.020,",' nspin=2,']
    for i,s in enumerate(species,1):
        mag=.50 if s=='Cu' else (.08 if s in {'N','Sc'} else 0.0)
        L.append(f' starting_magnetization({i})={mag:.3f},')
    L += ['/', '&ELECTRONS',' conv_thr=2.0d-8, mixing_beta=0.25, electron_maxstep=240,','/','ATOMIC_SPECIES']
    for s in species: L.append(f' {s} {MASS[s]:.8f} {pm[s]}')
    L.append('ATOMIC_POSITIONS crystal')
    for s,p in zip(sy,fr): L.append(f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}')
    L.append('CELL_PARAMETERS angstrom')
    for v in cell: L.append(' %.10f %.10f %.10f'%tuple(v))
    L += ['K_POINTS automatic',' 3 4 4 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:
        return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def last_float(text,pattern):
    m=re.findall(pattern,text,re.I); return float(m[-1]) if m else np.nan

def read_orb(prefix,el,orb):
    fs=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({orb})')
    E=None; y=None
    for f in fs:
        a=np.atleast_2d(np.loadtxt(f,comments='#'))
        yy=a[:,1:].sum(axis=1)
        if E is None: E=a[:,0]; y=yy
        elif len(E)==len(a): y += yy
    return E,y,len(fs)

def pdos_metrics(prefix,ef):
    E,cu,ncu=read_orb(prefix,'Cu','d'); EN,n,nn=read_orb(prefix,'N','p')
    if E is None or EN is None or not np.isfinite(ef): return {}
    if not np.allclose(E,EN): return {}
    out={'n_Cu_d_files':ncu,'n_N_p_files':nn}
    for w in [.05,.10,.20,.40,.60]:
        m=(E>=ef-w)&(E<=ef+w)
        if m.sum()<2: continue
        trap=np.trapezoid
        c=float(trap(cu[m],E[m])); p=float(trap(n[m],E[m]))
        h=float(trap(np.sqrt(np.clip(cu[m],0,None)*np.clip(n[m],0,None)),E[m]))
        tag=str(w).replace('.','p')
        out[f'Cu_d_pm{tag}_eV']=c; out[f'N_p_pm{tag}_eV']=p; out[f'CuN_geom_overlap_pm{tag}_eV']=h
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--name',required=True); ap.add_argument('--ppdir',required=True); ap.add_argument('--meta',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--scratch',required=True); ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x'); ap.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); scratch=Path(a.scratch); scratch.mkdir(parents=True,exist_ok=True)
    sy,fr,cell=build(a.name); els=sorted(set(sy)); pm=pmap(a.ppdir,a.meta,els); pref='schf_'+a.name
    inp=out/'scf.in'; oo=out/'scf.out'; write_scf(inp,pref,sy,fr,cell,pm,a.ppdir,str(scratch))
    rc=run(a.pw,inp,oo); rec={'name':a.name,'formula':'-'.join(sy),'scf_rc':rc}
    if rc==0:
        txt=oo.read_text(errors='ignore'); rec['scf_converged']='convergence has been achieved' in txt
        rec['fermi_eV']=last_float(txt,r'the Fermi energy is\s+([-0-9.Ee+]+)')
        rec['total_magnetization_Bohr']=last_float(txt,r'total magnetization\s+=\s+([-0-9.Ee+]+)\s+Bohr')
        rec['absolute_magnetization_Bohr']=last_float(txt,r'absolute magnetization\s+=\s+([-0-9.Ee+]+)\s+Bohr')
        forces=[]; conv=13.605693122994/0.529177210903
        for m in re.finditer(r'atom\s+\d+\s+type\s+\d+\s+force\s+=\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)',txt,re.I):
            forces.append(np.linalg.norm([float(m.group(i))*conv for i in (1,2,3)]))
        rec['max_residual_force_eV_A']=float(max(forces)) if forces else np.nan
        pi=out/'proj.in'; po=out/'proj.out'; fil=str(scratch/'pdos')
        pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{scratch}',\n filpdos='{fil}',\n DeltaE=0.025, degauss=0.01, ngauss=0,\n/\n")
        rc2=run(a.proj,pi,po); rec['proj_rc']=rc2
        if rc2==0: rec.update(pdos_metrics(fil,rec['fermi_eV']))
    (out/'result.json').write_text(json.dumps(rec,indent=2,default=float)); print(json.dumps(rec,indent=2,default=float))

if __name__=='__main__': main()
