#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

MASS={'Sc':44.95591,'Ti':47.867,'Cu':63.546,'N':14.007,'O':15.999}

def pmap(ppdir,meta,elements):
    md=json.load(open(meta)); out={}
    for e in elements:
        fn=md.get(e,{}).get('filename') if isinstance(md.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)): out[e]=fn; continue
        fs=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']: fs+=glob.glob(os.path.join(ppdir,pat))
        if not fs: raise RuntimeError(f'no pseudo {e}')
        out[e]=os.path.basename(sorted(fs)[0])
    return out

def write_scf(path,prefix,s,pm,ppdir,outdir,U):
    sy=[site.specie.symbol for site in s]; sp=[]
    for x in sy:
        if x not in sp: sp.append(x)
    a,b,c=s.lattice.abc
    # reciprocal resolution comparable across relaxed cells, capped for CPU runtime
    k=(max(2,min(5,round(15/a))),max(2,min(6,round(15/b))),max(4,min(10,round(24/c))))
    L=['&CONTROL'," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(s)}, ntyp={len(sp)},',' ecutwfc=55.0, ecutrho=440.0,'," occupations='smearing', smearing='mv', degauss=.018,",' nspin=2,']
    for i,x in enumerate(sp,1): L.append(f" starting_magnetization({i})={.55 if x=='Cu' else (.04 if x=='N' else 0):.3f},")
    L+=['/','&ELECTRONS',' conv_thr=1.d-8, mixing_beta=.22, electron_maxstep=320,','/']
    if U>0: L+=['HUBBARD (ortho-atomic)',f' U Cu-3d {U:.6f}']
    L+=['ATOMIC_SPECIES']+[f' {x} {MASS[x]:.7f} {pm[x]}' for x in sp]+['ATOMIC_POSITIONS crystal']
    for site in s: L.append(f' {site.specie.symbol} {site.frac_coords[0]:.12f} {site.frac_coords[1]:.12f} {site.frac_coords[2]:.12f}')
    L+=['CELL_PARAMETERS angstrom']+[' %.12f %.12f %.12f'%tuple(v) for v in s.lattice.matrix]+['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo: return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def num(pat,txt):
    m=re.findall(pat,txt,re.I); return float(m[-1]) if m else np.nan

def orb(prefix,el,kind):
    fs=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({kind})'); E=None; y=None
    for f in fs:
        a=np.atleast_2d(np.loadtxt(f,comments='#')); yy=a[:,1:].sum(1)
        if E is None: E=a[:,0]; y=yy
        elif len(E)==len(a): y+=yy
    return E,y,len(fs)

def metrics(prefix,ef):
    E,cu,ncu=orb(prefix,'Cu','d'); EN,n,nn=orb(prefix,'N','p'); EO,o,no=orb(prefix,'O','p')
    if E is None or not np.isfinite(ef): return {}
    lig=np.zeros_like(cu)
    if n is not None and len(n)==len(E): lig+=n
    if o is not None and len(o)==len(E): lig+=o
    tr=np.trapezoid; mw=(E>=ef-.6)&(E<=ef+.6); mc=(E>=ef-4)&(E<=ef+.5)
    def cent(y):
        d=tr(y[mc],E[mc]); return float(tr(E[mc]*y[mc],E[mc])/d) if abs(d)>1e-12 else np.nan
    cd,lp=cent(cu),cent(lig)
    return {'cu_d_EFwin':float(tr(cu[mw],E[mw])),'ligand_p_EFwin':float(tr(lig[mw],E[mw])),'pd_hybrid_overlap':float(tr(np.sqrt(np.clip(cu[mw],0,None)*np.clip(lig[mw],0,None)),E[mw])),'cu_d_centroid_eV':cd,'ligand_p_centroid_eV':lp,'pd_centroid_sep_eV':abs(cd-lp),'nCuD':ncu,'nNp':nn,'nOp':no}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cif',required=True);ap.add_argument('--pressure-atm',type=float,required=True);ap.add_argument('--ppdir',required=True);ap.add_argument('--meta',required=True);ap.add_argument('--scratch',required=True);ap.add_argument('--out',required=True);ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x');ap.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=ap.parse_args()
    s=Structure.from_file(a.cif); out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    pm=pmap(a.ppdir,a.meta,sorted({x.specie.symbol for x in s})); rows=[]
    for U in [0.,6.]:
        tag=f'U{int(U)}'; od=scratch/tag;od.mkdir(exist_ok=True);pref=f'sc3_{int(a.pressure_atm)}atm_{tag}'
        inp=out/f'scf_{tag}.in';oo=out/f'scf_{tag}.out';write_scf(inp,pref,s,pm,a.ppdir,str(od),U)
        rc=run(a.pw,str(inp),str(oo)); rec={'pressure_atm':a.pressure_atm,'U_Cu_eV':U,'scf_rc':rc,'natoms':len(s)}
        if rc==0:
            txt=oo.read_text(errors='ignore'); ef=num(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt)
            rec.update({'fermi_eV':ef,'energy_Ry':num(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt),'magnetization':num(r'total magnetization\s+=\s+([-0-9.Ee+]+)',txt)})
            pi=out/f'proj_{tag}.in';po=out/f'proj_{tag}.out';fil=str(od/'pdos');pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{od}',\n filpdos='{fil}',\n DeltaE=.04, degauss=.015, ngauss=0,\n/\n")
            prc=run(a.proj,str(pi),str(po)); rec['proj_rc']=prc
            if prc==0: rec.update(metrics(fil,ef))
        rows.append(rec);print(json.dumps(rec,default=float))
    keys=sorted({k for r in rows for k in r})
    with open(out/'qe_electronic.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
if __name__=='__main__': main()
