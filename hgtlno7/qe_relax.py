#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json,os,re,subprocess
from pathlib import Path
import numpy as np

A0=3.933714; C0=9.827602
MASS={'Ba':137.327,'Cu':63.546,'Hg':200.592,'Tl':204.38,'O':15.999,'N':14.007}
BASE=[('Ba',(.5,.5,.695597)),('Ba',(.5,.5,.304403)),('Cu',(0,0,.5)),('Hg',(0,0,0)),('O',(0,0,.794971)),('O',(0,0,.205029)),('O',(0,.5,.5)),('O',(.5,0,.5))]

def build(name):
    sy=[];fr=[];tags=[]
    for ix in range(2):
        for j,(s,p) in enumerate(BASE):
            sy.append(s);fr.append(((p[0]+ix)/2,p[1],p[2]));tags.append((ix,j))
    if name=='candidate':
        sy[tags.index((0,3))]='Tl'
        sy[tags.index((0,4))]='N'
    elif name!='parent': raise ValueError(name)
    cell=np.diag([2*A0,A0,C0])
    return sy,fr,cell

def pmap(ppdir,meta_path,elements):
    meta=json.load(open(meta_path)); out={}
    for e in elements:
        fn=meta.get(e,{}).get('filename') if isinstance(meta.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)): out[e]=fn; continue
        c=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']:
            c+=glob.glob(os.path.join(ppdir,pat))
        if not c: raise RuntimeError(f'no pseudo for {e}')
        out[e]=os.path.basename(sorted(c)[0])
    return out

def write_relax(path,prefix,sy,fr,cell,pm,ppdir,outdir):
    species=[]
    for s in sy:
        if s not in species: species.append(s)
    L=['&CONTROL'," calculation='vc-relax',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',"," forc_conv_thr=1.0d-3,","/",'&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(species)},',' ecutwfc=50.0, ecutrho=500.0,'," occupations='smearing', smearing='mv', degauss=0.02,"," nspin=2,"]
    for i,s in enumerate(species,1): L.append(f" starting_magnetization({i})={0.55 if s=='Cu' else (0.10 if s=='N' else 0.0):.2f},")
    L += ['/','&ELECTRONS',' conv_thr=2.0d-8, mixing_beta=0.25, electron_maxstep=220,','/','&IONS'," ion_dynamics='bfgs',",'/','&CELL'," cell_dynamics='bfgs', press=0.0, press_conv_thr=0.8, cell_dofree='all',",'/','ATOMIC_SPECIES']
    for s in species: L.append(f' {s} {MASS[s]:.8f} {pm[s]}')
    L.append('ATOMIC_POSITIONS crystal')
    for s,p in zip(sy,fr): L.append(f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}')
    L.append('CELL_PARAMETERS angstrom')
    for v in cell: L.append(' %.10f %.10f %.10f'%tuple(v))
    L += ['K_POINTS automatic',' 3 6 3 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo:
        return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def last_float(pattern,text):
    m=re.findall(pattern,text,re.I|re.M); return float(m[-1]) if m else np.nan

def parse_final_cell(text):
    # take the final CELL_PARAMETERS block emitted by vc-relax
    ms=list(re.finditer(r'CELL_PARAMETERS\s*\(angstrom\)\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)',text,re.I))
    if not ms:return None
    m=ms[-1]; return np.array([[float(x) for x in m.group(i).split()[:3]] for i in (1,2,3)])

def read_orb(prefix,el,orb):
    fs=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({orb})'); E=None;y=None
    for f in fs:
        a=np.atleast_2d(np.loadtxt(f,comments='#')); yy=a[:,1:].sum(1)
        if E is None:E=a[:,0];y=yy
        elif len(E)==len(a):y+=yy
    return E,y,len(fs)

def pdos_metrics(prefix,ef):
    E,cu,ncu=read_orb(prefix,'Cu','d');EO,o,no=read_orb(prefix,'O','p');EN,n,nn=read_orb(prefix,'N','p')
    if E is None or EO is None or not np.isfinite(ef): return {}
    lig=o.copy()
    if EN is not None and len(EN)==len(E):lig+=n
    trap=np.trapezoid; mask=(E>=ef-.6)&(E<=ef+.6)
    cuw=float(trap(cu[mask],E[mask]));ligw=float(trap(lig[mask],E[mask]));hov=float(trap(np.sqrt(np.clip(cu[mask],0,None)*np.clip(lig[mask],0,None)),E[mask]))
    m2=(E>=ef-3)&(E<=ef+.5)
    def cent(y):
        de=trap(y[m2],E[m2]); return float(trap(E[m2]*y[m2],E[m2])/de) if abs(de)>1e-12 else np.nan
    return {'cu_d_EFwin':cuw,'ligand_p_EFwin':ligw,'hybrid_overlap':hov,'pd_centroid_sep_eV':abs(cent(cu)-cent(lig)),'ncu':ncu,'no':no,'nn':nn}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['parent','candidate'],required=True);ap.add_argument('--ppdir',required=True);ap.add_argument('--meta',required=True);ap.add_argument('--output',required=True);ap.add_argument('--scratch',required=True);ap.add_argument('--pw',default='/opt/espresso/7.5/pw.x');ap.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    sy,fr,cell=build(a.name);pm=pmap(a.ppdir,a.meta,sorted(set(sy)));pref='rel'+a.name
    inp=out/'relax.in';oo=out/'relax.out';write_relax(inp,pref,sy,fr,cell,pm,a.ppdir,str(scratch));rc=run(a.pw,inp,oo)
    txt=oo.read_text(errors='ignore') if oo.exists() else ''
    rec={'name':a.name,'relax_rc':rc,'converged':('convergence has been achieved' in txt),'bfgs_done':('End of BFGS Geometry Optimization' in txt),'energy_Ry':last_float(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt),'fermi_eV':last_float(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt),'total_magnetization_Bohr':last_float(r'total magnetization\s+=\s+([-0-9.Ee+]+)\s+Bohr',txt)}
    fc=parse_final_cell(txt)
    if fc is not None:
        rec['final_cell_lengths_A']=[float(np.linalg.norm(x)) for x in fc];rec['final_volume_A3']=float(abs(np.linalg.det(fc)))
    # final pressure printed by QE in kbar
    rec['final_pressure_kbar']=last_float(r'P=\s*([-0-9.Ee+]+)',txt)
    if rc==0 and np.isfinite(rec['fermi_eV']):
        pi=out/'proj.in';po=out/'proj.out';fil=str(scratch/'pdos')
        pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{scratch}',\n filpdos='{fil}',\n DeltaE=0.05, degauss=0.02, ngauss=0,\n/\n")
        prc=run(a.proj,pi,po);rec['proj_rc']=prc
        if prc==0:rec.update(pdos_metrics(fil,rec['fermi_eV']))
    (out/'result.json').write_text(json.dumps(rec,indent=2,default=float));print(json.dumps(rec,indent=2,default=float))
if __name__=='__main__':main()
