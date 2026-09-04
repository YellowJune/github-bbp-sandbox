#!/usr/bin/env python3
import argparse, csv, glob, json, os, re, subprocess
from pathlib import Path
import numpy as np
from ase.io import read as ase_read

MASS={'Ca':40.078,'Ti':47.867,'Zr':91.224,'Hf':178.49,'Cu':63.546,'O':15.999,'N':14.007}
SPECS={
 'CaCuO2':('Ca','O',3.86,3.20),
 'TiCuN2':('Ti','N',3.72,3.08),
 'ZrCuN2':('Zr','N',3.84,3.32),
 'HfCuN2':('Hf','N',3.81,3.28),
}

def pseudo_map(ppdir,meta_path,elements):
    meta=json.load(open(meta_path)); out={}
    for e in elements:
        fn=meta.get(e,{}).get('filename') if isinstance(meta.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)): out[e]=fn; continue
        cand=[]
        for pat in [f'{e}.*UPF',f'{e}.*upf',f'{e}_*UPF',f'{e}_*upf']: cand+=glob.glob(os.path.join(ppdir,pat))
        if not cand: raise RuntimeError('missing pseudo '+e)
        out[e]=os.path.basename(sorted(cand)[0])
    return out

def base_structure(name):
    A,X,a,c=SPECS[name]
    sy=[A,'Cu',X,X]
    fr=np.array([[.5,.5,.5],[0,0,0],[.5,0,0],[0,.5,0]],float)
    cell=np.diag([a,a,c])
    return sy,fr,cell

def write_pw(path,prefix,sy,fr,cell,pmap,ppdir,outdir,calc='vc-relax',k=(6,6,6),press_kbar=0.0):
    sp=[]
    for s in sy:
        if s not in sp: sp.append(s)
    L=['&CONTROL',f" calculation='{calc}',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',",'/','&SYSTEM',' ibrav=0,',f' nat={len(sy)}, ntyp={len(sp)},'," ecutwfc=50.0, ecutrho=500.0, occupations='smearing', smearing='mv', degauss=0.02, nspin=2,"]
    for i,s in enumerate(sp,1): L.append(f" starting_magnetization({i})={0.55 if s=='Cu' else 0.0:.2f},")
    L += ['/','&ELECTRONS',' conv_thr=2.0d-8, mixing_beta=0.30, electron_maxstep=220,','/']
    if calc in ('relax','vc-relax'): L += ['&IONS'," ion_dynamics='bfgs',",'/']
    if calc=='vc-relax': L += ['&CELL'," cell_dynamics='bfgs',",f' press={press_kbar:.6f},',' press_conv_thr=0.5,',' cell_dofree=\'all\',','/']
    L += ['ATOMIC_SPECIES']
    for s in sp: L.append(f' {s} {MASS[s]:.8f} {pmap[s]}')
    L.append('ATOMIC_POSITIONS crystal')
    for s,p in zip(sy,fr): L.append(f' {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}')
    L.append('CELL_PARAMETERS angstrom')
    for v in cell: L.append(' %.10f %.10f %.10f'%tuple(v))
    L += ['K_POINTS automatic',f' {k[0]} {k[1]} {k[2]} 0 0 0']
    Path(path).write_text('\n'.join(L)+'\n')

def run(exe,inp,out):
    with open(inp,'rb') as fi,open(out,'wb') as fo: return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def parse_fermi(txt):
    m=re.findall(r'the Fermi energy is\s+([-0-9.Ee+]+)',txt,re.I); return float(m[-1]) if m else np.nan

def parse_energy(txt):
    m=re.findall(r'!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry',txt); return float(m[-1])*13.605693122994 if m else np.nan

def pdos(prefix,el,orb):
    fs=glob.glob(prefix+f'.pdos_atm#*({el})_wfc#*({orb})'); E=None;y=None
    for f in fs:
        a=np.atleast_2d(np.loadtxt(f,comments='#')); yy=a[:,1:].sum(1)
        if E is None: E=a[:,0]; y=yy
        elif len(E)==len(a): y+=yy
    return E,y,len(fs)

def metrics(prefix,ef,X):
    E,cu,ncu=pdos(prefix,'Cu','d'); EL,lig,nl=pdos(prefix,X,'p')
    if E is None or EL is None or not np.isfinite(ef): return {}
    mask=(E>=ef-.6)&(E<=ef+.6); wide=(E>=ef-4)&(E<=ef+1)
    trap=np.trapezoid
    def centroid(y):
        den=trap(y[wide],E[wide]); return float(trap(E[wide]*y[wide],E[wide])/den) if abs(den)>1e-12 else np.nan
    cd=centroid(cu); cp=centroid(lig)
    return {'cu_d_EFwin':float(trap(cu[mask],E[mask])),'ligand_p_EFwin':float(trap(lig[mask],E[mask])),'hybrid_overlap':float(trap(np.sqrt(np.clip(cu[mask],0,None)*np.clip(lig[mask],0,None)),E[mask])),'cu_d_centroid_eV':cd,'ligand_p_centroid_eV':cp,'pd_centroid_sep_eV':abs(cd-cp),'n_cu_d':ncu,'n_ligand_p':nl}

def main(a):
    sy,fr,cell=base_structure(a.variant); A,X,_,_=SPECS[a.variant]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True); scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    pm=pseudo_map(a.ppdir,a.meta,sorted(set(sy)))
    pref='il_'+a.variant
    vin=out/'vcrelax.in'; vout=out/'vcrelax.out'
    write_pw(vin,pref,sy,fr,cell,pm,a.ppdir,str(scratch),'vc-relax',(6,6,6),0.0)
    rc=run(a.pw,vin,vout); rec={'variant':a.variant,'vc_relax_rc':rc,'formal_scheme':f'{A}+Cu+{X}2','H_free':True}
    if rc!=0:
        rec['tail']='\n'.join(vout.read_text(errors='ignore').splitlines()[-20:]); (out/'result.json').write_text(json.dumps(rec,indent=2)); print(json.dumps(rec)); return
    at=ase_read(str(vout),format='espresso-out',index=-1)
    cell2=np.array(at.cell.array); fr2=np.array(at.get_scaled_positions(wrap=True)); sy2=at.get_chemical_symbols()
    rec.update({'a_A':float(np.linalg.norm(cell2[0])),'b_A':float(np.linalg.norm(cell2[1])),'c_A':float(np.linalg.norm(cell2[2])),'volume_A3':float(at.get_volume())})
    ds=at.get_all_distances(mic=True); vals=[]
    for i,s in enumerate(sy2):
        if s=='Cu':
            for j,t in enumerate(sy2):
                if t==X and i!=j: vals.append(ds[i,j])
    rec['min_Cu_ligand_A']=float(min(vals)) if vals else np.nan
    scfin=out/'scf.in';scfout=out/'scf.out';write_pw(scfin,pref,sy2,fr2,cell2,pm,a.ppdir,str(scratch),'scf',(8,8,8),0.0)
    rc2=run(a.pw,scfin,scfout);rec['scf_rc']=rc2
    if rc2==0:
        txt=scfout.read_text(errors='ignore');ef=parse_fermi(txt);rec['fermi_eV']=ef;rec['energy_eV']=parse_energy(txt)
        pin=out/'proj.in';pout=out/'proj.out';fp=str(scratch/'pdos');pin.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{scratch}',\n filpdos='{fp}',\n DeltaE=0.04, degauss=0.02, ngauss=0,\n/\n")
        rc3=run(a.proj,pin,pout);rec['proj_rc']=rc3
        if rc3==0: rec.update(metrics(fp,ef,X))
    (out/'result.json').write_text(json.dumps(rec,indent=2,default=float));
    with open(out/'result.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(rec));w.writeheader();w.writerow(rec)
    print(json.dumps(rec,indent=2,default=float))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=sorted(SPECS),required=True);p.add_argument('--ppdir',required=True);p.add_argument('--meta',required=True);p.add_argument('--output',required=True);p.add_argument('--scratch',required=True);p.add_argument('--pw',default='/opt/espresso/7.5/pw.x');p.add_argument('--proj',default='/opt/espresso/7.5/projwfc.x');main(p.parse_args())
