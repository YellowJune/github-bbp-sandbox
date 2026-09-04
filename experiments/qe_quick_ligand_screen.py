#!/usr/bin/env python3
import argparse, csv, glob, json, os, re, subprocess
from pathlib import Path
import numpy as np

A0=3.933714; C0=9.827602
MASS={"Ba":137.327,"Cu":63.546,"Hg":200.592,"O":15.999,"N":14.007,"La":138.90547,"Bi":208.9804,"Tl":204.38,"Y":88.90584}
BASE=[("Ba",(.5,.5,.695597)),("Ba",(.5,.5,.304403)),("Cu",(0,0,.5)),("Hg",(0,0,0)),("O",(0,0,.794971)),("O",(0,0,.205029)),("O",(0,.5,.5)),("O",(.5,0,.5))]

def build():
    sy=[]; fr=[]; tags=[]
    for ix in range(2):
        for j,(s,p) in enumerate(BASE):
            sy.append(s); fr.append(((p[0]+ix)/2,p[1],p[2])); tags.append((ix,j))
    cell=np.diag([2*A0,A0,C0]); out={"parent":(sy,fr,cell)}
    # O2- -> N3- needs +1 cation compensation.  La/Y replace Ba2+; Bi/Tl replace Hg2+.
    for cat,site in [("La",0),("Y",0),("Bi",3),("Tl",3)]:
        for pos,jO in [("apical",4),("planar",6)]:
            ss=list(sy); ss[tags.index((0,site))]=cat; ss[tags.index((0,jO))]="N"
            out[f"{cat}N_{pos}"]=(ss,list(fr),cell.copy())
    return out

def pmap(ppdir,meta_path,elements):
    meta=json.load(open(meta_path)); out={}
    for e in elements:
        fn=meta.get(e,{}).get("filename") if isinstance(meta.get(e),dict) else None
        if fn and os.path.exists(os.path.join(ppdir,fn)): out[e]=fn; continue
        c=[]
        for pat in [f"{e}.*UPF",f"{e}.*upf",f"{e}_*UPF",f"{e}_*upf"]: c+=glob.glob(os.path.join(ppdir,pat))
        if not c: raise RuntimeError(f"no pseudo {e}")
        out[e]=os.path.basename(sorted(c)[0])
    return out

def write_scf(path,prefix,sy,fr,cell,pm,ppdir,outdir):
    species=[]
    for s in sy:
        if s not in species: species.append(s)
    L=["&CONTROL"," calculation='scf',",f" prefix='{prefix}',",f" pseudo_dir='{ppdir}',",f" outdir='{outdir}',"," tstress=.true., tprnfor=.true., disk_io='low',","/","&SYSTEM"," ibrav=0,",f" nat={len(sy)}, ntyp={len(species)},"," ecutwfc=50.0, ecutrho=500.0,"," occupations='smearing', smearing='mv', degauss=0.025,"," nspin=2,"]
    for i,s in enumerate(species,1): L.append(f" starting_magnetization({i})={0.5 if s=='Cu' else (0.1 if s=='N' else 0.0):.2f},")
    L += ["/","&ELECTRONS"," conv_thr=2.0d-8, mixing_beta=0.30, electron_maxstep=200,","/","ATOMIC_SPECIES"]
    for s in species: L.append(f" {s} {MASS[s]:.8f} {pm[s]}")
    L.append("ATOMIC_POSITIONS crystal")
    for s,p in zip(sy,fr): L.append(f" {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    L.append("CELL_PARAMETERS angstrom")
    for v in cell: L.append(" %.10f %.10f %.10f"%tuple(v))
    L += ["K_POINTS automatic"," 4 8 4 0 0 0"]
    Path(path).write_text("\n".join(L)+"\n")

def run(exe,inp,out):
    with open(inp,"rb") as fi,open(out,"wb") as fo: return subprocess.run([exe],stdin=fi,stdout=fo,stderr=subprocess.STDOUT).returncode

def parse_fermi(t):
    m=re.findall(r"the Fermi energy is\s+([-0-9.Ee+]+)",t,re.I); return float(m[-1]) if m else np.nan

def parse_energy(t):
    m=re.findall(r"!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry",t); return float(m[-1])*13.605693122994 if m else np.nan

def read_orb(prefix,el,orb):
    fs=glob.glob(prefix+f".pdos_atm#*({el})_wfc#*({orb})"); E=None; y=None
    for f in fs:
        a=np.loadtxt(f,comments="#"); a=np.atleast_2d(a); yy=a[:,1:].sum(1)
        if E is None: E=a[:,0]; y=yy
        elif len(E)==len(a): y+=yy
    return E,y,len(fs)

def metrics(prefix,ef):
    E,cu,ncu=read_orb(prefix,"Cu","d"); EO,o,no=read_orb(prefix,"O","p"); EN,n,nn=read_orb(prefix,"N","p")
    if E is None or EO is None or not np.isfinite(ef): return {}
    lig=o.copy()
    if EN is not None and len(EN)==len(E): lig+=n
    mask=(E>=ef-.6)&(E<=ef+.6); trap=np.trapezoid
    cuw=float(trap(cu[mask],E[mask])); ligw=float(trap(lig[mask],E[mask])); hov=float(trap(np.sqrt(np.clip(cu[mask],0,None)*np.clip(lig[mask],0,None)),E[mask]))
    # occupied p/d centroids in a wider near-EF window, useful as a charge-transfer proxy
    m2=(E>=ef-3.0)&(E<=ef+.5)
    def cent(y):
        den=trap(y[m2],E[m2]); return float(trap(E[m2]*y[m2],E[m2])/den) if abs(den)>1e-12 else np.nan
    return {"cu_d_EFwin":cuw,"ligand_p_EFwin":ligw,"hybrid_overlap":hov,"cu_d_centroid_eV":cent(cu),"ligand_p_centroid_eV":cent(lig),"pd_centroid_sep_eV":abs(cent(cu)-cent(lig)),"ncu":ncu,"no":no,"nn":nn}

def main(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); scratch=Path(a.scratch); scratch.mkdir(parents=True,exist_ok=True)
    vv=build(); els=sorted({s for sy,_,_ in vv.values() for s in sy}); pm=pmap(a.ppdir,a.meta,els)
    rows=[]
    for name,(sy,fr,cell) in vv.items():
        vd=out/name; vd.mkdir(exist_ok=True); od=scratch/name; od.mkdir(exist_ok=True)
        pref=f"q{name}"; inp=vd/'scf.in'; oo=vd/'scf.out'; write_scf(inp,pref,sy,fr,cell,pm,a.ppdir,str(od))
        rc=run(a.pw,inp,oo); rec={"name":name,"scf_rc":rc,"formula":"".join(sy)}
        if rc==0:
            txt=oo.read_text(errors='ignore'); ef=parse_fermi(txt); rec.update({"fermi_eV":ef,"energy_eV":parse_energy(txt)})
            pi=vd/'proj.in'; po=vd/'proj.out'; fil=str(od/'pdos')
            pi.write_text(f"&PROJWFC\n prefix='{pref}',\n outdir='{od}',\n filpdos='{fil}',\n DeltaE=0.05, degauss=0.02, ngauss=0,\n/\n")
            rc2=run(a.proj,pi,po); rec['proj_rc']=rc2
            if rc2==0: rec.update(metrics(fil,ef))
        rows.append(rec); print(json.dumps(rec,default=float))
    par=next((x for x in rows if x['name']=='parent'),{})
    for r in rows:
        if r['name']!='parent':
            for k in ['hybrid_overlap','cu_d_EFwin','ligand_p_EFwin','pd_centroid_sep_eV']:
                if k in r and k in par and np.isfinite(r[k]) and np.isfinite(par[k]):
                    r['ratio_'+k]=r[k]/par[k] if par[k]!=0 else np.nan
                    r['delta_'+k]=r[k]-par[k]
    keys=sorted({k for r in rows for k in r});
    with open(out/'quick_screen.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2,default=float))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--ppdir',required=True); p.add_argument('--meta',required=True); p.add_argument('--output',default='artifacts/qe_quick_ligand'); p.add_argument('--scratch',default='/tmp/qquick'); p.add_argument('--pw',default='pw.x'); p.add_argument('--proj',default='projwfc.x'); main(p.parse_args())
