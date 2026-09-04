#!/usr/bin/env python3
import argparse, csv, glob, json, os, re, shutil, subprocess
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

MASS = {"Ba":137.327,"Cu":63.546,"Hg":200.592,"O":15.999,"N":14.007,"La":138.90547}
A0=3.933714
C0=9.827602
BASE=[
 ("Ba",(0.5,0.5,0.695597)),
 ("Ba",(0.5,0.5,0.304403)),
 ("Cu",(0.0,0.0,0.5)),
 ("Hg",(0.0,0.0,0.0)),
 ("O", (0.0,0.0,0.794971)),
 ("O", (0.0,0.0,0.205029)),
 ("O", (0.0,0.5,0.5)),
 ("O", (0.5,0.0,0.5)),
]


def supercell_2x1x1():
    syms=[]; frac=[]; tags=[]
    for ix in range(2):
        for j,(s,p) in enumerate(BASE):
            syms.append(s)
            frac.append(((p[0]+ix)/2.0,p[1],p[2]))
            tags.append((ix,j))
    cell=np.diag([2*A0,A0,C0])
    return syms,np.array(frac,float),cell,tags


def variants():
    syms,frac,cell,tags=supercell_2x1x1()
    out={"parent":(list(syms),frac.copy(),cell.copy())}
    # Formal charge compensation: O2- -> N3- paired with Ba2+ -> La3+.
    # This is a mechanistic stress test, not a prior claim of synthesizability.
    for name, oxygen_j in [("LaN_apical",4),("LaN_planar",6)]:
        ss=list(syms)
        ss[tags.index((0,0))]="La"
        ss[tags.index((0,oxygen_j))]="N"
        out[name]=(ss,frac.copy(),cell.copy())
    return out


def pseudo_map(pseudo_dir, sssp_json):
    meta=json.load(open(sssp_json)) if sssp_json and os.path.exists(sssp_json) else {}
    ans={}
    for e in MASS:
        fn=None
        if e in meta and isinstance(meta[e],dict):
            fn=meta[e].get("filename") or meta[e].get("pseudo")
        if fn and os.path.exists(os.path.join(pseudo_dir,fn)):
            ans[e]=fn; continue
        cand=[]
        for p in [f"{e}.*UPF",f"{e}.*upf",f"{e}_*UPF",f"{e}_*upf"]:
            cand += glob.glob(os.path.join(pseudo_dir,p))
        cand=sorted(set(cand))
        if not cand:
            raise FileNotFoundError(f"No pseudopotential found for {e} in {pseudo_dir}")
        ans[e]=os.path.basename(cand[0])
    return ans


def write_pw(path,prefix,syms,frac,cell,pmap,outdir,calc="relax",kgrid=(3,6,4)):
    species=[]
    for s in syms:
        if s not in species: species.append(s)
    lines=[]
    lines += ["&CONTROL",f" calculation='{calc}',",f" prefix='{prefix}',",f" pseudo_dir='{os.path.abspath(args.pseudo_dir)}',",f" outdir='{os.path.abspath(outdir)}',"," tstress=.true., tprnfor=.true., verbosity='high',"," disk_io='low',","/"]
    lines += ["&SYSTEM"," ibrav=0,",f" nat={len(syms)}, ntyp={len(species)},"," ecutwfc=60.0, ecutrho=600.0,"," occupations='smearing', smearing='mv', degauss=0.02,"," nspin=2,"]
    for i,s in enumerate(species,1):
        if s=="Cu": lines.append(f" starting_magnetization({i})=0.50,")
        elif s=="N": lines.append(f" starting_magnetization({i})=0.10,")
        else: lines.append(f" starting_magnetization({i})=0.00,")
    lines += ["/","&ELECTRONS"," conv_thr=1.0d-8, mixing_beta=0.30, electron_maxstep=240,","/"]
    if calc=="relax": lines += ["&IONS"," ion_dynamics='bfgs',","/"]
    lines += ["ATOMIC_SPECIES"]
    for s in species: lines.append(f" {s} {MASS[s]:.8f} {pmap[s]}")
    lines += ["ATOMIC_POSITIONS crystal"]
    for s,p in zip(syms,frac): lines.append(f" {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    lines += ["CELL_PARAMETERS angstrom"]
    for v in cell: lines.append(" %.10f %.10f %.10f"%tuple(v))
    lines += ["K_POINTS automatic",f" {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0"]
    Path(path).write_text("\n".join(lines)+"\n")


def run_cmd(cmd,stdin_path,stdout_path):
    with open(stdin_path,"rb") as fi, open(stdout_path,"wb") as fo:
        p=subprocess.run(cmd,stdin=fi,stdout=fo,stderr=subprocess.STDOUT)
    return p.returncode


def parse_fermi(text):
    for pat in [r"the Fermi energy is\s+([\-0-9.Ee+]+)",r"highest occupied level \(ev\):\s+([\-0-9.Ee+]+)"]:
        m=list(re.finditer(pat,text,re.I))
        if m: return float(m[-1].group(1))
    return float("nan")


def parse_energy(text):
    m=list(re.finditer(r"!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry",text))
    return float(m[-1].group(1))*13.605693122994 if m else float("nan")


def pdos_sum(prefix, element, orbital):
    files=glob.glob(prefix+f".pdos_atm#*({element})_wfc#*({orbital})")
    Es=None; total=None
    for fn in files:
        arr=np.loadtxt(fn,comments="#")
        if arr.ndim==1: arr=arr[None,:]
        e=arr[:,0]; y=np.sum(arr[:,1:],axis=1)
        if Es is None: Es=e; total=y
        elif len(e)==len(Es) and np.max(np.abs(e-Es))<1e-6: total+=y
    return Es,total,len(files)


def integrate_overlap(pdos_prefix,ef):
    E,cu,ncu=pdos_sum(pdos_prefix,"Cu","d")
    EO,op,no=pdos_sum(pdos_prefix,"O","p")
    EN,npd,nn=pdos_sum(pdos_prefix,"N","p")
    if E is None or cu is None or EO is None or op is None or not np.isfinite(ef):
        return {"cu_d_window":float("nan"),"o_p_window":float("nan"),"n_p_window":float("nan"),"ligand_p_window":float("nan"),"hybrid_overlap":float("nan"),"n_cu_d_files":ncu,"n_o_p_files":no,"n_n_p_files":nn}
    ligand=op.copy()
    if EN is not None and npd is not None and len(EN)==len(E) and np.max(np.abs(EN-E))<1e-6: ligand += npd
    mask=(E>=ef-0.5)&(E<=ef+0.5)
    trap=getattr(np,"trapezoid",np.trapz)
    cuint=float(trap(cu[mask],E[mask])); oint=float(trap(op[mask],E[mask])); nint=0.0
    if EN is not None and npd is not None:
        mn=(EN>=ef-0.5)&(EN<=ef+0.5); nint=float(trap(npd[mn],EN[mn]))
    lint=float(trap(ligand[mask],E[mask]))
    hov=float(trap(np.sqrt(np.clip(cu[mask],0,None)*np.clip(ligand[mask],0,None)),E[mask]))
    return {"cu_d_window":cuint,"o_p_window":oint,"n_p_window":nint,"ligand_p_window":lint,"hybrid_overlap":hov,"n_cu_d_files":ncu,"n_o_p_files":no,"n_n_p_files":nn}


def nearest_metrics(atoms):
    syms=atoms.get_chemical_symbols(); dists=atoms.get_all_distances(mic=True); vals=[]
    for i,s in enumerate(syms):
        if s!="Cu": continue
        for j,t in enumerate(syms):
            if t in ("O","N") and i!=j: vals.append((dists[i,j],t))
    vals=sorted(vals); near=[x for x in vals if x[0]<3.0]
    return {
        "min_Cu_ligand_A": float(near[0][0]) if near else float("nan"),
        "mean_4short_Cu_ligand_A": float(np.mean([x[0] for x in near[:8]])) if near else float("nan"),
        "min_Cu_N_A": float(min([x[0] for x in near if x[1]=="N"],default=float("nan"))),
        "volume_A3": float(atoms.get_volume()),
    }


def main():
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    scratch=Path(args.scratch_dir)
    if scratch.exists(): shutil.rmtree(scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    pmap=pseudo_map(args.pseudo_dir,args.sssp_json)
    (out/"pseudopotentials.json").write_text(json.dumps(pmap,indent=2))
    rows=[]
    for name,(syms,frac,cell) in variants().items():
        vdir=out/name; vdir.mkdir(exist_ok=True)
        odir=scratch/name; odir.mkdir(parents=True,exist_ok=True)
        prefix=f"csc_{name}"
        rel_in=vdir/"relax.in"; rel_out=vdir/"relax.out"
        write_pw(rel_in,prefix,syms,frac,cell,pmap,odir,"relax",(3,6,4))
        cmd=args.mpi_cmd.split()+["pw.x"] if args.mpi_cmd.strip() else ["pw.x"]
        rc=run_cmd(cmd,rel_in,rel_out); rec={"name":name,"relax_rc":rc}
        if rc!=0:
            rec["relax_tail"]="\n".join(rel_out.read_text(errors="ignore").splitlines()[-12:])
            rows.append(rec); print("RELAX_FAIL",name,rc); continue
        try:
            at=ase_read(str(rel_out),format="espresso-out",index=-1)
        except Exception as exc:
            rec["parse_error"]=repr(exc); rows.append(rec); continue
        rec.update(nearest_metrics(at))
        sy2=at.get_chemical_symbols(); cell2=np.array(at.cell.array,float); frac2=np.array(at.get_scaled_positions(wrap=True),float)
        scf_in=vdir/"scf.in"; scf_out=vdir/"scf.out"
        write_pw(scf_in,prefix,sy2,frac2,cell2,pmap,odir,"scf",(4,8,4))
        rc2=run_cmd(cmd,scf_in,scf_out); rec["scf_rc"]=rc2
        txt=scf_out.read_text(errors="ignore") if scf_out.exists() else ""
        rec["fermi_eV"]=parse_fermi(txt); rec["total_energy_eV"]=parse_energy(txt)
        if rc2==0:
            proj_in=vdir/"projwfc.in"; proj_out=vdir/"projwfc.out"; filpdos=os.path.join(str(odir),"pdos")
            proj_in.write_text("&PROJWFC\n prefix='%s',\n outdir='%s',\n filpdos='%s',\n DeltaE=0.05, degauss=0.02, ngauss=0,\n/\n"%(prefix,os.path.abspath(odir),os.path.abspath(filpdos)))
            pcmd=args.mpi_cmd.split()+["projwfc.x"] if args.mpi_cmd.strip() else ["projwfc.x"]
            rc3=run_cmd(pcmd,proj_in,proj_out); rec["projwfc_rc"]=rc3
            if rc3==0:
                rec.update(integrate_overlap(filpdos,rec["fermi_eV"]))
                pdos_out=vdir/"pdos_files"; pdos_out.mkdir(exist_ok=True)
                for fn in glob.glob(filpdos+"*"):
                    if os.path.isfile(fn): shutil.copy2(fn,pdos_out/os.path.basename(fn))
        rows.append(rec); print("DONE",json.dumps(rec,default=str))
    parent=next((r for r in rows if r.get("name")=="parent"),{})
    for r in rows:
        if r.get("name")!="parent":
            for k in ["hybrid_overlap","cu_d_window","ligand_p_window","min_Cu_ligand_A"]:
                if k in r and k in parent and isinstance(r[k],(int,float)) and isinstance(parent[k],(int,float)) and np.isfinite(r[k]) and np.isfinite(parent[k]):
                    r["delta_"+k]=r[k]-parent[k]
    keys=sorted({k for r in rows for k in r})
    with open(out/"qe_summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out/"summary.json").write_text(json.dumps({"constraint":{"H":False,"pressure_atm":0},"method":"spin-polarized PBE QE 6.7, SSSP 1.2.1 efficiency; same pipeline; PDOS overlap is a hybridization proxy, not Tc","rows":rows},indent=2,default=float))
    print("SUMMARY"); print(json.dumps(rows,indent=2,default=float))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir",required=True)
    ap.add_argument("--sssp-json",default="")
    ap.add_argument("--output-dir",default="artifacts/cipher_sc_qe_oxynitride")
    ap.add_argument("--scratch-dir",default="/tmp/cq")
    ap.add_argument("--mpi-cmd",default="mpirun --oversubscribe -np 4")
    args=ap.parse_args(); main()
