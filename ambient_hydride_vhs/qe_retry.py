from __future__ import annotations

import argparse
import glob
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

# This script is executed directly as `python ambient_hydride_vhs/qe_retry.py` in CI.
# In that mode Python places this directory, not the repository root, on sys.path.
import hydride_vhs_screen as base


def run_input(exe: str, inp: Path, out: Path) -> int:
    with open(inp, "rb") as fi, open(out, "wb") as fo:
        return subprocess.run([exe], stdin=fi, stdout=fo, stderr=subprocess.STDOUT).returncode


def last_float(pattern: str, text: str):
    m = re.findall(pattern, text, re.I)
    return float(m[-1]) if m else None


def last_int(pattern: str, text: str):
    m = re.findall(pattern, text, re.I)
    return int(float(m[-1])) if m else None


def disable_pw_symmetry(path: Path) -> None:
    """Force identity-only PW symmetry for relaxed near-symmetric cells.

    CHGNet-relaxed structures can be numerically close to a high-symmetry
    position without satisfying it exactly. QE projwfc then may abort in
    d_matrix while trying to symmetrize atomic projections.  Disabling PW
    symmetry consistently in both SCF and NSCF avoids that representation
    artifact and keeps all candidates/calibration on the same footing.
    """
    txt = path.read_text()
    marker = "&SYSTEM\n"
    if " nosym=.true.," not in txt:
        txt = txt.replace(
            marker,
            marker + " nosym=.true.,\n noinv=.true.,\n",
            1,
        )
    path.write_text(txt)


def write_nscf(path: Path, s: Structure, ppdir: str, scratch: Path, prefix: str,
               nbnd: int, kmesh: int, conv_thr: str = "1.d-7") -> None:
    species = []
    for site in s:
        z = site.specie.symbol
        if z not in species:
            species.append(z)
    pmap = {z: base.pseudo(ppdir, z) for z in species}
    lines = [
        "&CONTROL",
        " calculation='nscf',",
        f" prefix='{prefix}',",
        f" pseudo_dir='{ppdir}',",
        f" outdir='{scratch}',",
        " disk_io='low',",
        "/",
        "&SYSTEM",
        " ibrav=0,",
        " nosym=.true.,",
        " noinv=.true.,",
        f" nat={len(s)}, ntyp={len(species)},",
        " ecutwfc=50.0, ecutrho=400.0,",
        f" nbnd={nbnd},",
        " occupations='smearing', smearing='mv', degauss=0.02,",
        "/",
        "&ELECTRONS",
        f" conv_thr={conv_thr},",
        " diagonalization='cg',",
        " diago_thr_init=1.d-6,",
        " diago_full_acc=.true.,",
        " electron_maxstep=600,",
        "/",
        "ATOMIC_SPECIES",
    ]
    lines.extend(f" {z} {base.MASS[z]:.8f} {pmap[z]}" for z in species)
    lines.append("ATOMIC_POSITIONS crystal")
    for site in s:
        f = site.frac_coords
        lines.append(f" {site.specie.symbol} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}")
    lines.append("CELL_PARAMETERS angstrom")
    lines.extend(" %.12f %.12f %.12f" % tuple(v) for v in s.lattice.matrix)
    lines.append("K_POINTS automatic")
    lines.append(f" {kmesh} {kmesh} {kmesh} 0 0 0")
    path.write_text("\n".join(lines) + "\n")


def qe(candidate: str, cif: str, atm: int, ppdir: str, pw: str, outdir: str) -> None:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch = (out / "scratch").resolve()
    scratch.mkdir(exist_ok=True)
    s = Structure.from_file(cif)
    prefix = f"hyd_{candidate}_{atm}"

    scf_in, scf_out = out / "scf.in", out / "scf.out"
    base.write_pw(scf_in, s, ppdir, scratch, prefix, "scf")
    disable_pw_symmetry(scf_in)
    scf_rc = run_input(pw, scf_in, scf_out)
    scf_txt = scf_out.read_text(errors="ignore")
    e_ry = last_float(r"!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry", scf_txt)
    ef = last_float(r"the Fermi energy is\s+([-0-9.Ee+]+)\s+ev", scf_txt)
    nelec = last_float(r"number of electrons\s*=\s*([-0-9.Ee+]+)", scf_txt)
    scf_nbnd = last_int(r"number of Kohn-Sham states\s*=\s*([0-9]+)", scf_txt)

    min_occ = int(math.ceil((nelec or 0.0) / 2.0))
    attempts = []
    nscf_rc = 99
    nscf_txt = ""
    for idx, (kmesh, extra, conv) in enumerate(((8, 8, "1.d-7"), (6, 6, "5.d-7")), start=1):
        nbnd = max(min_occ + extra, min_occ + 2)
        if scf_nbnd is not None:
            nbnd = min(nbnd, scf_nbnd)
        nscf_in = out / f"nscf_attempt{idx}.in"
        nscf_out = out / f"nscf_attempt{idx}.out"
        write_nscf(nscf_in, s, ppdir, scratch, prefix, nbnd, kmesh, conv)
        rc = run_input(pw, nscf_in, nscf_out) if scf_rc == 0 else 99
        txt = nscf_out.read_text(errors="ignore") if nscf_out.exists() else ""
        attempts.append({
            "attempt": idx,
            "kmesh": kmesh,
            "nbnd": nbnd,
            "rc": rc,
            "c_bands_failure": "too many bands are not converged" in txt.lower(),
        })
        if rc == 0:
            nscf_rc = 0
            nscf_txt = txt
            break
        nscf_rc = rc
        nscf_txt = txt

    if attempts:
        (out / "nscf.in").write_text((out / f"nscf_attempt{attempts[-1]['attempt']}.in").read_text())
    (out / "nscf.out").write_text(nscf_txt)
    ef_n = last_float(r"the Fermi energy is\s+([-0-9.Ee+]+)\s+ev", nscf_txt)
    if ef_n is not None:
        ef = ef_n

    dos_exe = str(Path(pw).with_name("dos.x"))
    proj_exe = str(Path(pw).with_name("projwfc.x"))
    dos_file = out / "total.dos"
    dos_in = out / "dos.in"
    dos_in.write_text(
        "&DOS\n"
        f" prefix='{prefix}',\n outdir='{scratch}',\n fildos='{dos_file}',\n DeltaE=0.01,\n/\n"
    )
    dos_rc = run_input(dos_exe, dos_in, out / "dos.out") if nscf_rc == 0 else 99

    pdos_prefix = out / "pdos"
    proj_in = out / "projwfc.in"
    proj_in.write_text(
        "&PROJWFC\n"
        f" prefix='{prefix}',\n outdir='{scratch}',\n filpdos='{pdos_prefix}',\n DeltaE=0.01,\n/\n"
    )
    proj_rc = run_input(proj_exe, proj_in, out / "projwfc.out") if nscf_rc == 0 else 99

    total_ef = None
    h_ef = None
    descriptor = None
    hfiles = []
    if dos_rc == 0 and proj_rc == 0 and ef is not None and dos_file.exists():
        total_arr = base.load_numeric(dos_file)
        total_ef = max(0.0, base.interp_at_ef(total_arr, ef, 1))
        hfiles = [Path(p) for p in glob.glob(str(out / "pdos.pdos_atm#*(H)*"))]
        hvals = []
        for p in hfiles:
            arr = base.load_numeric(p)
            hvals.append(max(0.0, base.interp_at_ef(arr, ef, 1)))
        if hvals:
            h_ef = float(sum(hvals))
            descriptor = float(math.sqrt(total_ef * h_ef))

    result = {
        "candidate": candidate,
        "pressure_atm": atm,
        "natoms": len(s),
        "formula": s.composition.reduced_formula,
        "symmetry_mode": "PW nosym=.true., noinv=.true.; projwfc default lsym",
        "scf_rc": scf_rc,
        "nscf_rc": nscf_rc,
        "nscf_attempts": attempts,
        "dos_rc": dos_rc,
        "projwfc_rc": proj_rc,
        "total_energy_Ry": e_ry,
        "total_energy_eV": e_ry * base.RY_TO_EV if e_ry is not None else None,
        "fermi_eV": ef,
        "number_of_electrons": nelec,
        "scf_nbnd": scf_nbnd,
        "total_DOS_EF_states_per_eV_cell": total_ef,
        "H_projected_DOS_EF_states_per_eV_cell": h_ef,
        "xu_dos_descriptor": descriptor,
        "metallic_screen": bool(total_ef is not None and total_ef >= 0.05),
        "n_H_pdos_files": len(hfiles),
        "note": "Identity-only PW symmetry avoids near-symmetry projwfc d_matrix artifacts. DOS/H-DOS descriptor is ranking only; not EPC, lambda, phonon stability, or Tc evidence.",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if scf_rc != 0 or nscf_rc != 0 or dos_rc != 0 or proj_rc != 0 or descriptor is None:
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=sorted(base.CANDIDATES))
    ap.add_argument("--cif", required=True)
    ap.add_argument("--atm", type=int, required=True, choices=[0, 400])
    ap.add_argument("--ppdir", required=True)
    ap.add_argument("--pw", default="/opt/espresso/7.5/pw.x")
    ap.add_argument("--out", required=True)
    z = ap.parse_args()
    qe(z.candidate, z.cif, z.atm, z.ppdir, z.pw, z.out)


if __name__ == "__main__":
    main()
