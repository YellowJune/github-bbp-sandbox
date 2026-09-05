from __future__ import annotations

import argparse
import glob
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

ATM_TO_GPA = 0.000101325
GPA_TO_EV_A3 = 1.0 / 160.21766208
RY_TO_EV = 13.605693122994

# Ordered K2PtCl6-derived Mg(D)TMH6 cells. MgAlFeH6 is the literature
# calibration point; Ga/In and Ru/Os are donor/chemical-pressure counterfactuals.
CANDIDATES = {
    "MgAlFeH6": ("Al", "Fe", 6.24),
    "MgAlRuH6": ("Al", "Ru", 6.50),
    "MgAlOsH6": ("Al", "Os", 6.62),
    "MgGaFeH6": ("Ga", "Fe", 6.30),
    "MgGaRuH6": ("Ga", "Ru", 6.56),
    "MgGaOsH6": ("Ga", "Os", 6.68),
    "MgInFeH6": ("In", "Fe", 6.40),
    "MgInRuH6": ("In", "Ru", 6.66),
    "MgInOsH6": ("In", "Os", 6.78),
}

MASS = {
    "H": 1.00794,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Ga": 69.723,
    "In": 114.818,
    "Fe": 55.845,
    "Ru": 101.07,
    "Os": 190.23,
}


def build(candidate: str, xh: float = 0.240) -> Structure:
    donor, tm, a = CANDIDATES[candidate]
    # F-43m ordered derivative: Mg 4d, donor 4c, TM 4b, H 24f.
    conv = Structure.from_spacegroup(
        216,
        Lattice.cubic(a),
        ["Mg", donor, tm, "H"],
        [[0.75, 0.75, 0.75], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [xh, 0.0, 0.0]],
    )
    s = conv.get_primitive_structure(tolerance=1e-5)
    return s


def pair_min_distance(s: Structure) -> float:
    dmin = 99.0
    for i in range(len(s)):
        for j in range(i):
            dmin = min(dmin, float(s.get_distance(i, j)))
    return dmin


def h_metrics(s: Structure, tm: str) -> dict:
    hs = [i for i, site in enumerate(s) if site.specie.symbol == "H"]
    tms = [i for i, site in enumerate(s) if site.specie.symbol == tm]
    if not hs or not tms:
        raise RuntimeError("missing H or TM sites after relaxation")
    hh = [s.get_distance(i, j) for ii, i in enumerate(hs) for j in hs[:ii]]
    tmh = []
    for i in tms:
        tmh.extend(sorted(s.get_distance(i, j) for j in hs)[:6])
    return {
        "min_H_H_A": float(min(hh)) if hh else None,
        "TM_H_nearest6_mean_A": float(np.mean(tmh)),
        "TM_H_nearest6_max_A": float(np.max(tmh)),
    }


def relax(candidate: str, outdir: str) -> None:
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    donor, tm, _ = CANDIDATES[candidate]
    model = CHGNet.load()
    records = []

    for atm in (0, 400):
        starts = []
        # Two nearby internal-coordinate seeds reduce dependence on a single H start.
        for xh in (0.235, 0.245):
            s0 = build(candidate, xh=xh)
            at = AseAtomsAdaptor.get_atoms(s0)
            at.calc = CHGNetCalculator(model=model)
            v0 = float(at.get_volume())
            tag = f"{candidate}_xh{xh:.3f}_{atm}atm"
            filt = FrechetCellFilter(
                at,
                scalar_pressure=atm * ATM_TO_GPA * GPA_TO_EV_A3,
            )
            FIRE(filt, logfile=str(out / f"{tag}.log")).run(fmax=0.045, steps=600)
            sf = AseAtomsAdaptor.get_structure(at)
            forces = np.asarray(at.get_forces())
            fmax = float(np.linalg.norm(forces, axis=1).max())
            stress = np.asarray(at.get_stress(voigt=True)) * 160.21766208
            energy = float(at.get_potential_energy())
            vol = float(at.get_volume())
            p_gpa = atm * ATM_TO_GPA
            enthalpy = energy + p_gpa * vol * GPA_TO_EV_A3
            hm = h_metrics(sf, tm)
            dmin = pair_min_distance(sf)
            gross = bool(
                fmax <= 0.08
                and dmin >= 0.75
                and hm["min_H_H_A"] is not None
                and hm["min_H_H_A"] >= 0.90
                and 1.25 <= hm["TM_H_nearest6_mean_A"] <= 2.20
                and hm["TM_H_nearest6_max_A"] <= 2.35
                and abs(float(-np.mean(stress[:3])) - p_gpa) <= 0.8
            )
            cif = out / f"{tag}.cif"
            sf.to(filename=str(cif))
            rec = {
                "candidate": candidate,
                "donor": donor,
                "tm": tm,
                "pressure_atm": atm,
                "xh_start": xh,
                "formula": sf.composition.reduced_formula,
                "natoms": len(sf),
                "energy_eV_atom": energy / len(sf),
                "enthalpy_proxy_eV_atom": enthalpy / len(sf),
                "volume_A3": vol,
                "volume_ratio": vol / v0,
                "max_force_eV_A": fmax,
                "hydrostatic_GPa_from_stress": float(-np.mean(stress[:3])),
                "cell_lengths_A": [float(x) for x in at.cell.lengths()],
                "cell_angles_deg": [float(x) for x in at.cell.angles()],
                "min_pair_distance_A": dmin,
                **hm,
                "gross_pass": gross,
                "cif": str(cif),
            }
            records.append(rec)
            starts.append(rec)

        viable = [r for r in starts if r["gross_pass"]]
        selected = min(viable or starts, key=lambda r: r["enthalpy_proxy_eV_atom"])
        shutil.copy2(selected["cif"], out / f"selected_{atm}atm.cif")

    selected_by_p = {}
    for atm in (0, 400):
        rr = [r for r in records if r["pressure_atm"] == atm]
        viable = [r for r in rr if r["gross_pass"]]
        selected_by_p[str(atm)] = min(viable or rr, key=lambda r: r["enthalpy_proxy_eV_atom"])
    advance = bool(all(selected_by_p[str(p)]["gross_pass"] for p in (0, 400)))
    result = {
        "candidate": candidate,
        "selected": selected_by_p,
        "all_runs": records,
        "advance_struct": advance,
        "note": "CHGNet structural survival only; not phonon/dynamical stability and not superconductivity evidence.",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / ("ADVANCE_STRUCT" if advance else "REJECT_STRUCT")).write_text("1\n")
    print(json.dumps(result, indent=2))


def pseudo(ppdir: str, elem: str) -> str:
    root = Path(ppdir)
    el = elem.lower()
    cand = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        b = p.name.lower()
        if not b.endswith((".upf", ".upf.gz")):
            continue
        if b.startswith(el + ".") or b.startswith(el + "_") or b.startswith(el + "-"):
            cand.append(p)
    if not cand:
        cand = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.name.lower().startswith(el) and ".upf" in p.name.lower()
        ]
    if not cand:
        raise RuntimeError(f"missing SSSP pseudopotential for {elem}")
    p = sorted(cand, key=lambda x: (len(x.name), x.name.lower()))[0]
    dst = root / p.name
    if p.resolve() != dst.resolve():
        shutil.copy2(p, dst)
    return p.name


def write_pw(path: Path, s: Structure, ppdir: str, outdir: Path, prefix: str, mode: str) -> None:
    species = []
    for site in s:
        z = site.specie.symbol
        if z not in species:
            species.append(z)
    pmap = {z: pseudo(ppdir, z) for z in species}
    calc = "scf" if mode == "scf" else "nscf"
    lines = [
        "&CONTROL",
        f" calculation='{calc}',",
        f" prefix='{prefix}',",
        f" pseudo_dir='{ppdir}',",
        f" outdir='{outdir}',",
        " disk_io='low',",
        "/",
        "&SYSTEM",
        " ibrav=0,",
        f" nat={len(s)}, ntyp={len(species)},",
        " ecutwfc=50.0, ecutrho=400.0,",
        " occupations='smearing', smearing='mv', degauss=0.02,",
        "/",
        "&ELECTRONS",
        " conv_thr=1.d-8, mixing_beta=0.30, electron_maxstep=220,",
        "/",
        "ATOMIC_SPECIES",
    ]
    lines.extend(f" {z} {MASS[z]:.8f} {pmap[z]}" for z in species)
    lines.append("ATOMIC_POSITIONS crystal")
    for site in s:
        f = site.frac_coords
        lines.append(f" {site.specie.symbol} {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}")
    lines.append("CELL_PARAMETERS angstrom")
    lines.extend(" %.12f %.12f %.12f" % tuple(v) for v in s.lattice.matrix)
    lines.append("K_POINTS automatic")
    lines.append(" 6 6 6 0 0 0" if mode == "scf" else " 10 10 10 0 0 0")
    path.write_text("\n".join(lines) + "\n")


def run_input(exe: str, inp: Path, out: Path) -> int:
    with open(inp, "rb") as fi, open(out, "wb") as fo:
        return subprocess.run([exe], stdin=fi, stdout=fo, stderr=subprocess.STDOUT).returncode


def last_float(pattern: str, text: str) -> float | None:
    m = re.findall(pattern, text, re.I)
    return float(m[-1]) if m else None


def load_numeric(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            vals = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        raise RuntimeError(f"no numeric DOS rows in {path}")
    n = min(len(x) for x in rows)
    return np.asarray([x[:n] for x in rows], dtype=float)


def interp_at_ef(arr: np.ndarray, ef: float, col: int = 1) -> float:
    order = np.argsort(arr[:, 0])
    x = arr[order, 0]
    y = arr[order, col]
    return float(np.interp(ef, x, y))


def qe(candidate: str, cif: str, atm: int, ppdir: str, pw: str, outdir: str) -> None:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch = (out / "scratch").resolve()
    scratch.mkdir(exist_ok=True)
    s = Structure.from_file(cif)
    prefix = f"hyd_{candidate}_{atm}"

    scf_in, scf_out = out / "scf.in", out / "scf.out"
    write_pw(scf_in, s, ppdir, scratch, prefix, "scf")
    scf_rc = run_input(pw, scf_in, scf_out)
    scf_txt = scf_out.read_text(errors="ignore")
    e_ry = last_float(r"!\s+total energy\s+=\s+([-0-9.Ee+]+)\s+Ry", scf_txt)
    ef = last_float(r"the Fermi energy is\s+([-0-9.Ee+]+)\s+ev", scf_txt)

    nscf_in, nscf_out = out / "nscf.in", out / "nscf.out"
    write_pw(nscf_in, s, ppdir, scratch, prefix, "nscf")
    nscf_rc = run_input(pw, nscf_in, nscf_out) if scf_rc == 0 else 99
    nscf_txt = nscf_out.read_text(errors="ignore") if nscf_out.exists() else ""
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
        total_arr = load_numeric(dos_file)
        total_ef = max(0.0, interp_at_ef(total_arr, ef, 1))
        # QE projwfc filenames contain atom labels such as '(H)'. H has only 1s,
        # so the LDOS column can be summed across H-projected files without
        # orbital double counting.
        hfiles = [Path(p) for p in glob.glob(str(out / "pdos.pdos_atm#*(H)*"))]
        hvals = []
        for p in hfiles:
            arr = load_numeric(p)
            hvals.append(max(0.0, interp_at_ef(arr, ef, 1)))
        if hvals:
            h_ef = float(sum(hvals))
            descriptor = float(math.sqrt(total_ef * h_ef))

    result = {
        "candidate": candidate,
        "pressure_atm": atm,
        "natoms": len(s),
        "formula": s.composition.reduced_formula,
        "scf_rc": scf_rc,
        "nscf_rc": nscf_rc,
        "dos_rc": dos_rc,
        "projwfc_rc": proj_rc,
        "total_energy_Ry": e_ry,
        "total_energy_eV": e_ry * RY_TO_EV if e_ry is not None else None,
        "fermi_eV": ef,
        "total_DOS_EF_states_per_eV_cell": total_ef,
        "H_projected_DOS_EF_states_per_eV_cell": h_ef,
        "xu_dos_descriptor": descriptor,
        "metallic_screen": bool(total_ef is not None and total_ef >= 0.05),
        "n_H_pdos_files": len(hfiles),
        "note": "DOS/H-DOS descriptor is a ranking aid only. It is not EPC, lambda, phonon stability, or Tc evidence.",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if scf_rc != 0 or nscf_rc != 0 or dos_rc != 0 or proj_rc != 0 or descriptor is None:
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relax")
    r.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    r.add_argument("--out", required=True)
    q = sub.add_parser("qe")
    q.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    q.add_argument("--cif", required=True)
    q.add_argument("--atm", type=int, required=True, choices=[0, 400])
    q.add_argument("--ppdir", required=True)
    q.add_argument("--pw", default="/opt/espresso/7.5/pw.x")
    q.add_argument("--out", required=True)
    z = ap.parse_args()
    if z.cmd == "relax":
        relax(z.candidate, z.out)
    else:
        qe(z.candidate, z.cif, z.atm, z.ppdir, z.pw, z.out)


if __name__ == "__main__":
    main()
