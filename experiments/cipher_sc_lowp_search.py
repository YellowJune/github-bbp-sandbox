from __future__ import annotations

import json, math, os, re, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pymatgen.core import Composition, Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

OUT = Path("artifacts/cipher_sc_lowp")
OUT.mkdir(parents=True, exist_ok=True)

MAX_ATM = 400.0
ATM_TO_GPA = 0.000101325
MAX_GPA = MAX_ATM * ATM_TO_GPA
GPA_TO_EV_A3 = 1.0 / 160.21766208
MAX_PRESSURE_EV_A3 = MAX_GPA * GPA_TO_EV_A3
PARENT_TC = 133.5

SUPERCON_URL = (
    "https://raw.githubusercontent.com/aimat-lab/3DSC/"
    "8d69ca9c94b83d387378549225d3b7c3af85ca42/"
    "superconductors_3D/data/source/SuperCon/raw/Supercon_data_by_2018_Stanev.csv"
)
PARENT_CIF_URL = (
    "https://raw.githubusercontent.com/aimat-lab/3DSC/"
    "8d69ca9c94b83d387378549225d3b7c3af85ca42/"
    "superconductors_3D/data/source/MP/raw/cifs/mp-22601.cif"
)


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def comp_features(formula_or_comp):
    comp = formula_or_comp if isinstance(formula_or_comp, Composition) else Composition(formula_or_comp)
    el_amt = comp.get_el_amt_dict()
    if "H" in el_amt and el_amt["H"] > 1e-12:
        raise ValueError("hydrogen-containing composition")
    total = sum(el_amt.values())
    els = []
    weights = []
    for sym, amt in el_amt.items():
        e = Element(sym)
        els.append(e)
        weights.append(amt / total)
    w = np.array(weights, float)

    def arr(attr, fallback=0.0):
        vals = []
        for e in els:
            try:
                val = getattr(e, attr)
                vals.append(safe_float(val, fallback))
            except Exception:
                vals.append(fallback)
        return np.array(vals, float)

    props = [arr("Z"), arr("atomic_mass"), arr("X"), arr("row"), arr("group")]
    feat = []
    for v in props:
        m = float(np.sum(w * v))
        s = float(np.sqrt(np.sum(w * (v - m) ** 2)))
        feat.extend([m, s, float(v.min()), float(v.max())])
    entropy = float(-np.sum(w * np.log(w + 1e-12)))
    frac_o = el_amt.get("O", 0.0) / total
    frac_cu = el_amt.get("Cu", 0.0) / total
    frac_b = el_amt.get("B", 0.0) / total
    frac_c = el_amt.get("C", 0.0) / total
    feat.extend([len(els), entropy, frac_o, frac_cu, frac_b, frac_c, total])
    return np.array(feat, float)


def canonical(comp):
    c = Composition(comp)
    return c.reduced_formula.replace(" ", "")


def load_training():
    df = pd.read_csv(SUPERCON_URL)
    rows, X, y = [], [], []
    for _, r in df.iterrows():
        try:
            c = Composition(str(r["name"]))
            if any(e.symbol == "H" for e in c.elements):
                continue
            tc = float(r["Tc"])
            if not np.isfinite(tc) or tc < 0:
                continue
            f = comp_features(c)
        except Exception:
            continue
        rows.append((str(r["name"]), tc, canonical(c)))
        X.append(f); y.append(tc)
    return pd.DataFrame(rows, columns=["formula", "Tc", "canonical"]), np.vstack(X), np.array(y)


def build_models(X, y, seed=20260904):
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    ridge.fit(X, y)
    tree = ExtraTreesRegressor(
        n_estimators=350, min_samples_leaf=3, max_features=0.8,
        random_state=seed, n_jobs=-1
    )
    tree.fit(X, y)
    return ridge, tree


def candidate_specs():
    # All are generated interventions on HgBa2Ca2Cu3O8. None contains H.
    return [
        ("Hg_to_Tl_25", [("Hg", "Tl", 1)]),
        ("Hg_to_Bi_25", [("Hg", "Bi", 1)]),
        ("Hg_to_Au_25", [("Hg", "Au", 1)]),
        ("Hg_to_Re_25", [("Hg", "Re", 1)]),
        ("Ba_to_Sr_12p5", [("Ba", "Sr", 1)]),
        ("Ba_to_K_12p5", [("Ba", "K", 1)]),
        ("Ca_to_Sr_12p5", [("Ca", "Sr", 1)]),
        ("Ca_to_Y_12p5", [("Ca", "Y", 1)]),
        ("dual_Tl_K", [("Hg", "Tl", 1), ("Ba", "K", 1)]),
        ("dual_Bi_K", [("Hg", "Bi", 1), ("Ba", "K", 1)]),
        ("dual_Tl_Y", [("Hg", "Tl", 1), ("Ca", "Y", 1)]),
    ]


def substitute(struct, ops):
    s = struct.copy()
    for host, dop, nrep in ops:
        inds = [i for i, site in enumerate(s) if site.specie.symbol == host]
        if len(inds) < nrep:
            raise RuntimeError(f"not enough {host} sites")
        # deterministic spread: first site, then maximally separated in index ordering.
        picks = inds[:nrep]
        for i in picks:
            s.replace(i, dop)
    return s


def bootstrap_delta(X, y, f_parent, f_cand, n=36, seed=4404):
    rng = np.random.default_rng(seed)
    vals = []
    for b in range(n):
        idx = rng.integers(0, len(y), len(y))
        m = make_pipeline(StandardScaler(), Ridge(alpha=10 ** rng.uniform(0.8, 1.8)))
        m.fit(X[idx], y[idx])
        vals.append(float(m.predict(f_cand[None])[0] - m.predict(f_parent[None])[0]))
    return np.array(vals)


def relax_structure(struct, model, pressure_gpa, tag):
    atoms = AseAtomsAdaptor.get_atoms(struct)
    calc = CHGNetCalculator(model=model)
    atoms.calc = calc
    init_vol = float(atoms.get_volume())
    filt = FrechetCellFilter(atoms, scalar_pressure=pressure_gpa * GPA_TO_EV_A3)
    opt = FIRE(filt, logfile=str(OUT / f"{tag}_{pressure_gpa:.5f}GPa.log"))
    opt.run(fmax=0.08, steps=180)
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
    energy_pa = float(atoms.get_potential_energy() / len(atoms))
    final_vol = float(atoms.get_volume())
    st = AseAtomsAdaptor.get_structure(atoms)
    st.to(filename=str(OUT / f"{tag}_{pressure_gpa:.5f}GPa_relaxed.cif"))
    return st, {
        "pressure_GPa": pressure_gpa,
        "pressure_atm": pressure_gpa / ATM_TO_GPA,
        "converged_force": bool(max_force < 0.10),
        "max_force_eV_A": max_force,
        "energy_eV_atom": energy_pa,
        "volume_ratio": final_vol / init_vol,
        "natoms": len(atoms),
    }


def gamma_hessian(relaxed_struct, model, tag, delta=0.012):
    atoms0 = AseAtomsAdaptor.get_atoms(relaxed_struct)
    n = len(atoms0)
    nd = 3 * n
    H = np.zeros((nd, nd), float)
    for j in range(nd):
        ai, ax = divmod(j, 3)
        ap = atoms0.copy(); am = atoms0.copy()
        pp = ap.get_positions(); pm = am.get_positions()
        pp[ai, ax] += delta; pm[ai, ax] -= delta
        ap.set_positions(pp); am.set_positions(pm)
        ap.calc = CHGNetCalculator(model=model)
        am.calc = CHGNetCalculator(model=model)
        fp = ap.get_forces().reshape(-1)
        fm = am.get_forces().reshape(-1)
        H[:, j] = -(fp - fm) / (2 * delta)
    H = 0.5 * (H + H.T)
    masses = np.repeat(atoms0.get_masses(), 3)
    D = H / np.sqrt(np.outer(masses, masses))
    ev = eigvalsh(D)
    # Remove three modes closest to zero as translations; this is a Gamma-only check.
    keep = np.ones(len(ev), dtype=bool)
    keep[np.argsort(np.abs(ev))[:3]] = False
    phys = ev[keep]
    np.savetxt(OUT / f"{tag}_gamma_eigenvalues.txt", ev)
    return {
        "min_mass_weighted_hessian": float(phys.min()),
        "negative_modes_below_-1e-3": int(np.sum(phys < -1e-3)),
        "negative_modes_below_-1e-2": int(np.sum(phys < -1e-2)),
        "gamma_pass_loose": bool(np.sum(phys < -1e-2) == 0),
        "note": "Gamma-point finite-displacement Hessian only; not a full phonon-dispersion proof",
    }


def main():
    train, X, y = load_training()
    ridge, tree = build_models(X, y)
    urllib.request.urlretrieve(PARENT_CIF_URL, OUT / "parent_mp22601.cif")
    parent = Structure.from_file(OUT / "parent_mp22601.cif")
    parent.make_supercell([2, 2, 1])
    assert all(site.specie.symbol != "H" for site in parent)
    f_parent = comp_features(parent.composition)
    parent_pred_r = float(ridge.predict(f_parent[None])[0])
    parent_pred_t = float(tree.predict(f_parent[None])[0])

    seen = set(train["canonical"])
    candidates = []
    structs = {}
    for name, ops in candidate_specs():
        s = substitute(parent, ops)
        if any(site.specie.symbol == "H" for site in s):
            continue
        f = comp_features(s.composition)
        pr = float(ridge.predict(f[None])[0])
        pt = float(tree.predict(f[None])[0])
        # Anchor only the learned intervention delta to the measured parent Tc.
        delta_r = pr - parent_pred_r
        delta_t = pt - parent_pred_t
        tc_anchor_mean = PARENT_TC + 0.5 * (delta_r + delta_t)
        boot = bootstrap_delta(X, y, f_parent, f, seed=4404 + len(candidates))
        anchored_boot = PARENT_TC + boot
        canon = canonical(s.composition)
        candidates.append({
            "name": name,
            "formula": s.composition.reduced_formula,
            "canonical": canon,
            "in_training_corpus": canon in seen,
            "ridge_delta_K": delta_r,
            "tree_delta_K": delta_t,
            "Tc_anchor_mean_K": tc_anchor_mean,
            "Tc_anchor_p10_K": float(np.quantile(anchored_boot, 0.10)),
            "Tc_anchor_p90_K": float(np.quantile(anchored_boot, 0.90)),
        })
        structs[name] = s

    screen = pd.DataFrame(candidates)
    # Prefer novel, high lower-bound predictions. Novelty means exact reduced composition absent.
    screen["novel"] = ~screen["in_training_corpus"]
    screen = screen.sort_values(["novel", "Tc_anchor_p10_K", "Tc_anchor_mean_K"], ascending=False)
    screen.to_csv(OUT / "candidate_screen.csv", index=False)

    model = CHGNet.load()
    sim_rows = []
    relaxed_cache = {}

    # Always simulate parent plus the six best newly generated candidates.
    run_names = ["PARENT"] + screen[screen["novel"]].head(6)["name"].tolist()
    for nm in run_names:
        s = parent if nm == "PARENT" else structs[nm]
        for p in [0.0, MAX_GPA]:
            try:
                rst, rec = relax_structure(s, model, p, nm)
                rec.update({"name": nm, "formula": rst.composition.reduced_formula, "error": ""})
                relaxed_cache[(nm, p)] = rst
            except Exception as e:
                rec = {"name": nm, "formula": s.composition.reduced_formula, "pressure_GPa": p,
                       "pressure_atm": p / ATM_TO_GPA, "converged_force": False,
                       "max_force_eV_A": np.nan, "energy_eV_atom": np.nan,
                       "volume_ratio": np.nan, "natoms": len(s), "error": repr(e)}
            sim_rows.append(rec)
    sims = pd.DataFrame(sim_rows)
    sims.to_csv(OUT / "relaxation_results.csv", index=False)

    # Reality-ranked candidate must survive both 0 and 400 atm relaxations.
    pass_names = []
    for nm in run_names[1:]:
        g = sims[sims["name"] == nm]
        ok = len(g) == 2 and bool(g["converged_force"].all()) and bool(g["volume_ratio"].between(0.70, 1.30).all())
        if ok:
            pass_names.append(nm)

    if pass_names:
        rank = screen.set_index("name")
        top = max(pass_names, key=lambda n: rank.loc[n, "Tc_anchor_p10_K"])
    else:
        top = None

    gamma_rows = []
    # Full finite-displacement Gamma Hessian for parent and best survivor.
    for nm in ["PARENT", top] if top else ["PARENT"]:
        if nm is None or (nm, MAX_GPA) not in relaxed_cache:
            continue
        try:
            gr = gamma_hessian(relaxed_cache[(nm, MAX_GPA)], model, nm)
            gr.update({"name": nm, "error": ""})
        except Exception as e:
            gr = {"name": nm, "min_mass_weighted_hessian": np.nan,
                  "negative_modes_below_-1e-3": -1, "negative_modes_below_-1e-2": -1,
                  "gamma_pass_loose": False, "note": "failed", "error": repr(e)}
        gamma_rows.append(gr)
    gamma = pd.DataFrame(gamma_rows)
    gamma.to_csv(OUT / "gamma_hessian_results.csv", index=False)

    selected = None
    if top is not None:
        row = screen[screen["name"] == top].iloc[0].to_dict()
        grow = gamma[gamma["name"] == top]
        row["relaxation_pass_0_to_400atm"] = True
        row["gamma_pass_loose"] = bool(len(grow) and grow.iloc[0]["gamma_pass_loose"])
        row["credible_300K"] = bool(
            row["Tc_anchor_p10_K"] >= 300
            and row["relaxation_pass_0_to_400atm"]
            and row["gamma_pass_loose"]
        )
        selected = row

    summary = {
        "constraints": {"hydrogen_allowed": False, "max_pressure_atm": MAX_ATM, "max_pressure_GPa": MAX_GPA},
        "training_rows_hydrogen_free": int(len(train)),
        "generated_candidates": int(len(screen)),
        "simulated_candidates": run_names,
        "parent_formula": parent.composition.reduced_formula,
        "parent_reference_Tc_K": PARENT_TC,
        "selected": selected,
        "claim_policy": "No candidate is called 300 K unless lower-bound composition prediction >=300 K, both 0/400 atm CHGNet relaxations converge, and the Gamma Hessian has no substantial negative mode. Even then this is not experimental or full many-body validation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("MAX PRESSURE GPa", MAX_GPA)
    print("H-FREE TRAINING ROWS", len(train))
    print("\nTOP SCREEN")
    print(screen.head(11).to_string(index=False))
    print("\nRELAXATION")
    print(sims.to_string(index=False))
    print("\nGAMMA")
    print(gamma.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
