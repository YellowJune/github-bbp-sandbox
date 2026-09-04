from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ATM_TO_GPA = 0.000101325
GPA_TO_EV_A3 = 1 / 160.21766208

# 3x2 infinite-layer parent. Six A sites, six Cu, twelve ligand sites.
def base_sites(a0: float):
    c0 = 3.20
    sy, fr, tags = [], [], []
    for ix in range(3):
        for iy in range(2):
            sy += ['Mg', 'Cu', 'O', 'O']
            fr += [
                [(ix + .5) / 3, (iy + .5) / 2, .5],
                [ix / 3, iy / 2, 0],
                [(ix + .5) / 3, iy / 2, 0],
                [ix / 3, (iy + .5) / 2, 0],
            ]
            tags += [('A', ix, iy), ('Cu', ix, iy), ('Lx', ix, iy), ('Ly', ix, iy)]
    return sy, fr, tags, Lattice.orthorhombic(3*a0, 2*a0, c0)


def set_tag(sy, tags, tag, elem):
    sy[tags.index(tag)] = elem


def build(name: str, a0: float):
    sy, fr, tags, lat = base_sites(a0)
    # Control: one Na hole-dopant, otherwise Mg, all oxygen.
    if name == 'NaMg5Cu6O12':
        set_tag(sy, tags, ('A', 0, 0), 'Na')
    else:
        # Candidate composition is fixed: Na1 Mg2 Sc3 Cu6 N3 O9.
        # Only ordering is varied. These are intentionally distinct starting motifs.
        if name == 'candidate_dispersed':
            A_sc = [('A',0,1), ('A',1,0), ('A',2,1)]
            A_na = ('A',0,0)
            N_sites = [('Lx',0,0), ('Ly',1,1), ('Lx',2,0)]
        elif name == 'candidate_stripe':
            A_sc = [('A',0,0), ('A',1,0), ('A',2,0)]
            A_na = ('A',0,1)
            N_sites = [('Lx',0,0), ('Lx',1,0), ('Lx',2,0)]
        elif name == 'candidate_clustered':
            A_sc = [('A',0,0), ('A',1,0), ('A',0,1)]
            A_na = ('A',2,1)
            N_sites = [('Lx',0,0), ('Ly',0,0), ('Lx',1,0)]
        else:
            raise ValueError(name)
        set_tag(sy, tags, A_na, 'Na')
        for t in A_sc:
            set_tag(sy, tags, t, 'Sc')
        for t in N_sites:
            set_tag(sy, tags, t, 'N')

    s = Structure(lat, sy, fr)
    if any(x.specie.symbol == 'H' for x in s):
        raise RuntimeError('hydrogen forbidden')
    return s


def metrics(at):
    sy = at.get_chemical_symbols()
    d = at.get_all_distances(mic=True)
    cu_idx = [i for i,x in enumerate(sy) if x == 'Cu']
    lig_idx = [i for i,x in enumerate(sy) if x in ('O','N')]
    first = []
    cu_coord = []
    first_by_type = {'O': [], 'N': []}
    for i in cu_idx:
        pairs = sorted([(float(d[i,j]), sy[j]) for j in lig_idx], key=lambda z:z[0])[:4]
        first += [x[0] for x in pairs]
        cu_coord.append(sum(x[0] < 2.25 for x in pairs))
        for dist, elem in pairs:
            first_by_type[elem].append(dist)
    lig_lig = []
    cat_cat = []
    for i in range(len(sy)):
        for j in range(i+1, len(sy)):
            if sy[i] in ('O','N') and sy[j] in ('O','N'):
                lig_lig.append(float(d[i,j]))
            if sy[i] not in ('O','N') and sy[j] not in ('O','N'):
                cat_cat.append(float(d[i,j]))
    rec = {
        'median_first_CuLig_A': float(np.median(first)),
        'p90_first_CuLig_A': float(np.quantile(first,.90)),
        'min_first_CuLig_A': float(min(first)),
        'max_first_CuLig_A': float(max(first)),
        'mean_Cu_ligand_coord_lt2p25': float(np.mean(cu_coord)),
        'min_ligand_ligand_A': float(min(lig_lig)),
        'min_cation_cation_A': float(min(cat_cat)),
    }
    for elem in ('O','N'):
        arr = first_by_type[elem]
        rec[f'n_first_Cu{elem}_bonds'] = int(len(arr))
        rec[f'median_first_Cu{elem}_A'] = float(np.median(arr)) if arr else None
        rec[f'min_first_Cu{elem}_A'] = float(min(arr)) if arr else None
    return rec


def run(name, a0, pressure_gpa, out, model):
    at = AseAtomsAdaptor.get_atoms(build(name, a0))
    v0 = at.get_volume()
    m0 = metrics(at)
    at.calc = CHGNetCalculator(model=model)
    filt = FrechetCellFilter(at, scalar_pressure=pressure_gpa * GPA_TO_EV_A3)
    FIRE(filt, logfile=str(out/f'{name}_a{a0:.2f}_p{pressure_gpa:.5f}.log')).run(
        fmax=.055, steps=300
    )
    forces = at.get_forces()
    stress = at.get_stress(voigt=True) * 160.21766208
    m = metrics(at)
    rec = {
        'name': name,
        'initial_a_A': a0,
        'pressure_GPa_target': pressure_gpa,
        'pressure_atm_target': pressure_gpa / ATM_TO_GPA,
        'formula': at.get_chemical_formula(),
        'natoms': len(at),
        'max_force_eV_A': float(np.linalg.norm(forces,axis=1).max()),
        'energy_eV_atom': float(at.get_potential_energy()/len(at)),
        'volume_ratio': float(at.get_volume()/v0),
        'cell_lengths_A': [float(x) for x in at.cell.lengths()],
        'cell_angles_deg': [float(x) for x in at.cell.angles()],
        'stress_GPa_voigt': [float(x) for x in stress],
    }
    rec.update({'initial_'+k:v for k,v in m0.items()})
    rec.update({'final_'+k:v for k,v in m.items()})
    # Structural screen only. It is not a phonon-stability or superconductivity claim.
    rec['gross_structure_pass'] = bool(
        rec['max_force_eV_A'] < .09 and
        .65 < rec['volume_ratio'] < 1.35 and
        m['mean_Cu_ligand_coord_lt2p25'] >= 3.5 and
        1.55 < m['min_first_CuLig_A'] < 2.20 and
        m['p90_first_CuLig_A'] < 2.15 and
        m['min_ligand_ligand_A'] > 1.40
    )
    # Purely geometric energy-scale target from the prior mechanism screen.
    rec['bandwidth_geometry_target_pass'] = bool(m['median_first_CuLig_A'] <= 1.8622)
    AseAtomsAdaptor.get_structure(at).to(
        filename=str(out/f'{name}_a{a0:.2f}_p{pressure_gpa:.5f}_relaxed.cif')
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', choices=['NaMg5Cu6O12','candidate_dispersed','candidate_stripe','candidate_clustered'], required=True)
    ap.add_argument('--a0', type=float, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = CHGNet.load()
    rows = [
        run(args.name,args.a0,0.0,out,model),
        run(args.name,args.a0,400*ATM_TO_GPA,out,model),
    ]
    (out/'result.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
