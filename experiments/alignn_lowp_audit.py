from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from alignn.deprecated.pretrained import get_figshare_model_pure, _atom_features_from_config
from alignn.torch_graph_builder import build_pure_torch_graph

OUT = Path('artifacts/cipher_sc_lowp')


def load(name):
    model, cfg = get_figshare_model_pure(name, device='cpu')
    return model.eval(), cfg


def predict(model, cfg, atoms):
    cutoff = float(cfg.get('cutoff', 8.0))
    max_neighbors = int(cfg.get('max_neighbors', 12))
    af = _atom_features_from_config(cfg)
    g, lg = build_pure_torch_graph(
        atoms=atoms,
        two_body_cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features=af,
        compute_line_graph=True,
        device='cpu',
    )
    lat = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())
    with torch.no_grad():
        out = model([g, lg, lat])
    if isinstance(out, dict):
        out = out['out']
    return float(np.asarray(out.detach().cpu()).reshape(-1)[0])


def main():
    models = {}
    for name in ['Tc_supercon', 'ehull', 'optb88vdw_bandgap', 'formation_energy_peratom']:
        print('LOADING', name, flush=True)
        models[name] = load(name)

    rows = []
    for cif in sorted(OUT.glob('*_0.04053GPa_relaxed.cif')):
        atoms = Atoms.from_cif(str(cif))
        formula = atoms.composition.reduced_formula
        if 'H' in [str(x) for x in atoms.elements]:
            raise RuntimeError('Hydrogen found in supposedly H-free structure')
        rec = {'file': cif.name, 'formula': formula}
        for name, (m, cfg) in models.items():
            try:
                rec[name] = predict(m, cfg, atoms)
            except Exception as e:
                rec[name] = np.nan
                rec[name + '_error'] = repr(e)
        rows.append(rec)
        print('ALIGNN', rec, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'alignn_independent_audit.csv', index=False)

    # Conservative cross-model status. ALIGNN is independent of the composition model,
    # but this is still an ML prediction, not a many-body proof.
    if len(df):
        df['alignn_structure_plausible'] = (
            (df['ehull'].fillna(999) <= 0.10) &
            (df['optb88vdw_bandgap'].fillna(999) <= 0.50)
        )
        df['alignn_300K'] = df['Tc_supercon'].fillna(-999) >= 300
        df.to_csv(OUT / 'alignn_independent_audit.csv', index=False)

    summary = {
        'n_structures': int(len(df)),
        'max_alignn_Tc_K': float(df['Tc_supercon'].max()) if len(df) else None,
        'max_alignn_Tc_formula': (
            str(df.loc[df['Tc_supercon'].idxmax(), 'formula'])
            if len(df) and df['Tc_supercon'].notna().any() else None
        ),
        'n_alignn_300K': int(df['alignn_300K'].sum()) if len(df) else 0,
        'note': 'Independent ALIGNN structure-property audit; not experimental or full many-body validation.'
    }
    (OUT / 'alignn_summary.json').write_text(json.dumps(summary, indent=2))
    print('ALIGNN SUMMARY', json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
