"""
reestimar_base_ampliada.py — Reestimacao com base ampliada (V2 corrigido)
============================================================================
Reestima os 4 parametros faltantes (CCOO/CNH2 e CNH2/COO) usando
os 3 sistemas de Souza et al. (2025): butil + pentil + hexil acetato
(24 tie-lines no total).

CORRECAO V2: a contagem de CH3 nos acetatos era 2, deveria ser 1.
  O CH3 do lado do acetato ja esta dentro do grupo CH3COO (sg21).
  Decomposicao correta:
    Butil acetato  (C6H12O2): 1 CH3 + 3 CH2 + 1 CH3COO  -> 6 C
    Pentil acetato (C7H14O2): 1 CH3 + 4 CH2 + 1 CH3COO  -> 7 C
    Hexil acetato  (C8H16O2): 1 CH3 + 5 CH2 + 1 CH3COO  -> 8 C

PRE-REQUISITOS:
  Mesma pasta que params.py, gamma.py, flash.py, stability.py, diagram.py

EXECUCAO:
    python reestimar_base_ampliada_v2.py
"""

import numpy as np
import time
import json
import os
import importlib

import params as p
import gamma
import flash
import stability

T = 298.15
OUTDIR = 'resultados'
os.makedirs(OUTDIR, exist_ok=True)


# ===========================================================================
# DADOS EXPERIMENTAIS — Souza et al. (2025)
# (raf, ext) = (acetate-rich, IL-rich)
# ===========================================================================

EXP_BUTIL = [  # Tabela 4 — 9 tie-lines (Butil acetato C6H12O2)
    (np.array([0.961,  0.0224, 0.017]),   np.array([0.124,  0.768,  0.108])),
    (np.array([0.952,  0.0225, 0.025]),   np.array([0.139,  0.665,  0.196])),
    (np.array([0.942,  0.0225, 0.035]),   np.array([0.161,  0.550,  0.289])),
    (np.array([0.937,  0.0225, 0.040]),   np.array([0.1732, 0.505,  0.322])),
    (np.array([0.8840, 0.0223, 0.0936]),  np.array([0.2251, 0.3886, 0.386])),
    (np.array([0.920,  0.0225, 0.0573]),  np.array([0.2002, 0.4350, 0.365])),
    (np.array([0.8530, 0.0226, 0.1245]),  np.array([0.2513, 0.3493, 0.399])),
    (np.array([0.8189, 0.0234, 0.1577]),  np.array([0.2823, 0.3098, 0.4078])),
    (np.array([0.8096, 0.0238, 0.1666]),  np.array([0.2963, 0.2938, 0.4099])),
]

EXP_PENTIL = [  # Tabela 5 — 6 tie-lines (Pentil acetato C7H14O2)
    (np.array([0.968, 0.0250, 0.007]),   np.array([0.102,  0.775,  0.123])),
    (np.array([0.950, 0.0247, 0.025]),   np.array([0.1165, 0.642,  0.241])),
    (np.array([0.935, 0.0246, 0.041]),   np.array([0.1266, 0.575,  0.298])),
    (np.array([0.926, 0.0246, 0.049]),   np.array([0.1339, 0.540,  0.326])),
    (np.array([0.908, 0.0245, 0.067]),   np.array([0.1497, 0.4843, 0.3659])),
    (np.array([0.877, 0.0247, 0.098]),   np.array([0.1679, 0.4390, 0.3931])),
]

EXP_HEXIL = [  # Tabela 6 — 9 tie-lines (Hexil acetato C8H16O2)
    (np.array([0.942,  0.0242, 0.033]),  np.array([0.072,  0.829,  0.100])),
    (np.array([0.915,  0.0236, 0.061]),  np.array([0.084,  0.733,  0.183])),
    (np.array([0.904,  0.0234, 0.073]),  np.array([0.093,  0.649,  0.259])),
    (np.array([0.887,  0.0233, 0.090]),  np.array([0.1003, 0.580,  0.320])),
    (np.array([0.863,  0.0231, 0.113]),  np.array([0.1140, 0.501,  0.385])),
    (np.array([0.879,  0.0232, 0.098]),  np.array([0.1102, 0.518,  0.371])),
    (np.array([0.850,  0.0232, 0.127]),  np.array([0.1198, 0.4779, 0.402])),
    (np.array([0.820,  0.0236, 0.156]),  np.array([0.1444, 0.4103, 0.4452])),
    (np.array([0.841,  0.0233, 0.135]),  np.array([0.1336, 0.4360, 0.4304])),
]


# ===========================================================================
# DECOMPOSICAO EM GRUPOS UNIFAC (CORRIGIDA)
# ===========================================================================
# Estrutura: CH3-(CH2)n-O-CO-CH3
# O grupo CH3COO (sg21) ja inclui um CH3 + um COO juntos
# Logo, so precisamos contar 1 CH3 (do lado da cadeia alifatica) + CH2 + CH3COO

NU_BUTIL  = {1: 1, 2: 3, 21: 1}  # 1 CH3 + 3 CH2 + 1 CH3COO  -> C = 1+3+2 = 6
NU_PENTIL = {1: 1, 2: 4, 21: 1}  # 1 CH3 + 4 CH2 + 1 CH3COO  -> C = 7
NU_HEXIL  = {1: 1, 2: 5, 21: 1}  # 1 CH3 + 5 CH2 + 1 CH3COO  -> C = 8

# Verificacao
def conta_C(nu_dict):
    c_per_sg = {1: 1, 2: 1, 3: 1, 4: 1, 6: 2, 14: 0, 16: 0, 21: 2, 22: 2, 77: 1, 29: 1, 32: 1}
    return sum(nu_dict.get(sg, 0) * c_per_sg.get(sg, 0) for sg in nu_dict)

assert conta_C(NU_BUTIL) == 6, f"Butil: C={conta_C(NU_BUTIL)} != 6"
assert conta_C(NU_PENTIL) == 7, f"Pentil: C={conta_C(NU_PENTIL)} != 7"
assert conta_C(NU_HEXIL) == 8, f"Hexil: C={conta_C(NU_HEXIL)} != 8"

print(f"Decomposicao verificada:")
print(f"  Butil  acetato: C={conta_C(NU_BUTIL)} = 6 OK")
print(f"  Pentil acetato: C={conta_C(NU_PENTIL)} = 7 OK")
print(f"  Hexil  acetato: C={conta_C(NU_HEXIL)} = 8 OK")
print()

SYSTEMS_FIT = [
    {'name': 'butil',  'nu': NU_BUTIL,  'data': EXP_BUTIL},
    {'name': 'pentil', 'nu': NU_PENTIL, 'data': EXP_PENTIL},
    {'name': 'hexil',  'nu': NU_HEXIL,  'data': EXP_HEXIL},
]

NU_2HEAB  = {1: 1, 2: 3, 14: 1, 77: 1, 29: 1}
NU_ETANOL = {1: 1, 2: 1, 14: 1}


# ===========================================================================
# FUNCOES
# ===========================================================================

def make_model(pv, nu_acetato):
    p.INTERACTION_PARAMS[(11, 14)] = (pv[0], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 11)] = (pv[1], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 41)] = (pv[2], 0.0, 0.0)
    p.INTERACTION_PARAMS[(41, 14)] = (pv[3], 0.0, 0.0)
    importlib.reload(gamma); importlib.reload(flash); importlib.reload(stability)
    from gamma import UNIFACLL
    return UNIFACLL([nu_acetato, NU_2HEAB, NU_ETANOL], T)


def rmsd_for_system(pv, nu_acetato, tie_lines):
    try:
        model = make_model(pv, nu_acetato)
    except:
        return 1.0
    from stability import is_unstable
    from flash import flash_ell

    sse = 0.0
    for xa_e, xb_e in tie_lines:
        z = 0.5 * xa_e + 0.5 * xb_e
        z = np.clip(z, 1e-6, 1); z /= z.sum()
        bif = is_unstable(z, model)
        if not bif:
            for psi_t in [0.3, 0.4, 0.6, 0.7]:
                z2 = psi_t * xb_e + (1 - psi_t) * xa_e
                z2 = np.clip(z2, 1e-6, 1); z2 /= z2.sum()
                if is_unstable(z2, model):
                    z = z2; bif = True; break
        if not bif:
            sse += 0.3; continue
        xa, xb, _, ok, _ = flash_ell(z, model)
        if not ok:
            sse += 0.3; continue
        if xa[0] < xb[0]:
            xa, xb = xb, xa
        sse += np.sum((xa - xa_e) ** 2) + np.sum((xb - xb_e) ** 2)
    return np.sqrt(sse / (2 * len(tie_lines) * 3))


def rmsd_combined(pv):
    sses = []
    n_total = 0
    for s in SYSTEMS_FIT:
        n = len(s['data'])
        rmsd_s = rmsd_for_system(pv, s['nu'], s['data'])
        sses.append((rmsd_s ** 2) * (2 * n * 3))
        n_total += n
    return np.sqrt(sum(sses) / (2 * n_total * 3))


def main():
    print("=" * 70)
    print("REESTIMACAO COM BASE AMPLIADA — 3 SISTEMAS DE SOUZA et al. (2025)")
    print("=" * 70)
    print(f"  Butil acetato  + 2-HEAB + Etanol: {len(EXP_BUTIL)} tie-lines")
    print(f"  Pentil acetato + 2-HEAB + Etanol: {len(EXP_PENTIL)} tie-lines")
    print(f"  Hexil acetato  + 2-HEAB + Etanol: {len(EXP_HEXIL)} tie-lines")
    n_total = len(EXP_BUTIL) + len(EXP_PENTIL) + len(EXP_HEXIL)
    print(f"  TOTAL: {n_total} tie-lines")
    print()

    x0 = [2000.0, 2000.0, -2000.0, -300.0]

    print("Avaliacao inicial (parametros do hexil-only):")
    print(f"  x0 = [{x0[0]}, {x0[1]}, {x0[2]}, {x0[3]}]")
    rmsd_per_sys_init = {}
    for s in SYSTEMS_FIT:
        rmsd_s = rmsd_for_system(x0, s['nu'], s['data'])
        rmsd_per_sys_init[s['name']] = rmsd_s
        print(f"  RMSD em {s['name']:6s}: {rmsd_s:.4f}")
    rmsd_init = rmsd_combined(x0)
    print(f"  RMSD combinado:    {rmsd_init:.4f}")
    print()

    from scipy.optimize import minimize

    print("Otimizacao Nelder-Mead com multiplos pontos de partida...")
    print()

    starts = [
        [2000, 2000, -2000, -300],
        [1500, 1500, -1500, -200],
        [1000, 1000, -1000, -200],
        [800,  800,  -800,  -300],
        [500,  500,  -500,  -300],
    ]

    best_rmsd = np.inf
    best_x = None
    t0 = time.time()
    for i, xs in enumerate(starts):
        res = minimize(rmsd_combined, xs, method='Nelder-Mead',
                       options={'xatol': 30.0, 'fatol': 5e-4, 'maxiter': 50})
        rmsd_s = res.fun
        print(f"  Start {i+1}: x0=[{xs[0]:>5}, {xs[1]:>5}, {xs[2]:>5}, {xs[3]:>5}]")
        print(f"    -> a=[{res.x[0]:>+7.0f}, {res.x[1]:>+7.0f}, "
              f"{res.x[2]:>+7.0f}, {res.x[3]:>+7.0f}]  "
              f"RMSD={rmsd_s:.4f}  ({time.time()-t0:.0f}s)")
        if rmsd_s < best_rmsd:
            best_rmsd = rmsd_s
            best_x = res.x

    print()
    print("=" * 70)
    print("MELHOR ESTIMATIVA — BASE AMPLIADA")
    print("=" * 70)
    print(f"  a(CCOO, CNH2) = {best_x[0]:>+8.1f} K")
    print(f"  a(CNH2, CCOO) = {best_x[1]:>+8.1f} K")
    print(f"  a(CNH2, COO)  = {best_x[2]:>+8.1f} K")
    print(f"  a(COO, CNH2)  = {best_x[3]:>+8.1f} K")
    print()
    print(f"  RMSD combinado: {best_rmsd:.4f}")
    print()

    print("Detalhamento por sistema:")
    print(f"  {'Sistema':<10}  {'n_TL':>5}  {'RMSD':>8}")
    rmsd_per_sys = {}
    for s in SYSTEMS_FIT:
        rmsd_s = rmsd_for_system(best_x, s['nu'], s['data'])
        rmsd_per_sys[s['name']] = rmsd_s
        print(f"  {s['name']:<10}  {len(s['data']):>5}  {rmsd_s:>8.4f}")
    print()

    print("=" * 70)
    print("COMPARATIVO: hexil-only vs base ampliada")
    print("=" * 70)
    print(f"  Hexil-only (parametros 2000/2000/-2000/-300):")
    print(f"    RMSD em hexil:    0.0474")
    print(f"    Magnitudes |a|:    ~2000 (saturadas)")
    print()
    print(f"  Base ampliada (parametros estimados acima):")
    print(f"    RMSD combinado:   {best_rmsd:.4f}")
    print(f"    RMSD em hexil:    {rmsd_per_sys['hexil']:.4f}")
    print(f"    Magnitude max |a|: {max(abs(best_x)):.0f}")
    print()

    out = {
        'best_params': list(best_x),
        'rmsd_combined': float(best_rmsd),
        'rmsd_per_system': {k: float(v) for k, v in rmsd_per_sys.items()},
        'n_tie_lines': {
            'butil':  len(EXP_BUTIL),
            'pentil': len(EXP_PENTIL),
            'hexil':  len(EXP_HEXIL),
            'total':  n_total,
        },
    }
    with open('reestim_base_ampliada.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Salvo: reestim_base_ampliada.json")

    # Diagrama
    print()
    print("Gerando diagrama comparativo dos 3 sistemas...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def t2c(x1, x2, x3):
        X = 0.5 * (2.0 * x3 + x2)
        Y = (np.sqrt(3.0) / 2.0) * x2
        return X, Y

    def draw_panel(ax, sys_data, params_vec, title):
        verts = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2],[0,0]])
        ax.plot(verts[:,0], verts[:,1], 'k-', lw=1.5, zorder=5)
        for f in [0.2, 0.4, 0.6, 0.8]:
            for triple in [
                ((f, 1-f, 0), (f, 0, 1-f)),
                ((1-f, f, 0), (0, f, 1-f)),
                ((1-f, 0, f), (0, 1-f, f)),
            ]:
                X1, Y1 = t2c(*triple[0])
                X2, Y2 = t2c(*triple[1])
                ax.plot([X1, X2], [Y1, Y2], '-', color='#dddddd', lw=0.5, zorder=1)
        for xa_e, xb_e in sys_data['data']:
            Xa, Ya = t2c(*xa_e); Xb, Yb = t2c(*xb_e)
            ax.plot([Xa,Xb],[Ya,Yb],'o--',color='#e74c3c',lw=1,ms=5,
                    alpha=0.85, zorder=4)
        from stability import is_unstable
        from flash import flash_ell
        model = make_model(params_vec, sys_data['nu'])
        for xa_e, xb_e in sys_data['data']:
            z = 0.5*xa_e + 0.5*xb_e; z = np.clip(z, 1e-6, 1); z /= z.sum()
            bif = is_unstable(z, model)
            if not bif:
                for psi_t in [0.3, 0.4, 0.6, 0.7]:
                    z2 = psi_t*xb_e + (1-psi_t)*xa_e; z2 = np.clip(z2,1e-6,1); z2 /= z2.sum()
                    if is_unstable(z2, model): z = z2; bif = True; break
            if not bif: continue
            xa, xb, _, ok, _ = flash_ell(z, model)
            if not ok: continue
            if xa[0] < xb[0]: xa, xb = xb, xa
            Xa, Ya = t2c(*xa); Xb, Yb = t2c(*xb)
            ax.plot([Xa,Xb],[Ya,Yb],'s-',color='#2980b9',lw=1,ms=5, zorder=5)
        h = np.sqrt(3)/2
        labels = {'butil': 'Butil acet.', 'pentil': 'Pentil acet.', 'hexil': 'Hexil acet.'}
        ax.text(-0.04, -0.04, labels[sys_data['name']], ha='right', va='top',
                fontsize=9, fontweight='bold')
        ax.text(1.04, -0.04, 'Etanol', ha='left', va='top',
                fontsize=9, fontweight='bold')
        ax.text(0.5, h+0.04, '2-HEAB', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.10, h + 0.12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, s in enumerate(SYSTEMS_FIT):
        rmsd_s = rmsd_per_sys[s['name']]
        labels_map = {'butil': 'Butil acetato', 'pentil': 'Pentil acetato', 'hexil': 'Hexil acetato'}
        draw_panel(axes[i], s, best_x,
                   f"{labels_map[s['name']]} + 2-HEAB + Etanol\n"
                   f"{len(s['data'])} TLs | RMSD = {rmsd_s:.4f}")

    fig.suptitle("Validacao com base ampliada (Souza et al., 2025) — T = 298,15 K\n"
                 f"a(11,14)={best_x[0]:.0f}, a(14,11)={best_x[1]:.0f}, "
                 f"a(14,41)={best_x[2]:.0f}, a(41,14)={best_x[3]:.0f}",
                 fontsize=11, y=1.04)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#e74c3c', lw=1, marker='o', ms=5, ls='--', label='Experimental'),
        Line2D([0],[0], color='#2980b9', lw=1, marker='s', ms=5, ls='-', label='UNIFAC (calc.)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)
    plt.tight_layout()

    outpath = f'{OUTDIR}/comparativo_3_sistemas.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {outpath}")

    # Diagnostico em texto
    diag_lines = []
    diag_lines.append("=" * 70)
    diag_lines.append("DIAGNOSTICO — REESTIMACAO COM BASE AMPLIADA")
    diag_lines.append("=" * 70)
    diag_lines.append(f"")
    diag_lines.append(f"Sistemas usados na calibracao (Souza et al. 2025):")
    diag_lines.append(f"  Butil acetato  (C6) + 2-HEAB + Etanol: {len(EXP_BUTIL)} tie-lines")
    diag_lines.append(f"  Pentil acetato (C7) + 2-HEAB + Etanol: {len(EXP_PENTIL)} tie-lines")
    diag_lines.append(f"  Hexil acetato  (C8) + 2-HEAB + Etanol: {len(EXP_HEXIL)} tie-lines")
    diag_lines.append(f"  TOTAL:                                 {n_total} tie-lines")
    diag_lines.append(f"")
    diag_lines.append(f"Parametros estimados:")
    diag_lines.append(f"  a(CCOO, CNH2) = {best_x[0]:>+8.1f} K")
    diag_lines.append(f"  a(CNH2, CCOO) = {best_x[1]:>+8.1f} K")
    diag_lines.append(f"  a(CNH2, COO)  = {best_x[2]:>+8.1f} K")
    diag_lines.append(f"  a(COO, CNH2)  = {best_x[3]:>+8.1f} K")
    diag_lines.append(f"")
    diag_lines.append(f"Magnitude maxima: {max(abs(best_x)):.0f} K")
    diag_lines.append(f"  (vs ~2000 na estimacao com hexil-only)")
    diag_lines.append(f"")
    diag_lines.append(f"RMSD por sistema:")
    for s in SYSTEMS_FIT:
        diag_lines.append(f"  {s['name']:<10}: {rmsd_per_sys[s['name']]:.4f}")
    diag_lines.append(f"  Combinado: {best_rmsd:.4f}")

    with open(f'{OUTDIR}/diagnostico_base_ampliada.txt', 'w') as f:
        f.write('\n'.join(diag_lines))
    print(f"  Salvo: {OUTDIR}/diagnostico_base_ampliada.txt")

    print()
    print("=" * 70)
    print("PROXIMO PASSO:")
    print("Atualizar params.py com os novos valores e rodar gerar_resultados_tcc.py")
    print("=" * 70)


if __name__ == '__main__':
    main()