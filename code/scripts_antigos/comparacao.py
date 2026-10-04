"""
comparar_3_versoes_tcc.py — Comparativo das 3 versoes de parametros nos sistemas-alvo
=======================================================================================
Aplica aos sistemas {Triolina + Etanol + LIP} as tres versoes de parametros
estimados ao longo do TCC, para que possamos avaliar:
  1. Qual produz regiao bifasica fisicamente plausivel
  2. Qual responde corretamente a [DETA] vs [2-HEA] (Trindade 2019)
  3. Qual e o trade-off entre RMSD em Souza vs transferabilidade

VERSOES COMPARADAS:
  V1 - Hexil-only:    a = (+2000, +2000, -2000,  -300)  -> RMSD Souza-hexil = 0.047
  V2 - Base ampliada: a = (+3567, +3113, -3302,   +81)  -> RMSD Souza-3sis = 0.045
  V3 - Bounded:       a =  (+480,  +719,  -792,  -680)  -> RMSD Souza-hexil = 0.055

PRE-REQUISITOS:
  Mesma pasta que params.py, gamma.py, flash.py, stability.py, diagram.py

EXECUCAO:
    python comparar_3_versoes_tcc.py

SAIDAS:
    resultados/comparativo_TCC_2HEA_Hx.png
    resultados/comparativo_TCC_DETA_Hx.png
    resultados/comparativo_TCC_resumo.txt
    comparativo_3_versoes.json

TEMPO ESTIMADO: ~3-5 minutos (3 versoes x 2 sistemas x 100 tie-lines em grid 20x20)
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
import diagram

T = 298.15
OUTDIR = 'resultados'
os.makedirs(OUTDIR, exist_ok=True)


# ===========================================================================
# AS TRES VERSOES DE PARAMETROS
# ===========================================================================

VERSIONS = {
    'V1_hexil_only': {
        'label': 'V1: Hexil-only (RMSD$_{Souza}$=0.047)',
        'short': 'V1',
        'color': '#16a085',
        'params': (2000.0, 2000.0, -2000.0, -300.0),
        'description': 'Estimado contra 9 tie-lines de hexil acetato',
    },
    'V2_base_ampliada': {
        'label': 'V2: Base ampliada (RMSD$_{Souza}$=0.045)',
        'short': 'V2',
        'color': '#c0392b',
        'params': (3567.0, 3113.0, -3302.0, 81.0),
        'description': 'Estimado contra 24 tie-lines de butil+pentil+hexil',
    },
    'V3_bounded': {
        'label': r'V3: Bounded $|a|\leq 800$ K (RMSD$_{Souza}$=0.055)',
        'short': 'V3',
        'color': '#2980b9',
        'params': (479.6, 718.9, -792.0, -679.5),
        'description': 'Estimado com bound rigido, magnitudes moderadas',
    },
}

SYSTEMS = [
    {
        'name': '2HEA_Hx',
        'title': r'{Trioleina + Etanol + [2-HEA][Hx]}',
        'labels': ('Trioleina', 'Etanol', '[2-HEA][Hx]'),
        'nu_li': p.NU_2HEA_HX,
    },
    {
        'name': 'DETA_Hx',
        'title': r'{Trioleina + Etanol + [DETA][Hx]}',
        'labels': ('Trioleina', 'Etanol', '[DETA][Hx]'),
        'nu_li': p.NU_DETA_HX,
    },
]


# ===========================================================================
# FUNCOES
# ===========================================================================

def set_params(values):
    p.INTERACTION_PARAMS[(11, 14)] = (values[0], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 11)] = (values[1], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 41)] = (values[2], 0.0, 0.0)
    p.INTERACTION_PARAMS[(41, 14)] = (values[3], 0.0, 0.0)
    importlib.reload(gamma)
    importlib.reload(flash)
    importlib.reload(stability)
    importlib.reload(diagram)


def calc_tie_lines_for(sys_cfg, params_values, n_grid=20):
    set_params(params_values)
    from gamma import UNIFACLL
    from diagram import calc_tie_lines
    nu_list = [p.NU_TRIOLINA, p.NU_ETANOL, sys_cfg['nu_li']]
    model = UNIFACLL(nu_list, T)
    return calc_tie_lines(model, n_grid=n_grid)


def metrics_from_tie_lines(tie_lines):
    if not tie_lines:
        return {'n_tl': 0, 'bif_area_frac': 0.0,
                'K_triol_mean': None, 'x_triol_oleosa_mean': None,
                'x_triol_polar_mean': None}
    xa_arr = np.array([tl[0] for tl in tie_lines])
    xb_arr = np.array([tl[1] for tl in tie_lines])
    swap_idx = xa_arr[:, 0] < xb_arr[:, 0]
    if swap_idx.any():
        xa_arr[swap_idx], xb_arr[swap_idx] = xb_arr[swap_idx].copy(), xa_arr[swap_idx].copy()
    K_triol = xa_arr[:, 0] / np.maximum(xb_arr[:, 0], 1e-6)
    return {
        'n_tl': len(tie_lines),
        'bif_area_frac': float(len(tie_lines) / 210.0),
        'K_triol_mean': float(np.mean(K_triol)),
        'K_triol_min': float(np.min(K_triol)),
        'K_triol_max': float(np.max(K_triol)),
        'x_triol_oleosa_mean': float(xa_arr[:, 0].mean()),
        'x_triol_oleosa_min': float(xa_arr[:, 0].min()),
        'x_triol_oleosa_max': float(xa_arr[:, 0].max()),
        'x_triol_polar_mean': float(xb_arr[:, 0].mean()),
        'x_triol_polar_min': float(xb_arr[:, 0].min()),
        'x_triol_polar_max': float(xb_arr[:, 0].max()),
    }


# ===========================================================================
# DIAGRAMA
# ===========================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def t2c(x1, x2, x3):
    X = 0.5 * (2.0 * x3 + x2)
    Y = (np.sqrt(3.0) / 2.0) * x2
    return X, Y


def draw_panel(ax, tie_lines, labels, title, color):
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(verts[:, 0], verts[:, 1], 'k-', lw=1.5, zorder=5)
    for f in [0.2, 0.4, 0.6, 0.8]:
        for triple in [
            ((f, 1 - f, 0), (f, 0, 1 - f)),
            ((1 - f, f, 0), (0, f, 1 - f)),
            ((1 - f, 0, f), (0, 1 - f, f)),
        ]:
            X1, Y1 = t2c(*triple[0])
            X2, Y2 = t2c(*triple[1])
            ax.plot([X1, X2], [Y1, Y2], '-', color='#dddddd', lw=0.5, zorder=1)
    for xa, xb in tie_lines:
        Xa, Ya = t2c(*xa)
        Xb, Yb = t2c(*xb)
        ax.plot([Xa, Xb], [Ya, Yb], '-', color=color, lw=0.7, alpha=0.6, zorder=4)
        ax.plot([Xa, Xb], [Ya, Yb], 's', color=color, ms=2.0, zorder=5)
    h = np.sqrt(3) / 2
    ax.text(-0.04, -0.04, labels[0], ha='right', va='top',
            fontsize=8, fontweight='bold')
    ax.text(1.04, -0.04, labels[2], ha='left', va='top',
            fontsize=8, fontweight='bold')
    ax.text(0.5, h + 0.04, labels[1], ha='center', va='bottom',
            fontsize=8, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.10, h + 0.12)


# ===========================================================================
# EXECUCAO PRINCIPAL
# ===========================================================================

def main():
    print("=" * 75)
    print("COMPARATIVO DE 3 VERSOES DE PARAMETROS NOS SISTEMAS-ALVO DO TCC")
    print("Sistemas com Trioleina (T = 298,15 K)")
    print("=" * 75)
    print()
    print("Versoes comparadas:")
    for k, v in VERSIONS.items():
        a = v['params']
        print(f"  {v['short']}: a=({a[0]:>+5.0f}, {a[1]:>+5.0f}, {a[2]:>+5.0f}, {a[3]:>+5.0f})")
        print(f"      {v['description']}")
    print()

    all_results = {}
    all_metrics = {}
    t_start = time.time()

    for sys_cfg in SYSTEMS:
        print(f"--- {sys_cfg['title']} ---")
        all_results[sys_cfg['name']] = {}
        all_metrics[sys_cfg['name']] = {}

        for vk, v in VERSIONS.items():
            t0 = time.time()
            tl = calc_tie_lines_for(sys_cfg, v['params'], n_grid=20)
            print(f"  {v['short']}: {len(tl):>4} tie-lines  ({time.time()-t0:.0f}s)")
            all_results[sys_cfg['name']][vk] = [
                (list(tli[0]), list(tli[1])) for tli in tl
            ]
            all_metrics[sys_cfg['name']][vk] = metrics_from_tie_lines(tl)
        print()

    print(f"Tempo total: {time.time()-t_start:.0f}s\n")

    # JSON
    with open('comparativo_3_versoes.json', 'w') as f:
        json.dump({
            'tie_lines': all_results,
            'metrics': all_metrics,
            'versions': {k: {'params': v['params'], 'description': v['description']}
                         for k, v in VERSIONS.items()},
        }, f, indent=2)

    # ============================================================
    # FIGURAS
    # ============================================================
    print("Gerando figuras...")
    for sys_cfg in SYSTEMS:
        name = sys_cfg['name']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
        for i, vk in enumerate(['V1_hexil_only', 'V2_base_ampliada', 'V3_bounded']):
            v = VERSIONS[vk]
            tl = [(np.array(t[0]), np.array(t[1]))
                  for t in all_results[name][vk]]
            n_tl = len(tl)
            title = f"{v['label']}\n{n_tl} tie-lines"
            draw_panel(axes[i], tl, sys_cfg['labels'], title, v['color'])
        fig.suptitle(sys_cfg['title'] + " — T = 298,15 K",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        outpath = f'{OUTDIR}/comparativo_TCC_{name}.png'
        plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Salvo: {outpath}")

    # ============================================================
    # RESUMO COMPARATIVO EM TEXTO
    # ============================================================
    lines = []
    lines.append("=" * 80)
    lines.append("COMPARATIVO QUANTITATIVO DAS 3 VERSOES — SISTEMAS-ALVO")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Notacao:")
    lines.append("  n_tl       = numero de tie-lines bifasicas em grid 20x20")
    lines.append("  bif_frac   = fracao da regiao bifasica (~ n_tl / 210)")
    lines.append("  K_triol    = coef. particao da trioleina = x_triol(oleosa) / x_triol(polar)")
    lines.append("  x_T_oleosa = x_trioleina medio na fase oleosa")
    lines.append("")

    for sys_name in ['2HEA_Hx', 'DETA_Hx']:
        title = '[2-HEA][Hx]' if sys_name == '2HEA_Hx' else '[DETA][Hx]'
        lines.append(f"--- Trioleina + Etanol + {title} ---")
        lines.append(f"  {'Versao':<6}  {'Params':<28}  {'n_tl':>5}  {'bif_frac':>9}  "
                     f"{'K_triol_med':>12}  {'x_T_oleosa':>12}")
        for vk in ['V1_hexil_only', 'V2_base_ampliada', 'V3_bounded']:
            v = VERSIONS[vk]
            m = all_metrics[sys_name][vk]
            a = v['params']
            params_str = f"({a[0]:+.0f},{a[1]:+.0f},{a[2]:+.0f},{a[3]:+.0f})"
            ntl = m['n_tl']
            bf = m['bif_area_frac']
            if ntl > 0:
                Km = f"{m['K_triol_mean']:.1f}"
                xom = f"{m['x_triol_oleosa_mean']:.3f}"
            else:
                Km = "—"
                xom = "—"
            lines.append(f"  {v['short']:<6}  {params_str:<28}  {ntl:>5}  {bf:>9.3f}  "
                         f"{Km:>12}  {xom:>12}")
        lines.append("")

    # Comparacao [2-HEA] vs [DETA] em cada versao
    lines.append("=" * 80)
    lines.append("COMPARACAO ENTRE LIPs PARA CADA VERSAO")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Hipotese (Trindade et al. 2019): [DETA][Hx] e cossolvente MAIS eficiente")
    lines.append("que [2-HEA][Hx] -> espera-se MENOR regiao bifasica para [DETA].")
    lines.append("")
    lines.append(f"  {'Versao':<6}  {'n_TLs [2-HEA]':>14}  {'n_TLs [DETA]':>14}  {'Trindade?':>12}")
    for vk in ['V1_hexil_only', 'V2_base_ampliada', 'V3_bounded']:
        v = VERSIONS[vk]
        n2h = all_metrics['2HEA_Hx'][vk]['n_tl']
        ndt = all_metrics['DETA_Hx'][vk]['n_tl']
        if n2h == 0 and ndt == 0:
            verdict = "ambos monofasicos"
        elif n2h > 0 and ndt > 0:
            if ndt < n2h:
                verdict = "OK consistente"
            elif ndt > n2h:
                verdict = "X contradiz"
            else:
                verdict = "= empate"
        else:
            verdict = "parcial"
        lines.append(f"  {v['short']:<6}  {n2h:>14}  {ndt:>14}  {verdict:>12}")
    lines.append("")

    # Veredito final
    lines.append("=" * 80)
    lines.append("ANALISE QUALITATIVA")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Criterios para escolha da versao a apresentar no TCC:")
    lines.append("")
    lines.append("1. PLAUSIBILIDADE FISICA da regiao bifasica:")
    lines.append("   - bif_frac proximo de 0  ->  monofasico (improvavel para LIP)")
    lines.append("   - bif_frac proximo de 1  ->  bifasico em quase todo simplex (irreal)")
    lines.append("   - bif_frac em 0.2-0.6   ->  faixa fisicamente esperada")
    lines.append("")
    lines.append("2. CONSISTENCIA COM TRINDADE: regiao bifasica menor para [DETA] que [2-HEA]")
    lines.append("")
    lines.append("3. AJUSTE AOS DADOS DE SOUZA (calibracao):")
    lines.append("   V1 (hexil-only):    RMSD = 0.047 (mas em 1 sistema apenas)")
    lines.append("   V2 (base ampliada): RMSD = 0.045 (em 3 sistemas, melhor robustez)")
    lines.append("   V3 (bounded):       RMSD = 0.055 (15% pior que V1)")
    lines.append("")

    txt = "\n".join(lines)
    print()
    print(txt)

    with open(f'{OUTDIR}/comparativo_TCC_resumo.txt', 'w') as f:
        f.write(txt)
    print(f"\nResumo salvo: {OUTDIR}/comparativo_TCC_resumo.txt")
    print(f"JSON salvo:   comparativo_3_versoes.json")
    print()
    print("PROXIMO PASSO: analisar visualmente os 2 PNGs e o resumo,")
    print("decidir qual versao sera apresentada como principal no TCC.")


if __name__ == '__main__':
    main()