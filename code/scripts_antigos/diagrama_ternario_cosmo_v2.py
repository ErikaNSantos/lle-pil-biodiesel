"""
diagrama_ternario_cosmo_v3.py — Diagrama ternario COSMO-SAC, do zero
================================================================================
POR QUE REESCREVER
    A v2 reaproveitou diagram.plot_ternary() inteiro, sem perceber que essa
    funcao tem TEXTO ESPECIFICO DO UNIFAC hardcoded dentro dela:
      - titulo: "...UNIFAC-Dortmund (b = c = 0)"
      - legenda: "Binodal (UNIFAC-Do)"
      - nota de rodape sobre parametros de grupo Psi ausentes (CCOO/CNH2),
        um conceito que nem existe no COSMO-SAC (que nao tem parametros
        de grupo -- usa sigma-profiles continuos).
    Isso produziu uma figura COSMO-SAC com rotulos do UNIFAC coladas nela.
    Este arquivo reescreve a parte de PLOT do zero, sem nenhum texto
    hardcoded de modelo. O calculo das tie-lines (calc_tie_lines) e a
    extracao da binodal (extract_binodal) SAO reaproveitados de diagram.py
    porque essas duas funcoes sao genuinamente genericas -- nao tem
    nenhuma string ou suposicao especifica de UNIFAC dentro delas (apenas
    chamam is_unstable/flash_ell, que tambem sao genericos).

ACAO NECESSARIA ANTES DE RODAR
    Confirme NOME_2HEAHX e, se disponivel, NOME_DETAHX (nomes exatos no
    JCOSMO). Mesmos placeholders da v2.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py/flash.py/
diagram.py/stability_cosmo.py na mesma pasta.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from py4j.java_gateway import JavaGateway

from stability_cosmo import COSMOSAC, TRIOLINA, ETANOL
from diagram import calc_tie_lines, extract_binodal, select_representative, export_csv

T = 298.15
OUTDIR = 'resultados'
os.makedirs(OUTDIR, exist_ok=True)

# --- AJUSTAR: nomes EXATOS dos compostos par-neutro no JCOSMO ---
NOME_2HEAHX = '2HEAHX'
NOME_DETAHX = 'DETAHX'
# ------------------------------------------------------------------

N_GRID = 20

SISTEMAS = [
    {
        'nome': '2HEAHX',
        'li_jcosmo': NOME_2HEAHX,
        'labels': ('Oleic Acid', 'Etanol', '[2-HEA][Hx]'),
        'title': '{Trioleína + Etanol + [2-HEA][Hx]}',
        'safe': '2HEA_Hx_COSMO_v3',
    },
]
if NOME_DETAHX:
    SISTEMAS.append({
        'nome': 'DETAHX',
        'li_jcosmo': NOME_DETAHX,
        'labels': ('Triolína', 'Etanol', '[DETA][Hx]'),
        'title': '{Triolína + Etanol + [DETA][Hx]}',
        'safe': 'DETA_Hx_COSMO_v3',
    })


# ===========================================================================
# CONVERSAO DE COORDENADAS (mesma convencao de diagram.py: x1=base-esq,
# x2=topo, x3=base-dir)
# ===========================================================================

def t2c(x1, x2, x3):
    X = 0.5 * (2.0 * x3 + x2)
    Y = (np.sqrt(3.0) / 2.0) * x2
    return X, Y


def _draw_triangle(ax):
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(verts[:, 0], verts[:, 1], 'k-', lw=1.5, zorder=10)


def _draw_grid(ax, fracs=(0.2, 0.4, 0.6, 0.8)):
    for f in fracs:
        for triple in [
            ((f, 1-f, 0), (f, 0, 1-f)),
            ((1-f, f, 0), (0, f, 1-f)),
            ((1-f, 0, f), (0, 1-f, f)),
        ]:
            X1, Y1 = t2c(*triple[0]); X2, Y2 = t2c(*triple[1])
            ax.plot([X1, X2], [Y1, Y2], '-', color='#dddddd', lw=0.5, zorder=1)


def _draw_axis_ticks(ax, labels):
    fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    h = np.sqrt(3) / 2
    tick_len = 0.025
    for f in fracs:
        Xe, Ye = t2c(f, 1-f, 0)
        ax.plot([Xe-tick_len*0.5, Xe+tick_len*0.5], [Ye, Ye], 'k-', lw=0.8, zorder=9)
        ax.text(Xe-0.045, Ye, f'{f:.1f}', ha='right', va='center', fontsize=7.5, color='#333333')

        Xr, Yr = t2c(1-f, 0, f)
        ax.plot([Xr, Xr], [Yr-tick_len*0.5, Yr+tick_len*0.5], 'k-', lw=0.8, zorder=9)
        ax.text(Xr+0.005, Yr-0.04, f'{f:.1f}', ha='center', va='top', fontsize=7.5, color='#333333')

        Xt, Yt = t2c(0, f, 1-f)
        dx, dy = tick_len*np.cos(np.pi/6), tick_len*np.sin(np.pi/6)
        ax.plot([Xt+dx, Xt-dx], [Yt+dy, Yt-dy], 'k-', lw=0.8, zorder=9)
        ax.text(Xt+0.055, Yt+0.01, f'{f:.1f}', ha='left', va='center', fontsize=7.5, color='#333333')

    off = 0.07
    ax.text(-off, -0.04, labels[0], ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(1+off, -0.04, labels[2], ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(0.5, h+off, labels[1], ha='center', va='bottom', fontsize=12, fontweight='bold')


def plot_ternary_cosmo(tie_lines, labels, title, outfile, model_name,
                       n_rep=7, n_calls=None, elapsed=None):
    """
    Plot LIMPO, sem nenhum texto hardcoded de modelo. model_name e
    elapsed/n_calls (se fornecidos) aparecem no titulo/rodape de forma
    GENERICA e CORRETA -- nunca um texto fixo de outro modelo.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))
    _draw_triangle(ax)
    _draw_grid(ax)
    _draw_axis_ticks(ax, labels)

    branch_a, branch_b = extract_binodal(tie_lines)
    if len(branch_a) > 1:
        Xa = [t2c(*p)[0] for p in branch_a]; Ya = [t2c(*p)[1] for p in branch_a]
        ax.plot(Xa, Ya, 'b-', lw=1.5, zorder=5, label=f'Binodal ({model_name})')
    if len(branch_b) > 1:
        Xb = [t2c(*p)[0] for p in branch_b]; Yb = [t2c(*p)[1] for p in branch_b]
        ax.plot(Xb, Yb, 'b-', lw=1.5, zorder=5)

    rep = select_representative(tie_lines, n_lines=n_rep)
    for xa, xb in rep:
        Xa, Ya = t2c(*xa); Xb, Yb = t2c(*xb)
        ax.plot([Xa, Xb], [Ya, Yb], 'b-', lw=1.0, alpha=0.85, zorder=4)
        ax.plot(Xa, Ya, 'bs', ms=5, zorder=6)
        ax.plot(Xb, Yb, 'bs', ms=5, zorder=6)

    handles = [
        plt.Line2D([0], [0], color='b', lw=1.5, label=f'Binodal ({model_name})'),
        plt.Line2D([0], [0], color='b', lw=1.0, alpha=0.85, marker='s', ms=5,
                  label='Tie-lines (calc.)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8.5,
             framealpha=0.9, edgecolor='#cccccc')

    subtitulo = f'{model_name}, T = 298,15 K'
    rodape_partes = []
    if n_calls is not None:
        rodape_partes.append(f'{n_calls} chamadas ao JCOSMO')
    if elapsed is not None:
        rodape_partes.append(f'{elapsed:.0f}s')
    if rodape_partes:
        ax.text(0.5, -0.08, '  |  '.join(rodape_partes), ha='center', va='top',
               fontsize=8, color='#666666', transform=ax.transAxes, style='italic')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{title}\n{subtitulo}', fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(outfile, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figura: {outfile}")

def print_tie_lines_detalhado(tl, comp_names, zero_alert=1e-8):
    """Imprime x1 (trioleina), x2 (etanol), x3 (LIP) de ambas as fases,
    sem arredondamento, para diagnosticar se a binodal 'encostada' na
    aresta etanol+LIP e artefato numerico ou predicao fisica legitima."""
    n1, n2, n3 = comp_names
    print(f"\n  {'#':>3}  {'fase':<5} {n1:>14} {n2:>14} {n3:>14} {'soma':>12}")
    for i, (xa, xb) in enumerate(tl):
        for fase, x in (('I', xa), ('II', xb)):
            soma = float(x[0] + x[1] + x[2])
            print(f"  {i:>3}  {fase:<5} {x[0]:14.6e} {x[1]:14.6e} {x[2]:14.6e} {soma:12.8f}")
            for nome, v in zip((n1, n2, n3), x):
                if v < zero_alert:
                    print(f"       -> ALERTA: x_{nome} = {v:.3e} na fase {fase}, tie-line {i}")

if __name__ == '__main__':
    print("=" * 70)
    print("DIAGRAMA TERNARIO — COSMO-SAC (par neutro), plot reescrito do zero")
    print(f"T = {T} K   n_grid = {N_GRID}")
    print("=" * 70)

    gw = JavaGateway(auto_field=True)

    for s in SISTEMAS:
        print(f"\n{'─'*55}")
        print(f" {s['title']}")
        print(f"{'─'*55}")

        model = COSMOSAC([TRIOLINA, ETANOL, s['li_jcosmo']], T=T, gateway=gw)


        t0 = time.time()
        tl = calc_tie_lines(model, n_grid=N_GRID)
        elapsed = time.time() - t0

        print(f"  Tie-lines encontradas: {len(tl)}")
        print(f"  Chamadas ao JCOSMO:    {model.n_calls}")
        print(f"  Tempo:                 {elapsed:.0f}s")

        if tl:
            print_tie_lines_detalhado(tl, comp_names=('triolina', 'etanol', s['nome']))
            xa_arr = np.array([t[0] for t in tl])
            xb_arr = np.array([t[1] for t in tl])
            print(f"  Fase rica em triolína: x_triolína em "
                  f"[{xa_arr[:,0].min():.3f}, {xa_arr[:,0].max():.3f}]")
            print(f"  Fase rica em LI:       x_triolína em "
                  f"[{xb_arr[:,0].min():.4f}, {xb_arr[:,0].max():.4f}]")
            frac_bifasica = len(tl) / sum(
                1 for i in range(N_GRID+1) for j in range(N_GRID+1-i)
                if np.all(np.array([i, j, N_GRID-i-j], float)/N_GRID >= 0.02)
            )
            print(f"  Fracao da grade bifasica: ~{100*frac_bifasica:.0f}%")
        else:
            print("  AVISO: nenhuma tie-line encontrada.")

        png = f"{OUTDIR}/diagrama_{s['safe']}.png"
        plot_ternary_cosmo(tl, labels=s['labels'], title=s['title'],
                           outfile=png, model_name='COSMO-SAC-HB2 (par neutro)',
                           n_rep=7, n_calls=model.n_calls, elapsed=elapsed)

        csv = f"{OUTDIR}/tielines_{s['safe']}.csv"
        export_csv(tl, outfile=csv, comp_names=('triolina', 'etanol', s['nome']))

    print("\n" + "=" * 70)
    print("CONCLUIDO. Esta figura NAO tem nenhum texto/legenda do UNIFAC.")
    print("=" * 70)