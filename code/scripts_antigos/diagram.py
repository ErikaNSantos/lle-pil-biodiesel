"""
diagram.py — Diagrama de fases ternário ELL (UNIFAC-LL)
=============================================================
Gera o diagrama no formato padrão da literatura de ELL:
  - Triângulo equilátero com eixos escalonados (0 a 1, intervalos de 0.2)
  - Grade interna em 0.2, 0.4, 0.6, 0.8
  - 5–8 tie-lines representativas igualmente espaçadas ao longo da binodal
  - Curva binodal tracejada
  - Região bifásica identificada
  - Pontos experimentais sobrepostos (quando disponíveis)

Convenção de vértices (igual à referência):
  Vértice inferior-esquerdo : Componente 1 (Trioleína)
  Vértice inferior-direito  : Componente 3 (LIP)
  Vértice superior          : Componente 2 (Etanol)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import csv
import os

from gamma    import UNIFACLL
from stability import is_unstable
from flash    import flash_ell, check_mass_balance
from params   import NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX


# ---------------------------------------------------------------------------
# Conversão de coordenadas ternárias → cartesianas
# Convenção: x1 = base-esq, x2 = topo, x3 = base-dir
# ---------------------------------------------------------------------------

def t2c(x1, x2, x3):
    """
    Ternário (x1, x2, x3) → cartesiano (X, Y).
    x1 = componente 1 (base-esquerda)
    x2 = componente 2 (topo)
    x3 = componente 3 (base-direita)
    """
    X = 0.5 * (2.0 * x3 + x2)
    Y = (np.sqrt(3.0) / 2.0) * x2
    return X, Y


def c2t_grid(n_div=5):
    """Gera grade de pontos no simplex para n_div divisões."""
    pts = []
    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            k = n_div - i - j
            pts.append((i / n_div, j / n_div, k / n_div))
    return pts


# ---------------------------------------------------------------------------
# Construção das tie-lines (grade densa → selecionar representativas)
# ---------------------------------------------------------------------------

def calc_tie_lines(model, n_grid=40):
    """
    Calcula todas as tie-lines válidas varrendo o simplex.
    Retorna lista de (xa, xb) onde xa é fase rica em comp.1 (trioleína).
    """
    tie_lines = []

    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            if k < 0:
                continue
            z = np.array([i, j, k], float) / n_grid
            if np.any(z < 0.02):
                continue
            if not is_unstable(z, model):
                continue
            xa, xb, psi, ok, _ = flash_ell(z, model)
            if not ok or psi < 0.01 or psi > 0.99:
                continue
            if np.linalg.norm(xa - xb) < 0.02:
                continue
            mb_ok, _ = check_mass_balance(z, xa, xb, psi, tol=1e-3)
            if not mb_ok:
                continue
            # garantir que xa é a fase rica em comp.1
            if xa[0] < xb[0]:
                xa, xb = xb, xa
            tie_lines.append((xa.copy(), xb.copy()))

    return tie_lines


def select_representative(tie_lines, n_lines=7):
    """
    Seleciona n_lines tie-lines igualmente espaçadas ao longo
    da curva binodal (ordenadas por x1 da fase α).
    """
    if not tie_lines:
        return []
    # Ordenar por x_comp1 da fase α (trioleína)
    sorted_tl = sorted(tie_lines, key=lambda t: t[0][0])
    # Selecionar igualmente espaçadas
    idxs = np.linspace(0, len(sorted_tl) - 1, n_lines, dtype=int)
    return [sorted_tl[i] for i in idxs]


# ---------------------------------------------------------------------------
# Extração da curva binodal
# ---------------------------------------------------------------------------

def extract_binodal(tie_lines, n_pts=150, tol=0.008):
    """
    Retorna dois arrays de pontos para a curva binodal, com duplicatas removidas.
    branch_a: fase α, ordenada por x1; branch_b: fase β, ordenada por x2.
    n_pts: máximo de pontos por ramo (default mais alto para curva mais lisa
           ao desenhar; os pontos em si continuam sendo os calculados,
           nenhum dado é inventado).
    tol: distância mínima entre pontos mantidos (deduplicação).
    """
    if not tie_lines:
        return np.array([]), np.array([])

    a_pts = np.array([tl[0] for tl in tie_lines])
    b_pts = np.array([tl[1] for tl in tie_lines])

    def deduplicate(pts, tol=tol):
        if len(pts) == 0: return pts
        keep = [pts[0]]
        for p in pts[1:]:
            if np.min([np.linalg.norm(p - k) for k in keep]) > tol:
                keep.append(p)
        return np.array(keep)

    def subsample(pts, n):
        if len(pts) <= n: return pts
        idxs = np.linspace(0, len(pts)-1, n, dtype=int)
        return pts[idxs]

    a_sorted = subsample(deduplicate(a_pts[np.argsort(a_pts[:,0])]), n_pts)
    b_sorted = subsample(deduplicate(b_pts[np.argsort(b_pts[:,1])]), n_pts)
    return a_sorted, b_sorted


def smooth_curve_xy(X, Y, n_out=300):
    """
    Recebe pontos cartesianos (X, Y) já convertidos do ternário e devolve
    uma versão suavizada (spline cúbica parametrica) com n_out pontos,
    para desenhar a binodal como curva continua em vez de segmentos retos
    entre pontos esparsos.

    Importante: isto e SO para o desenho da curva. Os pontos calculados
    (tie-lines, CSV exportado) nao mudam -- a spline so passa por eles
    com mais resolucao visual entre um ponto e outro.

    Se scipy nao estiver disponivel, cai de volta para interpolacao
    linear simples (mais pontos, mas sem suavizacao de curvatura).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if len(X) < 4:
        return X, Y  # poucos pontos -- spline cubica nao se aplica

    try:
        from scipy.interpolate import splprep, splev
        # s=0 -> spline interpola exatamente os pontos (nao "smoothing"
        # estatistico, e so curva suave passando por eles)
        tck, _ = splprep([X, Y], s=0, k=min(3, len(X) - 1))
        u_fine = np.linspace(0, 1, n_out)
        X_smooth, Y_smooth = splev(u_fine, tck)
        return np.array(X_smooth), np.array(Y_smooth)
    except Exception:
        # fallback: interpolacao linear com mais pontos (sem scipy)
        t = np.linspace(0, 1, len(X))
        t_fine = np.linspace(0, 1, n_out)
        return np.interp(t_fine, t, X), np.interp(t_fine, t, Y)


# ---------------------------------------------------------------------------
# Plot principal
# ---------------------------------------------------------------------------

def _draw_triangle(ax):
    """Desenha o triângulo externo."""
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(verts[:, 0], verts[:, 1], 'k-', lw=1.5, zorder=10)


def _draw_grid(ax, fracs=(0.2, 0.4, 0.6, 0.8)):
    """Desenha linhas de grade internas para cada fração."""
    for f in fracs:
        # Paralelas ao lado oposto a cada vértice
        # Paralela à base (x2 = constante): de (x1=f,x2=0,x3=1-f) a (x1=0,x2=f,x3=1-f)? 
        # Usando convenção: x1=base-esq, x2=topo, x3=base-dir
        
        # Linhas de iso-x1 (paralelas ao lado x1=0, i.e., lado x2-x3)
        # x1 = f: de (f, 1-f, 0) a (f, 0, 1-f)
        X1, Y1 = t2c(f, 1-f, 0)
        X2, Y2 = t2c(f, 0, 1-f)
        ax.plot([X1, X2], [Y1, Y2], '-', color='#cccccc', lw=0.5, zorder=1)

        # Linhas de iso-x2 (paralelas ao lado x2=0, i.e., lado x1-x3)
        X1, Y1 = t2c(1-f, f, 0)
        X2, Y2 = t2c(0, f, 1-f)
        ax.plot([X1, X2], [Y1, Y2], '-', color='#cccccc', lw=0.5, zorder=1)

        # Linhas de iso-x3 (paralelas ao lado x3=0, i.e., lado x1-x2)
        X1, Y1 = t2c(1-f, 0, f)
        X2, Y2 = t2c(0, 1-f, f)
        ax.plot([X1, X2], [Y1, Y2], '-', color='#cccccc', lw=0.5, zorder=1)


def _draw_axis_ticks(ax, labels):
    """
    Desenha marcações e rótulos nos três eixos.
    Convenção: comp1=base-esq, comp2=topo, comp3=base-dir
    """
    fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    h = np.sqrt(3) / 2
    tick_len = 0.025

    for f in fracs:
        # Eixo esquerdo (comp1, de topo p/ base-esq): x1 cresce de 0→1
        # Ponto no eixo: (x1=f, x2=1-f, x3=0)
        Xe, Ye = t2c(f, 1-f, 0)
        # Tick perpendicular ao eixo esquerdo (apontando p/ fora)
        ax.plot([Xe - tick_len*0.5, Xe + tick_len*0.5],
                [Ye, Ye], 'k-', lw=0.8, zorder=9)
        ax.text(Xe - 0.045, Ye, f'{f:.1f}',
                ha='right', va='center', fontsize=7.5, color='#333333')

        # Eixo direito (comp3, de base-esq p/ base-dir): x3 cresce de 0→1
        Xr, Yr = t2c(1-f, 0, f)
        ax.plot([Xr, Xr], [Yr - tick_len*0.5, Yr + tick_len*0.5],
                'k-', lw=0.8, zorder=9)
        ax.text(Xr + 0.005, Yr - 0.04, f'{f:.1f}',
                ha='center', va='top', fontsize=7.5, color='#333333')

        # Eixo superior (comp2, de base-dir p/ topo): x2 cresce de 0→1
        Xt, Yt = t2c(0, f, 1-f)
        dx = tick_len * np.cos(np.pi/6)
        dy = tick_len * np.sin(np.pi/6)
        ax.plot([Xt + dx, Xt - dx], [Yt + dy, Yt - dy],
                'k-', lw=0.8, zorder=9)
        ax.text(Xt + 0.055, Yt + 0.01, f'{f:.1f}',
                ha='left', va='center', fontsize=7.5, color='#333333')

    # Rótulos dos vértices
    off = 0.07
    h = np.sqrt(3) / 2
    ax.text(-off, -0.04,  labels[0], ha='center', va='top',
            fontsize=12, fontweight='bold')
    ax.text(1+off, -0.04, labels[2], ha='center', va='top',
            fontsize=12, fontweight='bold')
    ax.text(0.5, h+off,   labels[1], ha='center', va='bottom',
            fontsize=12, fontweight='bold')


def plot_ternary(tie_lines,
                 labels,
                 title,
                 outfile,
                 n_rep=7,
                 exp_data=None):
    """
    Plota o diagrama ternário ELL no formato padrão da literatura.

    Parâmetros
    ----------
    tie_lines : lista de (xa, xb)
    labels    : (comp1, comp2, comp3) — nomes dos vértices
    title     : título do diagrama
    outfile   : caminho do arquivo PNG
    n_rep     : número de tie-lines representativas
    exp_data  : lista de (xa_exp, xb_exp) para pontos experimentais
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))

    _draw_triangle(ax)
    _draw_grid(ax)
    _draw_axis_ticks(ax, labels)

    # Curva binodal
    branch_a, branch_b = extract_binodal(tie_lines)

    if len(branch_a) > 1:
        Xa = [t2c(*p)[0] for p in branch_a]
        Ya = [t2c(*p)[1] for p in branch_a]
        Xa_s, Ya_s = smooth_curve_xy(Xa, Ya)
        ax.plot(Xa_s, Ya_s, 'b-', lw=1.5, zorder=5, label='Binodal (calc.)')

    if len(branch_b) > 1:
        Xb = [t2c(*p)[0] for p in branch_b]
        Yb = [t2c(*p)[1] for p in branch_b]
        Xb_s, Yb_s = smooth_curve_xy(Xb, Yb)
        ax.plot(Xb_s, Yb_s, 'b-', lw=1.5, zorder=5)

    # Tie-lines representativas
    rep = select_representative(tie_lines, n_lines=n_rep)
    for xa, xb in rep:
        Xa, Ya = t2c(*xa)
        Xb, Yb = t2c(*xb)
        ax.plot([Xa, Xb], [Ya, Yb], 'b-', lw=1.0, alpha=0.85, zorder=4)
        ax.plot(Xa, Ya, 'bs', ms=5, zorder=6)
        ax.plot(Xb, Yb, 'bs', ms=5, zorder=6)

    # Pontos experimentais (se disponíveis)
    if exp_data:
        for xa_e, xb_e in exp_data:
            Xa, Ya = t2c(*xa_e)
            Xb, Yb = t2c(*xb_e)
            ax.plot([Xa, Xb], [Ya, Yb], 'r-', lw=0.8, alpha=0.7, zorder=7)
            ax.plot(Xa, Ya, 'ro', ms=6, mfc='none', mew=1.5, zorder=8)
            ax.plot(Xb, Yb, 'ro', ms=6, mfc='none', mew=1.5, zorder=8)

    # Legenda
    handles = [
        plt.Line2D([0],[0], color='b', lw=1.5, label='Binodal (UNIFAC-LLE)'),
        plt.Line2D([0],[0], color='b', lw=1.0, alpha=0.85,
                   marker='s', ms=5, label='Tie-lines (calc.)'),
    ]
    if exp_data:
        handles.append(
            plt.Line2D([0],[0], color='r', lw=0.8, alpha=0.7,
                       marker='o', ms=6, mfc='none', label='Exp. (literatura)')
        )
    ax.legend(handles=handles, loc='upper right',
              fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{title}\nT = 298,15 K   UNIFAC-LLE',
                 fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(outfile, dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Figura: {outfile}")


# ---------------------------------------------------------------------------
# Exportar CSV
# ---------------------------------------------------------------------------

def export_csv(tie_lines, outfile, comp_names):
    c1, c2, c3 = comp_names
    with open(outfile, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'xa_{c1}', f'xa_{c2}', f'xa_{c3}',
                    f'xb_{c1}', f'xb_{c2}', f'xb_{c3}'])
        for xa, xb in tie_lines:
            w.writerow([f'{v:.6f}' for v in list(xa)+list(xb)])
    print(f"  CSV: {outfile}  ({len(tie_lines)} tie-lines)")


# ===========================================================================
# VALIDAÇÃO
# ===========================================================================
if __name__ == '__main__':
    print("=" * 58)
    print("VALIDAÇÃO: diagram.py")
    print("=" * 58)

    T = 298.15
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
    os.makedirs(OUT, exist_ok=True)

    systems = [
        dict(name='[2-HEA][Hx]', nu_li=NU_2HEA_HX,
             labels=('Trioleína', 'Etanol', '[2-HEA][Hx]'),
             title='{Trioleína + Etanol + [2-HEA][Hx]}',
             safe='2HEA_Hx'),
        dict(name='[DETA][Hx]', nu_li=NU_DETA_HX,
             labels=('Trioleína', 'Etanol', '[DETA][Hx]'),
             title='{Trioleína + Etanol + [DETA][Hx]}',
             safe='DETA_Hx'),
    ]

    for s in systems:
        print(f"\n{'─'*50}")
        print(f" {s['title']}")
        nu_list = [NU_TRIOLINA, NU_ETANOL, s['nu_li']]
        model   = UNIFACLL(nu_list, T)

        tl = calc_tie_lines(model, n_grid=35)
        print(f" Tie-lines calculadas: {len(tl)}")

        if tl:
            ba, bb = extract_binodal(tl)
            xa_arr = np.array([t[0] for t in tl])
            xb_arr = np.array([t[1] for t in tl])
            print(f" Fase α: x_trioleína ∈ [{xa_arr[:,0].min():.3f}, {xa_arr[:,0].max():.3f}]")
            print(f" Fase β: x_trioleína ∈ [{xb_arr[:,0].min():.4f}, {xb_arr[:,0].max():.4f}]")

        plot_ternary(tl,
                     labels=s['labels'],
                     title=s['title'],
                     outfile=f"{OUT}/diagrama_{s['safe']}.png",
                     n_rep=7)

        export_csv(tl,
                   outfile=f"{OUT}/tielines_{s['safe']}.csv",
                   comp_names=('trioleína','etanol',
                                s['name'].replace('[','').replace(']','')))

    print("\n" + "="*58)
    print("Concluído.")