# -*- coding: utf-8 -*-
"""
Diagrama ternario de equilibrio liquido-liquido. Agnostico ao modelo:
recebe a lista de tie-lines produzida pelo motor termodinamico e renderiza
o diagrama, amostra tie-lines representativas e exporta PNG e CSV.

Convencao de vertices:
  inferior esquerdo -> componente 1 (oleo/acido graxo)
  superior          -> componente 2 (alcool)
  inferior direito  -> componente 3 (PIL)
Coordenadas cartesianas:
  X = 0.5*(2*x3 + x2)   ;   Y = (sqrt(3)/2)*x2
"""

import os
import re
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sem tela (seguro em thread/servidor)
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


SQRT3_2 = math.sqrt(3.0) / 2.0
OUTDIR = "resultados"


def _clamp_to_triangle(X, Y):
    """Projeta pontos (X, Y) de volta para dentro do triangulo equilatero
    de composicao. Um ponto do binodal com fracao molar negativa ou > 1 e
    fisicamente impossivel; isso ocorre apenas como overshoot numerico do
    spline quando o ponto critico esta muito perto de um vertice. A projecao
    converte cada ponto para (x1,x2,x3), corta as fracoes em [0,1],
    renormaliza para somar 1 e reconverte para (X,Y). Pontos ja internos
    ficam inalterados (o corte nao os toca)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = Y / SQRT3_2
    x3 = X - 0.5 * x2
    x1 = 1.0 - x2 - x3
    M = np.vstack([x1, x2, x3])
    M = np.clip(M, 0.0, 1.0)
    s = M.sum(axis=0)
    s[s == 0] = 1.0
    M = M / s
    Xc = 0.5 * (2.0 * M[2] + M[1])
    Yc = SQRT3_2 * M[1]
    return Xc, Yc


def to_cartesian(x):
    """x = (x1, x2, x3) -> (X, Y) no triangulo equilatero."""
    x = np.asarray(x, dtype=float)
    X = 0.5 * (2.0 * x[2] + x[1])
    Y = SQRT3_2 * x[1]
    return X, Y


def _decimate_branch(order_idx, max_points):
    """Escolhe no maximo max_points indices igualmente espacados dentro de
    order_idx, sempre incluindo o primeiro e o ultimo. Usado para reduzir a
    quantidade de tie-lines brutas usadas para desenhar CADA ramo do
    binodal: a curva e uma envoltoria suave (tipo Bezier), nao uma poligonal
    que precisa tocar toda tie-line individualmente. Perto do ponto critico
    o flash tende a gerar muitas tie-lines curtas e proximas entre si, cujo
    ruido numerico (convergencia mais dificil ali) vira ondulacao visivel se
    a curva for forcada a passar por cada uma."""
    n = len(order_idx)
    if n <= max_points:
        return order_idx
    pick = np.linspace(0, n - 1, max_points).round().astype(int)
    pick = np.unique(pick)
    return [order_idx[i] for i in pick]


def build_smooth_binodal_through_plait(tie_lines, plait, n_samples=300,
                                       verify=False, max_points_per_branch=12):
    """Curva do binodal que PASSA pelo ponto critico estimado, sem bico.

    A curva e uma envoltoria SUAVE (conceitualmente como uma curva de
    Bezier: um arco continuo definido pela tendencia geral dos dados, nao
    uma poligonal forcada a tocar cada tie-line bruta individualmente).
    Por isso cada ramo e antes DECIMADO para no maximo
    max_points_per_branch pontos bem espacados (ver _decimate_branch) —
    sem isso, o aglomerado de tie-lines curtas e ruidosas perto do critico
    (onde o flash converge com mais dificuldade) produz uma ondulacao
    visivel em vez de um arco liso.

    Parametrizacao por COMPRIMENTO DE ARCO ACUMULADO (distancia percorrida
    ao longo da sequencia ordenada de pontos), estritamente crescente por
    construcao — a parametrizacao padrao para suavizar uma nuvem de pontos
    2D. Sequencia: ramo do oleo (do mais distante ate o mais proximo do
    ponto critico) -> ponto critico -> ramo do PIL (do mais proximo ate o
    mais distante). Um UNICO ajuste PCHIP em funcao do comprimento de arco
    produz uma curva C1 (derivada continua) ao longo de toda a sequencia,
    inclusive no no do ponto critico.

    Se verify=True, tambem retorna (X, Y, discrepancia_graus), onde
    discrepancia_graus e o angulo entre a tangente calculada um passo antes
    e um passo depois do no do ponto critico — deve ser ~0 se a curva
    estiver de fato suave ali (sem bico)."""
    lengths = np.array([float(np.linalg.norm(r["xa"] - r["xb"]))
                        for r in tie_lines])
    order_len = list(np.argsort(-lengths))   # do mais longo ao mais curto

    order_oil = _decimate_branch(order_len, max_points_per_branch)
    order_pil = _decimate_branch(order_len[::-1], max_points_per_branch)

    oil_xy = np.array([to_cartesian(tie_lines[i]["xa"]) for i in order_oil])
    pil_xy = np.array([to_cartesian(tie_lines[i]["xb"]) for i in order_pil])
    plait_xy = np.array(to_cartesian(plait))

    pts = np.vstack([oil_xy, plait_xy[None, :], pil_xy])

    # remove pontos consecutivos duplicados/quase-duplicados (comprimento
    # de arco zero quebraria a monotonicidade estrita exigida pelo PCHIP)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-12:
            keep.append(i)
    pts = pts[keep]
    plait_idx = keep.index(len(oil_xy)) if len(oil_xy) in keep else None

    if len(pts) < 3:
        X, Y = pts[:, 0], pts[:, 1]
        return (X, Y, None) if verify else (X, Y)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])

    fx = PchipInterpolator(t, pts[:, 0])
    fy = PchipInterpolator(t, pts[:, 1])
    # amostragem DENSA: com muitos pontos, os segmentos de reta que o
    # matplotlib traca entre pontos consecutivos ficam curtos o bastante
    # para que o overshoot entre eles seja imperceptivel. Amostragem esparsa
    # deixava o arco entre dois pontos escapar do triangulo mesmo com os
    # pontos em si dentro dele (o clamp so ve os pontos, nao os segmentos).
    n_dense = max(n_samples, 1200)
    t_fine = np.linspace(t[0], t[-1], n_dense)
    t_plot = np.union1d(t_fine, t)   # garante passagem exata pelos nos originais
    X, Y = fx(t_plot), fy(t_plot)

    # projeta a curva de volta para DENTRO do triangulo. Com a amostragem
    # densa acima, clampar os pontos e suficiente: qualquer overshoot do
    # PCHIP e corrigido ponto a ponto e os segmentos remanescentes sao
    # curtos demais para sair visivelmente. Fisicamente, nenhuma composicao
    # pode ter fracao negativa ou > 1.
    X, Y = _clamp_to_triangle(X, Y)

    if not verify:
        return X, Y

    discrepancia = None
    if plait_idx is not None and 0 < plait_idx < len(t) - 1:
        dfx, dfy = fx.derivative(), fy.derivative()
        t_c = t[plait_idx]
        eps = min(t[plait_idx] - t[plait_idx - 1],
                 t[plait_idx + 1] - t[plait_idx]) * 1e-3
        v_antes = np.array([dfx(t_c - eps), dfy(t_c - eps)])
        v_depois = np.array([dfx(t_c + eps), dfy(t_c + eps)])
        cos_ang = np.dot(v_antes, v_depois) / (
            np.linalg.norm(v_antes) * np.linalg.norm(v_depois) + 1e-15)
        discrepancia = float(np.degrees(np.arccos(np.clip(cos_ang, -1, 1))))
    return X, Y, discrepancia


def _smooth_curve_xy(bx, by, n_samples=300):
    """Suaviza a poligonal (bx, by) via PCHIP (Hermite cubico monotonico),
    parametrizado pelo indice do ponto ao longo da curva. Usado como reserva
    quando NAO ha ponto critico estimado disponivel (ver
    build_smooth_binodal_through_plait para o caso com ponto critico).

    A poligonal bruta (pontos ligados por retas) mostra um "bico" sempre
    que a direcao muda bruscamente entre dois segmentos. O PCHIP ajusta uma
    curva suave (C1, sem overshoot) passando por todos os pontos na ordem
    dada. Com menos de 3 pontos nao ha o que suavizar, retorna os pontos
    originais."""
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    if len(bx) < 3:
        return bx, by
    t = np.arange(len(bx), dtype=float)
    fx = PchipInterpolator(t, bx)
    fy = PchipInterpolator(t, by)
    tt = np.linspace(0.0, len(bx) - 1.0, n_samples)
    return fx(tt), fy(tt)


# ---------------------------------------------------------------------------
# Estrutura do triangulo (arestas, linhas de grade internas, ticks)
# ---------------------------------------------------------------------------
def _draw_triangle(ax, labels):
    v1 = to_cartesian([1, 0, 0])   # inferior esquerdo (comp 1)
    v2 = to_cartesian([0, 1, 0])   # superior         (comp 2)
    v3 = to_cartesian([0, 0, 1])   # inferior direito (comp 3)

    # arestas
    tri = np.array([v1, v2, v3, v1])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.3, zorder=3)

    # linhas de grade internas (constante x1, x2, x3 em 0.2..0.8)
    for c in (0.2, 0.4, 0.6, 0.8):
        # constante x1 = c
        p_a = to_cartesian([c, 1 - c, 0.0])
        p_b = to_cartesian([c, 0.0, 1 - c])
        # constante x2 = c
        q_a = to_cartesian([1 - c, c, 0.0])
        q_b = to_cartesian([0.0, c, 1 - c])
        # constante x3 = c
        r_a = to_cartesian([1 - c, 0.0, c])
        r_b = to_cartesian([0.0, 1 - c, c])
        for (pa, pb) in ((p_a, p_b), (q_a, q_b), (r_a, r_b)):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color="0.80", lw=0.5, zorder=1)

    # ticks e rotulos dos eixos (0.0 a 1.0 em passos de 0.2)
    for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        # eixo inferior (x1 <-> x3): rotulo de x1 crescendo da direita p/ esquerda
        base = to_cartesian([c, 0.0, 1 - c])
        ax.text(base[0], base[1] - 0.035, "%.1f" % c,
                ha="center", va="top", fontsize=7.5, color="0.25")
        # aresta esquerda (x2 <-> x1): rotulo de x2
        left = to_cartesian([1 - c, c, 0.0])
        ax.text(left[0] - 0.030, left[1] + 0.010, "%.1f" % c,
                ha="right", va="center", fontsize=7.5, color="0.25")
        # aresta direita (x2 <-> x3): rotulo de x3
        right = to_cartesian([0.0, c, 1 - c])
        ax.text(right[0] + 0.030, right[1] + 0.010, "%.1f" % c,
                ha="left", va="center", fontsize=7.5, color="0.25")

    # rotulos dos vertices
    ax.text(v1[0] - 0.02, v1[1] - 0.055, labels[0],
            ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(v2[0], v2[1] + 0.04, labels[1],
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(v3[0] + 0.02, v3[1] - 0.055, labels[2],
            ha="center", va="top", fontsize=12, fontweight="bold")

    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.14, SQRT3_2 + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Amostragem de tie-lines e construcao do binodal
# ---------------------------------------------------------------------------
def sample_tie_lines(tie_lines, n=7):
    """Seleciona n tie-lines igualmente espacadas, ordenadas por x1 da fase
    rica em oleo."""
    if len(tie_lines) <= n:
        return list(tie_lines)
    tl = sorted(tie_lines, key=lambda r: r["xa"][0])
    idx = np.linspace(0, len(tl) - 1, n).round().astype(int)
    return [tl[i] for i in idx]


def estimate_plait_point(tie_lines, n_use=3):
    """Estima o ponto critico (plait point) por extrapolacao linear das
    tie-lines mais CURTAS ate comprimento zero.

    O criterio de aceitacao do flash (DIST_MIN=0.02 em thermo_engine.py)
    impede que qualquer tie-line calculada tenha comprimento |xa-xb| menor
    que isso, entao o ponto critico real NUNCA e capturado diretamente pela
    varredura — precisa ser extrapolado a partir das tie-lines mais
    proximas dele.

    Metodo: toma as n_use tie-lines de menor comprimento, ajusta uma reta
    de cada componente de xa e de xb contra o comprimento da tie-line, e
    extrapola ambas para comprimento zero. O ponto critico e a media dessas
    duas extrapolacoes (deveriam coincidir se a tendencia for linear).

    ATENCAO: e uma ESTIMATIVA VISUAL por extrapolacao simples, nao um
    calculo termodinamico rigoroso do ponto critico (que exigiria a
    condicao de estabilidade de ordem superior). Tratar como aproximacao,
    sobretudo se as tie-lines mais curtas disponiveis ainda estiverem
    longe do critico ou se a tendencia nao for aproximadamente linear.
    Retorna None se houver tie-lines insuficientes ou a extrapolacao
    produzir composicao invalida (negativa ou soma nao positiva)."""
    if len(tie_lines) < 2:
        return None
    tl_sorted = sorted(tie_lines,
                       key=lambda r: np.linalg.norm(r["xa"] - r["xb"]))
    n = min(n_use, len(tl_sorted))
    sub = tl_sorted[:n]
    L = np.array([np.linalg.norm(r["xa"] - r["xb"]) for r in sub])
    xa = np.array([r["xa"] for r in sub])
    xb = np.array([r["xb"] for r in sub])

    if n == 1 or np.ptp(L) < 1e-9:
        # sem variacao de comprimento suficiente para extrapolar: usa o
        # ponto medio da tie-line mais curta como estimativa grosseira
        p = 0.5 * (xa[0] + xb[0])
    else:
        def extrap_to_zero(comp):
            out = np.zeros(3)
            for k in range(3):
                coef = np.polyfit(L, comp[:, k], deg=1)
                out[k] = np.polyval(coef, 0.0)
            return out
        p = 0.5 * (extrap_to_zero(xa) + extrap_to_zero(xb))

    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        return None
    return p / s


def build_binodal(tie_lines, plait_point=None):
    """Envelope do binodal construido respeitando o pareamento fisico de
    cada tie-line (xa e xb da MESMA tie-line estao em equilibrio um com o
    outro; nao podem ser reordenados de forma independente).

    Ordena as tie-lines por comprimento |xa-xb| decrescente (da mais
    "larga", afastada do ponto critico, para a mais "curta", proxima do
    ponto critico/plait point). O ramo do oleo percorre essa ordem; o ramo
    do PIL percorre a ordem INVERSA, de modo que a juncao entre os dois
    ramos ocorre sempre na tie-line mais curta (continuidade fisica no
    ponto critico), em vez de conectar pontos de tie-lines nao relacionadas.

    Se plait_point for fornecido (ver estimate_plait_point), ele e inserido
    exatamente na costura entre os dois ramos, dando o cuspide caracteristico
    do ponto critico em vez de um corte reto na tie-line mais curta
    disponivel (que tem comprimento >= DIST_MIN por construcao do flash)."""
    if not tie_lines:
        return None
    tl = sorted(tie_lines,
               key=lambda r: -np.linalg.norm(r["xa"] - r["xb"]))
    oil = [r["xa"] for r in tl]
    pil = [r["xb"] for r in reversed(tl)]
    if plait_point is not None:
        branch = oil + [plait_point] + pil
    else:
        branch = oil + pil
    return np.array(branch)


# ---------------------------------------------------------------------------
# Sanitizacao de nome de arquivo
# ---------------------------------------------------------------------------
def _slug(text):
    t = text.replace("[", "").replace("]", "")
    t = re.sub(r"[^0-9A-Za-z]+", "_", t)
    return t.strip("_")


# ---------------------------------------------------------------------------
# Exportacao CSV
# ---------------------------------------------------------------------------
def export_csv(tie_lines, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xa_comp1", "xa_comp2", "xa_comp3",
                    "xb_comp1", "xb_comp2", "xb_comp3"])
        for r in sorted(tie_lines, key=lambda r: r["xa"][0]):
            xa, xb = r["xa"], r["xb"]
            w.writerow(["%.6f" % xa[0], "%.6f" % xa[1], "%.6f" % xa[2],
                        "%.6f" % xb[0], "%.6f" % xb[1], "%.6f" % xb[2]])


# ---------------------------------------------------------------------------
# Funcao principal de plotagem
# ---------------------------------------------------------------------------
def plot_ternary(tie_lines, comp_labels, model_name, system_name=None,
                 jcosmo_calls=None, elapsed=None, outdir=OUTDIR, save=True,
                 return_fig=False, model=None):
    """Renderiza o diagrama ternario. Trata graciosamente o caso sem regiao
    bifasica. Retorna (png_path, csv_path, plait_point); se
    return_fig=True, retorna (png_path, csv_path, plait_point, fig) sem
    fechar a figura (responsabilidade do chamador fecha-la).

    plait_point: se 'model' (objeto com .gamma/.n) for fornecido, o ponto
    critico e CALCULADO pelo solver rigoroso (plait_solver, encolhimento da
    regiao instavel). Se o solver nao convergir, ou se 'model' for None,
    cai na ESTIMATIVA por extrapolacao das tie-lines (estimate_plait_point).
    None se nao houver tie-lines suficientes."""
    os.makedirs(outdir, exist_ok=True)
    if system_name is None:
        system_name = " + ".join(comp_labels)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    _draw_triangle(ax, comp_labels)

    n_tl = len(tie_lines)
    plait = None

    if n_tl == 0:
        # caso sem gap: informa no proprio grafico, nao desenha binodal
        ax.text(0.5, SQRT3_2 / 2.0,
                "Curva não obtida",
                ha="center", va="center", fontsize=10, color="firebrick",
                bbox=dict(boxstyle="round", fc="white", ec="firebrick", alpha=0.9))
        csv_path = None
    else:
        # ponto critico (plait point): tenta o solver rigoroso quando o
        # modelo esta disponivel; senao (ou se nao convergir), cai na
        # estimativa por extrapolacao das tie-lines mais curtas.
        plait = None
        plait_calculado = False
        if model is not None:
            try:
                from plait_solver import solve_plait_point
                res = solve_plait_point(model)
                if res.get("converged") and res.get("x") is not None:
                    xr = res["x"]
                    if np.all(np.isfinite(xr)) and np.all(xr >= -1e-6) \
                       and np.all(xr <= 1 + 1e-6):
                        plait = np.clip(xr, 0.0, 1.0)
                        plait = plait / plait.sum()
                        plait_calculado = True
            except Exception:
                plait = None
        if plait is None:
            plait = estimate_plait_point(tie_lines)

        # binodal (azul): quando ha ponto critico estimado, a curva PASSA
        # por ele sem bico (ajuste por comprimento de tie-line, ver
        # build_smooth_binodal_through_plait); sem ponto critico, usa a
        # suavizacao generica sobre os dados pareados das tie-lines.
        if plait is not None:
            bx, by = build_smooth_binodal_through_plait(tie_lines, plait)
        else:
            binodal = build_binodal(tie_lines)
            bx_raw = [to_cartesian(p)[0] for p in binodal]
            by_raw = [to_cartesian(p)[1] for p in binodal]
            bx, by = _smooth_curve_xy(bx_raw, by_raw)
        ax.plot(bx, by, color="tab:blue", lw=1.6, zorder=4,
                label="Binodal (%s)" % model_name)

        # 7 tie-lines representativas (marcadores quadrados nas pontas)
        for r in sample_tie_lines(tie_lines, 7):
            ax_, ay_ = to_cartesian(r["xa"])
            bx_, by_ = to_cartesian(r["xb"])
            ax.plot([ax_, bx_], [ay_, by_], color="tab:red", lw=0.9,
                    marker="s", markersize=4, markerfacecolor="tab:red",
                    zorder=5)

        ax.plot([], [], color="tab:red", lw=0.9, marker="s", markersize=4,
                label="Tie-lines")

        if plait is not None:
            pcx, pcy = to_cartesian(plait)
            rotulo = ("Ponto critico (calculado)" if plait_calculado
                      else "Ponto critico (estimado)")
            ax.plot([pcx], [pcy], marker="o", markersize=5,
                   markerfacecolor="tab:blue", markeredgecolor="black",
                   markeredgewidth=0.6, linestyle="none", zorder=6,
                   label=rotulo)

        ax.legend(loc="upper right", fontsize=8, frameon=True)

        csv_path = os.path.join(outdir, "ELL_%s_%s.csv"
                                % (_slug(system_name), _slug(model_name)))
        if save:
            export_csv(tie_lines, csv_path)

    # titulo dinamico
    ax.set_title("%s\nModelo: %s" % (system_name, model_name),
                 fontsize=11, pad=12)

    # rodape (chamadas JCOSMO e tempo, quando aplicavel)
    footer = []
    if jcosmo_calls is not None:
        footer.append("Chamadas JCOSMO: %d" % jcosmo_calls)
    if elapsed is not None:
        footer.append("Tempo: %.1f s" % elapsed)
    if footer:
        fig.text(0.5, 0.015, "   |   ".join(footer),
                 ha="center", fontsize=7.5, color="0.4")

    fig.tight_layout()

    png_path = os.path.join(outdir, "ELL_%s_%s.png"
                            % (_slug(system_name), _slug(model_name)))
    if save:
        fig.savefig(png_path, dpi=180, bbox_inches="tight")

    if return_fig:
        return png_path, csv_path, plait, fig   # chamador fecha a figura
    plt.close(fig)
    return png_path, csv_path, plait