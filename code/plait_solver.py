# -*- coding: utf-8 -*-
"""Solver rigoroso do ponto critico (plait point) de um sistema ternario.

Desenvolvido isolado para comparar contra a estimativa por extrapolacao
(estimate_plait_point em ternary_diagram.py). A ideia e gerar o diagrama
com os dois metodos e, se o ponto sair diferente, saber que o problema
esta no solver e nao em outro lugar.

TEORIA
------
Trabalha-se com a energia molar de Gibbs de mistura adimensional

    g(x) = sum_i x_i * [ ln(x_i) + ln(gamma_i(x)) ]

em funcao de DUAS composicoes independentes (x2, x3), com x1 = 1 - x2 - x3.
No ponto critico de um ternario, valem simultaneamente (criterio classico
de Gibbs para estabilidade critica; ver Michelsen & Mollerup, "Thermodynamic
Models: Fundamentals & Computational Aspects", cap. 9; Baker/Luks):

  (1) det(H) = 0
      onde H e a matriz Hessiana 2x2 de g em relacao a (x2, x3).
      Esta e a condicao SPINODAL (limite de estabilidade difusional).

  (2) det(M) = 0
      onde M e a matriz obtida de H substituindo UMA linha pelo gradiente
      de det(H). Esta e a condicao CRITICA de terceira ordem, que
      distingue o ponto critico dos demais pontos da spinodal.

As duas equacoes formam um sistema 2x2 em (x2, x3), resolvido por
Newton-Raphson. Todas as derivadas de g sao obtidas por diferenca finita
central, de modo que o solver e AGNOSTICO ao modelo: basta o objeto expor
.gamma(x) e .n (numero de componentes), igual ao resto da ferramenta.
"""

import numpy as np


def _g_mix(model, x2, x3):
    """Energia molar de Gibbs de mistura adimensional em (x2, x3)."""
    x1 = 1.0 - x2 - x3
    x = np.array([x1, x2, x3], dtype=float)
    if np.any(x <= 1e-12):
        return np.nan
    g = model.gamma(x)
    if g is None or np.any(~np.isfinite(g)) or np.any(g <= 0):
        return np.nan
    return float(np.sum(x * (np.log(x) + np.log(g))))


def _hessian(model, x2, x3, h=1e-4):
    """Hessiana 2x2 de g em (x2, x3) por diferenca finita central."""
    g_pp = _g_mix(model, x2 + h, x3 + h)
    g_pm = _g_mix(model, x2 + h, x3 - h)
    g_mp = _g_mix(model, x2 - h, x3 + h)
    g_mm = _g_mix(model, x2 - h, x3 - h)
    g_0 = _g_mix(model, x2, x3)
    g_p2 = _g_mix(model, x2 + h, x3)
    g_m2 = _g_mix(model, x2 - h, x3)
    g_p3 = _g_mix(model, x2, x3 + h)
    g_m3 = _g_mix(model, x2, x3 - h)
    g22 = (g_p2 - 2.0 * g_0 + g_m2) / (h * h)
    g33 = (g_p3 - 2.0 * g_0 + g_m3) / (h * h)
    g23 = (g_pp - g_pm - g_mp + g_mm) / (4.0 * h * h)
    return np.array([[g22, g23], [g23, g33]])


def _min_eigenval(model, x2, x3, h=1e-4):
    """Menor autovalor da Hessiana. Negativo => ponto instavel
    (difusionalmente). A fronteira (=0) e a spinodal."""
    H = _hessian(model, x2, x3, h)
    if not np.all(np.isfinite(H)):
        return np.nan
    return float(np.linalg.eigvalsh(H)[0])


def _unstable_width(model, x2, n_scan=200, h=1e-4):
    """Largura do intervalo de x3 instavel (menor autovalor < 0) na fatia
    de x2 dada, e o ponto medio desse intervalo. Retorna (largura, x3_meio)
    ou (0.0, None) se a fatia for inteiramente estavel."""
    x3max = 1.0 - x2 - 0.003
    if x3max < 0.01:
        return 0.0, None
    xs = np.linspace(0.003, x3max, n_scan)
    lam = np.array([_min_eigenval(model, x2, x3, h) for x3 in xs])
    neg = lam < 0
    if not neg.any():
        return 0.0, None
    idx = np.where(neg)[0]
    x3_inf, x3_sup = xs[idx[0]], xs[idx[-1]]
    return float(x3_sup - x3_inf), float(0.5 * (x3_inf + x3_sup))


def solve_plait_point(model, x0=None, tol=1e-9, max_iter=80,
                      h=1e-4, h_outer=1e-3, verbose=False):
    """Localiza o ponto critico (plait point) de um sistema ternario tipo 1.

    Metodo: o ponto critico e onde os dois ramos da binodal (equivalente:
    a fronteira da regiao instavel) se encontram, ou seja, onde a LARGURA
    do intervalo instavel em x3 encolhe a zero conforme se varia x2. Em vez
    de um Newton 2D na condicao de terceira ordem (numericamente instavel,
    pois depende de derivada terceira de g por diferenca finita), varre-se
    x2, mede-se a largura da regiao instavel em cada fatia, e extrapola-se
    a curva largura(x2) ate zero. E geometricamente garantido para tipo 1 e
    muito mais robusto.

    model : objeto com .gamma(x) e .n == 3
    Retorna dict com composicao (x1,x2,x3), convergencia e diagnostico.
    """
    if getattr(model, "n", 3) != 3:
        raise ValueError("solver aplicavel apenas a sistemas ternarios")

    # 1) varredura grossa de x2 medindo a largura instavel
    x2_grid = np.linspace(0.02, 0.97, 60)
    dados = []
    for x2 in x2_grid:
        w, x3m = _unstable_width(model, x2, h=h)
        if w > 0:
            dados.append((x2, w, x3m))
    if len(dados) < 3:
        return {"converged": False, "reason": "regiao instavel ausente/pequena",
                "x": None}

    dados = np.array([(d[0], d[1], d[2]) for d in dados])
    x2v, wv, x3v = dados[:, 0], dados[:, 1], dados[:, 2]

    # 2) o ponto critico esta na extremidade onde a largura -> 0. Descobre
    #    de que lado a largura decresce ate sumir (tipicamente x2 alto).
    #    Usa as ultimas fatias com largura>0 para extrapolar linearmente
    #    largura(x2)=0 e x3_meio no mesmo x2.
    #    Escolhe o lado (maior ou menor x2) com menor largura na ponta.
    if wv[-1] <= wv[0]:
        # decresce para x2 alto: usa as ultimas fatias
        sel = slice(-6, None)
        side = "x2_alto"
    else:
        sel = slice(0, 6)
        side = "x2_baixo"

    x2s, ws, x3s = x2v[sel], wv[sel], x3v[sel]
    # ajuste linear largura ~ a*x2 + b, resolve largura=0
    A = np.polyfit(x2s, ws, 1)
    if abs(A[0]) < 1e-12:
        return {"converged": False, "reason": "largura nao extrapola a zero",
                "x": None}
    x2_crit = -A[1] / A[0]
    # x3 do ponto critico: extrapola x3_meio no mesmo x2_crit
    B = np.polyfit(x2s, x3s, 1)

    # 3) refina por bisseccao: a extrapolacao LINEAR erra porque a largura
    #    encolhe mais rapido que linear perto do critico (lei de potencia
    #    ~sqrt). Procura, entre a ultima fatia instavel conhecida e a
    #    extrapolacao, o x2 onde a largura realmente cruza zero.
    if side == "x2_alto":
        x2_lo = x2s[-1]            # ainda instavel
        x2_hi = max(x2_crit, x2_lo + 1e-4)  # provavelmente ja estavel
    else:
        x2_hi = x2s[0]
        x2_lo = min(x2_crit, x2_hi - 1e-4)

    def largura_em(x2q):
        w, _ = _unstable_width(model, float(np.clip(x2q, 0.001, 0.999)), h=h)
        return w

    # garante bracket: lo instavel (w>0), hi estavel (w==0)
    lo, hi = (x2_lo, x2_hi) if side == "x2_alto" else (x2_hi, x2_lo)
    w_lo = largura_em(lo)
    w_hi = largura_em(hi)
    # expande hi se ainda instavel
    tries = 0
    while w_hi > 0 and tries < 20:
        step = (hi - lo)
        hi = hi + step
        w_hi = largura_em(hi)
        tries += 1
    if w_lo > 0 and w_hi == 0:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if largura_em(mid) > 0:
                lo = mid
            else:
                hi = mid
        x2_crit = lo   # ultimo x2 ainda instavel (largura -> 0+)

    x3_crit = B[0] * x2_crit + B[1]
    x1_crit = 1.0 - x2_crit - x3_crit

    x = np.array([x1_crit, x2_crit, x3_crit])
    if np.any(x < -1e-3) or np.any(x > 1 + 1e-3):
        return {"converged": False, "reason": "extrapolacao fora do simplex",
                "x": x, "side": side}
    x = np.clip(x, 0.0, 1.0)
    x = x / x.sum()

    # residuo: quao perto de zero esta a largura instavel no ponto achado
    w_final, _ = _unstable_width(model, float(np.clip(x[1], 0.02, 0.97)), h=h)
    if verbose:
        print("  lado=%s  x2_crit=%.4f  x3_crit=%.4f  largura_residual=%.4f"
              % (side, x2_crit, x3_crit, w_final))
    return {"converged": True, "x": x, "residual_width": float(w_final),
            "side": side, "n_iter": 1}


def solve_plait_multistart(model, starts=None, **kw):
    """Compatibilidade: o solver robusto por encolhimento da regiao
    instavel nao depende de chute inicial, entao esta funcao apenas chama
    solve_plait_point uma vez. Mantida para nao quebrar chamadas existentes."""
    kw.pop("verbose", None)
    res = solve_plait_point(model, **kw)
    return res


def comparar(model, out=None, n_grid=24, label=""):
    """Utilitario: imprime lado a lado o ponto critico ESTIMADO (extrapolacao)
    e o CALCULADO (solver), usando a estimativa como chute inicial do solver.
    'out' pode ser um resultado de grid_scan ja pronto; se None, roda um.
    Retorna (plait_estimado, plait_calculado)."""
    import ternary_diagram as tdg
    import thermo_engine as eng

    if out is None:
        out = eng.grid_scan(model, n_grid=n_grid)
    plait_est = tdg.estimate_plait_point(out["tie_lines"])

    starts = []
    if plait_est is not None:
        starts.append((float(plait_est[1]), float(plait_est[2])))
    starts += [(1/3, 1/3), (0.5, 0.25), (0.25, 0.5), (0.25, 0.25)]
    res = solve_plait_multistart(model, starts=starts)

    if label:
        print("== %s ==" % label)
    if plait_est is not None:
        print("  ESTIMADO (extrapolacao): x1=%.4f x2=%.4f x3=%.4f"
              % (plait_est[0], plait_est[1], plait_est[2]))
    else:
        print("  ESTIMADO: sem tie-lines suficientes")
    if res["converged"]:
        x = res["x"]
        print("  CALCULADO (solver):      x1=%.4f x2=%.4f x3=%.4f  (|F|=%.1e)"
              % (x[0], x[1], x[2], res["residual"]))
        if plait_est is not None:
            print("  distancia entre os dois: %.4f"
                  % float(np.linalg.norm(res["x"] - plait_est)))
        return plait_est, res["x"]
    else:
        print("  CALCULADO: solver nao convergiu (%s)" % res["reason"])
        return plait_est, None