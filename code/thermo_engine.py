# -*- coding: utf-8 -*-
"""
Motor termodinamico para calculo de equilibrio liquido-liquido (ELL) em
sistemas ternarios. Agnostico ao modelo: recebe qualquer objeto `model`
que exponha `model.gamma(x)` (vetor de coeficientes de atividade) e
`model.n` (numero de componentes). Assim, a mesma logica serve ao
UNIFAC-LL e ao adaptador COSMO-SAC.

Tres blocos:
  1. Estabilidade de fase pelo criterio do plano tangente (TPD).
  2. Flash ELL por Rachford-Rice + substituicoes sucessivas.
  3. Varredura da grade triangular do simplex.
"""

import numpy as np
from scipy.optimize import minimize, brentq


# ---------------------------------------------------------------------------
# Parametros numericos (conforme especificacao)
# ---------------------------------------------------------------------------
D_THRESHOLD = -5.0e-3      # limiar do plano tangente para instabilidade
N_RANDOM_STARTS = 3        # sorteios de Dirichlet na busca de estabilidade
RANDOM_SEED = 42           # semente para reprodutibilidade

# Estimativa inicial do flash (ajustada para estes sistemas)
XA0 = np.array([0.88, 0.06, 0.06])   # fase rica em oleo
XB0 = np.array([0.04, 0.22, 0.74])   # fase rica em PIL

SS_TOL = 1.0e-10           # criterio de convergencia das substituicoes sucessivas
SS_MAX_ITER = 500          # maximo de iteracoes
PSI_MIN, PSI_MAX = 0.01, 0.99   # fracao de fase nao trivial
DIST_MIN = 0.02            # separacao minima entre fases
MB_TOL = 1.0e-3            # residuo maximo de balanco de massa
COMP_FLOOR = 0.02          # fracao minima por componente na grade


# ---------------------------------------------------------------------------
# 1. Estabilidade de fase (criterio do plano tangente)
# ---------------------------------------------------------------------------
def _ln_activity(model, x):
    x = np.clip(np.asarray(x, dtype=float), 1e-300, None)
    x = x / x.sum()
    return np.log(x) + np.log(model.gamma(x))


def tangent_plane_distance(model, z, w):
    """D(w) = sum_i w_i [ln a_i(w) - ln a_i(z)]."""
    w = np.clip(np.asarray(w, dtype=float), 1e-300, None)
    w = w / w.sum()
    ln_a_z = _ln_activity(model, z)
    ln_a_w = _ln_activity(model, w)
    return float(np.sum(w * (ln_a_w - ln_a_z)))


def _softmax_fixed(u_free):
    """w = softmax([0, u1, u2]); remove a redundancia de deslocamento."""
    u = np.concatenate(([0.0], u_free))
    u = u - u.max()
    e = np.exp(u)
    return e / e.sum()


def is_unstable(model, z, log_callback=None):
    """Testa a estabilidade do feed z pelo plano tangente.

    Minimiza D(w) com L-BFGS-B a partir de multiplos pontos de partida
    (cada vertice puro + N_RANDOM_STARTS sorteios de Dirichlet). Retorna
    (instavel: bool, w_min: array, D_min: float, invalido: bool).

    'invalido' = True significa que TODOS os pontos de partida produziram
    D nao finito (NaN/Inf), tipicamente por overflow/NaN vindo do modelo
    (ex.: gamma extremo do COSMO-SAC em cantos do simplex). Nesse caso o
    ponto NAO deve ser silenciosamente tratado como estavel; o chamador
    deve contabilizar e reportar.
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    ln_a_z = _ln_activity(model, z)
    n = model.n

    if not np.all(np.isfinite(ln_a_z)):
        if log_callback is not None:
            log_callback("AVISO: ln(atividade) nao finita no feed z=%s "
                        "(gamma retornou valor invalido). Ponto ignorado."
                        % np.round(z, 4))
        return False, z, np.nan, True

    def obj(u_free):
        w = _softmax_fixed(u_free)
        wc = np.clip(w, 1e-300, None)
        ln_a_w = np.log(wc) + np.log(model.gamma(w))
        return float(np.sum(w * (ln_a_w - ln_a_z)))

    # pontos de partida: proximos a cada vertice + Dirichlet
    starts = []
    big = 8.0
    for i in range(n):
        w0 = np.full(n, np.exp(-big))
        w0[i] = 1.0
        w0 = w0 / w0.sum()
        starts.append(w0)
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(N_RANDOM_STARTS):
        starts.append(rng.dirichlet(np.ones(n)))

    best_D = np.inf
    best_w = z.copy()
    any_finite = False
    for w0 in starts:
        w0 = np.clip(w0, 1e-12, None)
        w0 = w0 / w0.sum()
        u0 = np.log(w0[1:] / w0[0])
        res = minimize(obj, u0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
        w = _softmax_fixed(res.x)
        D = obj(res.x)
        if np.isfinite(D):
            any_finite = True
            if D < best_D:
                best_D = D
                best_w = w

    if not any_finite:
        if log_callback is not None:
            log_callback("AVISO: todas as tentativas de minimizacao do plano "
                        "tangente produziram D nao finito para z=%s "
                        "(gamma provavelmente estourou). Ponto ignorado, "
                        "NAO contabilizado como estavel." % np.round(z, 4))
        return False, z, np.nan, True

    return (best_D < D_THRESHOLD), best_w, best_D, False


# ---------------------------------------------------------------------------
# 2. Flash ELL (Rachford-Rice + substituicoes sucessivas)
# ---------------------------------------------------------------------------
def _solve_rachford_rice(z, K):
    """Resolve f(psi)=sum z_i(K_i-1)/[1+psi(K_i-1)]=0 na janela fisica
    (flash negativo de Whitson-Michelsen). Retorna psi ou None."""
    K = np.asarray(K, dtype=float)
    if K.max() <= 1.0 or K.min() >= 1.0:
        return None  # sem inversao de K, nao ha bracket para separacao

    def f(psi):
        return np.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0)))

    lower = 1.0 / (1.0 - K.max())   # < 0
    upper = 1.0 / (1.0 - K.min())   # > 1
    a = lower + 1e-9
    b = upper - 1e-9
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return None
    try:
        return brentq(f, a, b, xtol=1e-14, maxiter=200)
    except Exception:
        return None


def flash(model, z, xa0=None, xb0=None):
    """Flash ELL de duas fases. Retorna dict com as fases ou None se o
    resultado nao passar nos criterios de aceitacao.

    Estimativa inicial: usa xa0/xb0 se fornecidos; caso contrario recorre a
    XA0/XB0 (afinadas para os sistemas ternarios oleo/alcool/PIL).
    Convencao: xa e a fase rica em oleo (xa[0] > xb[0])."""
    z = np.asarray(z, dtype=float)
    z = z / z.sum()

    xa_ini = np.asarray(xa0, float) if xa0 is not None else XA0
    xb_ini = np.asarray(xb0, float) if xb0 is not None else XB0
    xa_ini = xa_ini / xa_ini.sum()
    xb_ini = xb_ini / xb_ini.sum()

    K = xb_ini / xa_ini      # estimativa inicial de K a partir das composicoes-guia
    xa, xb, psi = xa_ini.copy(), xb_ini.copy(), 0.5
    converged = False

    for _ in range(SS_MAX_ITER):
        psi = _solve_rachford_rice(z, K)
        if psi is None:
            break

        denom = 1.0 + psi * (K - 1.0)
        xa = z / denom
        xb = K * xa
        # renormalizacao de seguranca (no zero de RR, ambos ja somam 1)
        xa = np.clip(xa, 0.0, None); xa = xa / xa.sum()
        xb = np.clip(xb, 0.0, None); xb = xb / xb.sum()

        ga = model.gamma(xa)
        gb = model.gamma(xb)
        K_new = ga / gb

        diff = float(np.sum((K_new - K) ** 2))
        K = K_new
        if diff < SS_TOL:
            converged = True
            break

    if not converged:
        return None

    # recomputa composicoes com o K final
    psi = _solve_rachford_rice(z, K)
    if psi is None:
        return None
    denom = 1.0 + psi * (K - 1.0)
    xa = z / denom; xa = np.clip(xa, 0.0, None); xa = xa / xa.sum()
    xb = K * (z / denom); xb = np.clip(xb, 0.0, None); xb = xb / xb.sum()

    # convencao: fase alfa rica em oleo (componente 1)
    if xa[0] < xb[0]:
        xa, xb = xb, xa
        psi = 1.0 - psi

    # --- criterios de aceitacao (todos obrigatorios) ---
    if not (PSI_MIN < psi < PSI_MAX):
        return None
    if np.max(np.abs(xa - xb)) <= DIST_MIN:
        return None
    mb = np.max(np.abs(z - ((1.0 - psi) * xa + psi * xb)))
    if mb >= MB_TOL:
        return None
    # residuo de isoatividade (diagnostico)
    ga = model.gamma(xa); gb = model.gamma(xb)
    iso = float(np.max(np.abs(xa * ga - xb * gb)))

    return {"xa": xa, "xb": xb, "psi": float(psi),
            "mass_balance": float(mb), "isoactivity": iso}


# ---------------------------------------------------------------------------
# 3. Varredura da grade triangular
# ---------------------------------------------------------------------------
def _triangular_grid(n_grid):
    pts = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            pts.append(np.array([i, j, k], dtype=float) / n_grid)
    return pts


def grid_scan(model, n_grid=20, progress_callback=None, log_callback=None):
    """Varre a grade do simplex, testa estabilidade e faz flash dos pontos
    instaveis. Retorna dict com as tie-lines aceitas (deduplicadas) e
    estatisticas da varredura, incluindo pontos invalidos (gamma nao
    finito) que NAO sao contados como estaveis."""
    grid = _triangular_grid(n_grid)
    grid = [p for p in grid if np.min(p) >= COMP_FLOOR]
    total = len(grid)

    tie_lines = []
    seen = set()
    n_unstable = 0
    n_invalid = 0
    invalid_examples = []

    for idx, z in enumerate(grid):
        unstable, _w, _D, invalid = is_unstable(model, z)
        if invalid:
            n_invalid += 1
            if len(invalid_examples) < 3:
                invalid_examples.append(tuple(np.round(z, 3)))
        elif unstable:
            n_unstable += 1
            res = flash(model, z)
            if res is not None:
                key = tuple(np.round(res["xa"], 3)) + tuple(np.round(res["xb"], 3))
                if key not in seen:
                    seen.add(key)
                    tie_lines.append(res)
        if progress_callback is not None:
            progress_callback(idx + 1, total)

    if log_callback is not None:
        log_callback("Grade: %d pontos testados, %d instaveis, %d tie-lines aceitas."
                     % (total, n_unstable, len(tie_lines)))
        if n_invalid > 0:
            log_callback("AVISO: %d de %d pontos tiveram gamma nao finito "
                        "(overflow/NaN) e foram DESCARTADOS da varredura, "
                        "nao contados como estaveis. Exemplos de z: %s. "
                        "Isso indica que o resultado 'sem regiao bifasica' "
                        "pode ser um artefato numerico, nao a previsao real "
                        "do modelo." % (n_invalid, total, invalid_examples))

    # ordena as tie-lines por x1 da fase rica em oleo (para amostragem posterior)
    tie_lines.sort(key=lambda r: r["xa"][0])

    return {"tie_lines": tie_lines,
            "n_tested": total,
            "n_unstable": n_unstable,
            "n_invalid": n_invalid,
            "n_accepted": len(tie_lines)}