"""
flash.py — Flash bifasico isotermico para equilibrio liquido-liquido (ELL)
=============================================================================
PROPOSITO
    Dada uma composicao global z e uma temperatura T, calcular as
    composicoes (xa, xb) das duas fases em equilibrio e a fracao psi
    da fase b. Esse calculo e o "coracao" do procedimento de geracao
    de tie-lines em diagramas ternarios.

    Este modulo so deve ser invocado para composicoes JA DETECTADAS
    como instaveis pelo modulo stability.py. Aplicar flash em
    composicoes monofasicas resulta em tie-lines colapsadas (xa = xb)
    ou em divergencia numerica.

CRITERIOS RESOLVIDOS
    O equilibrio liquido-liquido em z, T, P fixos corresponde a:

    (1) ISOFUGACIDADE de cada componente entre as duas fases:
            xa_i * gamma_i(xa) = xb_i * gamma_i(xb)
        Em termos de coeficientes de distribuicao K_i = xb_i/xa_i:
            K_i = gamma_i(xa) / gamma_i(xb)

    (2) BALANCO DE MASSA:
            z_i = psi * xb_i + (1-psi) * xa_i
        Substituindo xb_i = K_i * xa_i:
            xa_i = z_i / [1 + psi * (K_i - 1)]
            xb_i = K_i * z_i / [1 + psi * (K_i - 1)]

    (3) NORMALIZACAO: sum(xa) = sum(xb) = 1
        Subtraindo: sum(xb) - sum(xa) = 0:
            sum_i z_i * (K_i - 1) / [1 + psi*(K_i - 1)] = 0
        Esta e a EQUACAO DE RACHFORD-RICE — uma equacao escalar
        em psi para K_i fixo.

ALGORITMO DE SUBSTITUICOES SUCESSIVAS (Rachford-Rice + atualizacao de K)
    1. Estimativa inicial xa^0, xb^0
    2. K_i^0 = gamma_i(xa^0) / gamma_i(xb^0)
    3. Resolver Rachford-Rice em psi (metodo de Brent)
    4. Calcular xa, xb a partir de psi e K
    5. Atualizar K^new = gamma_i(xa) / gamma_i(xb)
    6. Se |K^new - K| < tol -> convergiu. Senao volta ao passo 3.

REFERENCIAS
    Rachford, H.H., Rice, J.D. (1952)
        "Procedure for use of electronic digital computers..."
        J. Petroleum Tech. 4, 19-20.
    Michelsen, M.L. (1982b)
        "The isothermal flash problem. Part II"
        Fluid Phase Equilib. 9, 21-40.
"""

import numpy as np
from scipy.optimize import brentq
from gamma  import UNIFACLL
from params import NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX


# ===========================================================================
# EQUACAO DE RACHFORD-RICE
# ===========================================================================

def rachford_rice_eq(psi: float, z: np.ndarray, K: np.ndarray) -> float:
    """
    f(psi) = sum_i z_i * (K_i - 1) / [1 + psi * (K_i - 1)]

    A raiz f(psi) = 0 em (0, 1) fornece a fracao molar da fase b para
    K_i e z_i fixos.

    PROPRIEDADES DA FUNCAO:
        - Decrescente em psi (em todo dominio onde e definida)
        - Tem assintotas verticais em psi = 1/(1-K_max) e 1/(1-K_min)
        - Cruza zero em pelo menos um ponto se min(K) < 1 < max(K)
          (condicao de bifase real — pelo menos um componente prefere
          cada fase)

    Por que essa formulacao?
        Inicialmente o problema parece ser duas equacoes (sum xa = 1
        e sum xb = 1). Subtraindo uma da outra, eliminamos o redundante
        e ficamos com UMA EQUACAO em UMA VARIAVEL (psi). E mais facil
        de resolver e numericamente estavel.
    """
    return float(np.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0))))


def solve_rachford_rice(z: np.ndarray, K: np.ndarray) -> float:
    """
    Resolve f(psi) = 0 pelo metodo de Brent no intervalo fisicamente valido.

    INTERVALO DE BUSCA:
        Para que xa_i, xb_i > 0 para todo i, precisamos:
            psi > 1/(1 - K_max)  se K_max > 1
            psi < 1/(1 - K_min)  se K_min < 1
        Isso garante que o denominador 1 + psi*(K_i - 1) > 0 para todo i.

    RETORNA:
        psi em (0, 1) — fracao molar da fase b
    """
    Kmax, Kmin = K.max(), K.min()

    # Limite inferior: psi minimo para xa_i positivo do componente com K maior
    lo = 1.0 / (1.0 - Kmax) + 1e-10 if Kmax > 1.0 else 1e-10
    # Limite superior: psi maximo para xa_i positivo do componente com K menor
    hi = 1.0 / (1.0 - Kmin) - 1e-10 if Kmin < 1.0 else 1.0 - 1e-10

    # Restringe ao dominio fisico [0, 1]
    lo = max(lo, 1e-10)
    hi = min(hi, 1.0 - 1e-10)

    if lo >= hi:
        # Caso degenerado: K nao prevê separacao real. Retorna ponto medio
        # como fallback — o flash externo ira detectar nao-convergencia
        # nas iteracoes seguintes.
        lo, hi = 1e-10, 1.0 - 1e-10

    # Verifica que ha mudanca de sinal (necessario para Brent)
    f_lo = rachford_rice_eq(lo, z, K)
    f_hi = rachford_rice_eq(hi, z, K)

    if f_lo * f_hi > 0:
        # Sem mudanca de sinal — RR nao tem raiz no intervalo. Indica
        # que K_i estao em estado inconsistente (provavelmente proximo
        # ao colapso de fases). Retorna 0.5 como fallback.
        return 0.5

    return brentq(rachford_rice_eq, lo, hi, args=(z, K), xtol=1e-14)


# ===========================================================================
# FLASH ELL PRINCIPAL
# ===========================================================================

def flash_ell(z: np.ndarray,
              model: UNIFACLL,
              tol: float = 1e-10,
              max_iter: int = 500) -> tuple:
    """
    Flash bifasico isotermico ELL pelo metodo de substituicoes sucessivas.

    PARAMETROS:
        z         : composicao global (N,)
        model     : instancia de UNIFACLL com T pre-definido
        tol       : tolerancia em |K^new - K|^2 para convergencia
        max_iter  : limite de iteracoes (raramente atinge)

    RETORNA:
        xa        : composicao da fase a (rica em comp. 1, por convencao)
        xb        : composicao da fase b (rica em outros comps.)
        psi       : fracao molar da fase b
        converged : True se convergiu dentro de tol
        n_iter    : iteracoes realizadas

    ESTIMATIVA INICIAL:
        Pre-definida para o tipo de sistema do TCC (triolina + etanol + LIP):
            xa^0 = [0.88, 0.06, 0.06]   (fase rica em triolina)
            xb^0 = [0.04, 0.22, 0.74]   (fase rica em LIP)
        Para outros tipos de sistema, esta inicializacao pode nao
        convergir. Em caso de problemas, ajustar conforme o sistema.
    """
    z = np.clip(np.asarray(z, float), 1e-15, 1.0)
    z = z / z.sum()
    N = len(z)

    # Estimativas iniciais ajustadas para sistemas do tipo do TCC
    xa0 = np.array([0.88, 0.06, 0.06])
    xb0 = np.array([0.04, 0.22, 0.74])

    # Adapta ao numero de componentes (caso N != 3)
    xa0 = np.clip(xa0[:N], 1e-15, 1.0); xa0 /= xa0.sum()
    xb0 = np.clip(xb0[:N], 1e-15, 1.0); xb0 /= xb0.sum()

    # K_i inicial = razao de gamma nas duas fases
    # Clipping evita explosoes numericas (K muito grande/pequeno)
    K   = np.clip(model.gamma(xa0) / model.gamma(xb0), 1e-10, 1e10)
    psi = 0.5

    for n_iter in range(1, max_iter + 1):
        # Passo 1: resolver Rachford-Rice -> psi
        psi = solve_rachford_rice(z, K)

        # Passo 2: composicoes das fases a partir de psi e K
        denom = 1.0 + psi * (K - 1.0)
        xa    = np.clip(z / denom,      1e-15, 1.0); xa /= xa.sum()
        xb    = np.clip(K * xa,         1e-15, 1.0); xb /= xb.sum()

        # Passo 3: atualizar K usando os novos gamma
        K_new = np.clip(model.gamma(xa) / model.gamma(xb), 1e-10, 1e10)

        # Criterio de convergencia: norma quadratica da variacao em K
        err = float(np.sum((K_new - K) ** 2))
        K   = K_new

        if err < tol:
            # Convergiu — verificar balanco de massa antes de aceitar
            mb_res = np.abs(z - (psi * xb + (1.0 - psi) * xa))
            if mb_res.max() > 1e-4:
                # Convergiu para um "minimo trivial" (xa ~= xb)
                # — comum quando z e na verdade monofasico
                return xa, xb, psi, False, n_iter
            return xa, xb, psi, True, n_iter

    # Nao convergiu em max_iter
    return xa, xb, psi, False, max_iter


# ===========================================================================
# VERIFICACOES POS-FLASH
# ===========================================================================
# Funcoes utilitarias para verificar a qualidade do resultado de flash_ell.
# Sao independentes do calculo principal — quem chama flash_ell pode usar
# essas verificacoes para decidir se aceita ou rejeita o resultado.
# ===========================================================================

def check_mass_balance(z: np.ndarray, xa: np.ndarray, xb: np.ndarray,
                       psi: float, tol: float = 1e-6) -> tuple[bool, np.ndarray]:
    """
    Verifica: z_i = psi * xb_i + (1-psi) * xa_i  para todo i.

    Esta e a EQUACAO DE LAVOISIER aplicada ao flash. A massa total
    de cada componente em z deve ser conservada na soma ponderada
    das duas fases.

    Retorna (ok, residual): booleano e vetor de residuos absolutos.
    """
    residual = np.abs(z - (psi * xb + (1.0 - psi) * xa))
    return bool(np.all(residual < tol)), residual


def check_isofugacity(xa: np.ndarray, xb: np.ndarray,
                      model: UNIFACLL,
                      tol: float = 1e-4) -> tuple[bool, np.ndarray]:
    """
    Verifica: xa_i * gamma_i(xa) = xb_i * gamma_i(xb)  para todo i.

    Esta e a CONDICAO DE EQUILIBRIO TERMODINAMICO. Se nao satisfeita,
    o flash nao convergiu para um equilibrio real — a solucao
    encontrada e numerica mas nao fisica.

    Retorna (ok, residual_relativo): booleano e vetor de residuos.
    """
    ga = model.gamma(xa)
    gb = model.gamma(xb)
    lhs = xa * ga
    rhs = xb * gb
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(lhs - rhs) / np.where(lhs > 0, lhs, 1.0)
    return bool(np.all(rel < tol)), rel


def check_rachford_rice_zero(psi: float, z: np.ndarray, K: np.ndarray,
                             tol: float = 1e-8) -> tuple[bool, float]:
    """
    Verifica que f(psi) ~= 0 na solucao encontrada.

    Sanidade do solver Rachford-Rice — em principio deve sempre passar
    se brentq convergiu corretamente.
    """
    val = rachford_rice_eq(psi, z, K)
    return abs(val) < tol, val


# ===========================================================================
# VALIDACAO INDEPENDENTE
# ===========================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("VALIDACAO ESTRUTURAL: flash.py")
    print("Testes do algoritmo de flash bifasico ELL")
    print("=" * 65)

    T = 298.15
    nu_list = [NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX]
    model   = UNIFACLL(nu_list, T)

    # ──────────────────────────────────────────────────────────────────
    # TESTE 1 — RESOLUCAO DA EQUACAO DE RACHFORD-RICE
    # ──────────────────────────────────────────────────────────────────
    # Verifica que solve_rachford_rice() retorna psi tal que f(psi) ~= 0,
    # para K_i e z_i pre-definidos onde sabemos que a raiz existe.
    #
    # FUNDAMENTO: f e estritamente decrescente entre suas assintotas, e
    # se K_max > 1 > K_min, ela cruza zero em (0, 1). Isso eh uma
    # propriedade matematica garantida — o teste verifica que nosso
    # solver implementa corretamente Brent.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 1 — Resolucao da equacao de Rachford-Rice")
    print("-" * 65)
    print("Fundamento: f(psi) = 0 tem raiz unica em (0,1) se min(K)<1<max(K)")
    print()

    K_test = np.array([10.0, 0.5, 0.2])
    z_test = np.array([0.4, 0.3, 0.3])

    psi_rr = solve_rachford_rice(z_test, K_test)
    f_rr   = rachford_rice_eq(psi_rr, z_test, K_test)

    print(f"  K = {K_test}  (K_1 > 1 > K_2 > K_3, valido para bifase)")
    print(f"  z = {z_test}")
    print(f"  psi resolvido = {psi_rr:.8f}")
    print(f"  f(psi) = {f_rr:.2e}")
    ok = abs(f_rr) < 1e-8
    print(f"  Criterio: |f(psi)| < 1e-8  -> {'[OK]' if ok else '[FALHOU]'}")

    # Verificacao adicional: f muda de sinal nos extremos
    f_lo = rachford_rice_eq(0.001, z_test, K_test)
    f_hi = rachford_rice_eq(0.999, z_test, K_test)
    print(f"\n  f(0.001) = {f_lo:+.4f}    f(0.999) = {f_hi:+.4f}")
    print(f"  Mudanca de sinal: {'[OK]' if f_lo * f_hi < 0 else '[FALHOU]'}")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 2 — FLASH EM COMPOSICOES BIFASICAS DO SISTEMA
    # ──────────────────────────────────────────────────────────────────
    # Para composicoes na regiao bifasica do sistema do TCC, o flash
    # deve:
    #   (a) convergir
    #   (b) satisfazer balanco de massa
    #   (c) satisfazer isofugacidade
    #   (d) produzir psi em (0, 1)
    #   (e) produzir xa, xb com soma = 1 e todos componentes positivos
    #
    # As composicoes testadas sao todas no interior da regiao bifasica
    # (rich-LIP), confirmadas em TESTE 3 do stability.py.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 2 — Flash em composicoes bifasicas do sistema")
    print("-" * 65)
    print("Fundamento: flash deve satisfazer conservacao + equilibrio")
    print()

    test_comps = {
        'z1 = [0.05, 0.10, 0.85]': np.array([0.05, 0.10, 0.85]),
        'z2 = [0.10, 0.05, 0.85]': np.array([0.10, 0.05, 0.85]),
        'z3 = [0.05, 0.20, 0.75]': np.array([0.05, 0.20, 0.75]),
    }

    pass_count = 0
    for name, z in test_comps.items():
        xa, xb, psi, ok, n_it = flash_ell(z, model)

        # Verificacao 1: balanco de massa
        mb_ok, mb_res = check_mass_balance(z, xa, xb, psi)

        # Verificacao 2: isofugacidade (equilibrio real)
        K_sol = model.gamma(xa) / model.gamma(xb)
        iso_ok, iso_res = check_isofugacity(xa, xb, model)

        # Verificacao 3: Rachford-Rice = 0
        rr_ok, rr_val = check_rachford_rice_zero(psi, z, K_sol, tol=1e-5)

        # Verificacao 4: psi fisicamente valido
        psi_ok = 0.0 < psi < 1.0

        # Verificacao 5: normalizacao e positividade
        norm_ok = (abs(xa.sum() - 1.0) < 1e-10 and
                   abs(xb.sum() - 1.0) < 1e-10 and
                   np.all(xa > 0) and np.all(xb > 0))

        passou = ok and mb_ok and iso_ok and rr_ok and psi_ok and norm_ok
        pass_count += passou

        status = "OK" if passou else "FALHOU"
        print(f"\n  [{status}] {name}  (convergiu={ok}, iter={n_it})")
        print(f"      xa = [{xa[0]:.4f}, {xa[1]:.4f}, {xa[2]:.4f}]")
        print(f"      xb = [{xb[0]:.4f}, {xb[1]:.4f}, {xb[2]:.4f}]")
        print(f"      psi = {psi:.4f}")
        print(f"      Balanco de massa: {'[OK]' if mb_ok else '[X]'}  "
              f"max|res|={mb_res.max():.2e}")
        print(f"      Isofugacidade:    {'[OK]' if iso_ok else '[X]'}  "
              f"max|res_rel|={iso_res.max():.2e}")
        print(f"      Rachford-Rice=0:  {'[OK]' if rr_ok else '[X]'}  "
              f"f(psi)={rr_val:.2e}")
        print(f"      psi em (0,1):     {'[OK]' if psi_ok else '[X]'}")
        print(f"      Normalizacao:     {'[OK]' if norm_ok else '[X]'}")

    print(f"\n  Resultado: {pass_count}/{len(test_comps)} testes aprovados")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 3 — COERENCIA FISICA DA SEPARACAO DE FASES
    # ──────────────────────────────────────────────────────────────────
    # Em equilibrio liquido-liquido com componente 1 = triolina (apolar)
    # e componente 3 = LIP (polar), esperamos que:
    #   - fase a (rotulada por convencao) seja RICA em triolina
    #   - fase b seja RICA em LIP
    # Logo: xa[0] > xb[0]  (mais triolina em a)
    #        xb[2] > xa[2]  (mais LIP em b)
    #
    # Esta NAO eh propriedade matematica do flash, mas decorre da
    # convencao de qual fase e' rotulada como "a" no codigo.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 3 — Coerencia fisica: fase a deve ser rica em triolina")
    print("-" * 65)
    print()

    pass_count = 0
    for name, z in test_comps.items():
        xa, xb, psi, ok, _ = flash_ell(z, model)
        coerente = xa[0] > xb[0]
        pass_count += coerente
        status = "OK" if coerente else "[ATENCAO: invertido]"
        print(f"  [{status:20s}] {name}: "
              f"xa_triol={xa[0]:.3f}  xb_triol={xb[0]:.3f}")
    print(f"\n  Resultado: {pass_count}/{len(test_comps)} aprovados")
    print("  Nota: inversao nao e bug — apenas convencao de rotulacao.")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 4 — COMPORTAMENTO EM EDGE CASES NUMERICOS
    # ──────────────────────────────────────────────────────────────────
    # Composicoes em regioes monofasicas ou proximas aos vertices podem
    # causar instabilidades numericas. O flash deve ser ROBUSTO:
    #   - nao lancar excecoes nao tratadas
    #   - retornar com converged=False quando apropriado
    #   - nao retornar resultados ABSURDOS (NaN, valores fora de [0,1])
    #
    # Para composicoes monofasicas, esperamos converged=False ou
    # tie-line colapsada (xa ~ xb).
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 4 — Robustez em edge cases")
    print("-" * 65)
    print("Fundamento: flash deve falhar de forma controlada em monofasico")
    print()

    edge_cases = [
        (np.array([0.97, 0.015, 0.015]), 'quase puro triolina (monofasico)'),
        (np.array([0.05, 0.05, 0.90]),   'rico em LIP (bifasico)'),
        (np.array([0.05, 0.90, 0.05]),   'rico em etanol (cossolvente)'),
    ]
    for z, desc in edge_cases:
        try:
            xa, xb, psi, ok, n_it = flash_ell(z, model)
            mb_ok, _ = check_mass_balance(z, xa, xb, psi)
            no_nan = not (np.any(np.isnan(xa)) or np.any(np.isnan(xb)))
            in_range = (0.0 <= psi <= 1.0)
            status = "OK" if (no_nan and in_range) else "FALHOU"
            print(f"  [{status}] {desc}")
            print(f"      conv={ok}  psi={psi:.3f}  iter={n_it}  "
                  f"MB={'[OK]' if mb_ok else '[X]'}")
            print(f"      xa={xa}  xb={xb}")
        except Exception as e:
            print(f"  [FALHOU - EXCECAO] {desc}: {e}")

    print("\n" + "=" * 65)
    print("FIM DA VALIDACAO ESTRUTURAL")
    print("=" * 65)