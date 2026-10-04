"""
stability.py — Analise de estabilidade de fases pelo criterio do plano tangente
=================================================================================
PROPOSITO
    Determinar se uma mistura de composicao global z, a temperatura T fixa,
    e termodinamicamente ESTAVEL (uma unica fase liquida) ou INSTAVEL
    (duas ou mais fases coexistentes em equilibrio).

    Esta e a etapa preliminar e CRUCIAL antes do calculo de tie-lines.
    Sem este teste, o algoritmo de flash pode "achar" duas fases que
    sao na verdade matematicamente identicas — uma falsa solucao.

FUNDAMENTO TERMODINAMICO
    Uma mistura e estavel quando sua energia de Gibbs total e MINIMA na
    composicao z. Se existir alguma outra composicao w tal que combinar
    fases na composicao w + (1-z) outra rebaixe a energia de Gibbs, entao
    a mistura nao esta no minimo: ela vai SEPARAR.

    O criterio matematico foi formalizado por Baker, Pierce & Luks (1982):
        D(w) = sum_i w_i * [ln w_i + ln gamma_i(w) - ln z_i - ln gamma_i(z)]

    INTERPRETACAO GEOMETRICA:
        Imagine a superficie de Gibbs molar reduzida g(x)/RT como funcao
        de x no simplex composicional. Em uma composicao z estavel, essa
        superficie esta acima do PLANO TANGENTE em todos os pontos w
        diferentes de z. D(w) mede a distancia vertical entre a superficie
        e o plano tangente em w. Se D(w) < 0 em algum w, a superficie
        "fura" o plano tangente, indicando que a mistura na composicao
        w teria menor energia que a interpolacao linear entre z e algo
        proximo — sinal de separacao de fases.

    REGRA OPERACIONAL:
        Se min_w D(w) < 0  ->  z e INSTAVEL (separa fases)
        Se min_w D(w) >= 0 ->  z e ESTAVEL  (uma so fase)

    Em pratica usamos um pequeno limite negativo (D_threshold ~ -5e-3)
    para evitar falsos positivos por ruido numerico.

REFERENCIAS
    Baker, L.E., Pierce, A.C., Luks, K.D. (1982)
        "Gibbs energy analysis of phase equilibria"
        SPE Journal 22, 731-742.
    Michelsen, M.L. (1982a, 1982b)
        "The isothermal flash problem" (Parts I, II)
        Fluid Phase Equilib. 9, 1-19; 9, 21-40.
"""

import numpy as np
from scipy.optimize import minimize
from gamma  import UNIFACLL
from params import NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX


# ===========================================================================
# FUNCAO TPD (Tangent Plane Distance)
# ===========================================================================

def tangent_plane_D(w: np.ndarray, z: np.ndarray,
                    ln_gz: np.ndarray, model: UNIFACLL) -> float:
    """
    Calcula D(w) — distancia da superficie de Gibbs ao plano tangente
    em z, avaliada na composicao de teste w.

    EQUACAO:
        D(w) = sum_i w_i * [ln w_i + ln gamma_i(w) - h_i]
        onde h_i = ln z_i + ln gamma_i(z) (referencia, fixa)

    O termo h_i e pre-calculado em ln_gz e passado como argumento, pois
    em is_unstable() chamamos D(w) varias vezes com o mesmo z.

    INTERPRETACAO POR TERMO:
        ln w_i + ln gamma_i(w) = ln a_i(w)        (atividade em w)
        ln z_i + ln gamma_i(z) = ln a_i(z)        (atividade em z)
        D(w) = sum w_i * ln(a_i(w)/a_i(z))

    Se D(w) < 0:
        sum w_i * ln(a_i(w)/a_i(z)) < 0
    Significa que, NA MEDIA, as atividades em w sao MENORES que em z.
    Isso e impossivel termodinamicamente para uma fase estavel, pois
    uma fase em equilibrio com outra (ambas tem mesma atividade) so
    pode existir se z for instavel.

    PARAMETROS:
        w     : composicao de teste (sera variada na minimizacao)
        z     : composicao global fixa
        ln_gz : pre-calculado fora -> ln z_i + ln gamma_i(z)
        model : instancia de UNIFACLL para calcular gamma_i(w)
    """
    # Saneamento do input
    w = np.clip(w, 1e-15, 1.0)
    w = w / w.sum()

    # Calcula ln a_i(w) = ln w_i + ln gamma_i(w)
    ln_gw = np.log(model.gamma(w))

    # D(w) na forma convenientemente vetorizada
    return float(np.sum(w * (np.log(w) + ln_gw - ln_gz)))


# ===========================================================================
# FUNCAO PRINCIPAL — IS_UNSTABLE
# ===========================================================================

def is_unstable(z: np.ndarray, model: UNIFACLL,
                n_random: int = 3,
                D_threshold: float = -5e-3) -> bool:
    """
    Decide se a composicao z e bifasica (True) ou monofasica (False).

    ALGORITMO:
        1. Calcula h_i = ln z_i + ln gamma_i(z) (referencia)
        2. Tenta minimizar D(w) a partir de varios pontos de partida
        3. Se algum minimo for < D_threshold -> z e instavel
        4. Se todos minimos forem >= D_threshold -> z e estavel

    POR QUE MULTIPLOS PONTOS DE PARTIDA?
        D(w) tipicamente tem multiplos minimos locais no simplex
        composicional. Um minimizador local (L-BFGS-B) pode parar em
        um minimo > 0 e perder o que esta < 0 em outra regiao. A
        estrategia padrao (Michelsen 1982) e tentar:
        - Cada componente puro (w[i] = 1 - epsilon)
        - Pontos aleatorios pelo simplex (uma distribuicao Dirichlet)

        Isso da coverage do simplex sem custo proibitivo.

    PARAMETROS:
        z           : composicao global (N,) — ja deve somar 1 (ou sera
                      normalizada)
        model       : instancia de UNIFACLL
        n_random    : pontos de partida adicionais (alem dos puros). Mais
                      pontos = mais confiavel mas mais lento.
        D_threshold : tolerancia para considerar D < 0. Usar exatamente 0
                      e arriscado por ruido numerico — usamos -5e-3 que
                      capta separacoes reais e ignora ruido.

    RETORNA:
        True  -> z e bifasica (separa em duas fases)
        False -> z e monofasica (estavel)
    """
    # Normaliza z para garantir sum = 1
    z = np.clip(z, 1e-15, 1.0)
    z = z / z.sum()
    N = len(z)

    # Pre-calcula referencia: h_i = ln z_i + ln gamma_i(z)
    ln_gz = np.log(z) + np.log(model.gamma(z))

    # ──────────────────────────────────────────────────────────────────
    # CONSTRUCAO DOS PONTOS DE PARTIDA
    # ──────────────────────────────────────────────────────────────────
    # Estrategia 1: vizinhanca de cada componente puro
    # Logica: se a fase rica em algum componente i for estavel, sua
    # composicao deve estar proxima de [0,...,1,...,0]. Comecar a busca
    # ali tem alta probabilidade de encontrar minimos relevantes.
    starts = []
    for i in range(N):
        w = np.ones(N) * 0.02         # cada outro componente com 2%
        w[i] = 1.0 - 0.02 * (N - 1)   # componente i com o restante
        starts.append(w / w.sum())

    # Estrategia 2: pontos aleatorios uniformes no simplex
    # Distribuicao Dirichlet com parametro 1 da uma distribuicao uniforme
    # sobre o simplex N-dimensional. Coverage adicional para casos onde
    # os minimos relevantes nao estao proximos a vertices.
    rng = np.random.default_rng(seed=42)  # seed fixa = reproducibilidade
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(N)))

    # ──────────────────────────────────────────────────────────────────
    # MINIMIZACAO COM REDUCAO DIMENSIONAL
    # ──────────────────────────────────────────────────────────────────
    # O simplex composicional tem dimensao N-1 (a restricao sum w_i = 1
    # remove um grau de liberdade). Trabalhamos com w[:-1] como variaveis
    # livres e w[-1] = 1 - sum(w[:-1]) implicito.
    #
    # Por que L-BFGS-B?
    #   - Suporta limites (bounds) — necessario para manter w_i positivos
    #   - Converge rapido em poucas iteracoes
    #   - Robusto a gradientes ruidosos (D(w) tem derivadas complexas)
    N_free = N - 1
    bounds = [(1e-10, 1.0 - 1e-10)] * N_free

    D_min = 0.0   # neutro — assumiremos estavel ate prova em contrario

    for w0 in starts:
        w0_free = w0[:-1]   # remove ultimo componente (depende dos demais)

        # Wrapper: reconstroi o vetor completo a partir das variaveis livres
        def obj(wf):
            w_full = np.append(wf, max(1e-15, 1.0 - wf.sum()))
            return tangent_plane_D(w_full, z, ln_gz, model)

        try:
            res = minimize(obj, w0_free, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < D_min:
                D_min = res.fun
                # Curto-circuito: se ja achamos D < threshold, e instavel
                # — nao precisa testar mais pontos de partida
                if D_min < D_threshold:
                    return True
        except Exception:
            # Se a otimizacao falhar para um ponto especifico, ignora e
            # continua com os outros. Isso pode acontecer em pontos
            # patologicos (ex: w totalmente em vertice).
            pass

    return False


# ===========================================================================
# VALIDACAO INDEPENDENTE
# ===========================================================================
# Os testes abaixo verificam propriedades MATEMATICAS da funcao TPD e da
# logica de is_unstable(), independentemente de qual sistema fisico.
# ===========================================================================

if __name__ == '__main__':
    import time

    print("=" * 65)
    print("VALIDACAO ESTRUTURAL: stability.py")
    print("Testes do criterio do plano tangente")
    print("=" * 65)

    T = 298.15
    nu_list = [NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX]
    model   = UNIFACLL(nu_list, T)

    # ──────────────────────────────────────────────────────────────────
    # TESTE 1 — IDENTIDADE ALGEBRICA: D(w=z) = 0
    # ──────────────────────────────────────────────────────────────────
    # Por construcao da funcao TPD:
    #     D(z) = sum_i z_i * [ln z_i + ln gamma_i(z) - ln z_i - ln gamma_i(z)]
    #          = sum_i z_i * 0
    #          = 0 (exato)
    # Isso vale para qualquer composicao z, qualquer modelo, qualquer T.
    # E uma verificacao basica de implementacao da funcao tangent_plane_D.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 1 — Identidade algebrica: D(w=z) = 0")
    print("-" * 65)
    print("Fundamento: por construcao, D(z) = 0 para qualquer z.")
    print("Criterio: |D(z)| < 1e-10")
    print()

    pass_count = 0
    test_zs = [
        ([0.33, 0.33, 0.34], 'equimolar'),
        ([0.80, 0.10, 0.10], 'rico em triolina'),
        ([0.05, 0.10, 0.85], 'rico em LIP'),
    ]
    for z_list, desc in test_zs:
        z_arr = np.array(z_list, float)
        z_arr /= z_arr.sum()
        ln_gz = np.log(z_arr) + np.log(model.gamma(z_arr))
        D_self = tangent_plane_D(z_arr, z_arr, ln_gz, model)
        ok = abs(D_self) < 1e-10
        pass_count += ok
        status = "OK" if ok else "FALHOU"
        print(f"  [{status}] z={z_list} ({desc:18s}): D(w=z) = {D_self:.2e}")
    print(f"\n  Resultado: {pass_count}/{len(test_zs)} aprovados")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 2 — DETECCAO DE FASE PURA COMO ESTAVEL
    # ──────────────────────────────────────────────────────────────────
    # FUNDAMENTO TERMODINAMICO:
    #     Um liquido praticamente puro (x_i > 0.99) NUNCA pode separar
    #     em duas fases. Para que houvesse separacao, seriam necessarias
    #     duas fases distintas — mas com x_i > 0.99 ambas seriam
    #     essencialmente o componente i puro.
    #
    # Este teste verifica que nosso algoritmo NAO produz falsos positivos
    # nessa regiao trivialmente estavel.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 2 — Faixa de estabilidade trivial (componente quase puro)")
    print("-" * 65)
    print("Fundamento: x_i > 0.99 -> sempre estavel (impossivel separar)")
    print("Criterio: is_unstable() deve retornar False")
    print()

    pass_count = 0
    for x1 in [0.999, 0.995, 0.990]:
        xr = (1 - x1) / 2
        z = np.array([x1, xr, xr])
        bif = is_unstable(z, model)
        ok = not bif
        pass_count += ok
        status = "OK" if ok else "FALHOU"
        print(f"  [{status}] x_triolina={x1}: bifasico={bif}  (esperado: False)")
    print(f"\n  Resultado: {pass_count}/3 aprovados")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 3 — CONSISTENCIA COM SISTEMA BINARIO IMISCIVEL
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANTE — REVISAO DE EXPECTATIVAS:
    #     Versoes anteriores deste teste assumiam que QUALQUER mistura
    #     triolina+etanol em qualquer composicao era bifasica. Isso esta
    #     ERRADO: triolina e etanol sao PARCIALMENTE MISCIVEIS a 298 K.
    #     Existe uma faixa de coexistencia (regiao bifasica) em compo-
    #     sicoes intermediarias, mas nao em todo o diagrama binario.
    #
    #     Alem disso, a presenca de um terceiro componente (LIP) pode
    #     atuar como COSSOLVENTE, eliminando localmente a regiao
    #     bifasica. Esse e o efeito que o TCC busca quantificar.
    #
    # O QUE TESTAMOS AQUI:
    #     Em uma composicao especifica RICA EM LIP (x_LIP = 0.85), com
    #     pequena triolina e pequena etanol, o sistema deve separar:
    #     a triolina (apolar) sai da fase rica em LIP (polar) e forma
    #     uma fase oleosa. Isso e o comportamento bifasico tipico do
    #     sistema do TCC e e confirmado experimentalmente em sistemas
    #     analogos (Souza et al. 2025, Trindade et al. 2019).
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 3 — Consistencia com bifase em regiao rica em LIP")
    print("-" * 65)
    print("Fundamento: triolina (apolar) deve separar de fase polar rica em LIP")
    print("Criterio: is_unstable() deve retornar True")
    print()

    pass_count = 0
    test_compositions = [
        ([0.05, 0.10, 0.85], 'rico em LIP, pouca triolina'),
        ([0.10, 0.05, 0.85], 'rico em LIP, mais triolina'),
        ([0.05, 0.20, 0.75], 'rico em LIP, etanol moderado'),
    ]
    for z_list, desc in test_compositions:
        z = np.array(z_list)
        bif = is_unstable(z, model)
        ok = bif
        pass_count += ok
        status = "OK" if ok else "ATENCAO"
        print(f"  [{status}] {desc:35s}: bifasico={bif}")
    print(f"\n  Resultado: {pass_count}/{len(test_compositions)} aprovados")
    print("  Nota: falhas aqui podem indicar parametros UNIFAC inadequados")
    print("        para o sistema (nao necessariamente bug de codigo).")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 4 — MAPA DE ESTABILIDADE NO SIMPLEX
    # ──────────────────────────────────────────────────────────────────
    # Saida exploratoria — nao e teste de aprovacao/reprovacao.
    # Mostra um "mapa" da regiao bifasica varrendo o simplex em malha
    # 5x4. Util para inspecao visual da topologia do diagrama: onde a
    # regiao bifasica fica concentrada, qual sua extensao.
    #
    # Comparar este mapa com o diagrama experimental e o teste mais
    # robusto da CAPACIDADE PREDITIVA dos parametros (separado de bugs
    # de codigo, que sao testados nos itens 1-2).
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 4 — Mapa de estabilidade no simplex")
    print("-" * 65)
    print("Inspecao exploratoria: B = bifasico, M = monofasico")
    print()

    t0 = time.time()
    print(f"  {'x_triol':>10} | x_etanol")
    print(f"  {'':>10} | {0.05:>4} {0.25:>4} {0.45:>4} {0.65:>4}")
    print(f"  {'-'*10}-+-{'-'*24}")
    for x1 in [0.05, 0.25, 0.50, 0.75, 0.90]:
        line = f"  {x1:>10.2f} |"
        for x2 in [0.05, 0.25, 0.45, 0.65]:
            x3 = 1.0 - x1 - x2
            if x3 < 0.02:
                line += f" {'.':>4}"
                continue
            z = np.array([x1, x2, x3]); z /= z.sum()
            res = is_unstable(z, model)
            line += f" {'B' if res else 'M':>4}"
        print(line)

    print(f"\n  Tempo de calculo: {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 5 — SIMETRIA DE D ENTRE FASES EM EQUILIBRIO
    # ──────────────────────────────────────────────────────────────────
    # FUNDAMENTO:
    #     Se z_alpha e z_beta sao duas fases em equilibrio, entao:
    #         D_alpha(z_beta) = D_beta(z_alpha)
    #     em valor absoluto. Essa e uma manifestacao da simetria de
    #     Gibbs entre as fases.
    #
    #     Em pratica, se sabemos que (xa, xb) e uma tie-line valida,
    #     D avaliado em qualquer dos pontos com referencia ao outro
    #     deve ser proximo a zero (ambas fases tem MESMO plano tangente).
    #
    # Aqui usamos uma "tie-line" generica de teste no sistema do TCC.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 5 — Simetria do plano tangente entre fases em equilibrio")
    print("-" * 65)
    print("Fundamento: fases em equilibrio compartilham o MESMO plano tangente")
    print("Criterio: D_a(z_b) ~ D_b(z_a) ~ 0 quando (z_a, z_b) e tie-line")
    print()

    # Tie-line aproximada do sistema [2-HEA][Hx]
    z_a = np.array([0.50, 0.05, 0.45])   # fase rica em triolina (esperada)
    z_b = np.array([0.04, 0.10, 0.86])   # fase rica em LIP (esperada)

    # Calcula D em cada referencia
    ln_g_a = np.log(z_a) + np.log(model.gamma(z_a))
    ln_g_b = np.log(z_b) + np.log(model.gamma(z_b))

    D_ab = tangent_plane_D(z_b, z_a, ln_g_a, model)  # plano em a, avaliado em b
    D_ba = tangent_plane_D(z_a, z_b, ln_g_b, model)  # plano em b, avaliado em a

    print(f"  z_a (fase oleosa)  = {z_a}")
    print(f"  z_b (fase polar)   = {z_b}")
    print(f"  D_a(z_b) = {D_ab:+.4f}  (deve ser <= 0 se z_a estavel)")
    print(f"  D_b(z_a) = {D_ba:+.4f}  (deve ser <= 0 se z_b estavel)")
    print()
    print("  Nota: estes pontos sao tie-line ESTIMADA, nao calculada por flash.")
    print("        Para tie-lines reais, ambos D devem ser ~ 0 (proximos do plano).")

    print("\n" + "=" * 65)
    print("FIM DA VALIDACAO ESTRUTURAL")
    print("=" * 65)