"""
gamma.py — Coeficientes de atividade pelo modelo UNIFAC
==========================================================================
PROPÓSITO
    Calcular o coeficiente de atividade γ_i de cada componente em uma
    mistura líquida, a partir da decomposição molecular em subgrupos
    funcionais. γ_i quantifica o desvio da idealidade da solução:
        μ_i = μ_i°(T,P) + RT·ln(x_i·γ_i)
    onde μ_i é o potencial químico, x_i a fração molar, e γ_i = 1 para
    soluções ideais (mistura ideal de componentes do mesmo tamanho e
    afinidades químicas iguais).

FUNDAMENTO TEÓRICO
    O UNIFAC (UNIQUAC Functional-group Activity Coefficients) decompõe
    a energia de Gibbs em excesso (G^E) em duas contribuições:

        ln γ_i = ln γ_i^C  +  ln γ_i^R
                 ───────     ───────
                 (combin.)   (residual)

    • Combinatorial (γ_i^C): captura efeitos entrópicos de tamanho e forma
      molecular. Vem do termo de Staverman-Guggenheim do UNIQUAC, que
      modela uma rede quasi-cristalina onde cada molécula ocupa um número
      r_i de sítios e tem área superficial q_i.

    • Residual (γ_i^R): captura efeitos energéticos de interação entre
      grupos funcionais (van der Waals, ligação de H, dipolo-dipolo).
      Esta é a "alma" do UNIFAC — a hipótese central é que a energia de
      interação A-B em qualquer mistura depende apenas dos GRUPOS de A
      interagindo com os GRUPOS de B, não das moléculas inteiras.

    A equação central é a de Fredenslund, Jones & Prausnitz (1975)
    AIChE J. 21, 1086, que é a versão "original" usada com parâmetros
    UNIFAC-LL de Magnussen et al. (1981) e parâmetros reajustados de
    Bessa et al. (2016) para sistemas de biodiesel.

DEPENDÊNCIAS
    params.py  — fornece SUBGROUPS, INTERACTION_PARAMS, e as decomposições
                 NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX
"""

import numpy as np
from params import (SUBGROUPS, INTERACTION_PARAMS,
                    NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX)


# ===========================================================================
# ESTRUTURAS DE ÍNDICE — pré-calculadas a partir de params.py
# ===========================================================================
# Por que pré-calcular? Em cálculos vetorizados de γ, vamos acessar muitas
# vezes os parâmetros R_k, Q_k de cada subgrupo. Convertê-los para arrays
# NumPy uma única vez evita acessos repetidos a dicionários (lentos) e
# permite uso de operações vetoriais (rápidas).

# Lista ordenada dos IDs de subgrupos definidos em params.py.
# Ordenar é importante: garante que o mesmo subgrupo sempre tenha o mesmo
# índice em todas as chamadas, em qualquer máquina.
SG_IDS       = sorted(SUBGROUPS.keys())
NSG          = len(SG_IDS)                        # número de subgrupos
SG2IDX       = {sg: i for i, sg in enumerate(SG_IDS)}  # mapa: ID → índice

# Vetores (NSG,) com R_k e Q_k de cada subgrupo, na ordem de SG_IDS.
# R_k é o "volume" reduzido do subgrupo (relativo ao CH₂ do n-alcano),
# calculado a partir do volume de van der Waals via R_k = V_wk / 15.17.
# Q_k é a "área superficial" reduzida, Q_k = A_wk / (2.5 × 10⁹).
R_VEC        = np.array([SUBGROUPS[sg][1] for sg in SG_IDS])
Q_VEC        = np.array([SUBGROUPS[sg][2] for sg in SG_IDS])

# Cada subgrupo pertence a um GRUPO PRINCIPAL (MG). A interação Ψ_mn é
# definida no nível do grupo principal: subgrupos do mesmo MG (ex: CH₃ e
# CH₂, ambos MG=1) compartilham os mesmos parâmetros de interação com
# qualquer outro grupo. Isso reflete a hipótese do UNIFAC de que apenas
# a NATUREZA QUÍMICA do grupo (não sua geometria) determina interações.
_MG_OF_SG    = [SUBGROUPS[sg][3] for sg in SG_IDS]
MAIN_GROUPS  = sorted(set(_MG_OF_SG))             # MGs únicos
NMG          = len(MAIN_GROUPS)
MG2IDX       = {mg: i for i, mg in enumerate(MAIN_GROUPS)}

# Array (NSG,) que mapeia cada subgrupo ao índice do seu MG na matriz Ψ.
# Ex: se sg=1 (CH₃) e sg=2 (CH₂) ambos pertencem ao MG=1, ambos terão o
# mesmo valor neste array. Isso é o que permite "compartilhar" parâmetros
# entre subgrupos do mesmo MG.
MG_OF_SG_IDX = np.array([MG2IDX[mg] for mg in _MG_OF_SG])


# ===========================================================================
# FUNÇÕES AUXILIARES
# ===========================================================================

def nu_to_vec(nu_dict):
    """
    Converte um dicionário {subgroup_id: count} em um array (NSG,).

    Por que precisamos disso?
        params.py define os componentes como dicts esparsos:
            NU_ETANOL = {1: 1, 2: 1, 14: 1}   # 1×CH3, 1×CH2, 1×OH
        Mas nos cálculos precisamos de um vetor denso indexado pela mesma
        ordem de SG_IDS, para que multiplicações matriciais funcionem.

    Subgrupos não presentes na molécula têm contagem 0 (preenchidos por
    np.zeros).
    """
    v = np.zeros(NSG)
    for sg, n in nu_dict.items():
        if sg in SG2IDX:                # ignora subgrupos não definidos
            v[SG2IDX[sg]] = float(n)
    return v


def build_psi_matrix(T: float) -> np.ndarray:
    """
    Monta a matriz de interação Psi[NMG x NMG] na temperatura T (K).

    EQUACAO:
        Psi_mn = exp[ -(a_mn + b_mn*T + c_mn*T^2) / T ]

    INTERPRETACAO TERMODINAMICA:
        Psi_mn é uma "constante de equilíbrio" relativa entre interações
        m-m e m-n. Na formulação UNIQUAC original:
            Psi_mn = exp[ -(u_mn - u_nn) / RT ]
        onde u_mn é a energia de interação entre grupos m e n. Logo,
            a_mn ~ (u_mn - u_nn) / R   (em Kelvin)
        Note que a matriz é ASSIMETRICA: a_mn != a_nm em geral, pois
        u_mn - u_nn != u_mn - u_mm.

    PARAMETROS USADOS NESTE TCC:
        Para o UNIFAC-LL original (Magnussen 1981) e Bessa et al. (2016),
        apenas a_mn é ajustado: b_mn = c_mn = 0, simplificando para
        Psi_mn = exp(-a_mn/T). Os termos b e c só são ativados em
        UNIFAC-Dortmund (extensão para ampla faixa de temperatura).

    DIAGONAL:
        Psi_mm = exp(0) = 1 sempre, pois a interação de um grupo consigo
        mesmo é tomada como referência.
    """
    # Inicializa matriz (NMG, NMG, 3): camada 0 = a, camada 1 = b, camada 2 = c
    A = np.zeros((NMG, NMG, 3))

    # Preenche a partir do dict INTERACTION_PARAMS de params.py
    for (m, n), (a, b, c) in INTERACTION_PARAMS.items():
        # Filtro defensivo: só usa pares cujos MGs estão no sistema atual
        if m in MG2IDX and n in MG2IDX:
            A[MG2IDX[m], MG2IDX[n]] = [a, b, c]

    # Separa as três camadas em matrizes independentes
    a_mat = A[:, :, 0]
    b_mat = A[:, :, 1]
    c_mat = A[:, :, 2]

    # Cálculo vetorizado: Psi_mn = exp[-(a + bT + cT^2)/T]
    PSI   = np.exp(-(a_mat + b_mat * T + c_mat * T**2) / T)

    # Garante diagonal = 1 (poderia estar ligeiramente diferente por
    # erros de arredondamento se A_mm fosse não-zero)
    np.fill_diagonal(PSI, 1.0)
    return PSI


def _theta_sg_from_nu(nu_mat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calcula a fração de área superficial theta_k de cada SUBGRUPO na mistura.

    EQUACOES:
        Primeiro a fração molar do subgrupo k na mistura:
            X_k = sum_i x_i*nu_k^(i)  /  sum_i sum_m x_i*nu_m^(i)
        Depois a fração de área (normalizada pela área total):
            theta_k = Q_k*X_k  /  sum_m Q_m*X_m

    INTERPRETACAO:
        theta_k é a "probabilidade" de que um ponto aleatório na superfície
        de contato entre moléculas pertença ao subgrupo k. Em uma mistura
        equimolar de etanol e n-hexano, por exemplo, a fração de área
        do grupo OH é pequena (apenas 1 OH por molécula de etanol contra
        muitos CH2/CH3), o que reflete a probabilidade real de uma OH
        encontrar outro grupo na vizinhança.

    USO NO RESIDUAL:
        theta_k entra na equação de ln Gamma_k (coef. de atividade do GRUPO),
        que é o cerne do termo residual.

    PARAMETROS:
        nu_mat: array (N, NSG) — linha i tem nu_k^(i) para o componente i
        x:      array (N,)     — frações molares dos componentes
    """
    # Soma ponderada: total_nu[k] = sum_i x_i * nu_k^(i)
    # É o numerador de X_k (sem o denominador comum)
    total_nu = x @ nu_mat

    # Q-ponderada: total_Q = sum_k Q_k * total_nu[k]
    # É o denominador da theta_k (área total da mistura)
    total_Q  = Q_VEC @ total_nu

    # Proteção contra divisão por zero (caso patológico: x=0 em todos)
    if total_Q < 1e-15:
        return np.zeros(NSG)

    return Q_VEC * total_nu / total_Q


def _ln_Gamma_sg(theta_sg: np.ndarray, PSI: np.ndarray) -> np.ndarray:
    """
    Calcula ln Gamma_k para todos os subgrupos.

    Gamma_k é o "coeficiente de atividade do grupo k" — a chave do método
    UNIFAC. Mede o quanto o ambiente da mistura desvia do ambiente que
    o grupo k teria em uma solução de referência.

    EQUACAO (Fredenslund 1975, eq. 7):
        ln Gamma_k = Q_k [ 1 - ln(sum_m theta_m*Psi_mk) - sum_m theta_m*Psi_km/(sum_n theta_n*Psi_nm) ]
                          \________A_k________/  \__________B_k__________/

    INTERPRETACAO DE A_k E B_k:
        - A_k = sum_m theta_m*Psi_mk -> "afinidade média" da vizinhança pelo
          grupo k. Se Psi_mk > 1 para muitos m presentes, A_k é grande,
          indicando que k é "bem-vindo" -> ln Gamma_k diminui (afinidade alta
          equivale a gamma baixo).

        - B_k = sum_m theta_m*Psi_km/A_m -> termo de correção que conta a
          competição: o grupo k "enxerga" outros grupos m, mas cada m
          também é influenciado por seus vizinhos. É o que torna o
          UNIFAC consistente com a equação de Gibbs-Duhem.

    DETALHE DE IMPLEMENTACAO — POR QUE ESTA VERSAO:
        A primeira versão deste código tinha um bug sutil: agregava as
        Q_k em "Q_mg médio" por grupo principal antes de multiplicar.
        Isso atribuía o MESMO ln Gamma a todos os subgrupos do mesmo MG
        (ex: CH3 com Q=0.848 e CH2 com Q=0.540 recebiam o mesmo valor).
        Mas a equação correta tem Q_k INDIVIDUAL fora do bracket. O
        bracket [1 - ln A_k - B_k] depende apenas do MG (pois Psi é
        indexado por MG), mas Q_k é específico de cada subgrupo.

    DESEMPENHO:
        Implementação totalmente vetorizada — sem loops Python.
        A é calculado por th_mg @ PSI (uma única multiplicação matricial).
        B é calculado por PSI @ (th_mg * inv_A).
    """
    # Passo 1: agregar theta por grupo principal (NMG,)
    # th_mg[m] = soma de theta_k para todos os subgrupos k pertencentes ao MG m
    th_mg = np.zeros(NMG)
    np.add.at(th_mg, MG_OF_SG_IDX, theta_sg)

    # Passo 2: A[m] = sum_n theta_n*Psi_nm  (vetor de "afinidade vizinha" por MG)
    # Em forma vetorial: A = th_mg @ PSI dá A[k] = sum_m th_mg[m]*PSI[m,k]
    A = th_mg @ PSI

    # Passo 3: B[m] — versão vetorizada do somatório com competição
    # Tratamento numérico: zeros em A (MGs ausentes) viram 0 em inv_A,
    # evitando NaN. Não afeta o resultado pois theta_m=0 -> contribuição zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_A = np.where(A > 0, 1.0 / A, 0.0)
        lnA   = np.where(A > 0, np.log(A), 0.0)

    # B[k] = sum_m PSI[k,m] * theta_m / A_m  (multiplicação PSI x vetor)
    B = PSI @ (th_mg * inv_A)

    # Passo 4: bracket termodinâmico [1 - ln A - B], um valor por MG
    bracket = 1.0 - lnA - B

    # Passo 5: ln Gamma_k = Q_k * bracket[MG_de_k]  para CADA SUBGRUPO
    # bracket[MG_OF_SG_IDX] expande o vetor (NMG,) para (NSG,) replicando
    # o valor do MG para todos os subgrupos pertencentes a ele.
    return Q_VEC * bracket[MG_OF_SG_IDX]


# ===========================================================================
# CLASSE PRINCIPAL — UNIFACLL
# ===========================================================================

class UNIFACLL:
    """
    Calculador de gamma_i pelo modelo UNIFAC-LL (Magnussen 1981) com
    combinatorial Fredenslund (1975).

    PROJETO ORIENTADO A OBJETO:
        Cada instância representa um SISTEMA específico (lista de
        componentes a uma temperatura T). Pré-computa tudo que não muda
        com x: matrizes Psi, parâmetros r_i, q_i, e ln Gamma_k^(i) para
        cada componente puro. Em chamadas a .gamma(x), só os termos que
        dependem de x são calculados.

    ECONOMIA COMPUTACIONAL:
        Em um cálculo de tie-line, .gamma() pode ser chamado milhares
        de vezes (otimização do flash, análise de estabilidade). Pré-
        computar reduz cada chamada a operações lineares em N e NSG.
    """

    def __init__(self, nu_list: list, T: float):
        """
        Constrói o modelo para uma mistura específica.

        PARAMETROS:
            nu_list : lista de N dicts {subgroup_id: count}, um por
                      componente. Ex: [NU_TRIOLINA, NU_ETANOL, NU_LIP]
            T       : temperatura em Kelvin (assumida constante na vida
                      desta instância)
        """
        self.T   = T
        self.N   = len(nu_list)

        # Matriz Psi — só depende de T, calculada uma vez
        self.PSI = build_psi_matrix(T)

        # Matriz NU[i, j] = nu_j^(i) — quantos subgrupos j na molécula i
        # Forma (N, NSG). Densamente preenchida (zero onde subgrupo ausente).
        self.NU = np.array([nu_to_vec(nu) for nu in nu_list])

        # Parâmetros moleculares r_i, q_i:
        #     r_i = sum_k nu_k^(i) * R_k    (volume da molécula i)
        #     q_i = sum_k nu_k^(i) * Q_k    (área da molécula i)
        # Ex: triolina tem r ~ 39 (molécula grande), etanol r ~ 2.6
        self.r = self.NU @ R_VEC
        self.q = self.NU @ Q_VEC

        # ln Gamma_k^(i) — REFERENCIA RESIDUAL.
        # Para cada componente i, calcula ln Gamma_k que cada subgrupo k
        # teria se a mistura fosse APENAS o componente i puro. Isso é
        # subtraído depois de ln Gamma_k da mistura, para garantir
        # gamma_i -> 1 quando x_i -> 1 (convenção de Lewis-Randall).
        # Pré-calculado pois não depende de x.
        self._lnG_pure = np.zeros((self.N, NSG))
        for i in range(self.N):
            total = Q_VEC @ self.NU[i]
            if total > 0:
                # theta no componente puro: Q_k*nu_k^(i) / sum_m Q_m*nu_m^(i)
                th_pure = Q_VEC * self.NU[i] / total
                self._lnG_pure[i] = _ln_Gamma_sg(th_pure, self.PSI)

    def gamma(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula gamma_i para todos os componentes na composição x.

        FLUXO:
            1. Normaliza x (evita inconsistências de input)
            2. Calcula ln gamma_i^C  — termo combinatorial
            3. Calcula ln gamma_i^R  — termo residual
            4. Retorna exp(lnC + lnR)

        SAIDA:
            array (N,) com gamma_i para cada componente
        """
        # Saneamento: clip evita zeros (logaritmos), normaliza para sum(x)=1
        x = np.clip(np.asarray(x, float), 1e-15, 1.0)
        x = x / x.sum()

        # ──────────────────────────────────────────────────────────────
        # TERMO COMBINATORIAL — Staverman-Guggenheim (UNIQUAC clássico)
        # ──────────────────────────────────────────────────────────────
        # Equação (Fredenslund, Jones & Prausnitz 1975, eq. 2):
        #     ln gamma_i^C = ln(phi_i/x_i) + 1 - phi_i/x_i
        #                  - 0.5*z*q_i*[ln(phi_i/theta_i) + 1 - phi_i/theta_i]
        # onde:
        #     phi_i = r_i*x_i / sum_j r_j*x_j   (fração de volume)
        #     theta_i = q_i*x_i / sum_j q_j*x_j (fração de área)
        #     z = 10                            (número de coordenação)
        #
        # INTERPRETACAO:
        #     Captura a CONTRIBUICAO ENTROPICA da mistura. Mede o
        #     "custo" de organizar moléculas de tamanhos e formas
        #     diferentes em uma rede. Para componentes idênticos
        #     (phi_i = theta_i = x_i), ln gamma_i^C = 0.
        #
        # NOTA HISTORICA — POR QUE NAO O r^{3/4} DO DORTMUND:
        #     UNIFAC-Dortmund usa V_i' = r_i^{0.75}*x_i / sum_j r_j^{0.75}*x_j
        #     no lugar de V_i. Esse expoente "amortece" diferenças de
        #     tamanho extremas (típicas em sistemas com triglicerídeos).
        #     POREM, o expoente 3/4 só vale com os PARAMETROS DORTMUND
        #     (a_mn, b_mn, c_mn) — esses foram regredidos assumindo o
        #     combinatorial modificado. Misturar combinatorial Dortmund
        #     com parâmetros Magnussen ou Bessa é incoerente: gera
        #     erros sistemáticos. Como usamos parâmetros Bessa et al.
        #     (2016), mantemos o combinatorial original.
        # ──────────────────────────────────────────────────────────────

        z = 10.0  # número de coordenação clássico do UNIQUAC

        # Notação: phi_x[i] = phi_i/x_i = r_i / sum_j r_j*x_j
        # (Trabalhamos com a razão phi_i/x_i diretamente, evitando dividir
        # por x_i depois — útil quando x_i é pequeno.)
        phi_x   = self.r / (x @ self.r)
        theta_x = self.q / (x @ self.q)

        # Logaritmos com proteção contra valores não-positivos
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_phi_x     = np.where(phi_x > 0, np.log(phi_x), 0.0)
            ln_phi_theta = np.where(
                (phi_x > 0) & (theta_x > 0),
                np.log(phi_x / theta_x), 0.0
            )

        # Equação completa do combinatorial
        lnC = (ln_phi_x + 1.0 - phi_x
               - 0.5 * z * self.q * (ln_phi_theta + 1.0 - phi_x / theta_x))

        # ──────────────────────────────────────────────────────────────
        # TERMO RESIDUAL — solução de grupos
        # ──────────────────────────────────────────────────────────────
        # Equação (Fredenslund 1975, eq. 6):
        #     ln gamma_i^R = sum_k nu_k^(i) * [ln Gamma_k(mix) - ln Gamma_k^(i)(puro)]
        #
        # INTERPRETACAO:
        #     Para cada grupo k presente em i:
        #     - nu_k^(i) é quantas vezes o grupo aparece em i
        #     - Gamma_k(mix) é como o grupo se sente NA MISTURA
        #     - Gamma_k^(i)(puro) é como ele se sentiria se a mistura
        #       fosse só componente i. Subtrair garante gamma_i -> 1
        #       em puro i.
        #
        # PASSOS:
        #     (a) calcula theta_k na mistura (depende de x via NU)
        #     (b) calcula ln Gamma_k(mix) — chama _ln_Gamma_sg
        #     (c) subtrai ln Gamma_k^(i)(puro) — pré-calculado em __init__
        #     (d) soma ponderada por nu_k^(i)
        # ──────────────────────────────────────────────────────────────

        th_mix  = _theta_sg_from_nu(self.NU, x)        # (NSG,)
        lnG_mix = _ln_Gamma_sg(th_mix, self.PSI)        # (NSG,)

        # diff[i,k] = ln Gamma_k(mix) - ln Gamma_k^(i)(puro)
        # Broadcasting: (1, NSG) - (N, NSG) -> (N, NSG)
        diff = lnG_mix[np.newaxis, :] - self._lnG_pure

        # Soma por componente: ln gamma_i^R = sum_k nu_k^(i) * diff[i,k]
        lnR = np.sum(self.NU * diff, axis=1)

        # ──────────────────────────────────────────────────────────────
        # COMBINACAO FINAL: gamma_i = exp(ln gamma_i^C + ln gamma_i^R)
        # ──────────────────────────────────────────────────────────────
        return np.exp(lnC + lnR)


# ===========================================================================
# VALIDACAO INDEPENDENTE
# ===========================================================================
# Estes testes verificam propriedades termodinâmicas FUNDAMENTAIS que QUALQUER
# implementação correta de modelo de coeficiente de atividade deve satisfazer.
# Eles são independentes dos parâmetros usados — são teoremas matemáticos,
# não comparações com dados experimentais.
#
# Se algum desses testes FALHAR, há bug no código. Se PASSAREM, o código
# está correto, mas isso não garante que os parâmetros sejam apropriados
# para o sistema (essa é uma questão separada, abordada em validacao_*.py).
# ===========================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("VALIDACAO ESTRUTURAL: gamma.py")
    print("Testes de propriedades termodinamicas fundamentais")
    print("=" * 65)

    T = 298.15
    nu_list = [NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX]
    model   = UNIFACLL(nu_list, T)
    names   = ['Triolina', 'Etanol', '[2-HEA][Hx]']

    # ──────────────────────────────────────────────────────────────────
    # TESTE 1 — NORMALIZACAO DE LEWIS-RANDALL: gamma_i -> 1 quando x_i -> 1
    # ──────────────────────────────────────────────────────────────────
    # Por definição da convenção de Lewis-Randall, o estado de referência
    # de cada componente é o LIQUIDO PURO. Logo, em x_i = 1, mu_i = mu_i°,
    # o que implica gamma_i = 1 exatamente.
    #
    # Em código, testamos com x_i ~= 0.997 (não exatamente 1, para evitar
    # divisões por zero em outras frações). Esperamos gamma_i muito próximo
    # de 1, com tolerância pequena.
    #
    # Se este teste falhar, o termo ln Gamma_k^(i)(puro) está mal calculado.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 1 — Normalizacao Lewis-Randall: gamma_i(x_i->1) -> 1")
    print("-" * 65)
    print("Criterio: |gamma_i - 1| < 0.01 quando x_i = 0.997")
    print()

    tol_pure = 0.01
    pass_count = 0
    for i, name in enumerate(names):
        x_pure    = np.ones(3) * 0.001
        x_pure[i] = 0.997
        x_pure   /= x_pure.sum()
        g         = model.gamma(x_pure)
        ok = abs(g[i] - 1.0) < tol_pure
        pass_count += ok
        status = "OK" if ok else "FALHOU"
        print(f"  [{status}] {name:14s}  x_i=0.997  gamma_i = {g[i]:.6f}  "
              f"|desvio| = {abs(g[i]-1.0):.2e}")
    print(f"\n  Resultado: {pass_count}/{len(names)} aprovados")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 2 — CONSISTENCIA DA EQUACAO DE GIBBS-DUHEM
    # ──────────────────────────────────────────────────────────────────
    # A equação de Gibbs-Duhem em condições isotérmicas e isobáricas:
    #     sum_i x_i * d(ln gamma_i)/dx_j = 0   para todo j (com sum x_i = 1)
    # Para uma mistura binária (x_2 = 1 - x_1):
    #     x_1 * d(ln gamma_1)/dx_1 + x_2 * d(ln gamma_2)/dx_1 = 0
    #
    # Verificamos numericamente por diferença finita central.
    # Esperamos resíduo < 1e-3 (limitado por precisão da derivada
    # numérica, não por erro do modelo).
    #
    # Se este teste falhar, há inconsistência interna no cálculo de gamma
    # (pode ser sinal trocado, fator errado em algum termo).
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 2 — Equacao de Gibbs-Duhem (consistencia interna)")
    print("-" * 65)
    print("Criterio: |sum x_i * d(ln gamma_i)/dx_1| < 1e-3 em mistura binaria")
    print("Sistema teste: triolina + etanol (x_3 = 1e-5, fixo)")
    print()

    h = 1e-5  # passo da derivada numérica
    print(f"  {'x_1':>6}  {'x_1*dlng_1/dx_1':>16}  {'x_2*dlng_2/dx_1':>16}  {'Soma':>10}")
    max_residue = 0.0
    for x1 in [0.2, 0.4, 0.6, 0.8]:
        # Composições com perturbação central em x_1
        x_mid = np.array([x1,        1 - x1 - 1e-5,    1e-5]); x_mid /= x_mid.sum()
        x_p   = np.array([x1 + h,    1 - x1 - h - 1e-5, 1e-5]); x_p   /= x_p.sum()
        x_m   = np.array([x1 - h,    1 - x1 + h - 1e-5, 1e-5]); x_m   /= x_m.sum()

        ln_g_p = np.log(model.gamma(x_p))
        ln_g_m = np.log(model.gamma(x_m))

        # Derivadas centrais: d(ln gamma)/dx_1
        dln_g_dx1 = (ln_g_p - ln_g_m) / (2 * h)

        # Equação de Gibbs-Duhem (componentes 1 e 2 dominam, 3 é traço)
        gd_1 = x_mid[0] * dln_g_dx1[0]
        gd_2 = x_mid[1] * dln_g_dx1[1]
        soma = gd_1 + gd_2
        max_residue = max(max_residue, abs(soma))
        print(f"  {x1:>6.2f}  {gd_1:>16.4e}  {gd_2:>16.4e}  {soma:>10.2e}")

    ok_gd = max_residue < 1e-3
    print(f"\n  Residuo maximo: {max_residue:.2e}  ->  "
          f"{'[OK] Aprovado' if ok_gd else '[FALHOU] Reprovado'}")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 3 — COMPORTAMENTO LIMITE: ln gamma_i -> 0 quando x_i -> 1
    # ──────────────────────────────────────────────────────────────────
    # Em x_i -> 1, ln gamma_i deve -> 0 (consequência de gamma_i -> 1).
    # NAO confundir com monotonicidade: ln gamma_i(x_i) NAO precisa ser
    # monotônico. Em sistemas com não-idealidade complexa, ln gamma_i pode
    # ter mínimos e máximos internos, atingir valores negativos
    # (atração mais forte que ideal), e depois subir até 0.
    #
    # O que verificamos aqui é apenas a consistência do limite.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 3 — Comportamento limite: ln gamma_i(x_i=0.99) ~= 0")
    print("-" * 65)
    print("Criterio: |ln gamma_i| < 0.01 quando x_i = 0.99")
    print()
    pass_count = 0
    for i, name in enumerate(names):
        x_lim = np.ones(3) * 0.005
        x_lim[i] = 0.99
        x_lim /= x_lim.sum()
        ln_g = np.log(model.gamma(x_lim)[i])
        ok = abs(ln_g) < 0.01
        pass_count += ok
        status = "OK" if ok else "FALHOU"
        print(f"  [{status}] {name:14s}  ln gamma_i = {ln_g:+.6f}")
    print(f"\n  Resultado: {pass_count}/{len(names)} aprovados")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 4 — CONSISTENCIA DE ATIVIDADE: a_i = x_i*gamma_i é monotonica
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANTE: NAO testamos monotonicidade de gamma_i (que pode oscilar),
    # mas sim de a_i = x_i*gamma_i, a ATIVIDADE.
    #
    # A atividade DEVE ser MONOTONICAMENTE CRESCENTE em x_i, em qualquer
    # mistura termodinamicamente estável (sem separação de fases na
    # direção testada). Isso é uma consequência direta de:
    #     d(mu_i)/dx_i >= 0   (estabilidade material)
    #     mu_i = mu_i° + RT*ln(a_i)
    #
    # Se a atividade NAO for monotônica em algum trecho, isso indica
    # que o sistema é INSTAVEL nessa região, o que pode ser fisicamente
    # correto (separação de fases). Em uma "tomada vertical" pelo
    # diagrama (movendo x_1 e mantendo x_3 fixo), a região de spinodal
    # corresponde a d(a_1)/d(x_1) < 0.
    #
    # Aqui testamos uma faixa rica em LIP (x_3 = 0.7), onde esperamos
    # estabilidade — então a atividade DEVE crescer monotonicamente.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 4 — Monotonicidade de a_i (atividade)")
    print("-" * 65)
    print("Criterio: a_i = x_i*gamma_i deve crescer monotonicamente com x_i")
    print("Sistema teste: variando x_triolina em x_LIP = 0.70 (estavel)")
    print()
    print(f"  {'x_triol':>8}  {'gamma_t':>10}  {'a_triol':>10}  {'Delta_a':>10}")

    a_prev = -np.inf
    monotone = True
    for x1 in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]:
        x = np.array([x1, 1 - x1 - 0.70, 0.70])
        x = np.clip(x, 1e-5, 1); x /= x.sum()
        g = model.gamma(x)
        a = x[0] * g[0]
        delta = a - a_prev
        if delta < 0 and a_prev > -np.inf:
            monotone = False
        print(f"  {x[0]:>8.4f}  {g[0]:>10.4f}  {a:>10.6f}  "
              f"{delta:>+10.2e}")
        a_prev = a

    print(f"\n  Atividade monotonica em regiao estavel: "
          f"{'[OK] Aprovado' if monotone else '[ATENCAO] nao-monotonica'}")
    print("  (Nao-monotonicidade aqui indicaria spinodal — improvavel em x_LIP=0.70)")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 5 — COMPORTAMENTO QUALITATIVO ESPERADO
    # ──────────────────────────────────────────────────────────────────
    # Para um sistema {triolina + etanol + LIP} a 298 K, esperamos:
    #
    #   gamma_triolina em ambiente predominantemente etanol > 1
    #     Fundamento: triolina (apolar, alto r) e etanol (polar, OH)
    #     têm afinidades diferentes. Triolina diluida em etanol "sente"
    #     um ambiente desfavorável -> gamma > 1.
    #
    #   gamma_etanol em si mesmo (x_etanol -> 1) ~= 1
    #     Já testado no teste 1.
    #
    #   gamma_etanol em ambiente predominantemente triolina > 1
    #     Por simetria do argumento acima.
    #
    # NAO esperamos gamma >> 100 (imiscibilidade total) — a triolina e o
    # etanol são PARCIALMENTE MISCIVEIS a 298 K (formam duas fases mas
    # com solubilidade mútua não-desprezível). gamma_inf típico de etanol
    # em triglicerideo é da ordem de 10-30, não de centenas.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 5 — Comportamento qualitativo esperado")
    print("-" * 65)

    x_etoh = np.array([0.01, 0.98, 0.01])
    g = model.gamma(x_etoh)
    ok_a = g[0] > 1.0
    print(f"  Triolina diluida em etanol (x = {x_etoh}):")
    print(f"    gamma_triol  = {g[0]:.3f}  -> "
          f"{'[OK]' if ok_a else '[FALHOU]'}  (esperado > 1)")
    print(f"    gamma_etanol = {g[1]:.3f}  (deve ser ~1, esta em si mesmo)")
    print(f"    gamma_LIP    = {g[2]:.3f}")

    x_oil = np.array([0.98, 0.01, 0.01])
    g = model.gamma(x_oil)
    ok_b = g[1] > 1.0
    print(f"\n  Etanol diluido em triolina (x = {x_oil}):")
    print(f"    gamma_triol  = {g[0]:.3f}  (deve ser ~1, esta em si mesmo)")
    print(f"    gamma_etanol = {g[1]:.3f}  -> "
          f"{'[OK]' if ok_b else '[FALHOU]'}  (esperado > 1)")
    print(f"    gamma_LIP    = {g[2]:.3f}")

    # ──────────────────────────────────────────────────────────────────
    # TESTE 6 — TABELA DE REFERENCIA EM COMPOSICOES-CHAVE
    # ──────────────────────────────────────────────────────────────────
    # Não é teste de aprovação/reprovação — é uma tabela de saída para
    # documentação e comparação visual. Útil para detectar regressões
    # quando os parâmetros forem alterados no futuro.
    # ──────────────────────────────────────────────────────────────────
    print("\nTESTE 6 — Coeficientes em composicoes de referencia (T=298.15 K)")
    print("-" * 65)
    print(f"  {'Composicao':<22}  {'gamma_t':>8}  {'gamma_e':>8}  {'gamma_L':>8}")
    test_comps = {
        'Equimolar':       [1/3, 1/3, 1/3],
        'Rico em oleo':    [0.80, 0.10, 0.10],
        'Rico em LIP':     [0.05, 0.20, 0.75],
        'Rico em etanol':  [0.05, 0.80, 0.15],
    }
    for desc, z in test_comps.items():
        g = model.gamma(z)
        print(f"  {desc:<22}  {g[0]:>8.3f}  {g[1]:>8.3f}  {g[2]:>8.3f}")

    print("\n" + "=" * 65)
    print("FIM DA VALIDACAO ESTRUTURAL")
    print("=" * 65)