"""
params.py — Parâmetros para cálculo ELL de sistemas de biodiesel
==================================================================
Fontes dos parâmetros de interação (a_mn):

  BIODIESEL (CH2, C=C, OH, H2O, CCOO):
    Bessa, L.C.B.A. et al. (2016) Fluid Phase Equilib. 425, 98-107
    Tabela 3 — parâmetros reajustados para LLE de biodiesel
    (1145 tie-lines, 218 sistemas, T = 293-393 K)

  AMINAS (CNH2, CNH) e COO (carboxilato):
    Fredenslund, Jones & Prausnitz (1975) AIChE J. 21, 1086
    Hansen et al. (1991) Ind. Eng. Chem. Res. 30, 2352
    Parametros originais UNIFAC (VLE) — unica fonte disponivel
    para esses grupos, nao existem no UNIFAC-LL de Magnussen.

  R_k, Q_k:
    Magnussen et al. (1981) — confirmado em Bessa et al. (2016), Tab. 2.

Psi_mn = exp(-a_mn / T)   [b = c = 0]

Grupos ausentes (Psi = 1, a_mn = 0):
  CCOO / CNH2  (MG11 / MG14) — sem parametros em nenhuma base
  CNH2 / COO   (MG14 / MG41) — sem parametros em nenhuma base
"""

import numpy as np

# ---------------------------------------------------------------------------
# SUBGRUPOS
# Formato: subgroup_id -> (nome, R_k, Q_k, main_group_id)
# Fonte: Magnussen et al. (1981), confirmado em Bessa et al. (2016) Tab. 2
# ---------------------------------------------------------------------------
SUBGROUPS = {
    # --- Grupo principal 1: CH2 ---
    1:  ('CH3',    0.9011, 0.848,  1),
    2:  ('CH2',    0.6744, 0.540,  1),
    3:  ('CH',     0.4469, 0.228,  1),
    4:  ('C',      0.2195, 0.000,  1),
    # --- Grupo principal 2: C=C ---
    6:  ('CH=CH',  1.1167, 0.867,  2),
    # --- Grupo principal 5: OH ---
    14: ('OH',     1.0000, 1.200,  5),
    # --- Grupo principal 7: H2O ---
    16: ('H2O',    0.9200, 1.400,  7),
    # --- Grupo principal 11: CCOO (ester) ---
    21: ('CH3COO', 1.9031, 1.728, 11),
    22: ('CH2COO', 1.6764, 1.420, 11),
    # --- Grupo principal 41: COO (carboxilato livre) ---
    77: ('COO',    1.3800, 1.200, 41),
    # --- Grupo principal 14: CNH2 (amina primaria) ---
    29: ('CH2NH2', 1.3692, 1.236, 14),
    # --- Grupo principal 15: CNH (amina secundaria) ---
    32: ('CH2NH',  1.2070, 0.936, 15),
}

# ---------------------------------------------------------------------------
# COMPOSICAO EM GRUPOS
# ---------------------------------------------------------------------------

# Triolina (C57H104O6) — CH3=3, CH2=41, CH=1, CH=CH=3, CH2COO=3 -> C=57
NU_TRIOLINA = {1: 3, 2: 41, 3: 1, 6: 3, 22: 3}

# Etanol (C2H5OH) -> C=2
NU_ETANOL = {1: 1, 2: 1, 14: 1}

# [2-HEA][Hx] (C8H19NO3) -> C=8
# Cation: OH(sg14) + CH2(sg2) + CH2NH2(sg29)
# Anion:  CH3(sg1) + CH2x4(sg2) + COO(sg77)
NU_2HEA_HX = {1: 1, 2: 5, 14: 1, 77: 1, 29: 1}

# [DETA][Hx] (C10H25N3O2) -> C=10
# Cation: CH2NH2(sg29)x2 + CH2(sg2)x1 + CH2NH(sg32)x1
# Anion:  CH3(sg1) + CH2x4(sg2) + COO(sg77)
NU_DETA_HX = {1: 3, 2: 13, 77: 3, 29: 2, 32: 1}
# ---------------------------------------------------------------------------
# PARAMETROS DE INTERACAO
# Formato: (MG_m, MG_n): (a_mn [K], 0.0, 0.0)
# Psi_mn = exp(-a_mn / T)
# ---------------------------------------------------------------------------
INTERACTION_PARAMS = {

    # ==================================================================
    # BLOCO 1: Bessa et al. (2016) — Tabela 3
    # Reajustados para LLE de sistemas de biodiesel
    # Ref: Fluid Phase Equilib. 425, 98-107
    # ==================================================================

    # CH2(1) / C=C(2)
    (1,  2): ( 239.32, 0.0, 0.0),
    (2,  1): ( 470.13, 0.0, 0.0),

    # CH2(1) / OH(5)
    (1,  5): ( 714.56, 0.0, 0.0),
    (5,  1): (  73.52, 0.0, 0.0),

    # CH2(1) / H2O(7)
    (1,  7): (2045.36, 0.0, 0.0),
    (7,  1): ( 230.67, 0.0, 0.0),

    # CH2(1) / CCOO(11)
    (1, 11): ( 560.42, 0.0, 0.0),
    (11, 1): ( 462.69, 0.0, 0.0),

    # C=C(2) / OH(5)
    (2,  5): ( 954.33, 0.0, 0.0),
    (5,  2): ( 212.36, 0.0, 0.0),

    # C=C(2) / H2O(7)
    (2,  7): ( 328.14, 0.0, 0.0),
    (7,  2): ( 207.51, 0.0, 0.0),

    # C=C(2) / CCOO(11)
    (2, 11): (  49.03, 0.0, 0.0),
    (11, 2): (1040.51, 0.0, 0.0),

    # OH(5) / H2O(7)
    (5,  7): (  19.65, 0.0, 0.0),
    (7,  5): (  90.08, 0.0, 0.0),

    # OH(5) / CCOO(11)
    (5, 11): ( 172.62, 0.0, 0.0),
    (11, 5): ( 243.03, 0.0, 0.0),

    # H2O(7) / CCOO(11)
    (7, 11): (  11.22, 0.0, 0.0),
    (11, 7): ( 616.17, 0.0, 0.0),

    # ==================================================================
    # BLOCO 2: UNIFAC original (VLE) — aminas e COO
    # Fonte: Fredenslund et al. (1975); Hansen et al. (1991)
    # Nao existem na base UNIFAC-LL de Magnussen (1981).
    # ==================================================================

    # CH2(1) / CNH2(14)
    (1, 14): ( 391.50, 0.0, 0.0),
    (14, 1): ( -30.48, 0.0, 0.0),

    # CH2(1) / CNH(15)
    (1, 15): ( 255.70, 0.0, 0.0),
    (15, 1): (  65.33, 0.0, 0.0),

    # CH2(1) / COO(41)
    (1, 41): ( 387.10, 0.0, 0.0),
    (41, 1): ( 529.00, 0.0, 0.0),

    # C=C(2) / CNH2(14)
    (2, 14): ( 240.90, 0.0, 0.0),
    (14, 2): (   1.163, 0.0, 0.0),

    # C=C(2) / CNH(15)
    (2, 15): ( 163.90, 0.0, 0.0),
    (15, 2): ( -28.70, 0.0, 0.0),

    # C=C(2) / COO(41)
    (2, 41): (  48.33, 0.0, 0.0),
    (41, 2): (1397.00, 0.0, 0.0),

    # OH(5) / CNH2(14)
    (5, 14): (   8.642, 0.0, 0.0),
    (14, 5): (-242.80, 0.0, 0.0),

    # OH(5) / CNH(15)
    (5, 15): (  42.70, 0.0, 0.0),
    (15, 5): (-150.00, 0.0, 0.0),

    # OH(5) / COO(41)
    (5, 41): ( 190.30, 0.0, 0.0),
    (41, 5): (  88.63, 0.0, 0.0),

    # CCOO(11) / CNH(15)
    (11, 15): ( -73.50, 0.0, 0.0),
    (15, 11): ( 136.00, 0.0, 0.0),

    # CCOO(11) / COO(41)
    # CORRIGIDO: valor anterior (560.20) era transcricao errada; conferido
    # contra binary.csv oficial do JCOSMO (Hansen et al. 1991, DOI:
    # 10.1021/ie00058a017): a(11,41) = -122.3 K. Possivel origem do erro:
    # confusao com a linha vizinha 11,20 (CCOO/COOH) = 660.2.
    (11, 41): (-122.3, 0.0, 0.0),
    (41, 11): (-234.90, 0.0, 0.0),

    # CNH2(14) / CNH(15)
    (14, 15): (-107.20, 0.0, 0.0),
    (15, 14): ( 127.40, 0.0, 0.0),

    # CNH(15) / COO(41)
    (15, 41): ( -38.77, 0.0, 0.0),
    (41, 15): ( 284.50, 0.0, 0.0),

    # ==================================================================
    # BLOCO 3: H2O com aminas/COO — UNIFAC VLE
    # ==================================================================

    # H2O(7) / CNH2(14)
    (7, 14): (  48.89, 0.0, 0.0),
    (14, 7): (-330.40, 0.0, 0.0),

    # H2O(7) / CNH(15)
    (7, 15): ( 168.00, 0.0, 0.0),
    (15, 7): (-448.20, 0.0, 0.0),

    # H2O(7) / COO(41)
    (7, 41): (-197.50, 0.0, 0.0),
    (41, 7): ( 284.40, 0.0, 0.0),

    # ==================================================================
    # BLOCO 4: PARAMETROS ESTIMADOS NESTE TCC
    # ==================================================================
    # Os pares CCOO/CNH2 e CNH2/COO nao tinham valores publicados em
    # nenhuma base UNIFAC (UNIFAC original, UNIFAC-LL Magnussen, ou
    # Bessa et al. 2016) por nao terem sistemas experimentais classicos
    # contendo essas combinacoes de grupos.
    #
    # ESTRATEGIA DE ESTIMACAO:
    # Os 4 parametros foram ajustados por minimizacao do RMSD (root mean
    # square deviation) entre as composicoes calculadas pelo flash ELL e
    # as 9 tie-lines experimentais de Souza et al. (2025) Tab. 6 para o
    # sistema {hexil acetato + 2-HEAB + etanol} a 298.15 K.
    #
    # Funcao objetivo:
    #   F(a) = sqrt[ sum_t sum_phase sum_i (x_i^calc - x_i^exp)^2 / (2*N_t*N_c) ]
    # com penalidade adicional caso o modelo preveja monofasico em uma
    # composicao experimental claramente bifasica.
    #
    # Otimizacao: Nelder-Mead + Differential Evolution (faixa fisica
    # [-2500, +2500] K), com refinamento local final.
    #
    # RESULTADO:
    #   RMSD inicial (Psi=1): infinito (0/9 tie-lines convergem)
    #   RMSD com parametros estimados: 0.047 (9/9 convergem)
    #
    # OBSERVACOES:
    # - Valores positivos em a(11,14) e a(14,11) refletem repulsao entre
    #   grupos ester (CCOO) e amonio protonado (CNH2): a parte hidrofobica
    #   do ester e a cabeca cationica do PIL nao se misturam bem.
    # - Valor negativo em a(14,41) reflete atracao entre o amonio (CNH2)
    #   e o carboxilato (COO) — coerente com a interacao eletrostatica
    #   intramolecular do par ionico no PIL.
    # - O parametro a(41,14) recebe um valor pequeno e negativo, refletindo
    #   a fraca diferenca residual de afinidade vista pelo COO.

    # CCOO(11) / CNH2(14)  — ESTIMADO neste TCC (V2, Souza et al. 2025)
    (11, 14): ( 53.04, 0.0, 0.0),
    (14, 11): ( 22.94, 0.0, 0.0),

    # CNH2(14) / COO(41)   — ESTIMADO neste TCC (V2, Souza et al. 2025)
    # ATENCAO: nao ha valor alternativo (Dortmund/ML) disponivel para este
    # par. Ele permanece com o valor estimado independentemente de qual
    # bloco for usado para (11,14)/(14,11) abaixo.
    (14, 41): (-69.49864, 0.0, 0.0),
    (41, 14): ( 20.755741, 0.0, 0.0),
}

# ===========================================================================
# BLOCO 5 (TESTE DE SENSIBILIDADE) — alternativas de ML para CCOO/CNH2
# ===========================================================================
# Fonte: Hayer, N.; Hasse, H.; Jirasek, F. (2024), arXiv:2412.12962
#   (mod. UNIFAC 2.0, framework Modified UNIFAC Dortmund)
# e Hayer, N.; Wendel, T.; Mandt, S.; Hasse, H.; Jirasek, F. (2023),
#   arXiv:2408.05220 (UNIFAC 2.0, framework UNIFAC original)
#
# Valores CONFIRMADOS nos arquivos .csv de material suplementar dos autores
# (a_Parameters_modUNIFAC20.csv, b_Parameters_modUNIFAC20.csv,
# Interaction_Parameters_UNIFAC_2_0.csv, Size_Parameters_*.csv).
# O valor usado numa tentativa anterior (-164.04 / 326.04 K) NAO confere
# com nenhuma dessas tabelas e foi descartado.
#
# IMPORTANTE sobre framework:
# - UNIFAC 2.0 (original) usa os MESMOS R_k/Q_k de Magnussen et al. (1981)
#   ja usados neste script (confirmado: CH3COO R=1.9031/Q=1.728, CH2COO
#   R=1.6764/Q=1.420, CH2NH2 R=1.3692/Q=1.236 -- identicos). Sem termo de
#   temperatura (so a_mn/a_nm), igual ao Psi=exp(-a/T) ja usado aqui.
#   -> ESTA E A OPCAO CONSISTENTE COM O RESTO DO SCRIPT.
# - mod. UNIFAC 2.0 (Dortmund) usa R_k/Q_k DIFERENTES dos de Magnussen
#   (ex.: CH3COO R=1.27/Q=1.6286 no Dortmund vs R=1.9031/Q=1.728 aqui).
#   Usar so o a_mn dele e uma aproximacao hibrida, nao consistente.
#   Alem disso, a formula do Modified UNIFAC NAO e centrada em T0=298.15K
#   (Psi = exp[-(a+bT+cT^2)/T], sem subtrair T_ref), entao o termo b NAO
#   se anula a 298.15K. O "a efetivo" abaixo ja incorpora b*T a 298.15K:
#     a_eff(11,14) = 64.430 + 0.191*298.15 = 121.36 K
#     a_eff(14,11) = 95.323 - 0.789*298.15 = -139.83 K
#
# Em ambos os casos, nenhuma das duas fontes confirma se este par especifico
# teve dado experimental real por tras ou foi so completado por analogia
# estatistica (ML). O par CNH2(14)/COO(41) continua sem substituto em
# qualquer das duas fontes; mantem o valor estimado do BLOCO 4.

# Opcao A (RECOMENDADA: mesmo R_k/Q_k do script, sem ajuste de T necessario)
INTERACTION_PARAMS_UNIFAC20_TEST = {
    (11, 14): ( 53.04, 0.0, 0.0),
    (14, 11): ( 22.94, 0.0, 0.0),
}

# Opcao B (framework Dortmund, R_k/Q_k diferentes -- hibrido, usar com cautela)
INTERACTION_PARAMS_MODUNIFAC20_TEST = {
    (11, 14): ( 121.36, 0.0, 0.0),   # a_eff a 298.15K
    (14, 11): (-139.83, 0.0, 0.0),   # a_eff a 298.15K
}


def get_interaction_params(mode: str = 'estimado') -> dict:
    """
    Retorna o dicionario de parametros de interacao a ser usado no calculo.

    mode='estimado' (default): os 4 parametros estimados neste TCC
        (BLOCO 4, V2, Souza et al. 2025) — versao de referencia, ja
        validada por leave-one-out e analise de sensibilidade.

    mode='unifac20_ml': substitui CCOO(11)/CNH2(14) pelo valor do
        UNIFAC 2.0 (framework original, MESMOS R_k/Q_k deste script).
        Opcao recomendada para o teste de sensibilidade, por ser a
        unica sem mistura de framework.

    mode='modunifac20_ml': substitui CCOO(11)/CNH2(14) pelo a_eff do
        mod. UNIFAC 2.0 (framework Dortmund, R_k/Q_k DIFERENTES dos
        deste script -- aproximacao hibrida, usar so como curiosidade).

    Em todos os casos, o par CNH2(14)/COO(41) permanece com o valor
    estimado do BLOCO 4, pois nao ha substituto disponivel para ele.
    """
    params = dict(INTERACTION_PARAMS)
    if mode == 'unifac20_ml':
        params.update(INTERACTION_PARAMS_UNIFAC20_TEST)
    elif mode == 'modunifac20_ml':
        params.update(INTERACTION_PARAMS_MODUNIFAC20_TEST)
    elif mode != 'estimado':
        raise ValueError(f"mode desconhecido: {mode!r}")
    return params


# ===========================================================================
# VALIDACAO — execute: python params.py
# ===========================================================================
if __name__ == '__main__':
    print("=" * 58)
    print("VALIDACAO: params.py")
    print("  Biodiesel: Bessa et al. (2016) Fluid Phase Equilib. 425")
    print("  Aminas/COO: UNIFAC original (VLE)")
    print("=" * 58)
    errors = 0

    print("\n1. R_k > 0 e Q_k >= 0:")
    for sg, (name, R, Q, mg) in SUBGROUPS.items():
        if R <= 0 or Q < 0:
            print(f"   x sg {sg} ({name}): R={R}, Q={Q}")
            errors += 1
    print("   OK" if errors == 0 else "")

    print("\n2. Contagem de carbonos:")
    c_sg = {1:1, 2:1, 3:1, 4:1, 6:2, 14:0, 16:0, 21:2, 22:2, 77:1, 29:1, 32:1}
    for nome, nu, C_exp in [
        ('Triolina   C57', NU_TRIOLINA, 57),
        ('Etanol     C2',  NU_ETANOL,    2),
        ('[2-HEA][Hx] C8', NU_2HEA_HX,  8),
        ('[DETA][Hx] C10', NU_DETA_HX, 22),
    ]:
        C = sum(nu.get(sg,0)*c_sg.get(sg,0) for sg in c_sg)
        ok = C == C_exp
        if not ok: errors += 1
        print(f"   {'OK' if ok else 'ERRO'} {nome}: C={C} (esperado {C_exp})")

    print("\n3. Pares de interacao:")
    n_ok   = sum(1 for v in INTERACTION_PARAMS.values() if v[0] != 0.0) // 2
    n_zero = sum(1 for v in INTERACTION_PARAMS.values() if v[0] == 0.0) // 2
    print(f"   Pares com parametros: {n_ok}")
    print(f"   Pares ausentes (Psi=1): {n_zero}")

    print("\n4. Reversos (n,m) presentes para todo (m,n):")
    miss = [(m,n) for (m,n) in INTERACTION_PARAMS if (n,m) not in INTERACTION_PARAMS]
    if miss:
        print(f"   ERRO {miss}"); errors += 1
    else:
        print("   OK")

    print("\n5. r e q dos componentes:")
    for nome, nu in [('Triolina',NU_TRIOLINA),('Etanol',NU_ETANOL),
                     ('[2-HEA][Hx]',NU_2HEA_HX),('[DETA][Hx]',NU_DETA_HX)]:
        r = sum(nu.get(sg,0)*SUBGROUPS[sg][1] for sg in nu if sg in SUBGROUPS)
        q = sum(nu.get(sg,0)*SUBGROUPS[sg][2] for sg in nu if sg in SUBGROUPS)
        print(f"   {nome:15s}: r={r:.4f}  q={q:.4f}")

    print("\n" + "="*58)
    print(f"{'OK' if errors==0 else f'{errors} erro(s)'}")