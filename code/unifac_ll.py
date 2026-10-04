# -*- coding: utf-8 -*-
"""
Nucleo UNIFAC-LL para calculo de coeficientes de atividade em sistemas
ternarios {oleo/acido graxo + alcool + liquido ionico protico}.

Formulacao: UNIFAC original (Fredenslund, Jones & Prausnitz, 1975),
termo combinatorial de Staverman-Guggenheim com z = 10, SEM a modificacao
r^(3/4) do UNIFAC-Dortmund. Parametros de interacao de LLE (Bessa et al.,
2016) para os grupos apolares/ester, UNIFAC classico para aminas e
carboxilato, e dois pares estimados neste trabalho (V2).

Este arquivo e a primeira camada da ferramenta unificada. Ele e
autossuficiente e pode ser executado diretamente para rodar os testes de
sanidade (python unifac_ll.py).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Subgrupos UNIFAC: id -> (nome, R_k, Q_k, id do grupo principal)
#    Valores de volume (R) e area (Q) de van der Waals reduzidos.
# ---------------------------------------------------------------------------
SUBGROUPS = {
    1:  ("CH3",     0.9011, 0.848, 1),   # grupo principal 1  (CH2)
    2:  ("CH2",     0.6744, 0.540, 1),
    3:  ("CH",      0.4469, 0.228, 1),
    6:  ("CH=CH",   1.1167, 0.867, 2),   # grupo principal 2  (C=C)
    14: ("OH",      1.0000, 1.200, 5),   # grupo principal 5  (OH)
    16: ("H2O",     0.9200, 1.400, 7),   # grupo principal 7  (H2O)
    22: ("CH2COO",  1.6764, 1.420, 11),  # grupo principal 11 (CCOO)
    29: ("CH2NH2",  1.3692, 1.236, 14),  # grupo principal 14 (CNH2)
    32: ("CH2NH",   1.2070, 0.936, 15),  # grupo principal 15 (CNH)
    43: ("COOH",    1.3013, 1.224, 20),  # grupo principal 20 (COOH)
    77: ("COO",     1.3800, 1.200, 41),  # grupo principal 41 (COO)
}

# Nome legivel de cada grupo principal (usado no detector de pares faltantes)
MAIN_GROUP_NAME = {
    1: "CH2", 2: "C=C", 5: "OH", 7: "H2O", 11: "CCOO",
    14: "CNH2", 15: "CNH", 20: "COOH", 41: "COO",
}


def main_group(sg_id):
    return SUBGROUPS[sg_id][3]


# ---------------------------------------------------------------------------
# 2. Decomposicoes moleculares (subgrupo id -> numero de ocorrencias)
#    Representacao de par ionico NEUTRO para os PILs (nao separar cation e
#    anion; a separacao produz gamma_inf de 10^11-10^15 nestes sistemas).
# ---------------------------------------------------------------------------
COMPONENTS = {
    # --- oleos / acidos graxos (vertice inferior esquerdo) ---
    # Trioleina C57H104O6
    "trioleina":        {1: 3, 2: 41, 3: 1, 6: 3, 22: 3},
    # Trilinoleina C57H98O6 (triglicerideo do acido linoleico, 2 insaturacoes/braco)
    "trilinoleina":     {1: 3, 2: 35, 3: 1, 6: 6, 22: 3},
    # Trilinolenina C57H92O6 (triglicerideo do acido linolenico, 3 insaturacoes/braco)
    "trilinolenina":    {1: 3, 2: 29, 3: 1, 6: 9, 22: 3},
    # Triricinoleina C57H104O9 (triglicerideo do acido ricinoleico; oleo de mamona)
    "triricinoleina":   {1: 3, 2: 38, 3: 4, 6: 3, 14: 3, 22: 3},
    # Tripalmitina C51H98O6 (triglicerideo do acido palmitico, saturado)
    "tripalmitina":     {1: 3, 2: 41, 3: 1, 22: 3},
    # Triestearina C57H110O6 (triglicerideo do acido estearico, saturado)
    "triestearina":     {1: 3, 2: 47, 3: 1, 22: 3},
    # Acido oleico C18:1  (C18H34O2)
    "acido_oleico":     {1: 1, 2: 14, 6: 1, 43: 1},
    # Acido linoleico C18:2 (C18H32O2)
    "acido_linoleico":  {1: 1, 2: 12, 6: 2, 43: 1},
    # Acido ricinoleico C18:1 + OH (C18H34O3)
    "acido_ricinoleico": {1: 1, 2: 13, 3: 1, 6: 1, 14: 1, 43: 1},

    # --- alcoois (vertice superior) ---
    # Metanol: aproximacao CH3+OH (UNIFAC classico usa subgrupo CH3OH dedicado)
    "metanol":          {1: 1, 14: 1},
    "etanol":           {1: 1, 2: 1, 14: 1},
    "1-propanol":       {1: 1, 2: 2, 14: 1},

    # --- liquidos ionicos / cossolventes (vertice inferior direito) ---
    # [2-HEA][Hx] 2-hidroxietilamonio hexanoato, par 1:1  (C8H19NO3)
    "2HEAHX":           {1: 1, 2: 5, 14: 1, 77: 1, 29: 1},
    # [DETA][Hx] dietilenotriamonio hexanoato, par 1:3 TRIPROTONADO
    #   par neutro C22H49N3O6 (80 atomos, ~451.6 g/mol)
    "DETAHX":           {1: 3, 2: 13, 77: 3, 29: 2, 32: 1},
}

# Rotulo de exibicao (para titulo de diagrama, legendas etc.)
COMPONENT_LABEL = {
    "trioleina": "Trioleina",
    "trilinoleina": "Trilinoleina",
    "trilinolenina": "Trilinolenina",
    "triricinoleina": "Triricinoleina",
    "tripalmitina": "Tripalmitina",
    "triestearina": "Triestearina",
    "acido_oleico": "Acido oleico",
    "acido_linoleico": "Acido linoleico",
    "acido_ricinoleico": "Acido ricinoleico",
    "metanol": "Metanol",
    "etanol": "Etanol",
    "1-propanol": "1-Propanol",
    "2HEAHX": "[2-HEA][Hx]",
    "DETAHX": "[DETA][Hx]",
}


# ---------------------------------------------------------------------------
# 3. Parametros de interacao a_mn (K), com b = c = 0  =>  Psi_mn = exp(-a_mn/T)
#    Notacao _pair("X/Y", a_xy, a_yx): define A[(X,Y)] e A[(Y,X)].
# ---------------------------------------------------------------------------
A = {}


def _pair(mg_i, mg_j, a_ij, a_ji):
    A[(mg_i, mg_j)] = a_ij
    A[(mg_j, mg_i)] = a_ji


# --- Bessa et al. (2016): parametros de LLE para biodiesel ---
_pair(1,  2,   239.32,  470.13)   # CH2 / C=C
_pair(1,  5,   714.56,   73.52)   # CH2 / OH
_pair(1,  7,  2045.36,  230.67)   # CH2 / H2O
_pair(1, 11,   560.42,  462.69)   # CH2 / CCOO
_pair(2,  5,   954.33,  212.36)   # C=C / OH
_pair(2,  7,   328.14,  207.51)   # C=C / H2O
_pair(2, 11,    49.03, 1040.51)   # C=C / CCOO
_pair(5,  7,    19.65,   90.08)   # OH  / H2O
_pair(5, 11,   172.62,  243.03)   # OH  / CCOO
_pair(7, 11,    11.22,  616.17)   # H2O / CCOO

# --- UNIFAC classico (VLE): aminas e carboxilato ---
_pair(1, 14,   391.50,  -30.48)   # CH2  / CNH2
_pair(1, 15,   255.70,   65.33)   # CH2  / CNH
_pair(1, 41,   387.10,  529.00)   # CH2  / COO
_pair(2, 14,   240.90,    1.163)  # C=C  / CNH2
_pair(2, 15,   163.90,  -28.70)   # C=C  / CNH
_pair(2, 41,    48.33, 1397.00)   # C=C  / COO
_pair(5, 14,     8.642, -242.80)  # OH   / CNH2
_pair(5, 15,    42.70, -150.00)   # OH   / CNH
_pair(5, 41,   190.30,   88.63)   # OH   / COO
_pair(11, 15,  -73.50,  136.00)   # CCOO / CNH
_pair(11, 41,  560.20, -234.90)   # CCOO / COO
# ATENCAO — pendencia em aberto (nao reintroduzir a "correcao" sem
# recalibrar o V2 junto):
#   O valor -122.3 (do binary.csv do JCOSMO) foi testado aqui e chegou a
#   ser adotado como "correcao" de uma suposta transcricao errada. Mas os
#   parametros V2 abaixo (CCOO/CNH2 e CNH2/COO) foram calibrados contra
#   Souza et al. (2025) usando ESTE valor (560.20) como base — trocar um
#   sem recalibrar o outro quebra a consistencia interna do conjunto.
#   Confirmado experimentalmente: com -122.3, TODOS os sistemas testados
#   (trioleina/trilinoleina x etanol/metanol x 2HEAHX/DETAHX) dao 0
#   tie-lines (miscibilidade total, sem distincao entre os dois PILs).
#   Com 560.20, os sistemas com [2-HEA][Hx] mostram binodal robusta
#   (82-117 tie-lines) e os com [DETA][Hx] nao mostram nenhuma — coerente
#   com o achado independente do COSMO-SAC de que [DETA][Hx] tolera mais
#   oleo antes de desestabilizar do que [2-HEA][Hx]. Mantido em 560.20
#   ate a Profa. Silvana confirmar a matriz corrigida E o V2 recalibrado
#   contra ela, como um par indissociavel.
_pair(14, 15, -107.20,  127.40)   # CNH2 / CNH
_pair(15, 41,  -38.77,  284.50)   # CNH  / COO
_pair(7, 14,    48.89, -330.40)   # H2O  / CNH2
_pair(7, 15,   168.00, -448.20)   # H2O  / CNH
_pair(7, 41,  -197.50,  284.40)   # H2O  / COO

# --- estimados neste trabalho (V2, calibrados contra Souza et al. 2025) ---
_pair(11, 14,   53.04,   22.94)      # CCOO / CNH2
_pair(14, 41,  -69.49864, 20.755741) # CNH2 / COO

# --- preditos por ML: mod. UNIFAC 2.0 (Hayer, Hasse & Jirasek, 2024) ---
# Extraidos de Interaction_Parameters_UNIFAC_2_0.csv (numeracao de grupos
# principais identica a UNIFAC classico/Magnussen 1981, conferida contra
# Size_Parameters_UNIFAC_2_0.csv). Preenchem TODOS os pares envolvendo
# COOH(20) que faltavam: os quatro classicos (com CH2, C=C, OH, H2O) e os
# tres especificos dos grupos do PIL (CNH2, CNH, COO). Com isso os sistemas
# com acido graxo deixam de ser sinalizados como quantitativamente nao
# confiaveis por parametro ausente.
_pair(20,  1,  168.72636,  470.64224)   # COOH / CH2
_pair(20,  2,  145.5463,   449.4651)    # COOH / C=C
_pair(20,  5, -128.54982,  -13.354626)  # COOH / OH
_pair(20,  7, -302.2765,    48.956203)  # COOH / H2O
_pair(20, 14,  -96.50031, -270.82874)   # COOH / CNH2
_pair(20, 15, -159.71945, -376.9019)    # COOH / CNH
_pair(20, 41,  -24.17741,  226.12439)   # COOH / COO

# ---------------------------------------------------------------------------
# Pares que faltavam envolvendo COOH — RESOLVIDO: todos os sete pares foram
# preenchidos com valores preditos por ML do mod. UNIFAC 2.0 (Hayer, Hasse &
# Jirasek, 2024), ver bloco acima. Nao ha mais pares ausentes envolvendo
# COOH neste conjunto de sistemas; o detector de pares faltantes so seria
# acionado se um grupo totalmente novo (fora dos 11 usados aqui) entrasse
# em cena.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Classe UNIFAC-LL. Instancia-se com o sistema (3 componentes) e T; o
#    metodo gamma(x) retorna o vetor de coeficientes de atividade.
# ---------------------------------------------------------------------------
Z_COORD = 10.0  # numero de coordenacao (Fredenslund original)


class UnifacLL:

    def __init__(self, comp_keys, T=298.15):
        if len(comp_keys) != 3:
            raise ValueError("sistema deve ter exatamente 3 componentes")
        for k in comp_keys:
            if k not in COMPONENTS:
                raise KeyError("componente desconhecido: %s" % k)

        self.comp_keys = list(comp_keys)
        self.T = float(T)
        self.n = 3

        # subgrupos presentes no sistema, em ordem estavel
        present = set()
        for k in self.comp_keys:
            present.update(COMPONENTS[k].keys())
        self.sg = sorted(present)
        self.nsg = len(self.sg)
        self._sg_index = {s: i for i, s in enumerate(self.sg)}

        # matriz de contagens nu[i, k]  (componente i, subgrupo k)
        self.nu = np.zeros((self.n, self.nsg))
        for i, key in enumerate(self.comp_keys):
            for s, count in COMPONENTS[key].items():
                self.nu[i, self._sg_index[s]] = count

        # vetores R_k e Q_k dos subgrupos presentes
        self.Rk = np.array([SUBGROUPS[s][1] for s in self.sg])
        self.Qk = np.array([SUBGROUPS[s][2] for s in self.sg])

        # r_i e q_i por componente
        self.r = self.nu @ self.Rk
        self.q = self.nu @ self.Qk

        # matriz Psi[m, k] = exp(-a_{mk}/T) sobre subgrupos presentes.
        # Registra os pares de grupo principal ausentes (Psi = 1).
        self._missing = set()
        self.Psi = np.ones((self.nsg, self.nsg))
        for a_i, s_m in enumerate(self.sg):
            mg_m = main_group(s_m)
            for b_i, s_k in enumerate(self.sg):
                mg_k = main_group(s_k)
                if mg_m == mg_k:
                    amk = 0.0
                elif (mg_m, mg_k) in A:
                    amk = A[(mg_m, mg_k)]
                else:
                    amk = 0.0
                    self._missing.add(frozenset((mg_m, mg_k)))
                self.Psi[a_i, b_i] = np.exp(-amk / self.T)

        # ln Gamma_k de referencia (componente puro), pre-computado
        self._ln_Gamma_pure = np.zeros((self.n, self.nsg))
        for i in range(self.n):
            Xi = self.nu[i] / self.nu[i].sum()
            self._ln_Gamma_pure[i] = self._ln_Gamma_group(Xi)

    # ---- coeficiente de atividade dos grupos, dado o vetor de fracao de grupo X ----
    def _ln_Gamma_group(self, X):
        theta = self.Qk * X
        s = theta.sum()
        if s <= 0.0:
            return np.zeros(self.nsg)
        theta = theta / s
        # S[k] = sum_m theta_m Psi_mk
        S = theta @ self.Psi
        # ln Gamma_k = Q_k [1 - ln S_k - sum_m Psi_km theta_m / S_m]
        second = self.Psi @ (theta / S)
        return self.Qk * (1.0 - np.log(S) - second)

    # ---- vetor de coeficientes de atividade dos componentes ----
    def gamma(self, x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, None)
        x = x / x.sum()

        # --- termo combinatorial (razoes computadas de forma robusta em x_i -> 0) ---
        sum_rx = float(self.r @ x)
        sum_qx = float(self.q @ x)
        phi_over_x = self.r / sum_rx                 # phi_i / x_i
        phi_over_theta = (self.r / self.q) * (sum_qx / sum_rx)  # phi_i / theta_i
        ln_gamma_c = (np.log(phi_over_x) + 1.0 - phi_over_x
                      - (Z_COORD / 2.0) * self.q
                      * (np.log(phi_over_theta) + 1.0 - phi_over_theta))

        # --- termo residual ---
        Xmix = self.nu.T @ x
        Xmix = Xmix / Xmix.sum()
        ln_Gamma_mix = self._ln_Gamma_group(Xmix)
        ln_gamma_r = np.array([
            float(self.nu[i] @ (ln_Gamma_mix - self._ln_Gamma_pure[i]))
            for i in range(self.n)
        ])

        return np.exp(ln_gamma_c + ln_gamma_r)

    # ---- deteccao de pares de grupo principal necessarios e ausentes ----
    def missing_pairs(self):
        """Lista de pares (mg_i, mg_j) presentes no sistema e ausentes da
        tabela de interacao. Retorna tuplas ordenadas com os nomes."""
        out = []
        for fs in self._missing:
            ids = sorted(fs)
            if len(ids) == 1:
                continue
            m, n = ids
            out.append((m, n))
        out.sort()
        return out

    def missing_pairs_report(self):
        pairs = self.missing_pairs()
        if not pairs:
            return "Nenhum par de interacao ausente para este sistema."
        linhas = ["Pares de interacao AUSENTES (calculo prossegue com Psi=1, "
                  "resultado quantitativamente nao confiavel):"]
        for m, n in pairs:
            linhas.append("  %s(%d) / %s(%d)"
                          % (MAIN_GROUP_NAME[m], m, MAIN_GROUP_NAME[n], n))
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# 5. Testes de sanidade (executados quando o arquivo roda como script)
# ---------------------------------------------------------------------------
def _fmt(v):
    return "[" + ", ".join("%.6f" % z for z in v) + "]"


if __name__ == "__main__":
    print("=" * 70)
    print("TESTES DE SANIDADE - NUCLEO UNIFAC-LL")
    print("=" * 70)

    sysA = ("trioleina", "etanol", "2HEAHX")
    m = UnifacLL(sysA, T=298.15)
    print("\nSistema A: %s + %s + %s  (T = %.2f K)"
          % (COMPONENT_LABEL[sysA[0]], COMPONENT_LABEL[sysA[1]],
             COMPONENT_LABEL[sysA[2]], m.T))
    print("  r =", _fmt(m.r), "  q =", _fmt(m.q))

    # Teste 1: limite gamma_i -> 1 no componente puro
    print("\n[Teste 1] Limite no vertice puro (gamma do componente dominante deve -> 1)")
    for i, key in enumerate(sysA):
        x = np.full(3, 0.0)
        x[i] = 1.0
        g = m.gamma(x)
        print("  puro %-12s x=%s  gamma=%s  -> gamma_%d = %.8f"
              % (COMPONENT_LABEL[key], _fmt(x), _fmt(g), i + 1, g[i]))

    # Teste 2: aproximacao continua ao limite
    print("\n[Teste 2] Aproximacao ao limite (x1 -> 1), gamma_1 deve -> 1")
    for e in (1e-1, 1e-2, 1e-3, 1e-4):
        x = np.array([1 - 2 * e, e, e])
        g = m.gamma(x)
        print("  x1 = %.4f   gamma_1 = %.8f" % (x[0], g[0]))

    # Teste 3: gamma em composicao intermediaria (para conferencia vs scripts)
    print("\n[Teste 3] gamma em composicoes intermediarias")
    for x in ([0.30, 0.40, 0.30], [0.50, 0.30, 0.20], [0.10, 0.45, 0.45]):
        g = m.gamma(x)
        print("  x=%s  ->  gamma=%s" % (_fmt(x), _fmt(g)))

    # Teste 4: consistencia do termo combinatorial (forma spec vs forma L_i)
    print("\n[Teste 4] Consistencia interna do termo combinatorial (spec vs L_i)")
    x = np.array([0.3, 0.4, 0.3])
    sum_rx = float(m.r @ x); sum_qx = float(m.q @ x)
    phi = m.r * x / sum_rx
    theta = m.q * x / sum_qx
    L = (Z_COORD / 2.0) * (m.r - m.q) - (m.r - 1.0)
    lnC_L = (np.log(phi / x) + (Z_COORD / 2.0) * m.q * np.log(theta / phi)
             + L - (phi / x) * float(L @ x))
    phi_over_x = m.r / sum_rx
    phi_over_theta = (m.r / m.q) * (sum_qx / sum_rx)
    lnC_spec = (np.log(phi_over_x) + 1.0 - phi_over_x
                - (Z_COORD / 2.0) * m.q
                * (np.log(phi_over_theta) + 1.0 - phi_over_theta))
    print("  ln gamma^C (spec) =", _fmt(lnC_spec))
    print("  ln gamma^C (L_i)  =", _fmt(lnC_L))
    print("  desvio max = %.3e" % np.max(np.abs(lnC_spec - lnC_L)))

    # Teste 5: detector de pares faltantes
    print("\n[Teste 5] Detector de pares de interacao ausentes")
    print("  -> Sistema com trioleina (esperado: 0 ausentes)")
    for sysT in [("trioleina", "etanol", "2HEAHX"),
                 ("trioleina", "etanol", "DETAHX"),
                 ("trioleina", "metanol", "DETAHX")]:
        mm = UnifacLL(sysT, 298.15)
        pares = mm.missing_pairs()
        print("     %-40s ausentes: %d %s"
              % (" + ".join(COMPONENT_LABEL[c] for c in sysT),
                 len(pares), pares if pares else ""))
    print("  -> Sistema com acido graxo (esperado: pares COOH ausentes)")
    for sysF in [("acido_oleico", "etanol", "2HEAHX"),
                 ("acido_ricinoleico", "etanol", "DETAHX")]:
        mm = UnifacLL(sysF, 298.15)
        print("     " + " + ".join(COMPONENT_LABEL[c] for c in sysF))
        print("     " + mm.missing_pairs_report().replace("\n", "\n     "))

    print("\n" + "=" * 70)
    print("FIM DOS TESTES")
    print("=" * 70)