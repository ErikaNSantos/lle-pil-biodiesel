import numpy as np

def calc_gama_unifac(x_tri, x_et, T_kelvin=318.15):
    """
    Calcula os coeficientes de atividade (combinatorial e residual) para o sistema binário {Trioleína + Etanol}.
    Baseado nas formulações do UNIFAC-LL clássico e matriz de Ferreira et al. (2018).
    """
    
    # Frações molares da mistura
    x = np.array([x_tri, x_et])
    
    # ---------------------------------------------------------
    # 1. DEFINIÇÃO DOS SUBGRUPOS (k) E GRUPOS PRINCIPAIS (m)
    # ---------------------------------------------------------
    # Índices dos subgrupos (k): 
    # 0: CH3, 1: CH2, 2: CH=CH, 3: OH, 4: CH2COO, 5: CHCOO
    
    # Frequência dos subgrupos em cada molécula (nu_ki)
    # Linha 0: Trioleína (C57H104O6) | Linha 1: Etanol (C2H6O)
    nu = np.array([
        [3, 42, 3, 0, 2, 1], 
        [1,  1, 0, 1, 0, 0]   
    ])
    
    # Parâmetros estruturais de van der Waals (Rk, Qk)
    Rk = np.array([0.9011, 0.6744, 1.1167, 1.0000, 1.6764, 1.3997])
    Qk = np.array([0.8480, 0.5400, 0.8670, 1.2000, 1.4200, 1.1600])
    
    # Mapeamento de subgrupos(k) para Grupos Principais(m) para uso na matriz Amn
    # Grupos Principais: 0: CH2(alifáticos), 1: C=C, 2: OH, 3: CCOO(éster)
    m_map = np.array([0, 0, 1, 2, 3, 3]) 
    
    # Matriz de Interação (Amn) em Kelvin - Fonte: Ferreira et al. (2018)
    a_mn = np.array([
        [0.00,   -10.73, 1304.58, 2590.68], # 0: CH2
        [902.32,   0.00,   93.41, 1755.53], # 1: C=C
        [882.11, 518.11,    0.00,  234.59], # 2: OH
        [627.98, 1924.10, 638.98,    0.00]  # 3: CCOO
    ])

    # ---------------------------------------------------------
    # 2. TERMO COMBINATORIAL (Staverman-Guggenheim)
    # ---------------------------------------------------------
    # Volume (r) e Área (q) de cada molécula puros
    r = np.sum(nu * Rk, axis=1)
    q = np.sum(nu * Qk, axis=1)
    
    # Frações de volume (Phi) e área (Theta) das moléculas na mistura
    Phi = (r * x) / np.sum(r * x)
    Theta = (q * x) / np.sum(q * x)
    
    # Parâmetro de massa (l) com coordenação z = 10
    l = (10 / 2) * (r - q) - (r - 1)
    
    # Cálculo de ln(gamma_comb)
    ln_gamma_comb = np.log(Phi / x) + (10 / 2) * q * np.log(Theta / Phi) + l - (Phi / x) * np.sum(x * l)

    # ---------------------------------------------------------
    # 3. TERMO RESIDUAL 
    # ---------------------------------------------------------
    Psi = np.exp(-a_mn / T_kelvin)
    
    def calc_lnGamma_k(f_molares):
        """Calcula a atividade dos subgrupos (ln Gamma_k) num ambiente específico."""
        # Fração molar de cada subgrupo na mistura (X_m)
        X_k = np.sum(nu * f_molares[:, np.newaxis], axis=0)
        X_k = X_k / np.sum(X_k) if np.sum(X_k) > 0 else X_k
        
        # Fração de área de cada subgrupo (Theta_k)
        Theta_k = (Qk * X_k) / np.sum(Qk * X_k) if np.sum(Qk * X_k) > 0 else np.zeros_like(Qk)
        
        # Consolidar Theta_k em Theta_m (Grupos Principais)
        Theta_m = np.zeros(4)
        for k in range(6):
            Theta_m[m_map[k]] += Theta_k[k]
            
        # Calcular ln Gamma_k para os 6 subgrupos
        lnGamma_k = np.zeros(6)
        for k in range(6):
            m = m_map[k] # Grupo principal do subgrupo k
            
            soma1 = np.sum(Theta_m * Psi[:, m])
            soma2 = 0
            for n in range(4):
                denominador = np.sum(Theta_m * Psi[:, n])
                if denominador > 0:
                    soma2 += (Theta_m[n] * Psi[m, n]) / denominador
                    
            lnGamma_k[k] = Qk[k] * (1.0 - np.log(soma1) - soma2)
            
        return lnGamma_k

    # Atividade dos grupos na MISTURA
    lnGamma_k_mistura = calc_lnGamma_k(x)
    
    # Atividade dos grupos nas MOLÉCULAS PURAS (x_i = 1)
    lnGamma_k_puro = np.zeros((2, 6))
    lnGamma_k_puro[0] = calc_lnGamma_k(np.array([1.0, 0.0])) # Trioleína pura
    lnGamma_k_puro[1] = calc_lnGamma_k(np.array([0.0, 1.0])) # Etanol puro
    
    # Cálculo de ln(gamma_res)
    ln_gamma_res = np.zeros(2)
    for i in range(2):
        ln_gamma_res[i] = np.sum(nu[i] * (lnGamma_k_mistura - lnGamma_k_puro[i]))

    # ---------------------------------------------------------
    # 4. RESULTADOS FINAIS
    # ---------------------------------------------------------
    gamma_comb = np.exp(ln_gamma_comb)
    gamma_res = np.exp(ln_gamma_res)
    gamma_total = gamma_comb * gamma_res
    ln_gamma_total = np.log(gamma_total)
    
    return {
        "gamma_comb": gamma_comb,
        "gamma_res": gamma_res,
        "gamma_total": gamma_total,
        "ln_gamma_total": ln_gamma_total
    }

# --- TESTES DE DILUIÇÃO INFINITA ---

print("=== Caso A: Trioleína infinitamente diluída em Etanol ===")
# x_trioleina -> 0, x_etanol -> 1
res_A = calc_gama_unifac(1e-6, 0.999999)
print(f"Trioleína | gamma^comb: {res_A['gamma_comb'][0]:.4f} | gamma^res: {res_A['gamma_res'][0]:.4e} | gamma^inf: {res_A['gamma_total'][0]:.4e} | ln(gamma^inf): {res_A['ln_gamma_total'][0]:.4e}")

print("\\n=== Caso B: Etanol infinitamente diluído em Trioleína ===")
# x_trioleina -> 1, x_etanol -> 0
res_B = calc_gama_unifac(0.999999, 1e-6)
print(f"Etanol    | gamma^comb: {res_B['gamma_comb'][1]:.4f} | gamma^res: {res_B['gamma_res'][1]:.4e} | gamma^inf: {res_B['gamma_total'][1]:.4e} | ln(gamma^inf): {res_B['ln_gamma_total'][1]:.4e}")
