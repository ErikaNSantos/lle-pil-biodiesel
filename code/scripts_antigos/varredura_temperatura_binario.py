"""
varredura_temperatura_binario.py — Existe UCST em TRIOLEIN+ETHANOL no COSMO-SAC?
==================================================================================
CONTEXTO
    O COSMO-SAC previu TRIOLEIN+ETHANOL como totalmente miscivel a 298.15 K
    (ln gamma^inf ~ 0.7-1.0 nos dois sentidos, sem gap na varredura de
    composicao). Evidencia convergente da literatura (Boocock 1996; Feuge &
    Gross apud Boocock 1996; Ma et al. 2016; Hanh et al. 2009) sugere que o
    sistema real e bifasico ou esta na fronteira a 298.15 K, e que pode ter
    uma UCST (upper critical solution temperature) entre 25 C e 50 C: Feuge &
    Gross relatam etanolise de oleo de amendoim a 50 C como UMA FASE, o que
    so faz sentido se a miscibilidade aumenta com T (comportamento UCST).

HIPOTESE TESTADA
    Se o COSMO-SAC entende a fisica do sistema mas erra a POSICAO da UCST,
    abaixar T a partir de 298.15 K deve abrir (ou alargar) um gap de
    miscibilidade -- mesmo que o gap nao apareca a 298.15 K.
    Se NENHUMA temperatura testada abre gap, a falha e estrutural (o
    sigma-profile nao captura a interacao de forma alguma), nao apenas um
    deslocamento da UCST.

LEITURA DO RESULTADO
    - Gap aparece e cresce ao diminuir T  -> modelo capta a fisica (UCST real
      um pouco acima de onde o COSMO-SAC coloca); resultado quantificavel e
      defensavel no TCC.
    - Gap nunca aparece, mesmo bem abaixo de 273 K -> falha estrutural do
      sigma-profile para este par; nao e so calibracao de T.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta.
"""

import numpy as np
from py4j.java_gateway import JavaGateway

from stability import is_unstable
from stability_cosmo import COSMOSAC, TRIOLINA, ETANOL

# Varredura de cima para baixo: comeca em 298.15 K (referencia do TCC) e desce.
# Inclui 323.15 K (50 C) como ponto de checagem positiva (Feuge & Gross: 1 fase).
TEMPERATURAS = [323.15, 298.15, 283.15, 273.15, 263.15, 253.15, 243.15]
N_SCAN = 31           # pontos de composicao por temperatura
X_MIN_SCAN = 0.02
X_MAX_SCAN = 0.98


def gamma_inf(model):
    eps = 1e-7
    g_a = model.gamma([eps, 1.0 - eps])[0]
    g_b = model.gamma([1.0 - eps, eps])[1]
    return float(g_a), float(g_b)


def varrer_T(T, gateway):
    model = COSMOSAC([TRIOLINA, ETANOL], T=T, gateway=gateway)
    g_a, g_b = gamma_inf(model)

    xs = np.linspace(X_MIN_SCAN, X_MAX_SCAN, N_SCAN)
    flags = []
    for x in xs:
        z = np.array([x, 1.0 - x])
        flags.append(bool(is_unstable(z, model, n_random=2)))

    n_bif = sum(flags)
    if n_bif == 0:
        faixa_str = "nenhuma (miscivel em toda a faixa)"
    else:
        idx = [i for i, f in enumerate(flags) if f]
        faixa_str = f"x_triolina em [{xs[idx[0]]:.2f}, {xs[idx[-1]]:.2f}]  ({n_bif}/{N_SCAN} pontos)"

    return {
        'T': T,
        'ln_g_triol_inf': np.log(g_a),
        'ln_g_etanol_inf': np.log(g_b),
        'n_bif': n_bif,
        'faixa': faixa_str,
        'n_calls': model.n_calls,
    }


if __name__ == '__main__':
    print("=" * 78)
    print("VARREDURA DE TEMPERATURA: TRIOLEIN + ETHANOL (existe UCST no COSMO-SAC?)")
    print("=" * 78)
    print(f"{'T (K)':>8}  {'T (C)':>7}  {'ln g_triol^inf':>15}  {'ln g_etOH^inf':>14}  {'gap de miscibilidade'}")
    print("-" * 78)

    gw = JavaGateway(auto_field=True)
    resultados = []
    for T in TEMPERATURAS:
        r = varrer_T(T, gw)
        resultados.append(r)
        print(f"{r['T']:>8.2f}  {r['T']-273.15:>7.1f}  {r['ln_g_triol_inf']:>15.3f}  "
              f"{r['ln_g_etanol_inf']:>14.3f}  {r['faixa']}")

    print("-" * 78)
    n_com_gap = sum(1 for r in resultados if r['n_bif'] > 0)
    print(f"\nTemperaturas com gap de miscibilidade: {n_com_gap}/{len(TEMPERATURAS)}")

    print("\nLEITURA:")
    if n_com_gap == 0:
        print("  Gap NUNCA aparece, mesmo nas T mais baixas testadas.")
        print("  -> Indicativo de FALHA ESTRUTURAL do sigma-profile para este par,")
        print("     nao apenas erro de posicionamento da UCST.")
    else:
        Ts_com_gap = [r['T'] for r in resultados if r['n_bif'] > 0]
        T_mais_alta_com_gap = max(Ts_com_gap)
        print(f"  Gap aparece para T <= {T_mais_alta_com_gap:.2f} K ({T_mais_alta_com_gap-273.15:.1f} C).")
        print("  -> O COSMO-SAC CAPTA a fisica do sistema (existe UCST), mas pode")
        print("     estar deslocando sua posicao em relacao ao valor real.")
        print("     Isso e uma limitacao QUANTIFICAVEL e defensavel no TCC,")
        print("     no mesmo espirito do artefato anti-cossolvente do UNIFAC-LL.")
    print("=" * 78)
