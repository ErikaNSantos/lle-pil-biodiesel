"""
diagnostico_binarios.py — O que o COSMO-SAC diz sobre as 3 arestas do ternario?
================================================================================
CONTEXTO
    O teste TPD esta CORRETO. O mapa todo-bifasico vem dos proprios gamma do
    COSMO-SAC, que prevem imiscibilidade quase total triolina/LI (gamma^inf
    da ordem de 1e9..1e16). Antes de confiar nisso, precisamos checar os
    binarios, onde:
      - triolina+etanol NAO envolve o LI (tira o confundidor do sigma-profile
        ionico) e tem comportamento experimental conhecido (miscibilidade
        limitada a 298 K — a premissa do TCC). Se o COSMO-SAC der "totalmente
        miscivel", os sigma-profiles nao servem para esses sistemas.

O QUE IMPRIME, para cada par:
    - gamma^inf nos dois sentidos (ln gamma^inf)
    - varredura de is_unstable ao longo do binario -> existe gap? em que faixa?

Regra de bolso (apenas indicativa; o veredito real e a varredura):
    LLE binario tende a existir quando ln gamma^inf > ~2 nos dois sentidos.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta.
"""

import numpy as np
from py4j.java_gateway import JavaGateway

from stability import is_unstable
from stability_cosmo import COSMOSAC, TRIOLINA, ETANOL, LI_2HEA, LI_DETA


PARES = [
    (TRIOLINA, ETANOL),      # <-- o teste decisivo (sem LI)
    (TRIOLINA, LI_2HEA),
    (ETANOL,   LI_2HEA),
    (TRIOLINA, LI_DETA),
    (ETANOL,   LI_DETA),
]

N_SCAN = 25
EPS_INF = 1e-7   # para sondar gamma^inf (acima do piso 1e-8 do modelo)


def gamma_inf(model):
    """Retorna (gamma_a^inf em b, gamma_b^inf em a)."""
    g_a = model.gamma([EPS_INF, 1.0 - EPS_INF])[0]   # a diluido em b
    g_b = model.gamma([1.0 - EPS_INF, EPS_INF])[1]   # b diluido em a
    return float(g_a), float(g_b)


def faixa_instavel(xs, flags):
    """Faixa de x_a (componente 1) onde is_unstable=True (assume gap unico)."""
    idx = [i for i, f in enumerate(flags) if f]
    if not idx:
        return None
    return xs[idx[0]], xs[idx[-1]]


def analisar_par(a, b, gateway):
    model = COSMOSAC([a, b], gateway=gateway)
    print(f"\n--- {a}  +  {b} ---")

    g_a, g_b = gamma_inf(model)
    print(f"  gamma^inf({a} em {b}) = {g_a:.3e}   (ln = {np.log(g_a):+.2f})")
    print(f"  gamma^inf({b} em {a}) = {g_b:.3e}   (ln = {np.log(g_b):+.2f})")
    indicativo = "espera gap" if (np.log(g_a) > 2 and np.log(g_b) > 2) else "espera miscivel"
    print(f"  regra de bolso: {indicativo}")

    xs = np.linspace(0.02, 0.98, N_SCAN)
    flags = []
    for x in xs:
        z = np.array([x, 1.0 - x])
        flags.append(bool(is_unstable(z, model, n_random=2)))

    faixa = faixa_instavel(xs, flags)
    if faixa is None:
        print(f"  varredura: NENHUM ponto instavel -> COSMO-SAC preve MISCIVEL")
    else:
        print(f"  varredura: gap de miscibilidade em x_{a} ~ [{faixa[0]:.2f}, {faixa[1]:.2f}]")
        n_bif = sum(flags)
        print(f"             ({n_bif}/{N_SCAN} pontos instaveis)")
    print(f"  chamadas JCOSMO: {model.n_calls}")
    return faixa


if __name__ == '__main__':
    print("=" * 70)
    print("DIAGNOSTICO DOS BINARIOS (COSMO-SAC / JCOSMO)")
    print("=" * 70)

    gw = JavaGateway(auto_field=True)
    for a, b in PARES:
        analisar_par(a, b, gw)

    print("\n" + "=" * 70)
    print("LEITURA CRITICA:")
    print("  TRIOLEIN+ETHANOL e o par decisivo (sem LI).")
    print("  Se der 'MISCIVEL' aqui e voce tem dado experimental de gap a 298 K,")
    print("  os sigma-profiles do COSMO-SAC nao reproduzem o sistema -> bloqueio")
    print("  real no .Gout/GAMESS, nao no codigo de estabilidade.")
    print("=" * 70)
