"""
diagnostico_xu.py — Por que is_unstable deu False num feed que deveria ser bifasico?
======================================================================================
CONTEXTO
    validar_xu2022.py deu 0/8 em DOIS sistemas diferentes ([EMIM][PF6] e
    [BMIM][PF6]), de forma uniforme. Um zero limpo e simetrico em dois
    sistemas quimicamente distintos e mais compativel com um problema no
    MINIMIZADOR (nao encontrando a regiao de D negativo) do que com "o
    COSMO-SAC nao entende a quimica" -- ja que IPAC/IPA/IL imidazolico e
    quimica bem mais simples que triolina+PIL.

O QUE ESTE SCRIPT FAZ
    Reproduz a minimizacao do TPD ponto a ponto (mesmos starts de
    stability.is_unstable), mas IMPRIME cada D(w) avaliado, para um feed
    Xu conhecido. Tambem avalia D diretamente nas composicoes das duas
    FASES EXPERIMENTAIS REPORTADAS (nao otimizadas) -- esse e o teste mais
    direto: se o COSMO-SAC "acredita" no equilibrio experimental, D
    avaliado exatamente nessas composicoes deve ser <= 0 (a fase
    experimental fica sobre ou abaixo do plano tangente).

LEITURA
    - Se D nas composicoes EXPERIMENTAIS ja for claramente negativo, mas o
      minimizador relata D_min ~ 0 -> bug de busca (minimizador raso,
      poucos starts, ou preso em ponto de sela). Conserto: mais starts,
      ou usar a fase experimental como chute inicial.
    - Se D nas composicoes experimentais tambem for ~0 ou positivo -> o
      proprio modelo COSMO-SAC nao reproduz a separacao aqui; aí sim e
      indicio de problema de sigma-profile/base de dados, nao do
      algoritmo.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py,
stability_cosmo.py e validar_xu2022.py (para os nomes/dados) na mesma pasta.
"""

import numpy as np
from scipy.optimize import minimize
from py4j.java_gateway import JavaGateway

from stability import tangent_plane_D, is_unstable
from stability_cosmo import COSMOSAC
from validar_xu2022 import IPAC, IPA, SISTEMAS_XU


def D_nas_fases_experimentais(model, x1_I, x2_I, x1_II, x2_II):
    """Avalia D(fase II) usando z=fase I como referencia, e vice-versa.
    Se as duas fases estao REALMENTE em equilibrio (isoatividade), o
    plano tangente em z=faseI deve tocar (quase) zero em w=faseII."""
    xI  = np.array([x1_I,  x2_I,  1.0 - x1_I  - x2_I])
    xII = np.array([x1_II, x2_II, 1.0 - x1_II - x2_II])

    ln_a_I  = np.log(xI)  + np.log(model.gamma(xI))
    ln_a_II = np.log(xII) + np.log(model.gamma(xII))

    D_em_II_usando_I  = float(np.sum(xII * (ln_a_II - ln_a_I)))
    D_em_I_usando_II  = float(np.sum(xI  * (ln_a_I  - ln_a_II)))
    return D_em_II_usando_I, D_em_I_usando_II


def minimizacao_instrumentada(model, z, n_random=5, seed=0):
    z = np.clip(z, 1e-15, 1.0); z = z / z.sum()
    N = len(z)
    ln_gz = np.log(z) + np.log(model.gamma(z))

    starts = []
    for i in range(N):
        w = np.ones(N) * 0.02
        w[i] = 1.0 - 0.02 * (N - 1)
        starts.append(w / w.sum())
    rng = np.random.default_rng(seed=seed)
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(N)))

    bounds = [(1e-10, 1.0 - 1e-10)] * (N - 1)

    melhor_D, melhor_w = 0.0, z.copy()
    for s, w0 in enumerate(starts):
        def obj(wf):
            w_full = np.append(wf, max(1e-15, 1.0 - wf.sum()))
            return tangent_plane_D(w_full, z, ln_gz, model)
        res = minimize(obj, w0[:-1], method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        w_full = np.append(res.x, max(1e-15, 1.0 - res.x.sum()))
        print(f"    start {s}: D = {res.fun:+.4e}   w = {np.round(w_full, 4)}")
        if res.fun < melhor_D:
            melhor_D, melhor_w = res.fun, w_full
    return melhor_D, melhor_w


if __name__ == '__main__':
    print("=" * 78)
    print("DIAGNOSTICO: feed Xu que falhou no controle")
    print("=" * 78)

    gw = JavaGateway(auto_field=True)

    # Usa o sistema [EMIM][PF6], primeira tie-line (a mais rica em IPAC,
    # caso mais "facil" de detectar -- se falhar aqui, falha em qualquer)
    nome_sistema = '[EMIM][PF6]'
    cfg = SISTEMAS_XU[nome_sistema]
    compostos = [IPAC, IPA, cfg['il']]
    model = COSMOSAC(compostos, gateway=gw)
    print(f"Sistema: {compostos}")

    x1_I, x2_I, x1_II, x2_II = cfg['tie_lines'][3]   # tie-line do meio
    print(f"\nFase I  (rica em IPAC) experimental: "
          f"x1={x1_I}, x2={x2_I}, x3={1-x1_I-x2_I:.4f}")
    print(f"Fase II (rica em IL)   experimental: "
          f"x1={x1_II}, x2={x2_II}, x3={1-x1_II-x2_II:.4f}")

    print("\n[1] D avaliado DIRETAMENTE nas composicoes experimentais")
    print("    (sem otimizacao -- testa se o COSMO-SAC 'acredita' no equilibrio)")
    D_II_de_I, D_I_de_II = D_nas_fases_experimentais(model, x1_I, x2_I, x1_II, x2_II)
    print(f"    D(faseII | referencia=faseI) = {D_II_de_I:+.4e}")
    print(f"    D(faseI  | referencia=faseII) = {D_I_de_II:+.4e}")
    if min(D_II_de_I, D_I_de_II) < -1e-3:
        print("    -> Negativo: o COSMO-SAC CONCORDA que ha 2 fases aqui.")
        print("       Se is_unstable deu False, o bug esta na BUSCA do minimizador.")
    else:
        print("    -> Proximo de zero/positivo: o COSMO-SAC NAO reproduz")
        print("       a separacao experimental neste ponto (problema de sigma-profile).")

    z = 0.5*np.array([x1_I, x2_I, 1-x1_I-x2_I]) + 0.5*np.array([x1_II, x2_II, 1-x1_II-x2_II])
    print(f"\n[2] Minimizacao instrumentada no feed (ponto medio da tie-line)")
    print(f"    z = {np.round(z, 4)}")
    D_min, w_min = minimizacao_instrumentada(model, z)
    print(f"\n    D_min GLOBAL encontrado pelo minimizador = {D_min:+.4e}")
    print(f"    is_unstable (threshold -5e-3) retornaria: {D_min < -5e-3}")

    print("\n" + "=" * 78)
    print("COMPARACAO:")
    print(f"  D nas fases experimentais (melhor dos dois): {min(D_II_de_I, D_I_de_II):+.4e}")
    print(f"  D_min achado pelo minimizador livre:          {D_min:+.4e}")
    if min(D_II_de_I, D_I_de_II) < -1e-3 and D_min > min(D_II_de_I, D_I_de_II) + 1e-3:
        print("\n  *** O minimizador NAO encontrou um minimo tao bom quanto a fase")
        print("      experimental conhecida. Isso e um BUG DE BUSCA, nao um")
        print("      problema do modelo COSMO-SAC. ***")
    print("=" * 78)