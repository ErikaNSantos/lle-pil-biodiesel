"""
diagnostico_tpd.py — Por que is_unstable() dispara em composicao quase-pura?
============================================================================
SINTOMA
    T2 do stability_cosmo falhou: x_triolina=0.999 deu bifasico=True, e o
    mapa inteiro deu B. Isso e FISICAMENTE IMPOSSIVEL (uma mistura 99,9%
    pura nao separa), logo o teste TPD esta MIS-DISPARANDO — nao e
    resultado real do COSMO-SAC.

POR QUE
    Num ponto estavel, o minimo GLOBAL do TPD e exatamente 0 (toca o plano
    em w=z). Qualquer micro-irregularidade na superficie de gamma faz o
    minimizador achar um w com D ligeiramente < 0. Se esse "ligeiramente"
    for maior que o guarda D_threshold=-5e-3, tudo vira instavel.

DUAS HIPOTESES (levam a consertos OPOSTOS — por isso medimos antes):
    H1 (referencia/estrutural): gamma_i(puro) != 1, ou sigma-profile
        inconsistente (historico do .Gout). Mascarar com threshold seria
        errado.
    H2 (consistencia/ruido): gamma do JCOSMO viola levemente Gibbs-Duhem
        (solver iterativo); o conserto legitimo e calibrar o threshold
        acima do piso de ruido.

O QUE ESTE SCRIPT MEDE
    (1) gamma nos tres vertices quase-puros -> testa H1
        criterio: gamma do PROPRIO componente puro deve ser ~1.0
    (2) o D_min GLOBAL e o w que o atinge, para um z quase-puro -> testa H2
        leitura:  |D_min| ~ 1e-3..1e-2  -> ruido/consistencia (H2)
                  |D_min| >> 1e-1  ou  gamma_puro != 1  -> estrutural (H1)

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta.
"""

import numpy as np
from scipy.optimize import minimize
from py4j.java_gateway import JavaGateway

from stability import tangent_plane_D
from stability_cosmo import construir_modelo_LI, LI_2HEA


def gamma_nos_vertices(model):
    """(1) Testa H1: gamma do componente puro deve ser ~1."""
    print("\n[1] gamma nos vertices quase-puros (testa referencia / H1)")
    print("    criterio: gamma do PROPRIO componente deve ser ~1.0000")
    eps = 1e-8
    for i, nome in enumerate(model.compostos):
        x = np.full(model.n, eps)
        x[i] = 1.0 - eps * (model.n - 1)
        x = x / x.sum()
        g = model.gamma(x)
        flag = "OK" if abs(g[i] - 1.0) < 1e-2 else "<-- SUSPEITO"
        print(f"    puro {nome:12s}: gamma_proprio = {g[i]:.4f}  {flag}")
        print(f"        vetor completo = {np.round(g, 3)}")


def diagnostico_min(model, z, n_random=3, D_threshold=-5e-3):
    """
    (2) Testa H2: reproduz EXATAMENTE a minimizacao de is_unstable
    (mesmos starts, bounds, metodo e opcoes), mas instrumentada para
    imprimir o D_min e o w de cada start e o minimo global.
    """
    z = np.clip(z, 1e-15, 1.0); z = z / z.sum()
    N = len(z)
    ln_gz = np.log(z) + np.log(model.gamma(z))

    # starts identicos a stability.is_unstable
    starts = []
    for i in range(N):
        w = np.ones(N) * 0.02
        w[i] = 1.0 - 0.02 * (N - 1)
        starts.append(w / w.sum())
    rng = np.random.default_rng(seed=42)
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(N)))

    bounds = [(1e-10, 1.0 - 1e-10)] * (N - 1)

    print(f"\n[2] Minimizacao do TPD instrumentada (testa ruido / H2)")
    print(f"    z = {np.round(z, 4)}    D_threshold = {D_threshold}")

    melhor_D, melhor_w = 0.0, z.copy()
    for s, w0 in enumerate(starts):
        def obj(wf):
            w_full = np.append(wf, max(1e-15, 1.0 - wf.sum()))
            return tangent_plane_D(w_full, z, ln_gz, model)
        res = minimize(obj, w0[:-1], method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        w_full = np.append(res.x, max(1e-15, 1.0 - res.x.sum()))
        print(f"    start {s}: D_min = {res.fun:+.3e}   w = {np.round(w_full, 4)}")
        if res.fun < melhor_D:
            melhor_D, melhor_w = res.fun, w_full

    print(f"    --> D_min GLOBAL = {melhor_D:+.3e}  em  w = {np.round(melhor_w, 4)}")
    print(f"        is_unstable retornaria: {melhor_D < D_threshold}")
    print(f"        |D_min| = {abs(melhor_D):.3e}")
    if abs(melhor_D) < 2e-2:
        print("        leitura: profundidade pequena -> compativel com H2 (ruido)")
    else:
        print("        leitura: profundidade GRANDE -> compativel com H1 (estrutural)")
    return melhor_D, melhor_w


if __name__ == '__main__':
    print("=" * 70)
    print("DIAGNOSTICO DO FALSO POSITIVO DO TPD (COSMO-SAC / JCOSMO)")
    print("=" * 70)

    gw = JavaGateway(auto_field=True)
    model = construir_modelo_LI(LI_2HEA, gateway=gw)
    print(f"Sistema: {model.compostos}")

    # (1) referencia
    gamma_nos_vertices(model)

    # (2) profundidade do D_min em pontos que DEVERIAM ser estaveis
    for z in ([0.999, 0.0005, 0.0005],     # quase-puro triolina (impossivel separar)
              [0.34, 0.33, 0.33]):          # equimolar
        diagnostico_min(model, np.array(z))

    print("\n" + "=" * 70)
    print("COMO LER:")
    print("  gamma_proprio != 1  OU  |D_min| >> 0.1  -> H1 estrutural")
    print("        (sigma-profile / .Gout / referencia — NAO mascarar com threshold)")
    print("  gamma_proprio ~ 1   E   |D_min| ~ 1e-3..1e-2 -> H2 ruido")
    print("        (calibrar D_threshold acima desse piso resolve)")
    print("=" * 70)
