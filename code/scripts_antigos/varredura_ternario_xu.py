"""
varredura_ternario_xu.py — O COSMO-SAC ve bifasico em ALGUM lugar deste
                            ternario, ou em lugar nenhum?
================================================================================
CONTEXTO
    diagnostico_modelo.py mostrou que os dois modelos que aceitam estes
    compostos (COSMO-SAC-HB2 GAMESS e COSMO-SAC puro) concordam entre si:
    ambos dao D positivo na tie-line experimental especifica testada. Isso
    mata a hipotese de descasamento de modelo, mas deixa em aberto uma
    pergunta importante: o COSMO-SAC esta com a fronteira de fases
    DESLOCADA (ve bifasico, mas em outro lugar do ternario -- erro de
    posicionamento, como o documentado na literatura para alguns ILs em
    Shah et al. 2012) ou simplesmente NAO VE separacao de fases nesse
    sistema (falha mais severa)?

O QUE ESTE SCRIPT FAZ
    Varre todo o simplex IPAC+IPA+IL com o is_unstable() ja validado
    (mesmo TPD usado no resto do TCC) e reporta se existe QUALQUER regiao
    bifasica, e onde.

LEITURA
    - Existe regiao bifasica (mesmo que deslocada da tie-line do Xu) ->
      erro QUANTITATIVO/de posicionamento, precedente documentado na
      literatura para COSMO-SAC com ILs (Shah 2012). Resultado defensavel,
      apresentavel como limitacao conhecida do modelo.
    - NENHUMA composicao testada e instavel -> falha mais severa (o
      modelo nao "ve" a fisica deste par de jeito nenhum); aí vale suspeitar
      de identidade do composto carregado (isomero errado, IL com atribuicao
      de carga estranha) antes de aceitar como limitacao do COSMO-SAC.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta.
"""

import numpy as np
from py4j.java_gateway import JavaGateway

from stability import is_unstable
from stability_cosmo import COSMOSAC

COMPOSTOS = ['ISOPROPYL_ACETATE', 'ISOPROPANOL', '[EMIM][PF6]']
N_GRID = 24   # resolucao do simplex (passo ~1/24 ~ 0.042)
X_MIN = 0.02  # evita vertices degenerados


def gerar_grade_triangular(n, x_min):
    pontos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            x1, x2, x3 = i / n, j / n, k / n
            if min(x1, x2, x3) < x_min:
                continue
            pontos.append((x1, x2, x3))
    return pontos


if __name__ == '__main__':
    print("=" * 78)
    print("VARREDURA TERNARIA: IPAC + IPA + [EMIM][PF6] — existe bifasico em")
    print("ALGUM lugar segundo o COSMO-SAC?")
    print("=" * 78)

    gw = JavaGateway(auto_field=True)
    model = COSMOSAC(COMPOSTOS, gateway=gw)
    print(f"Compostos: {COMPOSTOS}")

    pontos = gerar_grade_triangular(N_GRID, X_MIN)
    print(f"Pontos no simplex: {len(pontos)}")

    bifasicos = []
    for (x1, x2, x3) in pontos:
        z = np.array([x1, x2, x3])
        if is_unstable(z, model, n_random=2):
            bifasicos.append(z)

    print(f"\nPontos bifasicos encontrados: {len(bifasicos)}/{len(pontos)}")
    print(f"Chamadas ao JCOSMO: {model.n_calls}")

    if bifasicos:
        arr = np.array(bifasicos)
        print(f"\nFaixa da regiao bifasica encontrada:")
        print(f"  x_IPAC: [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}]")
        print(f"  x_IPA:  [{arr[:,1].min():.3f}, {arr[:,1].max():.3f}]")
        print(f"  x_IL:   [{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]")
        print("\nComparar com a tie-line experimental do Xu (TL4):")
        print("  faseI  x=(0.825, 0.171, 0.004)   faseII x=(0.085, 0.195, 0.721)")
        print("\nLEITURA: existe regiao bifasica segundo o COSMO-SAC. Verificar")
        print("se ela esta na vizinhanca da regiao experimental (erro de")
        print("posicionamento, precedente conhecido na literatura) ou em uma")
        print("zona completamente diferente do simplex (erro mais severo).")
    else:
        print("\nLEITURA: NENHUMA composicao testada no simplex foi instavel.")
        print("O COSMO-SAC, com estes sigma-profiles, preve este sistema como")
        print("totalmente miscivel em qualquer composicao -- nao e so um erro")
        print("de posicionamento da fronteira, e falha mais severa. Antes de")
        print("aceitar isso como limitacao do modelo, verificar a identidade")
        print("exata dos compostos carregados no JCOSMO (isomero do acetato,")
        print("atribuicao de carga do par EMIM+/PF6-).")
    print("=" * 78)