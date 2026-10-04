"""
validar_xu2022.py — Controle com sistema de CATALOGO (sem GAMESS customizado)
================================================================================
PROPOSITO
    Decidir entre duas hipoteses para qualquer discrepancia do COSMO-SAC no
    pipeline do TCC:
      H1: o CODIGO/pipeline (stability.py + stability_cosmo.py) tem um erro
          generico -> deveria falhar tambem aqui.
      H2: o problema esta nos COMPOSTOS CUSTOM (triolina, PILs) que exigiram
          o fluxo GAMESS/.Gout proprio -> aqui deveria passar, pois IPAC,
          IPA e os ILs imidazolicos abaixo sao moleculas "de catalogo",
          normalmente ja disponiveis nas bases padrao de sigma-profile do
          COSMO-SAC/JCOSMO, sem qualquer parametrizacao propria.

SISTEMA DE REFERENCIA
    Xu, D. et al. (2022) J. Taiwan Inst. Chem. Eng. 140, 104571.
    {Isopropyl acetate (IPAC) + Isopropanol (IPA) + IL} a 298.15 K, 0.1 MPa.
    Tres ILs testados: [Bmim][H2PO4], [Bmim][PF6], [Emim][PF6].
    Mesma temperatura do TCC (298.15 K); mesma familia de problema (eter/
    ester + alcool + IL imidazolico). Dados de tie-line direto da Tabela 1
    do artigo.

COMO OS FEEDS FORAM CONSTRUIDOS
    Para cada tie-line experimental, z = 0.5*(fase I) + 0.5*(fase II).
    Isso GARANTE que z esta na regiao de 2 fases (e o ponto medio de uma
    tie-line real medida), entao is_unstable(z, model) DEVE retornar True
    se o pipeline COSMO-SAC estiver funcionando para este sistema.

ACAO NECESSARIA ANTES DE RODAR
    Os nomes em XU_COMPOSTOS abaixo sao MELHORES PALPITES baseados na
    nomenclatura usual de bases de sigma-profile (ex.: VT-2005). Antes de
    rodar, verifique no JCOSMO (GUI, caixa de busca de compostos) os nomes
    EXATOS para:
      - Isopropyl acetate (acetato de isopropila)
      - Isopropanol / 2-Propanol
      - Os cations/anions do IL escolhido (provavelmente jah cadastrados
        como [EMIM][PF6] ou EMIM+ / PF6- separados, dependendo de como o
        JCOSMO organiza ILs de catalogo)
    Ajuste as strings em XU_COMPOSTOS_* se o JCOSMO nao reconhecer o nome
    (erro tipico: "sigma profile not found").

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta.
"""

import numpy as np
from py4j.java_gateway import JavaGateway

from stability_cosmo import COSMOSAC, validar_experimental


# ---------------------------------------------------------------------------
# NOMES DOS COMPOSTOS NO JCOSMO — VERIFICAR/AJUSTAR ANTES DE RODAR
# ---------------------------------------------------------------------------
IPAC = 'ISOPROPYL_ACETATE'      # <-- confirmar no JCOSMO
IPA  = 'ISOPROPANOL'            # <-- confirmar (pode ser '2-PROPANOL')

# Tres sistemas da Tabela 1 do Xu et al. (2022). Comece pelo [EMIM][PF6]
# (cation menor, sistema mais simples) se quiser rodar so um.
SISTEMAS_XU = {
    '[EMIM][PF6]': {
        'il': '[EMIM][PF6]',    # <-- confirmar nome exato no JCOSMO
        # Tabela 1, colunas (x1_I, x2_I, x1_II, x2_II) por tie-line:
        'tie_lines': [
            (0.9738, 0.0252, 0.0589, 0.0334),
            (0.9307, 0.0673, 0.0713, 0.0838),
            (0.8827, 0.1147, 0.0788, 0.1374),
            (0.8247, 0.1713, 0.0847, 0.1945),
            (0.7627, 0.2315, 0.0933, 0.2496),
            (0.7045, 0.2853, 0.1041, 0.3016),
            (0.6333, 0.3459, 0.1141, 0.3586),
            (0.5551, 0.4137, 0.1248, 0.4222),
        ],
    },
    '[BMIM][PF6]': {
        'il': '[BMIM][PF6]',    # <-- confirmar nome exato no JCOSMO
        'tie_lines': [
            (0.8759, 0.1240, 0.1401, 0.1408),
            (0.8177, 0.1821, 0.1527, 0.1988),
            (0.7670, 0.2318, 0.1620, 0.2435),
            (0.7232, 0.2747, 0.1761, 0.2860),
            (0.6749, 0.3212, 0.1961, 0.3307),
            (0.6268, 0.3690, 0.2272, 0.3760),
            (0.5767, 0.4086, 0.2569, 0.4141),
            (0.5313, 0.4410, 0.2830, 0.4420),
        ],
    },
    '[BMIM][H2PO4]': {
        'il': '[BMIM][H2PO4]',  # <-- confirmar nome exato no JCOSMO
        'tie_lines': [
            (0.9209, 0.0790, 0.0869, 0.1972),
            (0.8894, 0.1104, 0.0898, 0.2552),
            (0.8480, 0.1518, 0.0930, 0.3088),
            (0.8175, 0.1823, 0.0975, 0.3450),
            (0.7871, 0.2116, 0.0992, 0.3733),
            (0.7535, 0.2446, 0.1037, 0.4024),
            (0.7256, 0.2720, 0.1122, 0.4288),
            (0.6986, 0.2982, 0.1324, 0.4586),
            (0.6689, 0.3272, 0.1551, 0.4959),
            (0.6395, 0.3550, 0.1797, 0.5282),
            (0.6108, 0.3817, 0.2072, 0.5556),
        ],
    },
}


def feeds_a_partir_de_tie_lines(tie_lines):
    """z = ponto medio de cada tie-line experimental (garantidamente bifasico)."""
    feeds = []
    for x1_I, x2_I, x1_II, x2_II in tie_lines:
        x3_I  = 1.0 - x1_I  - x2_I
        x3_II = 1.0 - x1_II - x2_II
        z = 0.5 * np.array([x1_I, x2_I, x3_I]) + 0.5 * np.array([x1_II, x2_II, x3_II])
        feeds.append(z)
    return feeds


if __name__ == '__main__':
    print("=" * 78)
    print("VALIDACAO-CONTROLE: Xu et al. (2022) — sistema de CATALOGO")
    print("(sem GAMESS customizado — isola codigo vs. composto custom)")
    print("=" * 78)

    gw = JavaGateway(auto_field=True)
    resumo = {}

    for nome_sistema, cfg in SISTEMAS_XU.items():
        compostos = [IPAC, IPA, cfg['il']]
        feeds = feeds_a_partir_de_tie_lines(cfg['tie_lines'])

        print(f"\n>>> Sistema: IPAC + IPA + {nome_sistema}")
        print(f"    Compostos no JCOSMO: {compostos}")
        try:
            model = COSMOSAC(compostos, gateway=gw)
        except Exception as e:
            print(f"    ERRO ao montar o modelo (provavel nome de composto "
                  f"incorreto): {e}")
            resumo[nome_sistema] = 'ERRO_SETUP'
            continue

        ok = validar_experimental(model, feeds, nome=nome_sistema)
        resumo[nome_sistema] = 'PASSOU' if ok else 'FALHOU'

    print("\n" + "=" * 78)
    print("RESUMO")
    print("=" * 78)
    for nome_sistema, status in resumo.items():
        print(f"  {nome_sistema:20s}  {status}")

    n_pass = sum(1 for s in resumo.values() if s == 'PASSOU')
    n_total = len(resumo)
    print(f"\n{n_pass}/{n_total} sistemas de catalogo passaram no controle.")
    print()
    if n_pass == n_total and n_total > 0:
        print("LEITURA: o pipeline COSMO-SAC (codigo) esta correto para sistemas")
        print("de catalogo. Qualquer discrepancia observada nos sistemas do TCC")
        print("(triolina/PILs) e mais provavel de vir dos sigma-profiles CUSTOM")
        print("(fluxo GAMESS/.Gout), nao do codigo de estabilidade/flash.")
    elif n_pass == 0 and n_total > 0:
        print("LEITURA: falha tambem em sistema 100% de catalogo. Isso aponta")
        print("para um problema no PIPELINE (codigo, conexao py4j, ou parametros")
        print("do JCOSMO), nao especifico dos compostos do TCC. Revisar")
        print("stability_cosmo.py antes de confiar em qualquer resultado triolina/PIL.")
    else:
        print("LEITURA: resultado misto entre os tres ILs — investigar se a falha")
        print("se correlaciona com algum aniao/cation especifico antes de concluir.")
    print("=" * 78)
