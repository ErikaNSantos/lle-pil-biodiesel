"""
diagnostico_modelo.py — O modelo "COSMO-SAC-HB2 (GAMESS)" e compativel com
                         sigma-profiles de catalogo?
================================================================================
HIPOTESE TESTADA
    stability_cosmo.py fixa SELECTED_MODEL = 'COSMO-SAC-HB2 (GAMESS)' porque
    foi o escolhido para os compostos CUSTOM do TCC (triolina, PILs), cujos
    sigma-profiles foram gerados por voce via GAMESS. Os compostos do Xu
    (acetato de isopropila, isopropanol, ILs imidazolicos) sao "de catalogo"
    -- seus sigma-profiles no banco do JCOSMO provavelmente vieram de OUTRO
    metodo/nivel de calculo (nao do seu fluxo GAMESS). Se o JCOSMO permitir
    "misturar" essas fontes sob o rotulo "COSMO-SAC-HB2 (GAMESS)" sem
    avisar, isso produziria erros grosseiros como o observado (D fortemente
    positivo onde deveria ser negativo) -- nao por falha de fisica, mas por
    descasamento de parametrizacao.

O QUE ESTE SCRIPT FAZ
    1. Lista todos os modelos disponiveis no JCOSMO (JCOSMO.listModels()).
    2. Para CADA modelo, repete o teste do diagnostico_xu.py: avalia D
       diretamente nas composicoes experimentais do Xu (fase I e fase II).
    3. Reporta qual(is) modelo(s) dao D negativo (= modelo concorda com o
       experimento) vs. positivo (= modelo discorda).

LEITURA
    - Se ALGUM modelo (que NAO seja "...GAMESS") der D negativo nas fases
      experimentais -> confirma a hipotese de incompatibilidade de
      parametrizacao. Solucao: usar esse modelo para compostos de
      catalogo, e o "(GAMESS)" so para os compostos custom -- ou verificar
      se ha uma forma de misturar consistentemente.
    - Se NENHUM modelo reproduzir o experimento -> o problema nao e so a
      escolha do modelo; pode ser composto mal identificado no JCOSMO
      (nome bateu com estrutura errada) ou limitacao mais profunda.

PRE-REQUISITOS: JCOSMO aberto; stability.py/gamma.py/params.py e
stability_cosmo.py na mesma pasta. Roda independente de validar_xu2022.py
(os dados da tie-line estao copiados aqui para nao depender de import).
"""

import re
import numpy as np
from py4j.java_gateway import JavaGateway

# Sistema e tie-line de teste (mesma tie-line do diagnostico_xu.py)
COMPOSTOS = ['ISOPROPYL_ACETATE', 'ISOPROPANOL', '[EMIM][PF6]']
T = 298.15
# Lista manual de fallback, caso a extracao via regex nao funcione. Inclui
# nomes ja vistos no despejo bruto anterior (UNIFAC (Do), UNIFAC-NIST) mais
# o modelo customizado usado no resto do TCC.
MODELOS_MANUAIS = [
    'COSMO-SAC-HB2 (GAMESS)',
    'UNIFAC (Do)',
    'UNIFAC-NIST',
]
# (x1_I, x2_I, x1_II, x2_II) -- Xu et al. 2022, Tabela 1, [Emim][PF6], TL4
X1_I, X2_I, X1_II, X2_II = 0.8247, 0.1713, 0.0847, 0.1945


def gamma_com_modelo(gateway, JCOSMO, nome_modelo, x):
    model = JCOSMO.newModel(nome_modelo)
    comps = gateway.new_array(gateway.jvm.java.lang.String, len(COMPOSTOS))
    for i, nome in enumerate(COMPOSTOS):
        comps[i] = nome
    model.setCompounds(comps)
    model.setTemperature(T)
    xj = gateway.new_array(gateway.jvm.double, len(COMPOSTOS))
    for i in range(len(COMPOSTOS)):
        xj[i] = float(x[i])
    model.setComposition(xj)
    lnG = model.activityCoefficientLn()
    return np.exp(np.array([float(lnG[i]) for i in range(len(COMPOSTOS))]))


def D_nas_fases_experimentais(gateway, JCOSMO, nome_modelo):
    xI  = np.array([X1_I,  X2_I,  1.0 - X1_I  - X2_I])
    xII = np.array([X1_II, X2_II, 1.0 - X1_II - X2_II])

    g_I  = gamma_com_modelo(gateway, JCOSMO, nome_modelo, xI)
    g_II = gamma_com_modelo(gateway, JCOSMO, nome_modelo, xII)

    ln_a_I  = np.log(xI)  + np.log(g_I)
    ln_a_II = np.log(xII) + np.log(g_II)

    D_em_II_usando_I = float(np.sum(xII * (ln_a_II - ln_a_I)))
    D_em_I_usando_II = float(np.sum(xI  * (ln_a_I  - ln_a_II)))
    return D_em_II_usando_I, D_em_I_usando_II, g_I, g_II


if __name__ == '__main__':
    print("=" * 84)
    print("DIAGNOSTICO: o modelo 'COSMO-SAC-HB2 (GAMESS)' e compativel com")
    print("sigma-profiles de catalogo (acetato de isopropila/isopropanol/IL)?")
    print("=" * 84)

    gw = JavaGateway(auto_field=True)
    JCOSMO = gw.entry_point

    try:
        raw = JCOSMO.listModels()
    except Exception as e:
        print(f"ERRO: nao foi possivel listar modelos do JCOSMO: {e}")
        raise SystemExit(1)

    raw_str = str(raw)
    print("\n--- Saida BRUTA de JCOSMO.listModels() (primeiros 2000 caracteres) ---")
    print(raw_str[:2000])
    print("--- fim do trecho bruto ---\n")

    # listModels() aparentemente retorna uma STRING (nomes entre aspas simples
    # seguidos de referencia/DOI), nao uma lista de nomes. Extrai os nomes via
    # regex em vez de iterar a string caractere por caractere.
    candidatos = re.findall(r"'([^']+)'", raw_str)
    # Remove duplicatas preservando ordem
    vistos = set()
    modelos = []
    for c in candidatos:
        if c not in vistos:
            vistos.add(c)
            modelos.append(c)

    if not modelos:
        print("Nenhum nome de modelo extraido via regex de aspas simples.")
        print("Ajuste manualmente a lista MODELOS_MANUAIS abaixo com os nomes")
        print("exatos vistos no trecho bruto acima (ex.: copiar de 'COSMO-SAC-HB2 (GAMESS)').")
        modelos = MODELOS_MANUAIS

    print(f"\nModelos disponiveis no JCOSMO ({len(modelos)}):")
    for m in modelos:
        print(f"  - {m}")

    print(f"\nSistema de teste: {COMPOSTOS}")
    print(f"Fase I  exp.: x=({X1_I}, {X2_I}, {1-X1_I-X2_I:.4f})")
    print(f"Fase II exp.: x=({X1_II}, {X2_II}, {1-X1_II-X2_II:.4f})")
    print(f"\n{'Modelo':<35} {'D(II|ref=I)':>13} {'D(I|ref=II)':>13} {'concorda?':>10}")
    print("-" * 84)

    resultados = []
    for nome_modelo in modelos:
        try:
            D_II, D_I, g_I, g_II = D_nas_fases_experimentais(gw, JCOSMO, nome_modelo)
            concorda = "SIM" if min(D_II, D_I) < -1e-3 else "nao"
            print(f"{nome_modelo:<35} {D_II:>13.4f} {D_I:>13.4f} {concorda:>10}")
            resultados.append((nome_modelo, D_II, D_I, concorda))
        except Exception as e:
            msg = str(e).split('\n')[0][:60]
            print(f"{nome_modelo:<35} {'ERRO: ' + msg}")
            resultados.append((nome_modelo, None, None, 'ERRO'))

    print("-" * 84)
    concordam = [r for r in resultados if r[3] == 'SIM']
    print(f"\n{len(concordam)}/{len(resultados)} modelos concordam com o experimento "
          f"(D negativo nas fases reportadas).")

    if concordam:
        print("\nLEITURA: ha modelo(s) compativel(is) com sigma-profiles de catalogo:")
        for nome_modelo, D_II, D_I, _ in concordam:
            print(f"  - {nome_modelo}")
        print("Isso aponta para descasamento de parametrizacao no modelo")
        print("'COSMO-SAC-HB2 (GAMESS)' quando usado com compostos de catalogo,")
        print("nao para falha geral do COSMO-SAC. Considere usar um desses")
        print("modelos para validacoes com compostos de catalogo, e reservar")
        print("'(GAMESS)' apenas para os compostos custom (triolina/PILs) cujos")
        print("sigma-profiles vieram do seu proprio fluxo GAMESS.")
    else:
        print("\nLEITURA: NENHUM modelo disponivel reproduz a separacao experimental")
        print("para este sistema. Verificar se os nomes dos compostos bateram com")
        print("a estrutura correta no JCOSMO (ex.: 'ISOPROPYL ACETATE' pode ter")
        print("casado com um isomero ou composto errado).")
    print("=" * 84)