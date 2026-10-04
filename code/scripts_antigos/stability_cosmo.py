"""
stability_cosmo.py — Teste de estabilidade de fases (TPD) com gamma do COSMO-SAC
=================================================================================
PROPOSITO
    Reaproveitar EXATAMENTE o criterio do plano tangente (TPD) ja validado
    em stability.py, trocando apenas a FONTE dos coeficientes de atividade:
    em vez do UNIFAC implementado em Python (gamma.UNIFACLL), os gamma_i vem
    do COSMO-SAC calculado pelo JCOSMO, acessado via py4j.

    A logica do TPD (Baker-Pierce-Luks / Michelsen) NAO muda. As funcoes
    is_unstable() e tangent_plane_D() sao importadas de stability.py sem
    nenhuma alteracao; este modulo so fornece um objeto `model` com o
    metodo .gamma(x) compativel com a assinatura que elas esperam
    (e que flash.py tambem usa).

POR QUE ISSO CORRIGE O PIPELINE COSMO-SAC ANTERIOR
    Os scripts de varredura/bissecao anteriores marcavam a fronteira de
    fases onde DeltaGmix/RT trocava de sinal. Esse NAO e o criterio de
    estabilidade: DeltaGmix e tipicamente NEGATIVO dentro de toda a regiao
    bifasica, de modo que o contorno DeltaGmix=0 erra a binodal real e
    ignora as separacoes que ocorrem com DeltaGmix<0 (o caso usual).
    O criterio correto e o do plano tangente:

        z e INSTAVEL  <=>  existe w no simplex com D(w) < 0
        D(w) = sum_i w_i * [ln a_i(w) - ln a_i(z)],  a_i = x_i * gamma_i

    A unica diferenca em relacao ao UNIFAC e de onde vem gamma_i(.).

PRE-REQUISITOS
    1. JCOSMO ABERTO (GUI rodando) — funciona como servidor py4j.
    2. stability.py, gamma.py e params.py na mesma pasta (a logica do TPD
       e importada de stability.py; gamma/params sao dependencias dele).
    3. py4j instalado.

USO RAPIDO
    from stability_cosmo import construir_modelo_LI
    from stability import is_unstable
    import numpy as np

    model = construir_modelo_LI('[2HEA][HX]')          # JCOSMO aberto
    z = np.array([0.10, 0.05, 0.85])                   # [triolina, etanol, LI]
    print(is_unstable(z, model))                       # True -> bifasico

REFERENCIAS
    Baker, Pierce & Luks (1982) SPE J. 22, 731.
    Michelsen (1982) Fluid Phase Equilib. 9, 1; 9, 21.
"""

import time
import numpy as np
from py4j.java_gateway import JavaGateway

# A logica do plano tangente e REUTILIZADA sem modificacao alguma.
# (importar stability tambem carrega gamma/params, que precisam estar na pasta)
from stability import is_unstable, tangent_plane_D


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
SELECTED_MODEL = 'COSMO-SAC-HB2 (GAMESS)'
TEMPERATURE    = 298.15
X_MIN          = 1e-8   # piso por componente antes de enviar ao JCOSMO

# Nomes dos componentes no banco do JCOSMO (devem bater exatamente)
ETANOL    = 'ETHANOL'
TRIOLINA  = 'OLEIC_ACID'
LI_2HEA   = '2HEAHX'
LI_DETA   = '[DETA][HX]'


# ===========================================================================
# ADAPTADOR: gamma do COSMO-SAC com a mesma interface do UNIFACLL
# ===========================================================================

class COSMOSAC:
    """
    Expoe .gamma(x) -> np.ndarray a partir do JCOSMO, com a MESMA assinatura
    usada por stability.is_unstable() e flash.flash_ell().

    ORDEM DOS COMPONENTES
        `compostos` define a ordem. O vetor x passado a .gamma(x) deve
        seguir a MESMA ordem, e os gamma retornam nessa ordem. Para manter
        consistencia com o pipeline UNIFAC do TCC, use:
            ['TRIOLEIN', 'ETHANOL', '<LI>']
        (componente 1 = triolina, 2 = etanol, 3 = LI).

    DESEMPENHO
        Cada gamma e uma chamada ao JCOSMO (cara). Duas otimizacoes:
        - Compostos e temperatura sao definidos UMA vez no construtor
          (os scripts antigos re-setavam a cada chamada, sem necessidade).
        - Cache por composicao arredondada: o minimizador L-BFGS-B do TPD
          reavalia pontos vizinhos repetidamente; o cache evita recomputar.
        Se em algum cenario o JCOSMO exigir re-set de compostos a cada
        avaliacao, passe reseta_sempre=True (volta ao comportamento antigo).
    """

    def __init__(self, compostos, T=TEMPERATURE, modelo=SELECTED_MODEL,
                 gateway=None, usar_cache=True, reseta_sempre=False):
        self.compostos = list(compostos)
        self.n = len(self.compostos)
        self.T = float(T)
        self.modelo = modelo
        self.reseta_sempre = reseta_sempre

        # Conexao py4j (pode ser compartilhada entre instancias)
        self.gateway = gateway or JavaGateway(auto_field=True)
        JCOSMO = self.gateway.entry_point
        try:
            JCOSMO.listModels()
        except Exception as e:
            raise RuntimeError(
                "Nao foi possivel conectar ao JCOSMO. "
                "Abra a GUI do JCOSMO antes de rodar este script."
            ) from e

        self.model = JCOSMO.newModel(modelo)
        self._aplicar_compostos()        # define compostos + T uma vez

        self._cache = {} if usar_cache else None
        self.n_calls = 0                 # contador para diagnostico de custo

    # ---- internos -------------------------------------------------------

    def _aplicar_compostos(self):
        comps = self.gateway.new_array(
            self.gateway.jvm.java.lang.String, self.n)
        for i, nome in enumerate(self.compostos):
            comps[i] = nome
        self.model.setCompounds(comps)
        self.model.setTemperature(self.T)

    def _key(self, x):
        # gamma e suave o bastante para diferencas < 1e-9 serem irrelevantes
        return tuple(np.round(x, 9))

    # ---- interface publica ----------------------------------------------

    def gamma(self, x):
        """Coeficientes de atividade gamma na composicao x (mesma ordem de compostos)."""
        x = np.asarray(x, dtype=float)

        # Piso + renormalizacao: fracoes quase-nulas quebram o solver
        # interno do JCOSMO (NaN norm in solveGamma).
        x = np.clip(x, X_MIN, None)
        x = x / x.sum()

        if self._cache is not None:
            k = self._key(x)
            hit = self._cache.get(k)
            if hit is not None:
                return hit

        if self.reseta_sempre:
            self._aplicar_compostos()

        xj = self.gateway.new_array(self.gateway.jvm.double, self.n)
        for i in range(self.n):
            xj[i] = float(x[i])
        self.model.setComposition(xj)

        lnG = self.model.activityCoefficientLn()
        g = np.exp(np.array([float(lnG[i]) for i in range(self.n)], dtype=float))
        self.n_calls += 1

        if self._cache is not None:
            self._cache[k] = g
        return g


# ===========================================================================
# Construtores de conveniencia para os sistemas-alvo do TCC
# ===========================================================================

def construir_modelo_LI(li_nome, T=TEMPERATURE, gateway=None, **kw):
    """
    Cria o COSMOSAC para {triolina + etanol + LI}, na MESMA ordem do
    pipeline UNIFAC (comp.1 = triolina, comp.2 = etanol, comp.3 = LI).
    """
    return COSMOSAC([TRIOLINA, ETANOL, li_nome], T=T, gateway=gateway, **kw)


# ===========================================================================
# VALIDACAO
# ===========================================================================
# Os testes abaixo separam DOIS niveis, exatamente como em stability.py:
#   (A) Testes ESTRUTURAIS — propriedades matematicas do TPD, independem
#       de dado experimental. Confirmam que o elo com o JCOSMO funciona e
#       que a algebra do plano tangente esta correta.
#   (B) Teste-CONTROLE com dado experimental (Xu 2022) — confirma que o
#       criterio corrigido detecta como bifasico um sistema sabidamente
#       bifasico. Precisa dos nomes JCOSMO e de >=1 feed na regiao de 2 fases.
# ===========================================================================

def autoteste_tpd(model, nome=''):
    """(A) Testes estruturais — nao dependem de dados experimentais."""
    print(f"\n--- Autoteste TPD estrutural {('('+nome+')') if nome else ''} ---")

    # T1: identidade D(w=z) = 0 por construcao
    print("T1: identidade D(w=z) = 0   (criterio |D| < 1e-9)")
    zs = [
        np.array([0.34, 0.33, 0.33]),   # equimolar
        np.array([0.80, 0.10, 0.10]),   # rico em triolina
        np.array([0.05, 0.10, 0.85]),   # rico em LI
    ]
    ok1 = 0
    for z in zs:
        z = z / z.sum()
        ln_gz = np.log(z) + np.log(model.gamma(z))
        D_self = tangent_plane_D(z, z, ln_gz, model)
        bom = abs(D_self) < 1e-9
        ok1 += bom
        print(f"    [{'OK' if bom else 'FALHOU'}] z={np.round(z,2)}  D={D_self:+.2e}")
    print(f"    {ok1}/{len(zs)} aprovados")

    # T2: composicao quase-pura e estavel (impossivel separar)
    print("T2: componente quase-puro -> estavel   (is_unstable deve dar False)")
    ok2 = 0
    for x1 in [0.999, 0.995, 0.990]:
        xr = (1 - x1) / 2
        z = np.array([x1, xr, xr])
        inst = is_unstable(z, model)
        bom = not inst
        ok2 += bom
        print(f"    [{'OK' if bom else 'FALHOU'}] x_triolina={x1}  bifasico={inst}")
    print(f"    {ok2}/3 aprovados")

    return ok1 == len(zs) and ok2 == 3


def mapa_estabilidade(model, n_random=2):
    """(A) Mapa exploratorio B/M no simplex (coarse, so para inspecao)."""
    print("\n--- Mapa de estabilidade (B=bifasico, M=monofasico) ---")
    print(f"  {'x_triol':>8} | x_etanol: {0.10:>4} {0.30:>4} {0.50:>4}")
    print(f"  {'-'*8}-+-{'-'*20}")
    t0 = time.time()
    for x1 in [0.10, 0.30, 0.60, 0.85]:
        linha = f"  {x1:>8.2f} |          "
        for x2 in [0.10, 0.30, 0.50]:
            x3 = 1.0 - x1 - x2
            if x3 < 0.02:
                linha += f" {'.':>4}"
                continue
            z = np.array([x1, x2, x3]); z /= z.sum()
            inst = is_unstable(z, model, n_random=n_random)
            linha += f" {'B' if inst else 'M':>4}"
        print(linha)
    print(f"  tempo: {time.time()-t0:.1f}s   chamadas JCOSMO: {model.n_calls}")


def validar_experimental(model, feeds_bifasicos, nome='sistema'):
    """
    (B) Para cada composicao global SABIDAMENTE bifasica (dado experimental),
    o TPD corrigido deve retornar is_unstable == True.

    feeds_bifasicos : lista de composicoes z (mesma ordem dos compostos do
                      model), tipicamente o ponto medio de tie-lines
                      experimentais conhecidas como de 2 fases.
    """
    print(f"\n--- Validacao experimental: {nome} ---")
    print("  Esperado: is_unstable == True para todo feed bifasico")
    ok = 0
    for z in feeds_bifasicos:
        z = np.asarray(z, float); z = z / z.sum()
        inst = is_unstable(z, model)
        ok += bool(inst)
        print(f"    [{'OK' if inst else 'FALHOU'}] z={np.round(z,3)}  bifasico={inst}")
    print(f"  {ok}/{len(feeds_bifasicos)} aprovados")
    return ok == len(feeds_bifasicos)


# ---------------------------------------------------------------------------
# BLOCO XU (2022) — PREENCHER PARA FECHAR A VALIDACAO-CONTROLE
# ---------------------------------------------------------------------------
# Para o teste-controle do Xu sao necessarios:
#   (1) os NOMES EXATOS dos componentes do sistema do Xu no banco do JCOSMO
#       (na ordem em que voce vai montar o vetor z);
#   (2) ao menos uma composicao global z DENTRO da regiao de 2 fases
#       (idealmente o ponto medio de uma tie-line experimental do Xu).
# Enquanto (1)/(2) nao forem preenchidos, este bloco fica como esqueleto.
XU_COMPOSTOS       = None   # ex.: ['<comp1>', '<comp2>', '<comp3>']
XU_FEEDS_BIFASICOS = []     # ex.: [[0.40, 0.40, 0.20], [0.30, 0.50, 0.20]]


# ===========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("VALIDACAO: stability_cosmo.py  (TPD com gamma do COSMO-SAC/JCOSMO)")
    print("=" * 70)

    # Uma unica conexao py4j compartilhada entre os modelos
    gw = JavaGateway(auto_field=True)

    # ---- (A) Testes estruturais nos dois sistemas-alvo do TCC ----
    for li in (LI_2HEA, LI_DETA):
        model = construir_modelo_LI(li, gateway=gw)
        autoteste_tpd(model, nome=li)
        mapa_estabilidade(model)

    # ---- (B) Teste-controle Xu (2022) ----
    print("\n" + "=" * 70)
    if XU_COMPOSTOS and XU_FEEDS_BIFASICOS:
        xu = COSMOSAC(XU_COMPOSTOS, gateway=gw)
        validar_experimental(xu, XU_FEEDS_BIFASICOS, nome='Xu (2022)')
    else:
        print("BLOCO XU NAO PREENCHIDO.")
        print("Para fechar a validacao-controle, defina no topo do arquivo:")
        print("  XU_COMPOSTOS       = nomes JCOSMO dos componentes do Xu")
        print("  XU_FEEDS_BIFASICOS = >=1 composicao z na regiao de 2 fases")
        print("Sem isso, so os testes estruturais (A) foram executados.")
    print("=" * 70)
