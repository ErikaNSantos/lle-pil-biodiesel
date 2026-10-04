#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analise_cossolvencia.py

Extrai, dos CSVs de linhas de amarracao gerados pelo batch_runner, as duas
metricas necessarias a Secao 4.3 do TCC:

  (1) LARGURA DA LACUNA DE IMISCIBILIDADE em funcao da fracao de PIL
      Para cada nivel de fracao molar de PIL, mede-se a distancia entre os
      dois ramos da curva binodal (o rico em oleo e o rico em alcool),
      expressa como diferenca de fracao molar de oleo. Se a largura DIMINUI
      conforme x_PIL aumenta, ha cossolvencia; se AUMENTA, ha efeito
      anticossolvente.

  (2) SOLUBILIDADE DO OLEO NA FASE RICA EM ALCOOL
      Fracao molar de oleo no ramo rico em alcool, em funcao de x_PIL.
      Responde diretamente quanto de triglicerideo o meio alcoolico
      consegue dissolver, e como isso evolui com a adicao do PIL.

O ponto de referencia (x_PIL = 0) e a aresta oleo-alcool do diagrama, ou
seja, a solubilidade mutua na AUSENCIA de cossolvente. Ele e obtido por
extrapolacao dos pontos de menor x_PIL disponiveis, e nao medido
diretamente, o que e sinalizado na saida.

USO
---
    python3 analise_cossolvencia.py
    python3 analise_cossolvencia.py /outra/pasta
    python3 analise_cossolvencia.py --nbins 12
    python3 analise_cossolvencia.py --cols x1_I,x2_I,x3_I,x1_II,x2_II,x3_II

Sem argumento de pasta, usa PASTA_PADRAO (definida logo abaixo dos imports).

Sem --cols, o script tenta detectar as colunas automaticamente e imprime o
que encontrou para conferencia. A ordem assumida e (oleo, alcool, PIL) em
cada fase, seguindo a ordem de componentes do interface_ell.py.

SAIDA
-----
    analise_cossolvencia.csv   tabela consolidada, um bloco por sistema
    <sistema>_lacuna.png       grafico largura x fracao de PIL (se matplotlib)
"""

import os
import re
import csv
import sys
import glob
import argparse

import numpy as np


# Pasta consultada quando nenhuma e informada na linha de comando.
PASTA_PADRAO = "/home/erika/resultados"


# ---------------------------------------------------------------------------
# Leitura e deteccao de colunas
# ---------------------------------------------------------------------------

# padroes comuns de nomeacao; a primeira captura que casar e usada
PADROES_FASE = [
    # x1_I, x2_I, x3_I / x1_II, x2_II, x3_II
    (r"^x1_I$",   r"^x2_I$",   r"^x3_I$",   r"^x1_II$",  r"^x2_II$",  r"^x3_II$"),
    # xI_1 ... xII_3
    (r"^xI_1$",   r"^xI_2$",   r"^xI_3$",   r"^xII_1$",  r"^xII_2$",  r"^xII_3$"),
    # fase1_x1 ... fase2_x3
    (r"^fase1_x1$", r"^fase1_x2$", r"^fase1_x3$",
     r"^fase2_x1$", r"^fase2_x2$", r"^fase2_x3$"),
    # x_alpha_1 ... x_beta_3
    (r"^x_alpha_1$", r"^x_alpha_2$", r"^x_alpha_3$",
     r"^x_beta_1$",  r"^x_beta_2$",  r"^x_beta_3$"),
]


def detectar_colunas(header):
    """Devolve a lista de 6 nomes de coluna (3 da fase I, 3 da fase II) ou
    None se nenhum padrao casar."""
    h = [c.strip() for c in header]
    for padrao in PADROES_FASE:
        nomes = []
        ok = True
        for p in padrao:
            achados = [c for c in h if re.match(p, c, flags=re.I)]
            if len(achados) != 1:
                ok = False
                break
            nomes.append(achados[0])
        if ok:
            return nomes

    # fallback: pega as seis primeiras colunas numericas na ordem em que
    # aparecem, assumindo (I1, I2, I3, II1, II2, II3)
    if len(h) >= 6:
        return h[:6]
    return None


def ler_tielines(caminho, cols=None):
    """Le um CSV de tie-lines e devolve dois arrays (n, 3): fase I e fase II.
    As colunas sao interpretadas na ordem (oleo, alcool, PIL)."""
    with open(caminho, newline="", encoding="utf-8") as f:
        amostra = f.read(4096)
        f.seek(0)
        sep = ";" if amostra.count(";") > amostra.count(",") else ","
        reader = csv.DictReader(f, delimiter=sep)
        header = reader.fieldnames or []
        nomes = cols or detectar_colunas(header)
        if nomes is None:
            raise ValueError("nao foi possivel identificar as colunas em %s "
                             "(cabecalho: %s)" % (caminho, header))
        faseI, faseII = [], []
        for row in reader:
            try:
                a = [float(row[nomes[0]]), float(row[nomes[1]]), float(row[nomes[2]])]
                b = [float(row[nomes[3]]), float(row[nomes[4]]), float(row[nomes[5]])]
            except (TypeError, ValueError, KeyError):
                continue
            faseI.append(a)
            faseII.append(b)
    return np.array(faseI), np.array(faseII), nomes


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def separar_ramos(faseI, faseII):
    """Reorganiza as fases em ramo rico em oleo e ramo rico em alcool,
    independentemente de qual coluna do CSV corresponde a qual fase.
    Componente 0 = oleo, 1 = alcool, 2 = PIL."""
    ricos_oleo, ricos_alcool = [], []
    for a, b in zip(faseI, faseII):
        if a[0] >= b[0]:
            ricos_oleo.append(a)
            ricos_alcool.append(b)
        else:
            ricos_oleo.append(b)
            ricos_alcool.append(a)
    return np.array(ricos_oleo), np.array(ricos_alcool)


def perfil_lacuna(ramo_oleo, ramo_alcool, nbins=10):
    """Agrupa os pontos da binodal por faixa de fracao de PIL e mede, em cada
    faixa, a largura da lacuna e a solubilidade do oleo na fase alcoolica.

    A fracao de PIL de referencia da faixa e a media entre os dois ramos,
    ja que os dois extremos de uma mesma linha de amarracao nao tem, em
    geral, a mesma fracao de PIL."""
    x_pil = 0.5 * (ramo_oleo[:, 2] + ramo_alcool[:, 2])
    if len(x_pil) == 0:
        return []

    bordas = np.linspace(x_pil.min(), x_pil.max(), nbins + 1)
    linhas = []
    for i in range(nbins):
        lo, hi = bordas[i], bordas[i + 1]
        if i == nbins - 1:
            m = (x_pil >= lo) & (x_pil <= hi)
        else:
            m = (x_pil >= lo) & (x_pil < hi)
        if m.sum() == 0:
            continue
        largura = ramo_oleo[m, 0] - ramo_alcool[m, 0]
        linhas.append({
            "x_pil_medio": float(x_pil[m].mean()),
            "x_pil_min": float(lo),
            "x_pil_max": float(hi),
            "n_pontos": int(m.sum()),
            "largura_media": float(largura.mean()),
            "largura_dp": float(largura.std()),
            "x_oleo_fase_alcoolica": float(ramo_alcool[m, 0].mean()),
            "x_oleo_fase_oleosa": float(ramo_oleo[m, 0].mean()),
        })
    return linhas


def extrapolar_referencia(linhas, n_ajuste=3):
    """Estima a largura da lacuna em x_PIL = 0 por regressao linear sobre as
    n_ajuste faixas de menor fracao de PIL. Devolve (largura_0, inclinacao)
    ou (None, None) se nao houver pontos suficientes."""
    if len(linhas) < max(2, n_ajuste):
        return None, None
    sub = sorted(linhas, key=lambda d: d["x_pil_medio"])[:n_ajuste]
    x = np.array([d["x_pil_medio"] for d in sub])
    y = np.array([d["largura_media"] for d in sub])
    incl, interc = np.polyfit(x, y, 1)
    return float(interc), float(incl)


def tendencia_global(linhas):
    """Inclinacao da regressao linear da largura contra x_PIL usando todas as
    faixas. Negativa indica cossolvencia; positiva, efeito anticossolvente."""
    if len(linhas) < 2:
        return None
    x = np.array([d["x_pil_medio"] for d in linhas])
    y = np.array([d["largura_media"] for d in linhas])
    incl, _ = np.polyfit(x, y, 1)
    return float(incl)


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def processar(pasta, nbins=10, cols=None, plotar=True):
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    arquivos = [a for a in arquivos
                if not os.path.basename(a).startswith(("resumo", "analise"))]
    if not arquivos:
        print("Nenhum CSV de tie-lines encontrado em %s" % pasta)
        return

    saida = os.path.join(pasta, "analise_cossolvencia.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sistema", "x_pil_medio", "n_pontos", "largura_media",
                    "largura_dp", "x_oleo_fase_alcoolica", "x_oleo_fase_oleosa"])

        for arq in arquivos:
            nome = os.path.splitext(os.path.basename(arq))[0]
            try:
                fI, fII, nomes = ler_tielines(arq, cols)
            except Exception as e:
                print("  [pulado] %s -> %s" % (nome, e))
                continue
            if len(fI) < 5:
                print("  [pulado] %s -> apenas %d linha(s) de amarracao"
                      % (nome, len(fI)))
                continue

            ramo_o, ramo_a = separar_ramos(fI, fII)
            linhas = perfil_lacuna(ramo_o, ramo_a, nbins=nbins)
            larg0, _ = extrapolar_referencia(linhas)
            incl = tendencia_global(linhas)

            print("\n=== %s ===" % nome)
            print("  colunas usadas: %s" % ", ".join(nomes))
            print("  linhas de amarracao: %d" % len(fI))
            if larg0 is not None:
                print("  largura extrapolada em x_PIL=0: %.4f" % larg0)
            if incl is not None:
                sentido = ("DIMINUI (indicio de cossolvencia)" if incl < 0
                           else "AUMENTA (indicio de efeito anticossolvente)")
                print("  inclinacao d(largura)/d(x_PIL): %+.4f -> %s"
                      % (incl, sentido))
            print("  %-10s %-8s %-12s %-12s" %
                  ("x_PIL", "n", "largura", "x_oleo(alc)"))
            for d in linhas:
                print("  %-10.4f %-8d %-12.4f %-12.3e"
                      % (d["x_pil_medio"], d["n_pontos"],
                         d["largura_media"], d["x_oleo_fase_alcoolica"]))
                w.writerow([nome, "%.6f" % d["x_pil_medio"], d["n_pontos"],
                            "%.6f" % d["largura_media"],
                            "%.6f" % d["largura_dp"],
                            "%.6e" % d["x_oleo_fase_alcoolica"],
                            "%.6f" % d["x_oleo_fase_oleosa"]])

            if plotar:
                _plotar(nome, linhas, pasta)

    print("\nTabela consolidada salva em: %s" % saida)


def _plotar(nome, linhas, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not linhas:
        return
    x = [d["x_pil_medio"] for d in linhas]
    y = [d["largura_media"] for d in linhas]
    e = [d["largura_dp"] for d in linhas]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, linewidth=1.4)
    ax.set_xlabel(r"fração molar de LIP")
    ax.set_ylabel(r"largura da lacuna ($\Delta x_{\rm óleo}$)")
    ax.set_title(nome, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destino = os.path.join(pasta, "%s_lacuna.png" % nome)
    fig.savefig(destino, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Metricas de cossolvencia a partir dos CSVs de tie-lines")
    ap.add_argument("pasta", nargs="?", default=PASTA_PADRAO,
                    help="pasta com os CSVs (default: %s)" % PASTA_PADRAO)
    ap.add_argument("--nbins", type=int, default=10,
                    help="numero de faixas de fracao de PIL (default: 10)")
    ap.add_argument("--cols", default=None,
                    help="nomes das 6 colunas, separados por virgula, na ordem "
                         "oleo_I,alcool_I,pil_I,oleo_II,alcool_II,pil_II")
    ap.add_argument("--sem-grafico", action="store_true")
    args = ap.parse_args()

    cols = args.cols.split(",") if args.cols else None
    processar(args.pasta, nbins=args.nbins, cols=cols,
              plotar=not args.sem_grafico)