# -*- coding: utf-8 -*-
"""Diagnostico do COSMO-SAC via JCOSMO (rodar LOCALMENTE, com JCOSMO aberto).

Investiga se o resultado 'sem regiao bifasica' e miscibilidade real ou
artefato numerico (overflow/NaN vindo do JCOSMO em cantos do simplex).

Uso:
    python diagnostico_cosmo_local.py

Ajuste OIL_KEY, ALC_KEY, IL_KEY e os nomes JCOSMO conforme o sistema que
deu 'zero instaveis'.
"""

import numpy as np
from cosmo_sac_adapter import CosmoSacAdapter, CosmoSacError
import thermo_engine as eng

# ---- ajuste conforme o sistema testado ----
JCOSMO_NAMES = ["OLEIC_ACID", "ETHANOL", "2HEAHX"]   # nomes EXATOS no banco
LABELS = ["Acido oleico", "Etanol", "[2-HEA][Hx]"]
T = 298.15
# --------------------------------------------


def fmt(v):
    return "[" + ", ".join("%.4e" % x for x in v) + "]"


if __name__ == "__main__":
    print("=" * 74)
    print("DIAGNOSTICO COSMO-SAC (local, via JCOSMO)")
    print("Sistema: %s" % " + ".join(LABELS))
    print("=" * 74)

    try:
        model = CosmoSacAdapter(JCOSMO_NAMES, T=T)
    except CosmoSacError as exc:
        print("Falha ao conectar/configurar:", exc)
        raise SystemExit(1)

    print("\n--- (1) gamma em composicoes de teste (checando overflow/NaN) ---")
    casos = {
        "centro do simplex":      [0.3333, 0.3333, 0.3334],
        "quase puro oleo":        [0.96, 0.02, 0.02],
        "quase puro alcool":      [0.02, 0.96, 0.02],
        "quase puro PIL":         [0.02, 0.02, 0.96],
        "rico oleo+PIL":          [0.45, 0.10, 0.45],
        "rico oleo+PIL(2)":       [0.60, 0.05, 0.35],
        "rico alcool+PIL":        [0.05, 0.45, 0.50],
        "diluicao infinita oleo": [0.99, 0.005, 0.005],
        "diluicao infinita PIL":  [0.005, 0.005, 0.99],
    }
    algum_nao_finito = False
    for nome, x in casos.items():
        g = model.gamma(x)
        finito = np.all(np.isfinite(g))
        if not finito:
            algum_nao_finito = True
        marca = "" if finito else "  <<< NAO FINITO"
        print("  %-24s x=%s  gamma=%s%s" % (nome, fmt(x), fmt(g), marca))

    print("\n--- (2) estabilidade (plano tangente) nos mesmos pontos ---")
    n_invalido = 0
    n_instavel = 0
    for nome, x in casos.items():
        inst, w, D, invalido = eng.is_unstable(model, x, log_callback=None)
        if invalido:
            n_invalido += 1
            status = "INVALIDO (gamma nao finito em algum ponto do plano tangente)"
        elif inst:
            n_instavel += 1
            status = "INSTAVEL (2 fases)"
        else:
            status = "estavel"
        print("  %-24s D=%-14s -> %s" % (nome, ("%.4e" % D if np.isfinite(D) else "nan"), status))

    print("\n--- (3) varredura pequena (n_grid=10) com relatorio de invalidos ---")
    out = eng.grid_scan(model, n_grid=10, log_callback=print)
    print("  tie-lines aceitas: %d" % out["n_accepted"])

    print("\n" + "=" * 74)
    if algum_nao_finito or n_invalido > 0:
        print("DIAGNOSTICO: gamma nao finito detectado. O resultado 'sem gap' da")
        print("varredura completa e SUSPEITO DE ARTEFATO NUMERICO, nao deve ser")
        print("interpretado ainda como miscibilidade real. Envie esta saida.")
    else:
        print("DIAGNOSTICO: nenhum gamma nao finito nos pontos testados. Se a")
        print("varredura completa (n_grid=20) tambem seguir sem 'AVISO: gamma nao")
        print("finito' no log, o resultado de miscibilidade total e mais")
        print("provavelmente uma previsao genuina do modelo para este sistema.")
        print("Vale then aumentar N_RANDOM_STARTS em thermo_engine.py e/ou refinar")
        print("a grade perto dos vertices para descartar gap muito estreito.")
    print("=" * 74)
