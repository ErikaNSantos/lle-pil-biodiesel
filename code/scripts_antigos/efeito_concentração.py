"""
efeito_concentracao_LI.py — Efeito da concentracao de LIP sobre a miscibilidade
==================================================================================
Para cada um dos 2 sistemas-alvo do TCC, varia a fracao molar de LIP de 0 a 40%
mantendo razao molar fixa trioleina:etanol = 1:6 (estequiometrica para
transesterificacao alcalina), e analisa:

  - Se o sistema e monofasico ou bifasico em cada composicao
  - Coeficiente de particao K_triol em cada composicao bifasica
  - Concentracao critica de LIP (%LIP_c) na qual o sistema se torna monofasico

Esses calculos correspondem ao ultimo objetivo especifico do TCC:
"Analisar o efeito da concentracao de cossolvente sobre a extensao da regiao
de miscibilidade do sistema ternario".

PRE-REQUISITOS:
  Mesma pasta que params.py, gamma.py, flash.py, stability.py, diagram.py
  params.py deve ter os parametros V2 (base ampliada): a=(3567, 3113, -3302, 81)

EXECUCAO:
    python efeito_concentracao_LI.py

SAIDAS:
    resultados/efeito_concentracao_LI.png    (figura principal)
    resultados/efeito_concentracao_LI.txt    (tabela quantitativa)
    efeito_concentracao_LI.json              (dados completos)

TEMPO: ~1-2 minutos
"""

import numpy as np
import time
import json
import os
import importlib

import params as p
import gamma
import flash
import stability

T = 298.15
OUTDIR = 'resultados'
os.makedirs(OUTDIR, exist_ok=True)


# ===========================================================================
# CONFIGURACAO
# ===========================================================================

# Razao trioleina:etanol fixa em 1:6 (estequiometrica, considerando que cada
# trioleina precisa de 3 etanol para transesterificacao completa, com excesso
# de etanol = 100% para deslocar o equilibrio).
RAZAO_OLEO_ETANOL = 1.0 / 6.0   # x_trioleina / x_etanol

# Faixa de fracao molar de LIP a ser variada
PCT_LI_GRID = np.linspace(0.0, 0.40, 41)  # 0%, 1%, 2%, ..., 40%

# Parametros V2 — base ampliada (escolhidos como referencia principal)
PARAMS_V2 = (3567.0, 3113.0, -3302.0, 81.0)

# Sistemas-alvo
SYSTEMS = [
    {
        'name': '2HEA_Hx',
        'title': '[2-HEA][Hx]',
        'nu_li': p.NU_2HEA_HX,
        'color': '#16a085',
    },
    {
        'name': 'DETA_Hx',
        'title': '[DETA][Hx]',
        'nu_li': p.NU_DETA_HX,
        'color': '#c0392b',
    },
]


# ===========================================================================
# FUNCOES
# ===========================================================================

def set_params(values):
    p.INTERACTION_PARAMS[(11, 14)] = (values[0], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 11)] = (values[1], 0.0, 0.0)
    p.INTERACTION_PARAMS[(14, 41)] = (values[2], 0.0, 0.0)
    p.INTERACTION_PARAMS[(41, 14)] = (values[3], 0.0, 0.0)
    importlib.reload(gamma)
    importlib.reload(flash)
    importlib.reload(stability)


def composicao_global(pct_li):
    """
    Calcula z = [x_trioleina, x_etanol, x_LI] com:
      x_LI = pct_li
      x_trioleina + x_etanol = 1 - pct_li
      x_trioleina / x_etanol = RAZAO_OLEO_ETANOL = 1/6
    
    Logo:
      x_etanol = (1 - pct_li) / (1 + RAZAO_OLEO_ETANOL)
      x_trioleina = RAZAO_OLEO_ETANOL * x_etanol
    """
    x_etanol = (1.0 - pct_li) / (1.0 + RAZAO_OLEO_ETANOL)
    x_trioleina = RAZAO_OLEO_ETANOL * x_etanol
    return np.array([x_trioleina, x_etanol, pct_li])


def analyze_system(sys_cfg, params_values):
    """
    Para cada %LIP, determina se o sistema e bifasico e calcula metricas.
    """
    set_params(params_values)
    from gamma import UNIFACLL
    from stability import is_unstable
    from flash import flash_ell

    nu_list = [p.NU_TRIOLINA, p.NU_ETANOL, sys_cfg['nu_li']]
    model = UNIFACLL(nu_list, T)

    results = []
    for pct in PCT_LI_GRID:
        z = composicao_global(pct)
        z = np.clip(z, 1e-6, 1); z /= z.sum()

        # Verifica estabilidade
        bifasic = is_unstable(z, model)

        # Metricas
        rec = {
            'pct_li': float(pct),
            'z': list(z),
            'bifasic': bool(bifasic),
            'K_triol': None,
            'x_triol_oleosa': None,
            'x_triol_polar': None,
            'gap_x_triol': None,  # |x_triol(oleosa) - x_triol(polar)|
        }

        if bifasic:
            xa, xb, psi, ok, _ = flash_ell(z, model)
            if ok:
                if xa[0] < xb[0]:
                    xa, xb = xb, xa  # xa = oleosa, xb = polar
                rec['x_triol_oleosa'] = float(xa[0])
                rec['x_triol_polar'] = float(xb[0])
                rec['K_triol'] = float(xa[0] / max(xb[0], 1e-6))
                rec['gap_x_triol'] = float(xa[0] - xb[0])

        results.append(rec)
    return results


def find_critical_concentration(results):
    """
    Acha %LI critico onde sistema deixa de ser bifasico.
    Retorna None se nunca atinge monofasico no range testado.
    """
    for r in results:
        if not r['bifasic']:
            return r['pct_li']
    return None  # bifasico em todas as composicoes testadas


# ===========================================================================
# EXECUCAO
# ===========================================================================

def main():
    print("=" * 75)
    print("EFEITO DA CONCENTRACAO DE LIP SOBRE A MISCIBILIDADE DO SISTEMA")
    print("=" * 75)
    print(f"  Razao molar fixa: trioleina:etanol = 1:6 (estequiometrica)")
    print(f"  Faixa de %LIP: 0% a 40% (em incrementos de 1%)")
    print(f"  Parametros V2 (base ampliada): a={PARAMS_V2}")
    print()

    all_results = {}
    t0 = time.time()
    for sys_cfg in SYSTEMS:
        print(f"--- Calculando {sys_cfg['title']} ---")
        results = analyze_system(sys_cfg, PARAMS_V2)
        all_results[sys_cfg['name']] = results

        # Resumo
        n_bif = sum(1 for r in results if r['bifasic'])
        n_mono = len(results) - n_bif
        pct_c = find_critical_concentration(results)

        print(f"  Composicoes bifasicas:   {n_bif}/{len(results)}")
        print(f"  Composicoes monofasicas: {n_mono}/{len(results)}")
        if pct_c is not None:
            print(f"  Concentracao critica de LIP: {100*pct_c:.1f}%")
        else:
            print(f"  Bifasico em toda a faixa testada (0-40% LIP)")

        # Mostrar K_triol em algumas %LIP-chave
        print(f"  Coeficiente de particao K_triol em %LIP-chave:")
        for r in results:
            if abs(r['pct_li'] - 0.05) < 0.01 or \
               abs(r['pct_li'] - 0.15) < 0.01 or \
               abs(r['pct_li'] - 0.30) < 0.01:
                K = r['K_triol']
                K_str = f"{K:.1f}" if K is not None else "—"
                gap = r['gap_x_triol']
                gap_str = f"{gap:.4f}" if gap is not None else "—"
                print(f"    %LIP = {100*r['pct_li']:>4.1f}%  bif={r['bifasic']}  "
                      f"K_triol = {K_str}  Δx_triol = {gap_str}")
        print()

    print(f"Tempo: {time.time()-t0:.0f}s")
    print()

    # ============================================================
    # FIGURA PRINCIPAL
    # ============================================================
    print("Gerando figura...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Painel 1: gap entre fases (medida da intensidade da separacao)
    ax = axes[0]
    for sys_cfg in SYSTEMS:
        results = all_results[sys_cfg['name']]
        pct = np.array([r['pct_li'] for r in results]) * 100
        gap = np.array([r['gap_x_triol'] if r['gap_x_triol'] is not None else 0.0
                        for r in results])
        ax.plot(pct, gap, 'o-', color=sys_cfg['color'],
                label=sys_cfg['title'], lw=2, ms=4)

    ax.set_xlabel('Concentracao de LIP (% molar)', fontsize=11)
    ax.set_ylabel(r'$\Delta x_{trioleina} = x_{oleosa} - x_{polar}$', fontsize=11)
    ax.set_title('Intensidade da separacao de fases', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)

    # Painel 2: K_triol (escala log se variar muito)
    ax = axes[1]
    for sys_cfg in SYSTEMS:
        results = all_results[sys_cfg['name']]
        pct = []
        K_vals = []
        for r in results:
            if r['K_triol'] is not None:
                pct.append(r['pct_li'] * 100)
                K_vals.append(r['K_triol'])
        if pct:
            ax.semilogy(pct, K_vals, 'o-', color=sys_cfg['color'],
                       label=sys_cfg['title'], lw=2, ms=4)

    ax.set_xlabel('Concentracao de LIP (% molar)', fontsize=11)
    ax.set_ylabel(r'$K_{trioleina}$ = $x_{oleosa} / x_{polar}$', fontsize=11)
    ax.set_title('Coeficiente de particao da trioleina (escala log)',
                 fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')

    fig.suptitle("Efeito da concentracao de LIP sobre a miscibilidade — "
                 "Razao molar trioleina:etanol = 1:6 — T = 298,15 K",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    outpath = f'{OUTDIR}/efeito_concentracao_LI.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {outpath}")

    # ============================================================
    # TABELA
    # ============================================================
    lines = []
    lines.append("=" * 80)
    lines.append("EFEITO DA CONCENTRACAO DE LIP — TABELA QUANTITATIVA")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Razao molar trioleina:etanol = 1:6 (estequiometrica)")
    lines.append(f"Parametros UNIFAC: V2 base ampliada (a = {PARAMS_V2})")
    lines.append("")

    for sys_cfg in SYSTEMS:
        results = all_results[sys_cfg['name']]
        lines.append(f"--- Sistema: Trioleina + Etanol + {sys_cfg['title']} ---")
        lines.append(f"  {'%LIP':>6}  {'estado':<10}  {'K_triol':>10}  "
                     f"{'gap_x_triol':>12}  {'x_triol_oleosa':>15}  {'x_triol_polar':>14}")
        for r in results:
            if r['pct_li'] * 100 in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]:
                estado = 'bifasico' if r['bifasic'] else 'monofasico'
                K = f"{r['K_triol']:.2f}" if r['K_triol'] is not None else "—"
                gap = f"{r['gap_x_triol']:.4f}" if r['gap_x_triol'] is not None else "—"
                xo = f"{r['x_triol_oleosa']:.4f}" if r['x_triol_oleosa'] is not None else "—"
                xp = f"{r['x_triol_polar']:.4f}" if r['x_triol_polar'] is not None else "—"
                lines.append(f"  {100*r['pct_li']:>5.0f}%  {estado:<10}  "
                             f"{K:>10}  {gap:>12}  {xo:>15}  {xp:>14}")
        lines.append("")

        pct_c = find_critical_concentration(results)
        if pct_c is not None:
            lines.append(f"  Concentracao critica (transicao bifasico->monofasico): "
                         f"{100*pct_c:.1f}% LIP")
        else:
            lines.append(f"  Sistema permanece bifasico em toda a faixa 0-40% LIP")
        lines.append("")

    # Comparacao final
    lines.append("=" * 80)
    lines.append("COMPARACAO ENTRE LIPs")
    lines.append("=" * 80)
    lines.append("")
    pct_c_2hea = find_critical_concentration(all_results['2HEA_Hx'])
    pct_c_deta = find_critical_concentration(all_results['DETA_Hx'])
    if pct_c_2hea is not None and pct_c_deta is not None:
        lines.append(f"  Concentracao critica [2-HEA][Hx]: {100*pct_c_2hea:.1f}%")
        lines.append(f"  Concentracao critica [DETA][Hx]:  {100*pct_c_deta:.1f}%")
        if pct_c_deta < pct_c_2hea:
            lines.append(f"  -> [DETA][Hx] requer MENOS cossolvente para induzir miscibilidade,")
            lines.append(f"     consistente com Trindade et al. (2019).")
        else:
            lines.append(f"  -> [2-HEA][Hx] requer menos cossolvente. Inverte Trindade (2019).")
    else:
        lines.append(f"  Pelo menos um sistema permanece bifasico em toda a faixa testada.")
        lines.append(f"  Nesse caso, comparar pelo K_triol em %LIP fixos:")
        lines.append("")
        for pct_check in [0.05, 0.15, 0.30]:
            r_2hea = next((r for r in all_results['2HEA_Hx']
                           if abs(r['pct_li'] - pct_check) < 0.005), None)
            r_deta = next((r for r in all_results['DETA_Hx']
                           if abs(r['pct_li'] - pct_check) < 0.005), None)
            if r_2hea and r_deta:
                K2h = r_2hea['K_triol'] if r_2hea['K_triol'] else "monofasico"
                Kdt = r_deta['K_triol'] if r_deta['K_triol'] else "monofasico"
                lines.append(f"  %LIP = {100*pct_check:>4.1f}%:  "
                             f"K_triol([2-HEA]) = {K2h}  "
                             f"K_triol([DETA])  = {Kdt}")
    lines.append("")

    txt = "\n".join(lines)
    print()
    print(txt)

    with open(f'{OUTDIR}/efeito_concentracao_LI.txt', 'w') as f:
        f.write(txt)
    print(f"\nResumo: {OUTDIR}/efeito_concentracao_LI.txt")

    # JSON
    with open('efeito_concentracao_LI.json', 'w') as f:
        json.dump({
            'parameters': list(PARAMS_V2),
            'razao_oleo_etanol': RAZAO_OLEO_ETANOL,
            'pct_li_grid': list(PCT_LI_GRID),
            'systems': all_results,
        }, f, indent=2)
    print(f"JSON:   efeito_concentracao_LI.json")


if __name__ == '__main__':
    main()