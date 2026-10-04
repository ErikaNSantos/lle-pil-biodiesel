"""
main.py — Orquestração do cálculo ELL (UNIFAC-Dortmund)
========================================================
Executa o pipeline completo para os dois sistemas estudados:
  1. {Triolína + Etanol + [2-HEA][Hx]}
  2. {Triolína + Etanol + [DETA][Hx]}

Uso:
  python main.py              # execução padrão (n_grid=30)
  python main.py --grid 50    # grade mais fina
  python main.py --system 1   # só sistema 1
  python main.py --validate   # só validação sem diagrama

Saídas (em OUTPUT_DIR):
  diagrama_<sistema>.png
  tielines_<sistema>.csv
  relatorio_resultados.txt
"""

import argparse
import os
import time
import numpy as np

from params   import NU_TRIOLINA, NU_ETANOL, NU_2HEA_HX, NU_DETA_HX
from gamma    import UNIFACLL
from stability import is_unstable
from flash    import flash_ell, check_mass_balance, check_isofugacity
from diagram  import calc_tie_lines, plot_ternary, export_csv

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
T          = 298.15   # K

SYSTEMS = [
    {
        'id':      1,
        'name':    '[2-HEA][Hx]',
        'nu_li':   NU_2HEA_HX,
        'labels':  ('Triolína', 'Etanol', '[2-HEA][Hx]'),
        'title':   '{Triolína + Etanol + [2-HEA][Hx]}',
        'safe':    '2HEA_Hx',
    },
    {
        'id':      2,
        'name':    '[DETA][Hx]',
        'nu_li':   NU_DETA_HX,
        'labels':  ('Triolína', 'Etanol', '[DETA][Hx]'),
        'title':   '{Triolína + Etanol + [DETA][Hx]}',
        'safe':    'DETA_Hx',
    },
]

# ---------------------------------------------------------------------------
# Dados experimentais de validação
# Preencher com tie-lines do Souza et al. (2025) quando disponíveis.
# Formato: (xa_exp [array 3], xb_exp [array 3])
# ---------------------------------------------------------------------------
EXP_DATA = {
    1: [],   # sistema [2-HEA][Hx]
    2: [],   # sistema [DETA][Hx]
}

# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def calc_rmsd(tl_calc, tl_exp):
    """
    RMSD entre tie-lines calculadas e experimentais.

    RMSD = sqrt[ 1/(2·Nt·Nc) · ∑_{t,j,i} (x_calc − x_exp)² ]

    Para cada tie-line experimental encontra a calculada mais próxima
    (matching por distância mínima na composição das duas fases).
    """
    if not tl_calc or not tl_exp:
        return None

    Nc  = 3
    tot = 0.0
    for xa_e, xb_e in tl_exp:
        best = min(
            np.sum((xa_c - xa_e)**2) + np.sum((xb_c - xb_e)**2)
            for xa_c, xb_c in tl_calc
        )
        tot += best
    return float(np.sqrt(tot / (2 * len(tl_exp) * Nc)))

# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def print_separator(char='─', width=58):
    print(char * width)

def write_report(results: list[dict], outfile: str) -> None:
    """Escreve relatório de resultados em arquivo .txt."""
    lines = []
    lines.append("RELATÓRIO DE RESULTADOS — UNIFAC-Dortmund ELL")
    lines.append(f"T = {T} K")
    lines.append("=" * 58)

    for r in results:
        lines.append(f"\nSistema {r['id']}: {r['title']}")
        lines.append(f"  Tempo de cálculo : {r['elapsed']:.1f} s")
        lines.append(f"  Grid             : {r['n_grid']}×{r['n_grid']}")
        lines.append(f"  Pontos testados  : {r['n_pts']}")
        lines.append(f"  Bifásicos        : {r['n_bif']}")
        lines.append(f"  Tie-lines válidas: {r['n_tl']}")
        lines.append(f"  Falhas no flash  : {r['n_fail']}")
        if r['rmsd'] is not None:
            lines.append(f"  RMSD             : {r['rmsd']:.4f} (vs. dados exp.)")
        else:
            lines.append(f"  RMSD             : — (dados experimentais não carregados)")
        lines.append(f"  Figura           : {r['png']}")
        lines.append(f"  CSV tie-lines    : {r['csv']}")

    lines.append("\n" + "=" * 58)
    lines.append("NOTAS")
    lines.append("  Parâmetros CCOO/CNH2 e CCOO/CNH ausentes → Ψ=1 (ideal).")
    lines.append("  Área bifásica pode estar superestimada.")
    lines.append("  Para parâmetros completos: DDB / Gmehling (1993).")

    text = "\n".join(lines)
    with open(outfile, 'w') as f:
        f.write(text)
    print(f"\n  Relatório salvo: {outfile}")

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_system(sys_cfg: dict, n_grid: int, verbose: bool = True) -> dict:
    """Executa o pipeline completo para um sistema."""
    print_separator('═')
    print(f" Sistema {sys_cfg['id']}: {sys_cfg['title']}")
    print_separator('═')

    nu_list = [NU_TRIOLINA, NU_ETANOL, sys_cfg['nu_li']]
    model   = UNIFACLL(nu_list, T)

    # Sanity check rápido
    g_eq = model.gamma([1/3, 1/3, 1/3])
    print(f"\n  γ equimolar: "
          f"triolína={g_eq[0]:.3f}  etanol={g_eq[1]:.3f}  LIP={g_eq[2]:.3f}")

    # Calcular tie-lines
    print(f"\n  Calculando tie-lines (grid={n_grid})...")
    t0 = time.time()
    tl = calc_tie_lines(model, n_grid=n_grid)
    elapsed = time.time() - t0

    n_pts = sum(1 for i in range(n_grid+1)
                for j in range(n_grid+1-i)
                if n_grid-i-j >= 0
                and np.all(np.array([i, j, n_grid-i-j], float)/n_grid >= 0.03))
    n_bif  = sum(1 for z in [
        np.array([i, j, n_grid-i-j], float)/n_grid
        for i in range(n_grid+1) for j in range(n_grid+1-i)
        if n_grid-i-j >= 0
        and np.all(np.array([i, j, n_grid-i-j], float)/n_grid >= 0.03)
    ] if is_unstable(z, model))
    n_fail = n_bif - len(tl)

    # Caminhos de saída
    png = os.path.join(OUTPUT_DIR, f"diagrama_{sys_cfg['safe']}.png")
    csv = os.path.join(OUTPUT_DIR, f"tielines_{sys_cfg['safe']}.csv")

    # Gerar diagrama
    print(f"\n  Gerando diagrama...")
    plot_ternary(tl,
                 labels  = sys_cfg['labels'],
                 title   = sys_cfg['title'],
                 outfile = png)

    # Exportar CSV
    export_csv(tl,
               outfile    = csv,
               comp_names = ('triolina', 'etanol',
                             sys_cfg['name'].replace('[','').replace(']','')))

    # RMSD
    exp = EXP_DATA.get(sys_cfg['id'], [])
    rmsd = calc_rmsd(tl, exp)
    if rmsd is not None:
        print(f"\n  RMSD vs. dados experimentais: {rmsd:.4f}")
    else:
        print(f"\n  RMSD: dados experimentais não carregados.")
        print(f"  → Preencher EXP_DATA no topo de main.py com as")
        print(f"    tie-lines de Souza et al. (2025) para calcular.")

    print(f"\n  Tempo total: {elapsed:.1f} s")

    return {
        'id':      sys_cfg['id'],
        'title':   sys_cfg['title'],
        'elapsed': elapsed,
        'n_grid':  n_grid,
        'n_pts':   n_pts,
        'n_bif':   n_bif,
        'n_tl':    len(tl),
        'n_fail':  n_fail,
        'rmsd':    rmsd,
        'png':     png,
        'csv':     csv,
        'tl':      tl,
    }


def run_validation_only() -> None:
    """Executa só os módulos de validação, sem gerar diagramas."""
    print("Executando validações independentes dos módulos...\n")
    for module in ['params', 'gamma', 'stability', 'flash']:
        print_separator()
        print(f" python {module}.py")
        print_separator()
        os.system(f"cd /home/claude/unifac && python {module}.py")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cálculo ELL UNIFAC-Dortmund — TCC Érika'
    )
    parser.add_argument('--grid',     type=int, default=30,
                        help='Tamanho da grade ternária (padrão: 30)')
    parser.add_argument('--system',   type=int, choices=[1, 2], default=None,
                        help='Executar só o sistema 1 ou 2')
    parser.add_argument('--validate', action='store_true',
                        help='Executar só as validações dos módulos')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print_separator('═', 58)
    print("  UNIFAC-Dortmund — Cálculo de ELL Ternário")
    print(f"  T = {T} K")
    print_separator('═', 58)

    if args.validate:
        run_validation_only()
        return

    # Selecionar sistemas a executar
    systems_to_run = SYSTEMS if args.system is None \
                     else [s for s in SYSTEMS if s['id'] == args.system]

    results = []
    for sys_cfg in systems_to_run:
        r = run_system(sys_cfg, n_grid=args.grid)
        results.append(r)

    # Relatório final
    report_path = os.path.join(OUTPUT_DIR, 'relatorio_resultados.txt')
    write_report(results, report_path)

    print()
    print_separator('═', 58)
    print("  CONCLUÍDO")
    for r in results:
        print(f"  Sistema {r['id']}: {r['n_tl']} tie-lines  "
              f"RMSD={'%.4f' % r['rmsd'] if r['rmsd'] else '—'}")
    print_separator('═', 58)


if __name__ == '__main__':
    main()