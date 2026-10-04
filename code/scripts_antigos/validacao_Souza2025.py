"""
validacao_souza2025.py — Validação com dados de Souza et al. (2025)
=====================================================================
Sistema: {Hexil acetato (1) + 2-HEAB (2) + Etanol (3)} a 298.15 K

Referência:
  Souza, R.C.O. et al. (2025)
  "Liquid–Liquid Equilibria of Ternary Systems Containing Butyl Acetate,
   Pentyl Acetate, or Hexyl Acetate + 2-HEAB + Ethanol"
  J. Chem. Eng. Data 70, 477–487 — Tabela 6

Por que este sistema é adequado para validação:
  - 2-HEAB = [2-HEA]+ butanoato: mesmo cátion do [2-HEA][Hx] do TCC
  - Hexil acetato: éster C8, análogo simplificado à triolina
  - Grupos UNIFAC envolvidos: CH3, CH2, CH3COO, OH, COO, CH2NH2
  - Todos os parâmetros existem (exceto CCOO/CNH2 → Ψ=1)
  - NÃO envolve H2O → evita a fragilidade dos parâmetros de água

Decomposição UNIFAC dos componentes:
  Hexil acetato (C8H16O2):  1×CH3COO(sg21) + 5×CH2(sg2) + 1×CH3(sg1)
  2-HEAB (C6H15NO3):        1×CH3(sg1) + 3×CH2(sg2) + 1×OH(sg14)
                             + 1×COO(sg77) + 1×CH2NH2(sg29)
  Etanol (C2H6O):           1×CH3(sg1) + 1×CH2(sg2) + 1×OH(sg14)
"""

import sys
import os
import numpy as np

try:
    from gamma     import UNIFACLL
    from stability import is_unstable
    from flash     import flash_ell, check_mass_balance, check_isofugacity
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Componentes — decomposição em grupos UNIFAC
# ---------------------------------------------------------------------------
NU_HEXIL_ACETATO = {1: 1, 2: 5, 21: 1}     # CH3 + 5×CH2 + CH3COO → C=8 ✓
NU_2HEAB         = {1: 1, 2: 3, 14: 1, 77: 1, 29: 1}  # C=6 ✓
NU_ETANOL        = {1: 1, 2: 1, 14: 1}      # C=2 ✓

T = 298.15

# ---------------------------------------------------------------------------
# Dados experimentais — Souza et al. (2025) Tabela 6
# Sistema: {Hexil acetato (3) + 2-HEAB (4) + Etanol (5)} a 298.15 K
#
# ATENÇÃO: a numeração do artigo é (3,4,5); aqui usamos (1,2,3):
#   Componente 1 = Hexil acetato (3 no artigo)
#   Componente 2 = 2-HEAB        (4 no artigo)
#   Componente 3 = Etanol         (5 no artigo)
#
# Formato: (fase_rica_acetato [rafinado], fase_rica_IL [extrato])
# Cada fase: [x_hex_acet, x_2HEAB, x_etanol]
# ---------------------------------------------------------------------------
EXP_DATA = [
    # TL1
    (np.array([0.942,  0.0242, 0.033 ]),   # rafinado (acetato)
     np.array([0.072,  0.829,  0.100 ])),   # extrato (IL)
    # TL2
    (np.array([0.915,  0.0236, 0.061 ]),
     np.array([0.084,  0.733,  0.183 ])),
    # TL3
    (np.array([0.904,  0.0234, 0.073 ]),
     np.array([0.093,  0.649,  0.259 ])),
    # TL4
    (np.array([0.887,  0.0233, 0.090 ]),
     np.array([0.1003, 0.580,  0.320 ])),
    # TL5
    (np.array([0.863,  0.0231, 0.113 ]),
     np.array([0.1140, 0.501,  0.385 ])),
    # TL6
    (np.array([0.879,  0.0232, 0.098 ]),
     np.array([0.1102, 0.518,  0.371 ])),
    # TL7
    (np.array([0.850,  0.0232, 0.127 ]),
     np.array([0.1198, 0.4779, 0.402 ])),
    # TL8
    (np.array([0.820,  0.0236, 0.156 ]),
     np.array([0.1444, 0.4103, 0.4452])),
    # TL9
    (np.array([0.841,  0.0233, 0.135 ]),
     np.array([0.1336, 0.4360, 0.4304])),
]

# ---------------------------------------------------------------------------
# Criar modelo
# ---------------------------------------------------------------------------
print("=" * 65)
print("VALIDAÇÃO: {Hexil acetato + 2-HEAB + Etanol} a 298.15 K")
print("Modelo: UNIFAC (Bessa 2016 + VLE aminas)")
print("Dados:  Souza et al. (2025) J. Chem. Eng. Data 70, Tab. 6")
print("=" * 65)

nu_list = [NU_HEXIL_ACETATO, NU_2HEAB, NU_ETANOL]
model   = UNIFACLL(nu_list, T)

print(f"\nParâmetros moleculares:")
print(f"  Hexil acetato: r={model.r[0]:.4f}  q={model.q[0]:.4f}")
print(f"  2-HEAB:        r={model.r[1]:.4f}  q={model.q[1]:.4f}")
print(f"  Etanol:        r={model.r[2]:.4f}  q={model.q[2]:.4f}")

# ---------------------------------------------------------------------------
# Calcular tie-lines
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("Comparação tie-line a tie-line:")
print(f"  {'TL':>3}  {'Fase':>8}  {'x_acet':>8}  {'x_2HEAB':>8}  {'x_etOH':>8}  {'Fonte':>10}")
print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

rmsd_list = []

for i, (xa_exp, xb_exp) in enumerate(EXP_DATA):
    tl = i + 1

    # Composição global
    z = 0.5 * xa_exp + 0.5 * xb_exp
    z = np.clip(z, 1e-6, 1.0); z /= z.sum()

    # Verificar estabilidade
    bif = is_unstable(z, model)
    if not bif:
        for psi_try in [0.3, 0.4, 0.6, 0.7]:
            z2 = psi_try * xb_exp + (1-psi_try) * xa_exp
            z2 = np.clip(z2, 1e-6, 1.0); z2 /= z2.sum()
            if is_unstable(z2, model):
                z = z2; bif = True; break

    if not bif:
        print(f"  {tl:>3}  [modelo prevê monofásico — tie-line {tl} ignorada]")
        continue

    # Flash
    xa_c, xb_c, psi, ok, nit = flash_ell(z, model)
    if not ok:
        print(f"  {tl:>3}  [flash não convergiu — iter={nit}]")
        continue

    mb_ok, _ = check_mass_balance(z, xa_c, xb_c, psi)
    if not mb_ok:
        print(f"  {tl:>3}  [balanço de massa falhou]")
        continue

    # Garantir xa = fase rica em acetato (componente 1)
    if xa_c[0] < xb_c[0]:
        xa_c, xb_c = xb_c, xa_c

    # Imprimir
    print(f"  {tl:>3}  {'raf(exp)':>8}  {xa_exp[0]:>8.4f}  {xa_exp[1]:>8.4f}  {xa_exp[2]:>8.4f}  {'Souza2025':>10}")
    print(f"  {tl:>3}  {'raf(cal)':>8}  {xa_c[0]:>8.4f}  {xa_c[1]:>8.4f}  {xa_c[2]:>8.4f}  {'UNIFAC':>10}")
    print(f"  {tl:>3}  {'ext(exp)':>8}  {xb_exp[0]:>8.4f}  {xb_exp[1]:>8.4f}  {xb_exp[2]:>8.4f}  {'Souza2025':>10}")
    print(f"  {tl:>3}  {'ext(cal)':>8}  {xb_c[0]:>8.4f}  {xb_c[1]:>8.4f}  {xb_c[2]:>8.4f}  {'UNIFAC':>10}")
    print()

    sq = np.sum((xa_c - xa_exp)**2) + np.sum((xb_c - xb_exp)**2)
    rmsd_list.append(sq)

# ---------------------------------------------------------------------------
# RMSD global
# ---------------------------------------------------------------------------
print("=" * 65)
if rmsd_list:
    N_tl = len(rmsd_list)
    rmsd = np.sqrt(sum(rmsd_list) / (2 * N_tl * 3))
    print(f"Tie-lines calculadas: {N_tl} de {len(EXP_DATA)}")
    print(f"RMSD = {rmsd:.4f}  ({N_tl} tie-lines, 3 componentes)")
    print()
    print("Referência de qualidade:")
    print("  RMSD < 0.02 → excelente para modelo preditivo")
    print("  RMSD 0.02–0.05 → satisfatório")
    print("  RMSD > 0.05 → limitações paramétricas significativas")
    print()
    print("Nota: este é um sistema ANÁLOGO ao do TCC.")
    print("  Mesmo cátion ([2-HEA]+), ânion similar (butanoato vs hexanoato).")
    print("  Hexil acetato como substituto simplificado da triolina.")
else:
    print("⚠️  Nenhuma tie-line calculada.")
    print("O modelo prevê o sistema como monofásico em todas as composições.")

print("=" * 65)

# ---------------------------------------------------------------------------
# Diagrama ternário
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def t2c(x1, x2, x3):
        X = 0.5 * (2*x3 + x2)
        Y = (3**0.5 / 2) * x2
        return X, Y

    fig, ax = plt.subplots(figsize=(7, 7))
    verts = np.array([[0,0],[1,0],[0.5,3**0.5/2],[0,0]])
    ax.plot(verts[:,0], verts[:,1], 'k-', lw=1.5)

    # Experimentais
    for j, (xa_e, xb_e) in enumerate(EXP_DATA):
        Xa, Ya = t2c(*xa_e)
        Xb, Yb = t2c(*xb_e)
        ax.plot([Xa,Xb],[Ya,Yb],'o--',color='#e74c3c',lw=1,ms=5,
                label='Experimental (Souza 2025)' if j==0 else '')

    # Calculadas
    calc_count = 0
    for j, (xa_e, xb_e) in enumerate(EXP_DATA):
        z = 0.5*xa_e + 0.5*xb_e
        z = np.clip(z,1e-6,1.0); z/=z.sum()
        for psi in [0.5,0.3,0.4,0.6,0.7]:
            z2 = psi*xb_e+(1-psi)*xa_e; z2=np.clip(z2,1e-6,1.0); z2/=z2.sum()
            if is_unstable(z2,model): z=z2; break
        if not is_unstable(z,model): continue
        xa_c,xb_c,psi,ok,_=flash_ell(z,model)
        if not ok: continue
        if xa_c[0]<xb_c[0]: xa_c,xb_c=xb_c,xa_c
        Xa,Ya=t2c(*xa_c); Xb,Yb=t2c(*xb_c)
        ax.plot([Xa,Xb],[Ya,Yb],'s-',color='#2980b9',lw=1,ms=5,
                label='UNIFAC (calc.)' if calc_count==0 else '')
        calc_count += 1

    h = 3**0.5/2
    ax.text(0.0,-0.05,'Hexil acetato',ha='center',fontsize=10,fontweight='bold')
    ax.text(1.0,-0.05,'Etanol',ha='center',fontsize=10,fontweight='bold')
    ax.text(0.5,h+0.03,'2-HEAB',ha='center',fontsize=10,fontweight='bold')
    ax.set_title('{Hexil acetato + 2-HEAB + Etanol} — 298.15 K\n'
                 'UNIFAC vs. Souza et al. (2025)', fontsize=11)
    ax.set_aspect('equal'); ax.axis('off')
    handles,labels=ax.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax.legend(by_label.values(),by_label.keys(),loc='lower center',
              bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=9)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados','validacao_souza2025.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nDiagrama salvo em: {out}")
except Exception as e:
    print(f"\n(Diagrama não gerado: {e})")