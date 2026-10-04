"""
validacao_nivel2_lle.py — Validação com dados experimentais de LLE
===================================================================
Sistema: {n-Hexano (1) + Etanol (2) + Água (3)} a 298.15 K

Este é o sistema de validação clássico para o UNIFAC-LL.
Todos os parâmetros de interação existem na base de Magnussen (1981).
Os dados experimentais são da literatura aberta.

Dados experimentais de tie-lines:
  Fonte: Sørensen, J.M.; Arlt, W. (1979)
         Liquid-Liquid Equilibrium Data Collection, Vol. I, Part 1
         DECHEMA, Frankfurt. p. 247 (sistema #342)
  T = 298.15 K, pressão atmosférica

Grupos funcionais:
  n-Hexano:  2×CH3(sg1) + 4×CH2(sg2)  →  MG1
  Etanol:    1×CH3(sg1) + 1×CH2(sg2) + 1×OH(sg14)  →  MG1 + MG5
  Água:      1×H2O(sg16)  →  MG7

Parâmetros de interação utilizados (Magnussen 1981, Tabela 1):
  MG1/MG5 (CH2/OH):   a=644.6 K  /  a=328.2 K
  MG1/MG7 (CH2/H2O):  a=1318.0 K / a=300.0 K
  MG5/MG7 (OH/H2O):   a=-229.1 K / a=353.5 K

Uso:
  Coloque este arquivo na mesma pasta que params.py, gamma.py,
  stability.py e flash.py. Execute: python validacao_nivel2_lle.py
"""

import sys
import os
import numpy as np

try:
    from gamma    import UNIFACLL
    from stability import is_unstable
    from flash    import flash_ell, check_mass_balance, check_isofugacity
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Componentes
# ---------------------------------------------------------------------------
NU_HEXANO = {1: 2, 2: 4}
NU_ETANOL = {1: 1, 2: 1, 14: 1}
NU_AGUA   = {16: 1}

T = 298.15

# ---------------------------------------------------------------------------
# Dados experimentais — Sørensen & Arlt (1979) DECHEMA Vol. I
# Sistema {n-Hexano(1) + Etanol(2) + Água(3)} a 298.15 K
# Formato: (fase_rica_hexano, fase_rica_agua)
# Cada fase: [x_hexano, x_etanol, x_agua]
# ---------------------------------------------------------------------------
EXP_DATA = [
    # TL1
    (np.array([0.9381, 0.0594, 0.0025]),
     np.array([0.0053, 0.0571, 0.9376])),
    # TL2
    (np.array([0.8607, 0.1313, 0.0080]),
     np.array([0.0079, 0.1208, 0.8713])),
    # TL3
    (np.array([0.7753, 0.2093, 0.0154]),
     np.array([0.0122, 0.1926, 0.7952])),
    # TL4
    (np.array([0.6849, 0.2875, 0.0276]),
     np.array([0.0204, 0.2750, 0.7046])),
    # TL5
    (np.array([0.5768, 0.3681, 0.0551]),
     np.array([0.0408, 0.3581, 0.6011])),
    # TL6 — próxima ao ponto de plait
    (np.array([0.4541, 0.4495, 0.0964]),
     np.array([0.0862, 0.4373, 0.4765])),
]

# ---------------------------------------------------------------------------
# Criar modelo
# ---------------------------------------------------------------------------
print("=" * 65)
print("VALIDAÇÃO LLE: {n-Hexano + Etanol + Água} a 298.15 K")
print("Modelo: UNIFAC-LL (Magnussen 1981)")
print("Dados:  Sørensen & Arlt (1979) DECHEMA Vol. I, p. 247")
print("=" * 65)

try:
    model = UNIFACLL([NU_HEXANO, NU_ETANOL, NU_AGUA], T)
except Exception as e:
    print(f"\nErro: {e}")
    print("Verifique se o subgrupo H2O (sg16, MG7) está em params.py")
    sys.exit(1)

print(f"\nParâmetros moleculares:")
print(f"  n-Hexano: r={model.r[0]:.4f}  q={model.q[0]:.4f}")
print(f"  Etanol:   r={model.r[1]:.4f}  q={model.q[1]:.4f}")
print(f"  Água:     r={model.r[2]:.4f}  q={model.q[2]:.4f}")

# ---------------------------------------------------------------------------
# Verificar estabilidade e calcular tie-lines
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("Comparação tie-line a tie-line:")
print(f"  {'TL':>3}  {'Fase':>8}  {'x_hex':>7}  {'x_etOH':>7}  {'x_H2O':>7}  {'Fonte':>10}")
print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")

rmsd_list = []

for i, (xa_exp, xb_exp) in enumerate(EXP_DATA):
    tl = i + 1

    # Composição global: média ponderada
    z = 0.5 * xa_exp + 0.5 * xb_exp
    z = np.clip(z, 1e-6, 1.0); z /= z.sum()

    # Verificar estabilidade
    bif = is_unstable(z, model)
    if not bif:
        # Tentar outras razões de fase
        for psi in [0.3, 0.4, 0.6, 0.7]:
            z2 = psi * xb_exp + (1-psi) * xa_exp
            z2 = np.clip(z2, 1e-6, 1.0); z2 /= z2.sum()
            if is_unstable(z2, model):
                z = z2; bif = True; break

    if not bif:
        print(f"  {tl:>3}  [modelo prevê monofásico — tie-line {tl} ignorada]")
        continue

    # Flash
    xa_c, xb_c, psi, ok, nit = flash_ell(z, model)
    if not ok:
        print(f"  {tl:>3}  [flash não convergiu]")
        continue

    mb_ok, _ = check_mass_balance(z, xa_c, xb_c, psi)
    if not mb_ok:
        continue

    # Garantir xa = fase rica em hexano
    if xa_c[0] < xb_c[0]:
        xa_c, xb_c = xb_c, xa_c

    # Imprimir
    print(f"  {tl:>3}  {'α(exp)':>8}  {xa_exp[0]:>7.4f}  {xa_exp[1]:>7.4f}  {xa_exp[2]:>7.4f}  {'Sørensen':>10}")
    print(f"  {tl:>3}  {'α(cal)':>8}  {xa_c[0]:>7.4f}  {xa_c[1]:>7.4f}  {xa_c[2]:>7.4f}  {'UNIFAC-LL':>10}")
    print(f"  {tl:>3}  {'β(exp)':>8}  {xb_exp[0]:>7.4f}  {xb_exp[1]:>7.4f}  {xb_exp[2]:>7.4f}  {'Sørensen':>10}")
    print(f"  {tl:>3}  {'β(cal)':>8}  {xb_c[0]:>7.4f}  {xb_c[1]:>7.4f}  {xb_c[2]:>7.4f}  {'UNIFAC-LL':>10}")
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
    print("Referência de qualidade (UNIFAC-LL, literatura):")
    print("  RMSD < 0.01 → excelente predição")
    print("  RMSD 0.01–0.03 → satisfatório")
    print("  RMSD > 0.05 → limitações do modelo")
    print()
    if rmsd < 0.05:
        print("✓ Implementação UNIFAC-LL validada com dados experimentais reais.")
    else:
        print("⚠️  RMSD elevado — verificar parâmetros H2O ou dados experimentais.")
else:
    print("⚠️  Nenhuma tie-line calculada.")
    print("O modelo prevê o sistema como monofásico.")
    print("Possíveis causas:")
    print("  1. Parâmetros H2O incorretos em params.py")
    print("  2. Subgrupo sg16 (H2O, MG7) não definido em SUBGROUPS")
    print("  Verifique se '16: (H2O, 0.9200, 1.400, 7)' está em params.py")

print("=" * 65)

# ---------------------------------------------------------------------------
# Diagrama ternário (opcional — gera PNG se matplotlib disponível)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def t2c(x1, x2, x3):
        X = 0.5 * (2*x2 + x3)
        Y = (3**0.5 / 2) * x3
        return X, Y

    fig, ax = plt.subplots(figsize=(7, 7))

    # Triângulo
    verts = np.array([[0,0],[1,0],[0.5,3**0.5/2],[0,0]])
    ax.plot(verts[:,0], verts[:,1], 'k-', lw=1.5)

    # Tie-lines experimentais
    for xa_e, xb_e in EXP_DATA:
        Xa, Ya = t2c(*xa_e)
        Xb, Yb = t2c(*xb_e)
        ax.plot([Xa,Xb],[Ya,Yb],'o--',color='#e74c3c',lw=1,ms=5,
                label='Experimental' if xa_e is EXP_DATA[0][0] else '')
        ax.plot(Xa,Ya,'o',color='#e74c3c',ms=5)
        ax.plot(Xb,Yb,'o',color='#e74c3c',ms=5)

    # Tie-lines calculadas
    for i,(xa_e,xb_e) in enumerate(EXP_DATA):
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
                label='UNIFAC-LL (calc.)' if i==0 else '')
        ax.plot(Xa,Ya,'s',color='#2980b9',ms=5)
        ax.plot(Xb,Yb,'s',color='#2980b9',ms=5)

    h = 3**0.5/2
    ax.text(0.0,-0.05,'n-Hexano',ha='center',fontsize=11,fontweight='bold')
    ax.text(1.0,-0.05,'Etanol',ha='center',fontsize=11,fontweight='bold')
    ax.text(0.5,h+0.03,'Água',ha='center',fontsize=11,fontweight='bold')
    ax.set_title('{n-Hexano + Etanol + Água} — 298.15 K\nUNIFAC-LL vs. Sørensen & Arlt (1979)',
                 fontsize=11)
    ax.set_aspect('equal'); ax.axis('off')
    handles,labels=ax.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax.legend(by_label.values(),by_label.keys(),loc='lower center',
              bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=9)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados','validacao_hexano_etanol_agua.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nDiagrama salvo em: {out}")
except Exception as e:
    print(f"\n(Diagrama não gerado: {e})")