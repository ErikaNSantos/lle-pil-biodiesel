import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import brentq, fsolve

# ── Configuração visual ──────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['figure.dpi'] = 300

# ── Modelo de solução regular ────────────────────────────────────
def gmix(x1, A_RT):
    """ΔG_mix / RT para solução regular."""
    x1 = np.clip(x1, 1e-12, 1 - 1e-12)
    x2 = 1 - x1
    ideal = x1 * np.log(x1) + x2 * np.log(x2)
    excess = A_RT * x1 * x2
    return ideal + excess

def d2gmix(x1, A_RT):
    """Segunda derivada de ΔG_mix/RT em relação a x1."""
    x1 = np.clip(x1, 1e-12, 1 - 1e-12)
    x2 = 1 - x1
    return 1.0 / x1 + 1.0 / x2 - 2 * A_RT

def dg_dx(x1, A_RT):
    """Primeira derivada de ΔG_mix/RT em relação a x1."""
    x1 = np.clip(x1, 1e-12, 1 - 1e-12)
    x2 = 1 - x1
    return np.log(x1) - np.log(x2) + A_RT * (1 - 2 * x1)

# ── Encontrar pontos espinodais (d²G/dx² = 0) ───────────────────
def find_spinodal(A_RT):
    """Retorna as duas composições espinodais."""
    xs1 = brentq(d2gmix, 0.01, 0.49, args=(A_RT,))
    xs2 = brentq(d2gmix, 0.51, 0.99, args=(A_RT,))
    return xs1, xs2

# ── Encontrar pontos binodais (tangente dupla) ───────────────────
def find_binodal(A_RT):
    """Retorna as duas composições binodais."""
    xs1, xs2 = find_spinodal(A_RT)

    def system(vars):
        x1a, x1b = vars
        ga = gmix(x1a, A_RT)
        gb = gmix(x1b, A_RT)
        dga = dg_dx(x1a, A_RT)
        dgb = dg_dx(x1b, A_RT)
        eq1 = dga - dgb  # derivadas iguais
        eq2 = (gb - ga) / (x1b - x1a) - dga  # inclinação = derivada
        return [eq1, eq2]

    x0 = [xs1 * 0.7, xs2 + (1 - xs2) * 0.3]
    xb1, xb2 = fsolve(system, x0, full_output=False)
    return xb1, xb2

# ── Dados para os gráficos ───────────────────────────────────────
# Estendido um pouco mais o X para encostar perfeitamente nas bordas
x = np.linspace(1e-6, 1 - 1e-6, 500)

# (a) Sistema estável
A_RT_a = 1.0
g_a = gmix(x, A_RT_a)

# (b) Sistema com separação de fases
A_RT_b = 2.5
g_b = gmix(x, A_RT_b)
xs1, xs2 = find_spinodal(A_RT_b)
xb1, xb2 = find_binodal(A_RT_b)

# Tangente dupla
gb1 = gmix(xb1, A_RT_b)
gb2 = gmix(xb2, A_RT_b)
slope = (gb2 - gb1) / (xb2 - xb1)
x_tang = np.linspace(xb1, xb2, 100)
g_tang = gb1 + slope * (x_tang - xb1)

# ── Gerar figura ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.0), sharey=False)

# ── Painel (a): Sistema estável ──────────────────────────────────
ax1.plot(x, g_a, 'k-', linewidth=1.5)
ax1.set_xlabel(r'Fração molar $x_1$')
ax1.set_ylabel(r'$\Delta G_{mix}\,/\,RT$')
ax1.set_title('(a) Mistura estável', fontsize=11, pad=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.55, 0.05)  
ax1.text(0.5, gmix(0.5, A_RT_a) - 0.03, 'fase única\n(estável)',
         ha='center', va='top', fontsize=9.5, style='italic',
         color='#2166ac')
# Removida a linha: ax1.axhline(0, color='gray', linewidth=0.5, linestyle='-')

# ── Painel (b): Separação de fases ───────────────────────────────
ax2.plot(x, g_b, 'k-', linewidth=1.5, label=r'$\Delta G_{mix}/RT$')
ax2.plot(x_tang, g_tang, '--', color='#b2182b', linewidth=1.2,
         label='tangente dupla')

# Pontos binodais com flechas verticais (90°)
ax2.plot([xb1, xb2], [gb1, gb2], 'o', color='#b2182b', markersize=7, zorder=5)
ax2.annotate(r'$x_1^{\,I}$ (binodal)', xy=(xb1, gb1),
             xytext=(xb1, gb1 + 0.06), 
             fontsize=9, color='#b2182b', ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#b2182b', lw=0.8))
ax2.annotate(r'$x_1^{\,II}$ (binodal)', xy=(xb2, gb2),
             xytext=(xb2, gb2 + 0.06), 
             fontsize=9, color='#b2182b', ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#b2182b', lw=0.8))

# Pontos espinodais com flechas verticais (90°)
gs1, gs2 = gmix(xs1, A_RT_b), gmix(xs2, A_RT_b)
ax2.plot([xs1, xs2], [gs1, gs2], 's', color='#1a9850', markersize=6, zorder=5)
ax2.annotate('espinodal', xy=(xs1, gs1),
             xytext=(xs1, gs1 + 0.04), 
             fontsize=9, color='#1a9850', ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#1a9850', lw=0.8))
ax2.annotate('espinodal', xy=(xs2, gs2),
             xytext=(xs2, gs2 + 0.04), 
             fontsize=9, color='#1a9850', ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#1a9850', lw=0.8))

# Regiões - Instável
x_mid = 0.5
g_mid = gmix(x_mid, A_RT_b)
ax2.text(0.5, g_mid - 0.022, 'instável',
         ha='center', va='bottom', fontsize=9.5, style='italic',
         color='#1a9850')

# Regiões - Meta-estável
x_meta_l = (xb1 + xs1) / 2
x_meta_r = (xb2 + xs2) / 2
ax2.text(x_meta_l, gmix(x_meta_l, A_RT_b) + 0.025, 'meta-\nestável',
         ha='center', va='bottom', fontsize=8.5, style='italic',
         color='#762a83')
ax2.text(x_meta_r, gmix(x_meta_r, A_RT_b) + 0.025, 'meta-\nestável',
         ha='center', va='bottom', fontsize=8.5, style='italic',
         color='#762a83')

ax2.set_xlabel(r'Fração molar $x_1$')
ax2.set_ylabel(r'$\Delta G_{mix}\,/\,RT$')
ax2.set_title('(b) Separação de fases', fontsize=11, pad=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.14, 0.05) 
# Removida a linha: ax2.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax2.legend(fontsize=9, loc='lower center', framealpha=0.9, ncol=2)

# ── Ajustes finais ───────────────────────────────────────────────
plt.tight_layout(w_pad=3.0)
plt.savefig('fig1_gmix.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig1_gmix.png', bbox_inches='tight', dpi=300)