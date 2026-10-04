"""
Figura 2 — Diagrama Ternário Esquemático (Curva Suave)
Seção 2.1 do TCC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Configuração visual ──────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 300

def to_cartesian(xA, xB, xC):
    """Converte frações molares ternárias para coordenadas cartesianas."""
    tot = xA + xB + xC
    xA, xB, xC = xA/tot, xB/tot, xC/tot
    x = xB + 0.5 * xC
    y = xC * np.sqrt(3) / 2
    return x, y

def plot_ternary_grid(ax):
    """Desenha as linhas de grade internas do diagrama ternário."""
    ticks = np.linspace(0.1, 0.9, 9)
    for t in ticks:
        x1, y1 = to_cartesian(t, 1-t, 0)
        x2, y2 = to_cartesian(t, 0, 1-t)
        ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=0.3, alpha=0.5, zorder=0)
        
        x1, y1 = to_cartesian(1-t, t, 0)
        x2, y2 = to_cartesian(0, t, 1-t)
        ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=0.3, alpha=0.5, zorder=0)
        
        x1, y1 = to_cartesian(1-t, 0, t)
        x2, y2 = to_cartesian(0, 1-t, t)
        ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=0.3, alpha=0.5, zorder=0)

# ── Inicialização da Figura ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
plot_ternary_grid(ax)

# Triângulo Externo
corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
xc, yc = to_cartesian(corners[:,0], corners[:,1], corners[:,2])
ax.plot(xc, yc, 'k-', linewidth=1.5, zorder=1)

# ── Geração da Curva Binodal (Arredondada e Suave) ───────────────
# Vamos criar uma parábola invertida suave nas coordenadas (xB, xC)
t = np.linspace(0, 1, 200)

# Raízes na base (eixo xC = 0)
xB_start = 0.10
xB_end = 0.90
xB_mid = 0.45  # Centro da cúpula deslocado para a esquerda (típico de plait point)

# Altura máxima da cúpula (fração molar de C)
xC_max = 0.40 

# Construção de uma curva de Bézier quadrática suave para xB e xC
# P0 = inicio, P1 = controle, P2 = fim
P0 = np.array([xB_start, 0.0])
P2 = np.array([xB_end, 0.0])
P1 = np.array([xB_mid, xC_max * 2])  # O ponto de controle precisa ser mais alto para o máximo ser xC_max

# Curva paramétrica de Bezier
curve = np.outer((1-t)**2, P0) + np.outer(2*(1-t)*t, P1) + np.outer(t**2, P2)
xB_binodal = curve[:, 0]
xC_binodal = curve[:, 1]
xA_binodal = 1 - xB_binodal - xC_binodal

x_cart, y_cart = to_cartesian(xA_binodal, xB_binodal, xC_binodal)
ax.plot(x_cart, y_cart, '-', color='#b2182b', linewidth=2, label='Curva binodal', zorder=3)

# ── Ponto Plait (Pliegue) ────────────────────────────────────────
# O plait point fica próximo ao topo da curva suave
idx_plait = np.argmax(xC_binodal)  # Pegando o ponto mais alto para o plait point
xp, yp = x_cart[idx_plait], y_cart[idx_plait]
ax.plot(xp, yp, 'o', color='#1a9850', markersize=8, zorder=5, label='Ponto plait')

# ── Geração das Tie Lines ────────────────────────────────────────
# Distribuímos as tie lines entre as raízes e o ponto plait
num_ties = 5
idx_left = np.linspace(0, idx_plait - 5, num_ties + 1, dtype=int)[1:]  
idx_right = np.linspace(199, idx_plait + 5, num_ties + 1, dtype=int)[1:]

# Para dar inclinação termodinâmica (coeficiente de partição), 
# deslocamos os índices de forma assimétrica
for i in range(num_ties):
    i_L = idx_left[i]
    i_R = idx_right[i]
    
    # Criando uma leve inclinação ajustando os índices
    shift = int((num_ties - i) * 3)
    i_R = max(idx_plait + 2, i_R - shift)
    
    xl, yl = x_cart[i_L], y_cart[i_L]
    xr, yr = x_cart[i_R], y_cart[i_R]
    
    ax.plot([xl, xr], [yl, yr], '--', color='#2166ac', linewidth=1.5, zorder=2)

ax.plot([], [], '--', color='#2166ac', linewidth=1.5, label='Linhas de amarração')

# ── Textos e Legendas ────────────────────────────────────────────
offset = 0.03
ax.text(0 - offset, -offset, 'Solvente A', ha='right', va='top', fontsize=12)
ax.text(1 + offset, -offset, 'Solvente B', ha='left', va='top', fontsize=12)
ax.text(0.5, np.sqrt(3)/2 + offset, 'Soluto C', ha='center', va='bottom', fontsize=12)

bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)

ax.text(0.5, 0.12, 'Região\nbifásica', ha='center', va='center', 
        style='italic', bbox=bbox_props, zorder=4)
ax.text(0.5, 0.55, 'Fase única', ha='center', va='center', 
        style='italic', color='#2166ac', bbox=bbox_props, zorder=4)

ax.set_aspect('equal')
ax.axis('off')
ax.legend(loc='upper right', bbox_to_anchor=(1.05, 0.95), frameon=False)

# ── Salvar ───────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig('fig2_diagrama_ternario.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig2_diagrama_ternario.png', bbox_inches='tight', dpi=300)
print("Figuras salvas: fig2_diagrama_ternario.pdf e fig2_diagrama_ternario.png")
plt.show()