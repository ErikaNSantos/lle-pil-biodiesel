"""
Figura 3 — Esquema comparativo CA vs C+A (Correção Definitiva de Sobreposição)
Seção 2.4 do TCC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

# ── Configuração visual ──────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 300

# Criar a figura
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# Função auxiliar com COMPENSAÇÃO MATEMÁTICA de padding para evitar sobreposição
def draw_box(x, y, w, h, text, color):
    pad = 0.15
    # Subtraímos o dobro do padding para que a dimensão final com a borda seja exatamente 'w' e 'h'
    drawn_w = w - 2 * pad
    drawn_h = h - 2 * pad
    
    box = FancyBboxPatch((x - drawn_w/2, y - drawn_h/2), drawn_w, drawn_h, 
                         boxstyle=f"round,pad={pad}",
                         ec="#333333", fc=color, lw=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, zorder=3, color='black')

def draw_arrow(x1, y1, x2, y2):
    # zorder=1 garante que as pontas das setas fiquem por baixo da caixa, dando acabamento limpo
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='#555555'),
                zorder=1)

# ── Cores ────────────────────────────────────────────────────────
c_input = '#f4a582'   # Laranja/Salmão suave
c_process = '#d1e5f0' # Azul bem claro
c_output = '#92c5de'  # Azul médio

# ── Posições da Grade ────────────────────────────────────────────
x_col1 = 2.5
x_col2 = 7.5
y_title = 5.5
y_level1 = 4.2
y_level2 = 2.5
y_level3 = 1.0

# Novas dimensões controladas
box_h = 0.9        # Altura ligeiramente maior para o texto respirar
box_w_large = 4.0  # Largura das caixas grandes
box_w_small = 1.8  # Largura das caixas pequenas

# ── Títulos e Divisão ────────────────────────────────────────────
ax.text(x_col1, y_title, "Abordagem CA\n(Tratamento como molécula neutra)", 
        ha='center', va='center', fontsize=12, weight='bold')
ax.text(x_col2, y_title, "Abordagem C + A\n(Tratamento como mistura de íons)", 
        ha='center', va='center', fontsize=12, weight='bold')

# Linha divisória vertical
ax.plot([5, 5], [0, 6], ':', color='gray', lw=1.5, alpha=0.6)

# ── Coluna Esquerda: Abordagem CA ───────────────────────────────
draw_box(x_col1, y_level1, box_w_large, box_h, "[ C $\\cdot$ A ]\n(Par Iônico associado)", c_input)
draw_arrow(x_col1, y_level1 - box_h/2, x_col1, y_level2 + box_h/2)

draw_box(x_col1, y_level2, box_w_large, box_h, "Cálculo Quântico Único\n(Otimização do par)", c_process)
draw_arrow(x_col1, y_level2 - box_h/2, x_col1, y_level3 + box_h/2)

draw_box(x_col1, y_level3, box_w_large, box_h, "Geração de 1 único\n$\\sigma$-profile combinado", c_output)


# ── Coluna Direita: Abordagem C+A ───────────────────────────────
# Afastamento lateral das caixas pequenas calculado matematicamente (gap no meio será de 0.4)
x_col2_left = x_col2 - 1.1 
x_col2_right = x_col2 + 1.1

# Nível 1: Inputs isolados
draw_box(x_col2_left, y_level1, box_w_small, box_h, "[ C ]$^+$", c_input)
draw_box(x_col2_right, y_level1, box_w_small, box_h, "[ A ]$^-$", c_input)
ax.text(x_col2, y_level1, "+", ha='center', va='center', fontsize=16, weight='bold', color='#555555')

# Setas Nível 1 para 2
draw_arrow(x_col2_left, y_level1 - box_h/2, x_col2_left, y_level2 + box_h/2)
draw_arrow(x_col2_right, y_level1 - box_h/2, x_col2_right, y_level2 + box_h/2)

# Nível 2: Processos
draw_box(x_col2_left, y_level2, box_w_small, box_h, "Cálculo\nCátion", c_process)
draw_box(x_col2_right, y_level2, box_w_small, box_h, "Cálculo\nÂnion", c_process)

# Setas do Nível 2 para 3 (convergentes para o centro)
draw_arrow(x_col2_left, y_level2 - box_h/2, x_col2 - 0.2, y_level3 + box_h/2)
draw_arrow(x_col2_right, y_level2 - box_h/2, x_col2 + 0.2, y_level3 + box_h/2)

# Nível 3: Output misto
draw_box(x_col2, y_level3, box_w_large, box_h, "Mistura de 2 $\\sigma$-profiles\n(proporção 1:1)", c_output)


# ── Finalização ──────────────────────────────────────────────────
ax.axis('off')
plt.tight_layout()
plt.savefig('fig3_esquema_CA.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig3_esquema_CA.png', bbox_inches='tight', dpi=300)
print("Figuras salvas com sucesso.")
plt.show()