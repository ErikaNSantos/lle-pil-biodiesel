# lle-pil-biodiesel

TCC em Engenharia Química (UFBA) — modelagem termodinâmica de equilíbrio
líquido-líquido (ELL) de sistemas {óleo/ácido graxo + álcool + líquido iônico
protico}, testando a hipótese de cossolvência de líquidos iônicos próticos em
sistemas de biodiesel. Pesquisa financiada pela ANP.

Tese completa: [`TCC_Biodiesel_CONCLUIDO.pdf`](TCC_Biodiesel_CONCLUIDO.pdf).
Fonte LaTeX em [`thesis/`](thesis/).

## Ferramenta de Diagramas Ternários de ELL — UNIFAC-LL / COSMO-SAC

Aplicação única com interface gráfica para calcular diagramas ternários de
equilíbrio líquido-líquido dos sistemas acima, usando UNIFAC-LL (Python puro)
ou COSMO-SAC (via JCOSMO/py4j). Substitui os ~10 scripts separados anteriores
(mantidos em [`code/scripts_antigos/`](code/scripts_antigos/) por histórico).

### Como executar

```
python code/interface_ell.py
```

Dependências: `numpy`, `scipy`, `matplotlib` (e `python3-tk` para a interface).
Para o caminho COSMO-SAC: `py4j` instalado e **JCOSMO aberto** antes do
cálculo (o JCOSMO em si não faz parte deste repositório).

### Estrutura

| Arquivo | Papel |
|---|---|
| `code/interface_ell.py` | Interface Tkinter + lógica de controle. **Ponto de entrada.** |
| `code/unifac_ll.py` | Modelo UNIFAC-LL (grupos, decomposições, matriz de interação, γ). |
| `code/thermo_engine.py` | Estabilidade (plano tangente), flash (Rachford-Rice), varredura da grade. |
| `code/ternary_diagram.py` | Renderização do diagrama, amostragem de tie-lines, exportação PNG/CSV. |
| `code/cosmo_sac_adapter.py` | Adaptador COSMO-SAC via py4j, mesma assinatura `gamma(x)`. |
| `code/batch_runner.py` / `code/diagnostico_cosmo_local.py` / `code/plait_solver.py` / `code/analise_cossolvencia.py` | Execução em lote, diagnóstico local do COSMO, solver auxiliar e análise consolidada da hipótese de cossolvência. |
| `code/teoricos/` | Scripts ilustrativos usados nas figuras teóricas da tese (ΔGmix, diagrama ternário esquemático). |
| `code/scripts_antigos/` | Scripts de versões anteriores, mantidos por histórico (não são o caminho atual). |
| `code/resultados_antigos/` | Resultados de versões anteriores do modelo, mantidos por histórico. |
| `resultados/` | Diagramas (PNG) e tie-lines (CSV) atuais, um por sistema/modelo. |

Os módulos do motor, diagrama e modelos são **agnósticos ao modelo**: a mesma
lógica de estabilidade/flash/diagrama serve ao UNIFAC-LL e ao COSMO-SAC porque
ambos expõem `gamma(x)` idêntico.

### Estado de validação

Testado e confirmado:

- **UNIFAC-LL**: limite γᵢ→1 nos vértices puros (exato); consistência interna do
  termo combinatorial (spec vs L_i, desvio ~2×10⁻¹⁴); detector de pares faltantes.
- **Motor termodinâmico**: validado contra Margules com gap analítico conhecido —
  detecta o gap quando existe, recupera o binodal analítico com exatidão, não
  gera falso split e não colapsa em solução trivial (RMSD 2,50×10⁻⁷).
- **Validação externa**: contra dados de Sørensen & Arlt (1980) para o sistema
  n-hexano/etanol/água (RMSD 0,0872).
- **Diagrama**: pipeline completo exercitado com modelo sintético tipo 1
  (binodal, 7 tie-lines, exportações) e o caso miscível tratado com aviso.
- **Interface**: smoke test sob framebuffer — construção da janela, troca de
  modelo, verificação de parâmetros, cálculo em thread com fila, barra de
  progresso, figura embutida e reabilitação de botão.

### Resultado relevante (UNIFAC-LL nos sistemas com trioleína)

Com a representação de par neutro e o parâmetro (11,41) CCOO/COO já corrigido
(−122,3 K), o UNIFAC-LL prevê **fase única em todo o simplex** para
{trioleína + etanol + PIL} — sem região bifásica. Confirmado por três vias
independentes (varredura de estabilidade na grade, varredura das bordas
binárias, convexidade direta da energia de Gibbs de mistura).

Mecanismo: no par trioleína/etanol o termo residual dá ln γ∞ = +9,78 (que
sozinho geraria imiscibilidade), mas o combinatorial dá ln γ∞ = −7,56; a
assimetria de tamanho (r ≈ 39 da trioleína contra ≈ 2,6 do etanol) produz
estabilização entrópica que suprime o gap. É a limitação de extrapolação
(parâmetros regredidos em ésteres C4–C6) que sustenta a escolha do COSMO-SAC
como modelo principal. Os binodais da tese vêm do COSMO-SAC, rodado localmente,
usando a temperatura e as concentrações efetivamente empregadas nos ensaios
(diferentes das condições da literatura consultada).

**Conclusão da tese**: a hipótese de cossolvência não se confirmou nos termos
testados — os dois modelos divergem conforme o líquido iônico, e o gap entre
previsão e dado experimental amplia (2,7×–15×) em vez de fechar.

### Itens em aberto

1. **Pares COOH ausentes** (sistemas com ácido graxo): o detector sinaliza
   COOH com CH2, C=C, OH (clássicos, a preencher do Magnussen/`binary.csv`) e
   COOH com CNH2, CNH, COO (sem valor publicado; aguardam ML de Hayer et al.).
   Preencher via o `.csv` das matrizes faltantes assim que disponível — a
   estrutura já reserva o lugar. Nenhum valor foi inventado.
2. **Estequiometria do [DETA][Hx]**: resolvido — par 1:3 **triprotonado**,
   fórmula corrigida para **C₂₂H₄₉N₃O₆** (a decomposição em grupos já estava
   correta na spec).
3. **Recalibração V2**: os parâmetros V2 (CCOO/CNH2 e CNH2/COO) estão como na
   spec. Se a recalibração contra a matriz corrigida gerar valores novos, é
   troca de uma linha em `code/unifac_ll.py` (bloco "estimados neste trabalho").

## Licença

Código sob [MIT](LICENSE). O texto da tese (`thesis/`, PDF) é disponibilizado
para consulta acadêmica.
