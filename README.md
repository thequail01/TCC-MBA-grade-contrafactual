# Grade de decisão para central de água gelada

Código que reproduz os resultados do trabalho de conclusão de curso
*Grade de decisão para central de água gelada: sequenciamento supera reset de
setpoint em edifício comercial* (MBA em Data Science e Analytics, USP/Esalq, 2026).

O trabalho desenvolve um instrumento que estima o custo energético de cada
configuração operacional disponível ao operador de uma central de água gelada
(setpoint de água gelada e número de máquinas), a partir de 569 horas de dados
de automação predial distribuídas em três regimes de setpoint.

## Dados

A base (`base_cag_v2.csv`) não é distribuída, e o empreendimento não é
identificado. Os scripts documentam integralmente o procedimento, de modo que
ele pode ser replicado em instalação com instrumentação equivalente. O esquema
esperado da base (uma linha por hora, índice de data e hora) é o produzido por
`recalibrar_vazao.py`; as colunas usadas nas análises são:

| Coluna | Conteúdo |
|---|---|
| `kw_total` | potência elétrica total dos chillers (kW) |
| `n_chillers` | número de máquinas em operação |
| `sp_chw` | setpoint de água gelada (°C) |
| `dt_chw` | diferença entre retorno e envio de água gelada (°C) |
| `dp_bags1`, `V_est` | pressão diferencial (kPa) e vazão estimada (m³/h) |
| `Q_TR`, `Q_TR_cal` | carga térmica estimada com a vazão calibrada (TR) |
| `entalpia`, `T_ext`, `UR_ext` | condições do ar externo (INMET) |
| `hour`, `day_of_week`, `month` | calendário, no relógio do servidor de automação |
| `janela_analise` | 1 nas horas da janela (dias úteis, 12h às 20h, potência > 50 kW, ao menos duas máquinas) |

## Como reproduzir

```bash
pip install -r requirements.txt
python executar_tudo.py              # resultados do artigo
python executar_tudo.py --completo   # inclui os ensaios de robustez
```

Todos os scripts leem `base_cag_v2.csv` do diretório corrente e gravam suas
saídas ali; as figuras vão para `figuras/`, em 300 dpi. A semente aleatória é
fixa em 42. Ambiente de conferência: Python 3.12, pandas 2.3, scikit-learn 1.8
e xgboost 3.4.1.

## Organização

### Núcleo compartilhado

| Arquivo | Conteúdo |
|---|---|
| `comum.py` | carregamento da base e da janela, conjuntos de variáveis, restrições de monotonicidade, hiperparâmetros, seleção do número de árvores, construção das treze partições de validação, linhas de base |
| `estilo.py` | convenções gráficas: vírgula decimal nos eixos e rótulos, gravação das figuras |

Todos os scripts de modelagem importam de `comum.py`, de modo que a janela de
análise, as variáveis e a seleção do número de árvores são uma única
implementação.

### Preparação da base

| Script | Entrada | Saída |
|---|---|---|
| `consolidar_base.py` | exports brutos do sistema de automação e clima do INMET | `base_cag.csv` |
| `ajuste_vazao.py` | export com vazão e pressão diferencial simultâneas (argumento); sem argumento, demonstra a fórmula e a sensibilidade ao expoente | coeficientes da calibração |
| `recalibrar_vazao.py` | `base_cag.csv`, `bags_horario.csv` | `base_cag_v2.csv` |

### Resultados do artigo

| Script | Produz | Onde aparece |
|---|---|---|
| `comparacao_validacao.py` | `comparacao_validacao.csv`, `dispersao_arvores.csv` | Tabelas 5 a 8 |
| `sensibilidade.py` | resultados impressos | Tabela 9, varredura de hiperparâmetros e efeito das restrições do segundo modelo |
| `tabela10_comparacao_pareada.py` | `tabela10_pareada.csv`, `tabela10_estratos.csv` | Tabela 10 |
| `tabela11_grade.py` | `tabela11_grade.csv` | Tabela 11 |
| `acerto_nch.py` | resultados impressos | acurácia mínima exigida do número de máquinas |
| `analise_residuos.py` | `residuos_por_configuracao.csv` | Figura 8 e diagnóstico dos resíduos |

`analise_residuos.py` isola o erro do estágio de potência: o modelo é
alimentado pela carga medida, e não pela estimada, e usa partições próprias de
janela expansiva (ver o cabeçalho do script).

### Ensaios de robustez

| Script | Pergunta |
|---|---|
| `tabela11_grade_robusta.py` | a grade se mantém sem a restrição sobre o número de máquinas? Qual a incerteza por bootstrap de blocos diários? |
| `teste_crossfit_cv.py` | a carga estimada que treina o segundo modelo deve vir de dentro da amostra ou de cross-fitting? |
| `teste_lags.py` | quanto do desempenho do primeiro modelo vem das defasagens da carga? |
| `grid_search.py` | a configuração de hiperparâmetros é robusta a variações conjuntas (81 combinações)? |
| `sensibilidade_janela.py` | a escolha do recorte horário altera as conclusões? |
| `verificacoes_complementares.py` | efeito das restrições do primeiro modelo, retirada da variável mês, Ridge com validação temporal e persistência na primeira hora da janela |

### Figuras

| Figura do artigo | Arquivo | Script |
|---|---|---|
| 1. Perfil horário | `eda1_perfil_horario.png` | `gerar_eda.py` |
| 2. Distribuição da carga | `fig7_distribuicao_carga_IC.png` | `gerar_figuras.py` |
| 3. Regimes operacionais | `eda2_serie_temporal.png` | `gerar_eda.py` |
| 4. Entalpia do ar externo | `eda3_entalpia.png` | `gerar_eda.py` |
| 5. Potência por setpoint | `eda4_setpoint.png` | `gerar_eda.py` |
| 6. Acionamento das máquinas | `eda6_operacao.png` | `gerar_eda.py` |
| 7. Matriz de correlação | `eda5_correlacao.png` | `gerar_eda.py` |
| 8. Resíduos por configuração | `fig8_residuos_configuracao.png` | `analise_residuos.py` |

`gerar_figuras.py` produz ainda figuras complementares que não entram no
artigo. `gerar_eda.py` imprime as estatísticas citadas na análise exploratória,
para conferência do texto.

## Decisões metodológicas registradas no código

- As defasagens da carga são calculadas dentro de cada dia da janela, sem
  herdar horas anteriores à abertura (`comum.carregar`).
- A carga estimada que alimenta o segundo modelo é gerada dentro de cada
  partição, a partir de um primeiro modelo ajustado só no treino dela
  (`comum.cascata`).
- O número de árvores é escolhido por validação interna ao treino, com
  embargo de 24 h nas fronteiras; o teste não participa (`comum.n_arvores`).
- O embargo de 24 h entre treino e teste vale para os esquemas A, B, C e E e é
  dispensado no esquema por regime ausente (`comum.particoes`).
- A regressão Ridge escolhe a regularização por validação cruzada generalizada
  (padrão do scikit-learn); a variante temporal está em
  `verificacoes_complementares.py`.
- Na primeira hora de cada dia a persistência recebe a média do treino, pois a
  hora anterior está fora da janela (`comum.persistencia`).
