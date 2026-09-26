"""
=============================================================================
recalibrar_vazao.py — vazão estimada pela pressão diferencial (base v2)
=============================================================================

MOTIVAÇÃO
Sob a premissa de vazão constante de 350 m³/h, o COP de março (3,10) saía
sistematicamente melhor que o de abril (2,65) e o de maio (2,78), o que
contraria a física: março operava com setpoint mais baixo (5,5 °C), o que
deveria piorar o COP, e não melhorá-lo.

DIAGNÓSTICO
As tendências das bombas de água gelada mostraram que março operava com
pressão diferencial cerca de 20% menor (69,8 contra 85 a 87 kPa). Menor ΔP
implica menor vazão; com menos água circulando, a diferença de temperatura
sobe sem que a carga real tenha subido, inflando a carga estimada e o COP.

CALIBRAÇÃO EMPÍRICA (reproduzida em ajuste_vazao.py)
Um export complementar traz vazão medida (m³/h) e ΔP (kPa) simultâneos entre
21/05 e 04/06/2026. Ajuste log-log sobre 1.162 observações em operação
(vazão > 50 m³/h, ΔP > 10 kPa):

        V = 27,5 · ΔP^0,549          (r de Pearson contra √ΔP: 0,676)

O expoente, estimado e não imposto, fica próximo do valor teórico de 0,5
(curva de sistema, V ∝ √ΔP). Vazão média medida em operação: 322 m³/h
(mediana 316).

O sinal de ΔP desse export e o dos exports de março correspondem ao mesmo
ponto físico: correlação de 1,000 e erro médio absoluto de 0,015 kPa nas
1.988 horas sobrepostas, o que torna a calibração transferível para março.

O QUE ESTE SCRIPT PRODUZ
  Q_TR_350 / COP_350   versão com vazão fixa, mantida para comparação
  V_est                vazão estimada hora a hora pela calibração
  Q_TR_cal / COP_cal   versão calibrada; Q_TR passa a ser igual a Q_TR_cal

LIMITAÇÕES
1. A calibração cobre ΔP de 65 a 159 kPa. A média de março (~70 kPa) está
   dentro da faixa, mas na borda inferior, e parte das horas de março fica
   abaixo de 65 kPa (extrapolação). O relatório final lista, por mês, o ΔP
   mínimo e a fração de horas abaixo da faixa calibrada.
2. A calibração foi feita sob setpoint de 7,5 °C. Assume-se que a relação
   ΔP-vazão independe do setpoint (é curva de sistema, e não de máquina),
   hipótese fisicamente razoável, mas não verificada em outros regimes.
3. Opera sempre uma bomba por vez (titular e reserva alternando), o que foi
   verificado nas horas da janela; a calibração vale para essa configuração.
4. As horas com um único chiller são excluídas da janela (ver o filtro
   n_chillers >= 2 abaixo).

SAÍDA: base_cag_v2.csv
=============================================================================
"""

import numpy as np
import pandas as pd

CP, RHO, CONV = 4.187, 1000.0, 3.517
K_VAZAO, N_VAZAO = 27.5, 0.549          # Q = K · ΔP^N  (ajuste empírico)
VAZAO_ANTIGA = 350.0                     # premissa anterior, mantida p/ comparação

base = pd.read_csv("base_cag.csv", index_col=0, parse_dates=True)
bags = pd.read_csv("bags_horario.csv", index_col=0, parse_dates=True)

# --- traz ΔP e modulação das bombas para a base ---
for c in ["dp_bags1", "ao_bags1", "ao_bagsr"]:
    if c in bags.columns:
        base[c] = bags[c].reindex(base.index)

base["ao_bomba_ativa"] = base[["ao_bags1", "ao_bagsr"]].max(axis=1)

# --- vazão estimada e rótulos recalibrados ---
base["V_est"] = K_VAZAO * base["dp_bags1"].clip(lower=1) ** N_VAZAO
base["Q_TR_350"] = (VAZAO_ANTIGA * RHO / 3600 * CP * base["dt_chw"]) / CONV
base["Q_TR_cal"] = (base["V_est"] * RHO / 3600 * CP * base["dt_chw"]) / CONV
kw = base["kw_total"].replace(0, np.nan)
base["COP_350"] = base["Q_TR_350"] * CONV / kw
base["COP_cal"] = base["Q_TR_cal"] * CONV / kw

# Q_TR passa a ser o calibrado (rótulo principal dos modelos).
base["Q_TR"] = base["Q_TR_cal"]
base["COP"] = base["COP_cal"]

# --- FILTRO n_chillers >= 2 -------------------------------------------------
# Com um único chiller ativo, o dt_chw medido no barrilete comum não
# representa o ΔT do evaporador: há mistura com água que não passou pela
# máquina ativa, o que infla o ΔT e a carga estimada. Nas 9 horas afetadas
# (todas em 02/03/2026) o COP calculado fica entre 4,54 e 4,79, incoerente
# com o restante da série (média em torno de 2,7 sob a mesma calibração).
# Essas horas saem da janela de análise.
n_antes = int(base["janela_analise"].sum())
base.loc[base["n_chillers"] < 2, "janela_analise"] = 0
print(f"Filtro n_chillers>=2: {n_antes} -> {int(base['janela_analise'].sum())} horas "
      f"({n_antes - int(base['janela_analise'].sum())} removidas)")

# --- lags intra-dia recalculados sobre o novo rótulo ---
op = base[base["janela_analise"] == 1].copy().sort_index()
dia = op.index.normalize()
for col in ["Q_TR", "kw_total", "entalpia", "T_ext", "n_chillers"]:
    g = op.groupby(dia)[col]
    base.loc[op.index, f"{col}_L1"] = g.shift(1)
    base.loc[op.index, f"{col}_L2"] = g.shift(2)
    base.loc[op.index, f"{col}_L3"] = g.shift(3)
    base.loc[op.index, f"{col}_R3"] = g.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())

base.to_csv("base_cag_v2.csv")

# --- relatório ---
op = base[base["janela_analise"] == 1]
op = op.assign(mes=op.index.to_period("M").astype(str))
print("=" * 78)
print("EFEITO DA RECALIBRAÇÃO")
print("=" * 78)
print(op.groupby("mes")[["dp_bags1", "V_est", "dt_chw",
                         "Q_TR_350", "Q_TR_cal", "COP_350", "COP_cal"]]
      .mean().round(2).to_string())
print("\nkW por TR entregue:")
print(op.groupby("mes")[["kw_total", "Q_TR_350", "Q_TR_cal"]].apply(lambda d: pd.Series({
    "kW/TR_350": (d["kw_total"] / d["Q_TR_350"]).mean(),
    "kW/TR_cal": (d["kw_total"] / d["Q_TR_cal"]).mean()})).round(3).to_string())
print(f"\nDispersão do COP entre meses: "
      f"antes={op.groupby('mes')['COP_350'].mean().std():.3f}  "
      f"depois={op.groupby('mes')['COP_cal'].mean().std():.3f}")
LIM_INF, LIM_SUP = 65.0, 159.0     # faixa de ΔP coberta pela calibração (kPa)
print("\nΔP na janela de análise, por mês (faixa calibrada: 65 a 159 kPa):")
print(op.groupby("mes")["dp_bags1"].agg(
    minimo="min", media="mean", maximo="max",
    abaixo_da_faixa_pct=lambda x: 100 * (x < LIM_INF).mean()).round(1).to_string())
print("\nbase_cag_v2.csv gerado.")
