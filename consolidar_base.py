"""
=============================================================================
Consolidação da base da CAG — março + abril/maio numa base horária única
=============================================================================

ENTRADAS (todas na mesma pasta):
  base_abr_mai.csv                       -> abril/maio (já tratado)
  export_potencia_*.csv         -> março: kW e %RLA por chiller
  export_dados_gerais_*.csv         -> março: temperaturas + BAGS
  export_estado_*.csv                                -> março: estado on/off dos chillers
  clima_marco.csv                           -> março: clima INMET (horário, UTC)

SAÍDA:
  base_cag.csv   -> base horária unificada, com lags INTRA-DIA

-----------------------------------------------------------------------------
NOTAS METODOLÓGICAS IMPORTANTES (ler antes de usar):

1. ESQUEMA COMUM. Março NÃO possui as 6 temperaturas de evaporador por chiller
   (t_evap_sup/ret_ch1..3), que existem em abril/maio. Essas colunas ficam NaN
   em março, assim como `dt_evap_mean`. Todo o resto é comum aos dois períodos.

2. BASE DE MEDIÇÃO DE kW. Março usa kW por chiller (P03 / objeto AI-62), a MESMA
   base de abril/maio (soma kw_ch1+kw_ch2+kw_ch3). NÃO usar os trafos
   (medidor geral da CAG) para isso: eles medem circuitos adicionais (bombas,
   torres) e leem ~45-55% acima da soma dos compressores.

3. VAZÃO. Nesta etapa Q_TR ainda usa a premissa de vazão constante de
   350 m³/h. Ela é substituída pela vazão estimada a partir da pressão
   diferencial em recalibrar_vazao.py, que gera a base usada nas análises.

4. FUSO HORÁRIO. Os servidores de automação registram em horário de
   Brasília (UTC-3), padrão da rede corporativa. O CSV do INMET traz
   "Hora (UTC)" e é convertido com -3h para alinhar ao relógio do servidor.
   Não há horário de verão a considerar (extinto no Brasil em 2019). A
   escolha foi verificada pela correlação entre entalpia e carga, máxima
   com esse alinhamento.

5. SETPOINT DE MARÇO. Fixo em 5,5 °C durante todo o mês (informado pela
   operação). Não há sinal de setpoint no export de março.

6. LAGS. Todos calculados com groupby(dia) DENTRO da janela 12h-20h, de modo
   que a primeira hora de cada dia não herda o estado operacional de antes da
   abertura. Isso corrige um vazamento presente nas colunas de lag do arquivo
   original (que usavam .shift() sobre a série contínua de 24h e contaminavam
   22-33% das linhas).

7. COBERTURA DE MARÇO. O export P03 termina em 31/03 ao meio-dia e o CH-03 só
   passa a registrar a partir de 03/03 15:40. Resultam ~191h válidas na janela.
=============================================================================
"""

import csv
import glob
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import psychrolib as ps
    ps.SetUnitSystem(ps.SI)
    HAS_PSY = True
except ImportError:
    HAS_PSY = False
    print("psychrolib ausente — usando formulação Magnus/ASHRAE equivalente "
          "(diferença < 0,03 kJ/kg vs psychrolib).")

VAZAO_M3H, CP_AGUA, RHO_AGUA, CONV_TR = 350.0, 4.187, 1000.0, 3.517
SP_MARCO = 5.5


# =============================================================================
# Utilitários
# =============================================================================
def parse_trend(fn):
    """Lê export multi-header do Metasys (blocos de 4 colunas por sinal)."""
    with open(fn, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hi, ho, he, data = rows[4], rows[5], rows[6], rows[12:]
    out = {}
    for si in range(len(ho) // 4):
        i = si * 4
        key = (f"{he[i].replace('Equipment - ','').strip()}|"
               f"{ho[i].replace('Object Name - ','').strip()}|{hi[i]}")
        ts, v = [], []
        for r in data:
            if len(r) <= i + 2:
                continue
            t, val = r[i].strip(), r[i + 2].strip()
            if t == "" or val == "":
                continue
            ts.append(t)
            v.append(val)
        s = pd.Series(
            pd.to_numeric(pd.Series(v), errors="coerce").values,
            index=pd.to_datetime(pd.Series(ts), format="%m/%d/%Y %I:%M:%S %p",
                                 errors="coerce"))
        s = s[~s.index.isna()].sort_index()
        out[key] = s[~s.index.duplicated(keep="last")]
    return out


def entalpia_ar_umido(T, UR, P_hpa=1013.25):
    """Entalpia específica do ar úmido (kJ/kg ar seco)."""
    if HAS_PSY:
        vals = []
        for t, u in zip(T, UR):
            if pd.isna(t) or pd.isna(u):
                vals.append(np.nan)
            else:
                w = ps.GetHumRatioFromRelHum(float(t), float(u) / 100, P_hpa * 100)
                vals.append(ps.GetMoistAirEnthalpy(float(t), w) / 1000)
        return pd.Series(vals, index=T.index)
    Pws = 6.112 * np.exp(17.67 * T / (T + 243.5))
    Pw = (UR / 100) * Pws
    W = 0.622 * Pw / (P_hpa - Pw)
    return 1.006 * T + W * (2501 + 1.86 * T)


def achar(padrao):
    hits = glob.glob(padrao)
    if not hits:
        raise FileNotFoundError(padrao)
    return sorted(hits)[-1]


# =============================================================================
# 1. MARÇO — reconstrução a partir dos exports brutos
# =============================================================================
print("Reconstruindo março a partir dos exports brutos…")

grid = pd.date_range("2026-03-01", "2026-04-01", freq="5min", inclusive="left")


def to_grid(s, limit=12):
    return s.reindex(s.index.union(grid)).sort_index().ffill(limit=limit).reindex(grid)


sig = {}

# --- P03: %RLA (AI-61) e kW (AI-62) por chiller ---
for k, s in parse_trend(achar("export_potencia_*.csv")).items():
    ch = {"12001": "ch1", "12002": "ch2", "12003": "ch3"}[re.search(r"UC800_(\d+)", k).group(1)]
    sig[("rla_" if "RLA" in k else "kw_") + ch] = s

# --- P04: temperaturas gerais da CAG e bombas ---
for k, s in parse_trend(achar("export_dados_gerais_*.csv")).items():
    eq, obj = k.split("|")[0], k.split("|")[1]
    if "TE - 1" in obj:
        sig["t_chw_sup"] = s
    elif "TE - 2" in obj:
        sig["t_chw_ret"] = s
    elif "TE - 3" in obj:
        sig["t_cond_sup"] = s
    elif "TE - 4" in obj:
        sig["t_cond_ret"] = s
    elif "Dif" in obj and eq == "BAGS-1":
        sig["dp_ag_r"] = s
    elif "Modula" in obj and eq == "BAGS-1":
        sig["q_bags1"] = s
    elif "Modula" in obj and eq == "BAGS-R":
        sig["q_bagsr"] = s

# --- estado: estado on/off (COV, precisa ffill sem limite) ---
estado = {}
for k, s in parse_trend(achar("export_estado_*.csv")).items():
    if "Estado UR" in k:
        ch = {"12001": "ch1_on", "12002": "ch2_on", "12003": "ch3_on"}[
            re.search(r"UC800_(\d+)", k).group(1)]
        estado[ch] = s

df5 = pd.DataFrame({k: to_grid(v, limit=12) for k, v in sig.items()})
for k, v in estado.items():
    df5[k] = to_grid(v, limit=10 ** 6)

mar = df5.resample("h").mean()

# --- validação física cruzada do estado (regra da Metodologia: kW > 15 ou RLA > 5%) ---
for i, ch in enumerate(["ch1", "ch2", "ch3"], start=1):
    ativo = (mar[f"kw_{ch}"] > 15) | (mar[f"rla_{ch}"] > 5)
    mar[f"{ch}_on"] = ativo.astype(float)

mar["n_chillers"] = mar[["ch1_on", "ch2_on", "ch3_on"]].sum(axis=1)
mar["kw_total"] = mar[["kw_ch1", "kw_ch2", "kw_ch3"]].sum(axis=1)
mar["sp_chw"] = SP_MARCO

# --- clima INMET (horário, UTC -> relógio do servidor de automação, UTC-3) ---
w = pd.read_csv(achar("clima_marco*.csv"), sep=";", decimal=",",
                encoding="utf-8-sig")
w.columns = [c.strip('"') for c in w.columns]
w["ts"] = pd.to_datetime(
    w["Data"] + " " + w["Hora (UTC)"].astype(str).str.zfill(4),
    format="%d/%m/%Y %H%M") - pd.Timedelta(hours=3)
w = w.set_index("ts")
mar["T_ext"] = w["Temp. Ins. (C)"].reindex(mar.index)
mar["UR_ext"] = w["Umi. Ins. (%)"].reindex(mar.index)
mar["T_orv"] = w["Pto Orvalho Ins. (C)"].reindex(mar.index)
mar["entalpia"] = entalpia_ar_umido(mar["T_ext"], mar["UR_ext"])

mar["origem"] = "marco_reconstruido"


# =============================================================================
# 2. ABRIL/MAIO — carrega base já tratada
# =============================================================================
print("Carregando abril/maio…")
ab = pd.read_csv("base_abr_mai.csv", index_col=0, parse_dates=True)
ab = ab[[c for c in ab.columns
         if "lag" not in c and "roll" not in c and not c.startswith("d_")]].copy()
ab["origem"] = "abril_maio_original"


# =============================================================================
# 3. UNIFICAÇÃO
# =============================================================================
COLS = [
    "ch1_on", "ch2_on", "ch3_on", "n_chillers", "sp_chw",
    "kw_ch1", "kw_ch2", "kw_ch3", "kw_total",
    "t_chw_sup", "t_chw_ret", "t_cond_sup", "t_cond_ret",
    "t_evap_sup_ch1", "t_evap_ret_ch1", "t_evap_sup_ch2", "t_evap_ret_ch2",
    "t_evap_sup_ch3", "t_evap_ret_ch3",
    "rla_ch1", "rla_ch2", "rla_ch3",
    "dp_ag_r", "q_bags1", "q_bagsr",
    "T_ext", "UR_ext", "T_orv", "entalpia",
    "origem",
]
for c in COLS:
    for d in (mar, ab):
        if c not in d.columns:
            d[c] = np.nan

base = pd.concat([mar[COLS], ab[COLS]]).sort_index()
base = base[~base.index.duplicated(keep="last")]


# =============================================================================
# 4. DERIVADAS (calculadas de forma idêntica nos dois períodos)
# =============================================================================
base["dt_chw"] = base["t_chw_ret"] - base["t_chw_sup"]
base["dt_cond"] = base["t_cond_ret"] - base["t_cond_sup"]
base["thermal_lift"] = base["t_cond_ret"] - base["t_chw_sup"]
base["rla_mean"] = base[["rla_ch1", "rla_ch2", "rla_ch3"]].mean(axis=1)
base["dt_evap_mean"] = pd.concat(
    [base[f"t_evap_ret_ch{i}"] - base[f"t_evap_sup_ch{i}"] for i in (1, 2, 3)],
    axis=1).mean(axis=1)

# Q_TR sob premissa de vazão constante (ver nota 3)
base["Q_TR"] = (VAZAO_M3H * RHO_AGUA / 3600 * CP_AGUA * base["dt_chw"]) / CONV_TR
base["COP"] = (base["Q_TR"] * CONV_TR) / base["kw_total"].replace(0, np.nan)

# --- diagnóstico do desequilíbrio hidráulico do CH-03 ---
base["rla_ch12_mean"] = base[["rla_ch1", "rla_ch2"]].mean(axis=1)
base["gap_rla_ch3"] = base["rla_ch12_mean"] - base["rla_ch3"]
base["flag_desbal_ch3"] = ((base["n_chillers"] == 3) & (base["gap_rla_ch3"] > 30)).astype(int)

# calendário e janela de estudo
base["hour"] = base.index.hour
base["day_of_week"] = base.index.dayofweek
base["is_weekend"] = (base["day_of_week"] >= 5).astype(int)
base["month"] = base.index.month
base["operando"] = (base["kw_total"] > 50).astype(int)
base["horario_funcionamento"] = base["hour"].between(10, 22).astype(int)
base["janela_analise"] = (
    (base["day_of_week"] <= 4) & base["hour"].between(12, 20) & (base["kw_total"] > 50)
).astype(int)


# =============================================================================
# 5. LAGS INTRA-DIA (só dentro da janela; nunca herdam hora de fora — nota 6)
# =============================================================================
op = base[base["janela_analise"] == 1].copy()
dia = op.index.normalize()
for col in ["Q_TR", "kw_total", "entalpia", "T_ext", "n_chillers"]:
    g = op.groupby(dia)[col]
    base.loc[op.index, f"{col}_L1"] = g.shift(1)
    base.loc[op.index, f"{col}_L2"] = g.shift(2)
    base.loc[op.index, f"{col}_L3"] = g.shift(3)
    base.loc[op.index, f"{col}_R3"] = g.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())


# =============================================================================
# 6. EXPORTA E RESUME
# =============================================================================
base.to_csv("base_cag.csv")

op = base[base["janela_analise"] == 1]
print("\n" + "=" * 74)
print(f"Base consolidada: {len(base)} horas totais | {len(op)} horas na janela 12h-20h")
print("=" * 74)
print(op.groupby("origem").agg(
    horas=("kw_total", "size"),
    dias=("hour", lambda s: s.index.normalize().nunique()),
    kw_medio=("kw_total", "mean"),
    Q_TR_medio=("Q_TR", "mean"),
    COP_medio=("COP", "mean"),
    entalpia_media=("entalpia", "mean"),
).round(2).to_string())

print("\nCOP por nº de chillers e período:")
print(op.groupby(["origem", "n_chillers"])["COP"].agg(["mean", "count"]).round(3).to_string())

print("\nDesequilíbrio CH-03 (horas com 3 chillers):")
t = op[op["n_chillers"] == 3]
print(t.groupby("origem")[["rla_ch1", "rla_ch2", "rla_ch3", "gap_rla_ch3"]]
      .mean().round(1).to_string())
print(f"\nHoras sinalizadas com desequilíbrio (gap > 30pp): "
      f"{int(op['flag_desbal_ch3'].sum())} de {len(t)} horas com 3 chillers")
print("\n✅ Arquivo gerado: base_cag.csv")
