"""
=============================================================================
ANÁLISE EXPLORATÓRIA DE DADOS (AED)
=============================================================================

Gera as figuras da seção de caracterização do conjunto de dados, anteriores
aos resultados de modelagem.

ENTRADA : base_cag_v2.csv
SAÍDA   : figuras/eda*.png  (300 dpi, prontas para o Word)
          e, no terminal, as estatísticas citadas no texto da seção

CONVENÇÃO DE UNIDADES
A carga térmica é apresentada como índice de carga (IC), normalizado pela
média da série. A escala absoluta em TR não foi validada por balanço
energético, conforme declarado na Metodologia. Por isso o eixo secundário
em TR equivalentes vem desligado (EIXO_TR = False).

CORRESPONDÊNCIA COM O ARTIGO
  eda1_perfil_horario   Figura 1
  eda2_serie_temporal   Figura 3
  eda3_entalpia         Figura 4
  eda4_setpoint         Figura 5
  eda5_correlacao       Figura 7
  eda6_operacao         Figura 6
  (a Figura 2 é gerada por gerar_figuras.py)

FIGURAS PRODUZIDAS
  eda1_perfil_horario   Perfil médio por hora do dia (carga, potência, COP relativo)
  eda2_serie_temporal   Série completa de carga e potência, com marcação de regimes
  eda3_entalpia         Comportamento da entalpia externa: série e distribuição horária
  eda4_setpoint         Efeito do setpoint sobre a potência: distribuição e perfil
  eda5_correlacao       Matriz de correlação entre as variáveis candidatas
  eda6_operacao         Padrão de acionamento das máquinas ao longo do período
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from estilo import br, nota, salvar

warnings.filterwarnings("ignore")

OUT = "figuras"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

AZUL, LARANJA, VERDE, CINZA, ESC = "#1F4E79", "#DD8452", "#55A868", "#9AA5B1", "#C44E52"
ROXO = "#8172B3"
EIXO_TR = False     # eixo secundário em TR equivalentes na Figura 1

# -----------------------------------------------------------------------------
# Carga dos dados e derivadas de apoio
# -----------------------------------------------------------------------------
b = pd.read_csv("base_cag_v2.csv", index_col=0, parse_dates=True)
op = b[b["janela_analise"] == 1].copy().sort_index()

IC_REF = op["Q_TR_cal"].mean()          # 1,00 = carga média da série
op["IC"] = op["Q_TR_cal"] / IC_REF
op["mes"] = op.index.to_period("M").astype(str)
op["dia"] = op.index.normalize()
# eficiência relativa: carga entregue por unidade de potência, normalizada
op["ef_rel"] = (op["IC"] / op["kw_total"])
op["ef_rel"] = op["ef_rel"] / op["ef_rel"].mean()

MESES = {"2026-03": "Março (SP 5,5 °C)",
         "2026-04": "Abril (SP variável)",
         "2026-05": "Maio (SP 7,5 °C)"}
CORES_MES = {"2026-03": AZUL, "2026-04": LARANJA, "2026-05": VERDE}


# =============================================================================
# EDA 1 — Perfil horário médio
# =============================================================================
def eda1_perfil_horario():
    """
    Mostra como carga, potência e eficiência relativa variam ao longo do dia.
    Justifica a escolha da janela 12h-20h e evidencia o pico de abertura.
    """
    g = op.groupby("hour").agg(ic=("IC", "mean"), ic_sd=("IC", "std"),
                               kw=("kw_total", "mean"), kw_sd=("kw_total", "std"),
                               ef=("ef_rel", "mean"), n=("IC", "size"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))

    a1.plot(g.index, g["ic"], "-o", color=AZUL, linewidth=2, markersize=5, label="Índice de carga")
    a1.fill_between(g.index, g["ic"] - g["ic_sd"], g["ic"] + g["ic_sd"],
                    color=AZUL, alpha=0.13, label="± 1 desvio-padrão")
    a1.axhline(1.0, color=CINZA, linestyle=":", linewidth=1)
    a1.set_xlabel("Hora do dia"); a1.set_ylabel("Índice de carga (IC)")
    a1.set_title("Carga térmica cai ao longo da tarde")
    a1.set_xticks(range(12, 21))
    a1.legend(frameon=False, loc="upper right")
    if EIXO_TR:     # eixo secundário em TR equivalentes, apenas como referência
        a1b = a1.twinx()
        a1b.set_ylim(np.array(a1.get_ylim()) * IC_REF)
        a1b.set_ylabel("TR equivalentes", fontsize=8, color="#777777")
        a1b.tick_params(labelsize=7.5, colors="#777777")
        a1b.grid(False)

    a2.plot(g.index, g["kw"], "-o", color=ESC, linewidth=2, markersize=5, label="Potência")
    a2.fill_between(g.index, g["kw"] - g["kw_sd"], g["kw"] + g["kw_sd"],
                    color=ESC, alpha=0.13)
    a2.set_xlabel("Hora do dia"); a2.set_ylabel("Potência elétrica (kW)", color=ESC)
    a2.tick_params(axis="y", colors=ESC)
    a2.set_xticks(range(12, 21))
    a2.set_title("Potência acompanha a carga, com eficiência estável")
    a2e = a2.twinx()
    a2e.plot(g.index, g["ef"], "--s", color=VERDE, linewidth=1.7, markersize=4)
    a2e.set_ylabel("Eficiência relativa (média = 1,00)", color=VERDE, fontsize=8.5)
    a2e.tick_params(axis="y", colors=VERDE, labelsize=7.5)
    a2e.grid(False)
    a2e.set_ylim(0.85, 1.15)

    nota(fig, 0.5, -0.06, f"Janela de análise: dias úteis, 12h às 20h, com duas ou mais máquinas em operação (n = {len(op)} h).",
             ha="center", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda1_perfil_horario.png")
    plt.close(fig)


# =============================================================================
# EDA 2 — Série temporal e regimes
# =============================================================================
def eda2_serie_temporal():
    """
    Série diária de carga e potência, com as fronteiras dos três regimes de
    setpoint. Evidencia a estabilidade da carga frente à mudança de regime.
    """
    d = op.groupby("dia").agg(ic=("IC", "mean"), kw=("kw_total", "mean"),
                              sp=("sp_chw", "mean"), ent=("entalpia", "mean"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})

    for ax, col, cor, rot in [(a1, "ic", AZUL, "Índice de carga (IC)"),
                              (a2, "kw", ESC, "Potência média (kW)")]:
        ax.plot(d.index, d[col], "-o", color=cor, linewidth=1.4, markersize=3.2)
        ax.set_ylabel(rot)
        for lim in ["2026-04-01", "2026-05-01"]:
            ax.axvline(pd.Timestamp(lim), color=CINZA, linestyle="--", linewidth=1)

    a1.axhline(1.0, color=CINZA, linestyle=":", linewidth=1)
    a1.set_title("Índice de carga ao longo dos três regimes operacionais")
    a2.set_title("Potência média diária ao longo dos três regimes")
    a2.set_xlabel("Dia útil")

    # rótulos dos regimes
    ymax = a1.get_ylim()[1]
    for x, txt in [("2026-03-16", "Março\nSP 5,5 °C"),
                   ("2026-04-15", "Abril\nSP variável"),
                   ("2026-05-15", "Maio\nSP 7,5 °C")]:
        a1.text(pd.Timestamp(x), ymax * 0.985, txt, ha="center", va="top",
                fontsize=7.8, fontweight="bold", color="#44546A",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    a2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda2_serie_temporal.png")
    plt.close(fig)


# =============================================================================
# EDA 3 — Entalpia do ar externo
# =============================================================================
def eda3_entalpia():
    """
    Série horária de entalpia e sua distribuição por hora do dia.
    Sustenta a escolha da entalpia como variável exógena principal do M1.
    """
    e = b[b["entalpia"].notna()].copy()
    e = e[(e.index >= "2026-03-01") & (e.index < "2026-06-01")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9),
                                 gridspec_kw={"width_ratios": [1.7, 1]})

    a1.plot(e.index, e["entalpia"], color=ROXO, linewidth=0.55)
    a1.fill_between(e.index, e["entalpia"].min(), e["entalpia"], color=ROXO, alpha=0.10)
    med = op["entalpia"].mean()
    a1.axhline(med, color=ESC, linestyle="--", linewidth=1.2)
    a1.text(0.99, 0.97, f"média na janela de análise: {br(med, 1)} kJ/kg",
            transform=a1.transAxes, ha="right", va="top", fontsize=7.6,
            color=ESC, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))
    a1.set_ylabel("Entalpia do ar externo (kJ/kg)")
    a1.set_title("Entalpia externa ao longo do período")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    a1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))

    dados = [op.loc[op["hour"] == h, "entalpia"].dropna() for h in range(12, 21)]
    bp = a2.boxplot(dados, positions=range(12, 21), widths=0.62, patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=2.2, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(ROXO); patch.set_alpha(0.72); patch.set_edgecolor("none")
    a2.set_xlabel("Hora do dia"); a2.set_ylabel("Entalpia (kJ/kg)")
    a2.set_title("Distribuição por hora (12h–20h)")
    a2.set_xticks(range(12, 21))

    nota(fig, 0.5, -0.06,
             "A entalpia decresce ao longo da tarde, o que sustenta seu uso como variável exógena principal do modelo de carga.",
             ha="center", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda3_entalpia.png")
    plt.close(fig)


# =============================================================================
# EDA 4 — Efeito do setpoint
# =============================================================================
def eda4_setpoint():
    """
    Distribuição de potência por patamar de setpoint e perfil horário.
    ATENÇÃO INTERPRETATIVA: a comparação bruta entre patamares é confundida
    pela carga e pelo clima, uma vez que os patamares ocorreram em meses
    distintos. O painel direito controla parcialmente esse efeito ao
    restringir a comparação a estratos equivalentes de índice de carga.
    """
    d = op.copy()
    # patamares: até 6,0 °C, de 6,0 a 7,0 °C e acima de 7,0 °C. O rótulo mostra
    # a faixa efetivamente observada em cada patamar (ex.: "6,5 a 6,8 °C").
    faixa = pd.cut(d["sp_chw"], [0, 6.0, 7.0, 9], labels=False)
    rotulos = {}
    for k, s in d["sp_chw"].groupby(faixa):
        lo, hi = s.min(), s.max()
        rotulos[k] = (f"{br(lo, 1)} °C" if round(lo, 1) == round(hi, 1)
                      else f"{br(lo, 1)} a {br(hi, 1)} °C")
    d["sp_b"] = faixa.map(rotulos)
    d = d[d["sp_b"].notna()]
    cores = dict(zip([rotulos[k] for k in sorted(rotulos)], [AZUL, VERDE, LARANJA]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))

    grupos = [d.loc[d["sp_b"] == s, "kw_total"] for s in cores]
    bp = a1.boxplot(grupos, labels=list(cores), patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
    for patch, s in zip(bp["boxes"], cores):
        patch.set_facecolor(cores[s]); patch.set_alpha(0.85); patch.set_edgecolor("none")
    for i, s in enumerate(cores, start=1):
        sub = d.loc[d["sp_b"] == s, "kw_total"]
        a1.text(i, sub.max() + 18, f"μ = {sub.mean():.0f} kW\n(n = {len(sub)})",
                ha="center", fontsize=7.4, fontweight="bold", color=cores[s])
    a1.set_ylabel("Potência elétrica total (kW)")
    a1.set_xlabel("Setpoint de água gelada")
    a1.set_title("Distribuição bruta por setpoint")
    a1.set_ylim(min(450, d["kw_total"].min() - 30), d["kw_total"].max() + 90)

    # comparação controlada por estrato de carga
    d["bq"] = pd.qcut(d["IC"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = d.groupby(["bq", "sp_b"], observed=True)["kw_total"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 4]
    x = np.arange(4); larg = 0.26
    for i, s in enumerate(cores):
        sub = g[g["sp_b"] == s].set_index("bq").reindex(["Q1", "Q2", "Q3", "Q4"])
        a2.bar(x + (i - 1) * larg, sub["mean"].values, larg, label=s,
               color=cores[s], edgecolor="white", linewidth=0.6)
    a2.set_xticks(x); a2.set_xticklabels(["Q1\nmenor IC", "Q2", "Q3", "Q4\nmaior IC"])
    a2.set_ylabel("Potência média (kW)")
    a2.set_title("Comparação por quartil de índice de carga")
    a2.legend(frameon=False, title="Setpoint", title_fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    a2.set_ylim(0, 1000)

    nota(fig, 0.5, -0.12,
             "O painel esquerdo não controla carga nem clima: os patamares ocorreram em meses distintos. "
             "O painel direito estratifica por índice de carga, reduzindo esse confundimento.",
             ha="center", fontsize=7.4, style="italic", color="#555555")
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda4_setpoint.png")
    plt.close(fig)


# =============================================================================
# EDA 5 — Correlação entre variáveis
# =============================================================================
def eda5_correlacao():
    """
    Matriz de correlação de Pearson entre as variáveis candidatas.
    Serve para justificar a seleção de features e sinalizar redundâncias
    (por exemplo, entre entalpia e temperatura externa).
    """
    cols = {"IC": "Índice de carga", "kw_total": "Potência (kW)",
            "entalpia": "Entalpia", "T_ext": "Temp. externa", "UR_ext": "Umidade rel.",
            "sp_chw": "Setpoint", "n_chillers": "Nº de chillers",
            "dt_chw": "ΔT água gelada", "t_cond_sup": "Temp. condensação",
            "hour": "Hora do dia"}
    m = op[list(cols)].corr()
    m.index = list(cols.values()); m.columns = list(cols.values())

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    mask = np.triu(np.ones_like(m, dtype=bool), k=1)
    mm = np.ma.masked_array(m.values, mask)
    im = ax.imshow(mm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(m))); ax.set_yticks(range(len(m)))
    ax.set_xticklabels(m.columns, rotation=42, ha="right", fontsize=8)
    ax.set_yticklabels(m.index, fontsize=8)
    ax.grid(False)
    for i in range(len(m)):
        for j in range(len(m)):
            if not mask[i, j]:
                v = m.values[i, j]
                ax.text(j, i, br(v, 2), ha="center", va="center", fontsize=7.2,
                        color="white" if abs(v) > 0.55 else "#222222")
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Correlação de Pearson", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title(f"Correlação entre variáveis candidatas (n = {len(op)} h)", pad=12)
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda5_correlacao.png")
    plt.close(fig)


# =============================================================================
# EDA 6 — Padrão de acionamento das máquinas
# =============================================================================
def eda6_operacao():
    """
    Evidencia a alteração da política de acionamento em maio: o número de
    máquinas deixa de acompanhar a carga térmica.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    ct = pd.crosstab(op["mes"], op["n_chillers"], normalize="index") * 100
    x = np.arange(3); larg = 0.38
    a1.bar(x - larg / 2, ct[2.0], larg, label="2 máquinas", color=AZUL, edgecolor="white")
    a1.bar(x + larg / 2, ct[3.0], larg, label="3 máquinas", color=LARANJA, edgecolor="white")
    for i in range(3):
        a1.text(i - larg / 2, ct[2.0].iloc[i] + 1.5, f"{ct[2.0].iloc[i]:.0f}%",
                ha="center", fontsize=7.8, fontweight="bold", color=AZUL)
        a1.text(i + larg / 2, ct[3.0].iloc[i] + 1.5, f"{ct[3.0].iloc[i]:.0f}%",
                ha="center", fontsize=7.8, fontweight="bold", color=LARANJA)
    a1.set_xticks(x); a1.set_xticklabels([MESES[m] for m in ct.index], fontsize=8)
    a1.set_ylabel("Percentual das horas (%)")
    a1.set_title("Distribuição do número de máquinas em operação")
    a1.set_ylim(0, 100)
    a1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))

    for m, cor in CORES_MES.items():
        s = op[op["mes"] == m]
        r = s["n_chillers"].corr(s["IC"])
        # cor neutra: as cores do painel esquerdo representam o número de
        # máquinas, e não os meses
        a2.bar(MESES[m].split(" ")[0], r, 0.55, color="#6B7B8C", edgecolor="white")
        a2.text(MESES[m].split(" ")[0], r + 0.02, br(r, 2),
                ha="center", fontsize=8.4, fontweight="bold")
    a2.axhline(0, color="#444444", linewidth=0.9)
    a2.set_ylabel("Correlação entre nº de máquinas e IC")
    a2.set_title("Acionamento deixa de responder à carga")
    rs = [op[op["mes"] == m]["n_chillers"].corr(op[op["mes"] == m]["IC"]) for m in CORES_MES]
    a2.set_ylim(min(0, min(rs) - 0.08), max(0.75, max(rs) + 0.08))

    nota(fig, 0.5, -0.10,
             "Em maio, a decisão de acionamento passou a ser tomada por critério externo à demanda térmica, "
             "o que caracteriza alteração de política operacional.",
             ha="center", fontsize=7.4, style="italic", color="#555555")
    fig.tight_layout()
    salvar(fig, f"{OUT}/eda6_operacao.png")
    plt.close(fig)


def estatisticas_do_texto():
    """Imprime os números da seção de análise exploratória, para conferência do texto."""
    g = op.groupby("hour").agg(ic=("IC", "mean"), ef=("ef_rel", "mean"))
    ef = g["ef"]
    print("\nESTATÍSTICAS CITADAS NA ANÁLISE EXPLORATÓRIA")
    print(f"  horas na janela: {len(op)} | dias: {op['dia'].nunique()}")
    print(f"  índice de carga: dp = {op['IC'].std():.3f}, mín = {op['IC'].min():.2f}, "
          f"máx = {op['IC'].max():.2f}, assimetria = {op['IC'].skew():.3f}")
    print(f"  variação do IC entre 12h e 20h: {100 * (g['ic'].iloc[-1] / g['ic'].iloc[0] - 1):+.1f}%")
    print(f"  eficiência relativa por hora: mín = {ef.min():.3f}, máx = {ef.max():.3f}, "
          f"amplitude = {100 * (ef.max() - ef.min()):.1f}%, "
          f"coeficiente de variação entre horas = {100 * ef.std() / ef.mean():.1f}%")
    print(f"  entalpia na janela: média = {op['entalpia'].mean():.2f}, mediana = "
          f"{op['entalpia'].median():.2f}, mín = {op['entalpia'].min():.1f}, "
          f"máx = {op['entalpia'].max():.1f} kJ/kg")
    print(f"  temperatura externa na janela: média = {op['T_ext'].mean():.2f} °C, "
          f"mediana = {op['T_ext'].median():.2f} °C")
    print(f"  potência: mín = {op['kw_total'].min():.0f}, máx = {op['kw_total'].max():.0f}, "
          f"média = {op['kw_total'].mean():.0f}, dp = {op['kw_total'].std():.1f} kW")
    print(f"  setpoint: dp = {op['sp_chw'].std():.3f} °C")
    print("\n  horas por mês, setpoint e número de máquinas:")
    faixa = pd.cut(op["sp_chw"], [0, 6.0, 7.0, 9], labels=["5,5", "6,5-6,8", "7,5"])
    print(pd.crosstab([op["mes"], faixa], op["n_chillers"], margins=True).to_string())
    varia = op.groupby("dia")["sp_chw"].agg(lambda s: s.max() - s.min() > 0.05)
    print(f"\n  dias com variação de setpoint dentro da janela: {int(varia.sum())} "
          f"(por mês: {varia.groupby(pd.to_datetime(varia.index).to_period('M')).sum().to_dict()})")
    print("\n  correlação entre número de máquinas e IC por mês:")
    for m in CORES_MES:
        s = op[op["mes"] == m]
        print(f"    {m}: {s['n_chillers'].corr(s['IC']):.3f}")


for f in [eda1_perfil_horario, eda2_serie_temporal, eda3_entalpia,
          eda4_setpoint, eda5_correlacao, eda6_operacao]:
    f()
    print("ok:", f.__name__)
estatisticas_do_texto()
print("\nFiguras em", OUT)
