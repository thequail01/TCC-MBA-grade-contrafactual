"""
=============================================================================
gerar_figuras.py — Figura 2 do artigo e figuras complementares
=============================================================================
fig7_distribuicao_carga_IC   Figura 2 do artigo (distribuição da carga e IC)

As demais figuras não entram no artigo e ficam como material complementar:
fig1_fluxograma_cascata, fig2_esquemas_validacao, fig3_loro_regimes,
fig4_pareada_2v3, fig5_grade_contrafactual e fig6_desequilibrio_ch03.

ENTRADAS: base_cag_v2.csv, comparacao_validacao.csv, tabela11_grade.csv
SAÍDA   : figuras/*.png (300 dpi)
=============================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from estilo import br, nota, salvar

warnings.filterwarnings("ignore")
OUT = "figuras"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

AZUL, LARANJA, VERDE, CINZA = "#1F4E79", "#DD8452", "#55A868", "#9AA5B1"
ESC = "#C44E52"

b = pd.read_csv("base_cag_v2.csv", index_col=0, parse_dates=True)
op = b[b["janela_analise"] == 1].copy()
op["mes"] = op.index.to_period("M").astype(str)
IC_REF = op["Q_TR_cal"].mean()
op["IC"] = op["Q_TR_cal"] / IC_REF

# -----------------------------------------------------------------------------
# Resultados de validação, lidos diretamente da saída de comparacao_validacao.py.
# Nenhum valor de R² é digitado aqui: as figuras 2 e 3 passam a ser função do
# arquivo de resultados, o que impede que texto e figura divirjam entre rodadas.
# -----------------------------------------------------------------------------
VAL = pd.read_csv("comparacao_validacao.csv")

# Rótulo de exibição -> rótulo usado no script de validação.
ROTULOS = {"Persistência": "Persistencia", "Regressão Linear": "LinReg",
           "Ridge": "Ridge", "Random Forest": "RandomForest", "XGBoost": "XGB_mono"}
ORDEM_ESQUEMAS = ["A_expansivo_AM", "B_expansivo_MAM", "C_deslizante",
                  "D_LORO", "E_holdout_final"]
ORDEM_REGIMES = ["testa_R1_marco_SP5.5", "testa_R2_abril_SPvar", "testa_R3_maio_SP7.5"]


def r2_por_esquema(alvo):
    """R² médio por esquema de validação, na ordem de exibição das figuras."""
    p = (VAL[VAL["alvo"] == alvo]
         .pivot_table(index="modelo", columns="esquema", values="R2", aggfunc="mean"))
    faltantes = [e for e in ORDEM_ESQUEMAS if e not in p.columns]
    if faltantes:
        raise ValueError(f"esquemas ausentes em comparacao_validacao.csv: {faltantes}")
    return {rot: [float(p.loc[chave, e]) for e in ORDEM_ESQUEMAS]
            for rot, chave in ROTULOS.items()}


def r2_loro(alvo="M2_kW"):
    """R² por regime deixado de fora, no esquema D."""
    sub = VAL[(VAL["alvo"] == alvo) & (VAL["esquema"] == "D_LORO")]
    p = sub.pivot_table(index="modelo", columns="fold", values="R2", aggfunc="mean")
    faltantes = [r for r in ORDEM_REGIMES if r not in p.columns]
    if faltantes:
        raise ValueError(f"regimes ausentes em comparacao_validacao.csv: {faltantes}")
    return {rot: [float(p.loc[chave, r]) for r in ORDEM_REGIMES]
            for rot, chave in ROTULOS.items()}


# =============================================================================
# FIG 1 — Fluxograma da cascata de modelos
# =============================================================================
def fig_fluxograma():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 106); ax.set_ylim(0, 66); ax.axis("off"); ax.grid(False)

    def caixa(x, y, w, h, titulo, linhas, cor, cor_texto="white", fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    facecolor=cor, edgecolor="none"))
        ax.text(x + w/2, y + h - 3.2, titulo, ha="center", va="top",
                fontsize=9, fontweight="bold", color=cor_texto)
        if linhas:
            ax.text(x + w/2, y + h - 7.8, "\n".join(linhas), ha="center", va="top",
                    fontsize=fs, color=cor_texto, linespacing=1.5)

    def seta(x1, y1, x2, y2, curva=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.5, color="#44546A",
                                     connectionstyle=f"arc3,rad={curva}"))

    # coluna 1: entradas
    caixa(1, 41, 23, 22, "Previsão climática",
          ["Entalpia do ar externo", "Temperatura, umidade",
           "Hora, dia da semana, mês"], AZUL)
    caixa(1, 23, 23, 14, "Defasagens de IC",
          ["L1, L2, média móvel 3h", "(intradiárias)"], CINZA, "#1a1a1a", fs=7.8)
    caixa(1, 2, 23, 17, "Decisão do operador",
          ["Setpoint (5,5 / 6,5 / 7,5 °C)", "Número de máquinas (2 ou 3)"], LARANJA, fs=7.8)

    # coluna 2 e 3: modelos
    caixa(35, 34, 22, 24, "Modelo M1",
          ["XGBoost monotônico", "", "Alvo: índice de", "carga térmica (IC)"], "#2E5F8A")
    caixa(66, 17, 22, 24, "Modelo M2",
          ["XGBoost monotônico", "", "Alvo: potência", "elétrica total (kW)"], "#2E5F8A")

    caixa(94, 20, 11, 18, "Grade", ["kW por", "cenário"], VERDE, fs=7.8)

    # setas
    seta(24, 52, 35, 50)                 # clima -> M1
    seta(24, 30, 35, 40)                 # defasagens -> M1
    seta(24, 10, 66, 25)                 # decisão -> M2
    seta(57, 42, 66, 35)                 # M1 -> M2
    seta(88, 29, 94, 29)                 # M2 -> grade

    cx = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none")
    ax.text(62.0, 40.0, "IC previsto", fontsize=7.8, style="italic",
            color="#44546A", ha="center", va="center", rotation=-24, bbox=cx, zorder=5)
    ax.text(45.0, 15.6, "variáveis de decisão", fontsize=7.8, style="italic",
            color="#44546A", ha="center", va="center", rotation=9, bbox=cx, zorder=5)

    ax.text(53, 65, "Arquitetura em cascata: da previsão climática ao custo de cada cenário",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1F3864")
    ax.text(53, 0.2,
            "M1 emprega apenas variáveis exógenas, o que evita circularidade com M2 e viabiliza operação a partir de previsão meteorológica.",
            ha="center", va="center", fontsize=7.5, style="italic", color="#555555")
    salvar(fig, f"{OUT}/fig1_fluxograma_cascata.png")
    plt.close(fig)


# =============================================================================
# FIG 2 — Desempenho por esquema de validação (M2)
# =============================================================================
def fig_esquemas():
    esquemas = ["A\nexpansivo\n(abr-mai)", "B\nexpansivo\n(mar-mai)", "C\ndeslizante",
                "D\nLORO", "E\nholdout\nmaio"]
    dados = r2_por_esquema("M2_kW")
    cores = {"Persistência": CINZA, "Regressão Linear": "#8FB3D9", "Ridge": "#4C72B0",
             "Random Forest": VERDE, "XGBoost": AZUL}
    hachura = {"Persistência": "//", "Regressão Linear": "", "Ridge": "",
               "Random Forest": "", "XGBoost": ""}

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(esquemas)); larg = 0.16
    for i, (nome, v) in enumerate(dados.items()):
        ax.bar(x + (i - 2) * larg, v, larg, label=nome, color=cores[nome],
               hatch=hachura[nome], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(esquemas)
    ax.set_ylabel("Coeficiente de determinação (R²)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Modelo 2: desempenho por esquema de validação")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    # marca, por esquema, os modelos que superam a persistencia daquele esquema
    for j, pv in enumerate(dados["Persistência"]):
        ax.plot([j - 2.5 * larg, j + 2.5 * larg], [pv, pv], color=ESC,
                linewidth=1.3, linestyle=":", zorder=5)
    nota(fig, 0.5, -0.115, "Linha pontilhada vermelha: nível da persistência em cada esquema, para leitura direta de quais modelos a superam.",
             ha="center", fontsize=7.4, color="#555555", style="italic")
    salvar(fig, f"{OUT}/fig2_esquemas_validacao.png")
    plt.close(fig)


# =============================================================================
# FIG 3 — Robustez entre regimes (LORO)
# =============================================================================
def fig_loro():
    regimes = ["R1 março\nSP 5,5 °C", "R2 abril\nSP variável", "R3 maio\nSP 7,5 °C"]
    dados = r2_loro()
    cores = {"Persistência": CINZA, "Regressão Linear": "#8FB3D9", "Ridge": "#4C72B0",
             "Random Forest": VERDE, "XGBoost": AZUL}

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(regimes)); larg = 0.16
    for i, (nome, v) in enumerate(dados.items()):
        ax.bar(x + (i - 2) * larg, v, larg, label=nome, color=cores[nome],
               hatch="//" if nome == "Persistência" else "",
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_ylabel("Coeficiente de determinação (R²)")
    ax.set_ylim(0, 0.78)
    ax.set_title("Validação por regime ausente (LORO): generalização entre regimes")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    ax.annotate("regime intermediário:\núnico caso de interpolação",
                xy=(1 + 2 * larg, dados["XGBoost"][1]), xytext=(1.62, 0.735),
                fontsize=7.2, color="#44546A", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#44546A", linewidth=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    for j, pv in enumerate(dados["Persistência"]):
        ax.plot([j - 2.5 * larg, j + 2.5 * larg], [pv, pv], color=ESC,
                linewidth=1.3, linestyle=":", zorder=5)
    nota(fig, 0.5, -0.10, "Linha pontilhada vermelha: nível da persistência em cada regime.",
             ha="center", fontsize=7.2, color="#555555", style="italic")
    salvar(fig, f"{OUT}/fig3_loro_regimes.png")
    plt.close(fig)


# =============================================================================
# FIG 4 — Comparação pareada 2 vs 3 máquinas
# =============================================================================
def fig_pareada():
    d = op.dropna(subset=["Q_TR_cal", "kw_total"]).copy()
    d["bq"] = pd.qcut(d["IC"], 5, labels=False)
    d["bh"] = pd.qcut(d["entalpia"], 3, labels=False)
    g = d.groupby(["bq", "bh", "n_chillers"]).agg(kw=("kw_total", "mean"),
                                                  ic=("IC", "mean"),
                                                  n=("kw_total", "size")).reset_index()
    p = g.pivot_table(index=["bq", "bh"], columns="n_chillers", values=["kw", "ic", "n"]).dropna()
    p.columns = [f"{a}{int(c)}" for a, c in p.columns]
    p = p[(p["n2"] >= 5) & (p["n3"] >= 5)]
    r = p.groupby("bq").agg(kw2=("kw2", "mean"), kw3=("kw3", "mean"), ic=("ic2", "mean"))
    r["d"] = r["kw3"] - r["kw2"]

    rot = ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"]
    rot = [f"{rot[i]}\nIC ≈ {br(r['ic'].iloc[i])}" for i in range(len(r))]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(r)); larg = 0.36
    a1.bar(x - larg/2, r["kw2"], larg, label="2 máquinas", color=AZUL, edgecolor="white")
    a1.bar(x + larg/2, r["kw3"], larg, label="3 máquinas", color=LARANJA, edgecolor="white")
    a1.set_xticks(x); a1.set_xticklabels(rot, fontsize=7.6)
    a1.set_ylabel("Potência elétrica média (kW)")
    a1.set_title("Consumo em estratos equivalentes de carga e entalpia")
    a1.legend(frameon=False, loc="upper left")
    a1.set_ylim(0, 1080)

    a2.bar(x, r["d"], 0.6, color=ESC, edgecolor="white")
    for i, v in enumerate(r["d"]):
        a2.text(i, v + 3, f"+{v:.0f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels([br(v) for v in r["ic"]], fontsize=8)
    a2.set_xlabel("Índice de carga médio do estrato (IC)")
    a2.set_ylabel("Consumo adicional da 3ª máquina (kW)")
    a2.set_title("Penalidade cresce com a carga")
    a2.set_ylim(0, max(r["d"]) * 1.22)
    salvar(fig, f"{OUT}/fig4_pareada_2v3.png")
    plt.close(fig)


# =============================================================================
# FIG 5 — Grade contrafactual
# =============================================================================
def fig_grade():
    sp = [5.5, 6.5, 7.5]
    GR = pd.read_csv("tabela11_grade.csv", index_col=0)
    ic = [round(float(v), 2) for v in GR["IC"]]
    kw2 = [[float(GR.loc[r, f"{s}/2"]) for s in sp] for r in GR.index]
    kw3 = [[float(GR.loc[r, f"{s}/3"]) for s in sp] for r in GR.index]
    lo = min(min(min(a), min(b)) for a, b in zip(kw2, kw3))
    hi = max(max(max(a), max(b)) for a, b in zip(kw2, kw3))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    tons2 = ["#BBD0E6", "#7FA6CC", "#4478AE", "#1F4E79"]
    tons3 = ["#F5CDAE", "#EBA778", "#DD8452", "#B35F33"]
    for i in range(len(ic)):
        ax.plot(sp, kw2[i], "-o", color=tons2[i], linewidth=1.9, markersize=5,
                label=f"IC {br(ic[i])} · 2 máq.")
        ax.plot(sp, kw3[i], "--s", color=tons3[i], linewidth=1.9, markersize=4.5,
                label=f"IC {br(ic[i])} · 3 máq.")
    ax.set_xticks(sp); ax.set_xticklabels([f"{br(s, 1)} °C" for s in sp])
    ax.set_xlabel("Setpoint de água gelada")
    ax.set_ylabel("Potência elétrica prevista (kW)")
    ax.set_title("Grade contrafactual: custo de cada configuração operacional")
    ax.legend(ncol=2, frameon=False, fontsize=7.6, loc="upper right")
    ax.annotate("linha cheia: 2 máquinas\nlinha tracejada: 3 máquinas",
                xy=(5.55, lo - (hi - lo) * 0.06), fontsize=7.4, color="#555555", style="italic")
    ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.10)
    salvar(fig, f"{OUT}/fig5_grade_contrafactual.png")
    plt.close(fig)


# =============================================================================
# FIG 6 — Desequilíbrio do CH-03
# =============================================================================
def fig_ch03():
    t = op[op["n_chillers"] == 3]
    g = t.groupby("mes")[["rla_ch1", "rla_ch2", "rla_ch3"]].mean()
    meses = ["Março\nSP 5,5 °C", "Abril\nSP variável", "Maio\nSP 7,5 °C"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(3); larg = 0.26
    a1.bar(x - larg, g["rla_ch1"], larg, label="CH-01", color=AZUL, edgecolor="white")
    a1.bar(x, g["rla_ch2"], larg, label="CH-02", color="#4C72B0", edgecolor="white")
    a1.bar(x + larg, g["rla_ch3"], larg, label="CH-03", color=ESC, edgecolor="white")
    for i in range(3):
        a1.text(i + larg, g["rla_ch3"].iloc[i] + 1.5, f"{g['rla_ch3'].iloc[i]:.0f}%",
                ha="center", fontsize=8, fontweight="bold", color=ESC)
    a1.set_xticks(x); a1.set_xticklabels(meses)
    a1.set_ylabel("Corrente do motor (% de RLA)")
    a1.set_ylim(0, 108)
    a1.set_title("CH-03 opera abaixo da metade da carga dos pares")
    a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))

    ev = op.dropna(subset=["t_evap_sup_ch1", "t_evap_ret_ch1"]).copy()
    dts = []
    for i in [1, 2, 3]:
        m = ev[ev[f"ch{i}_on"] == 1]
        dts.append((m[f"t_evap_ret_ch{i}"] - m[f"t_evap_sup_ch{i}"]).mean())
    a2.bar(["CH-01", "CH-02", "CH-03"], dts, 0.55,
           color=[AZUL, "#4C72B0", ESC], edgecolor="white")
    for i, v in enumerate(dts):
        a2.text(i, v + 0.12, f"{br(v)} °C", ha="center", fontsize=8.4, fontweight="bold")
    a2.set_ylabel("ΔT do evaporador (°C)")
    a2.set_ylim(0, 6.3)
    a2.set_title("ΔT do CH-03 é um terço dos pares")
    salvar(fig, f"{OUT}/fig6_desequilibrio_ch03.png")
    plt.close(fig)


for f in [fig_fluxograma, fig_esquemas, fig_loro, fig_pareada, fig_grade, fig_ch03]:
    f()
    print("ok:", f.__name__)
print("\nArquivos em", OUT)


# =============================================================================
# FIG 7 — Distribuição de carga e relação entre TR teórico e IC
# =============================================================================
def fig_distribuicao():
    from scipy import stats as st
    q = op["Q_TR_cal"].dropna()
    mu, sd = q.mean(), q.std()
    ic_ref = mu

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------- painel A: histograma + densidade ----------
    a1.hist(q, bins=32, density=True, color="#BBD0E6", edgecolor="white",
            linewidth=0.7, alpha=0.95, label=f"Observações horárias (n = {len(q)})")
    xs = np.linspace(q.min(), q.max(), 400)
    kde = st.gaussian_kde(q)
    a1.plot(xs, kde(xs), color=AZUL, linewidth=2.1, label="Densidade estimada")
    a1.plot(xs, st.norm.pdf(xs, mu, sd), color=ESC, linewidth=1.5, linestyle="--",
            label=f"Normal ajustada (μ = {mu:.0f}; σ = {sd:.0f})")

    a1.axvline(mu, color="#1F3864", linewidth=1.4, linestyle=":")
    # rótulo da média à esquerda da linha; a legenda fica no canto superior direito
    a1.text(mu - 0.012 * (q.max() - q.min()), kde(xs).max() * 1.20, "média (IC = 1,00)",
            fontsize=7.6, color="#1F3864", fontweight="bold", va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    topo = kde(xs).max()
    # alturas alternadas para que rótulos de percentis próximos não se cubram
    for (p, lab), alt in zip([(25, "P25"), (75, "P75"), (90, "P90")], [0.80, 0.80, 0.56]):
        v = q.quantile(p / 100)
        a1.axvline(v, color=CINZA, linewidth=0.9, linestyle="-", alpha=0.8, ymax=0.63)
        a1.text(v, topo * alt, f"{lab}\nIC {br(v / ic_ref)}", fontsize=6.9,
                color="#555555", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none"))

    a1.set_xlabel("Carga térmica estimada (TR equivalentes, escala não validada)")
    a1.set_ylabel("Densidade")
    a1.set_title("Distribuição da carga na janela de análise")
    a1.legend(frameon=False, fontsize=7.6, loc="upper right")
    a1.set_ylim(0, kde(xs).max() * 1.28)

    # eixo superior em IC
    a1t = a1.twiny()
    a1t.set_xlim(a1.get_xlim())
    ticks = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    ticks = [t for t in ticks if q.min() <= t * ic_ref <= q.max()]
    a1t.set_xticks([t * ic_ref for t in ticks])
    a1t.set_xticklabels([br(t) for t in ticks], fontsize=8)
    a1t.set_xlabel("Índice de carga IC (adimensional)", labelpad=7)
    a1t.grid(False)
    a1t.spines["top"].set_visible(False)

    # ---------- painel B: relação linear TR <-> IC ----------
    x0, x1 = q.min() * 0.97, q.max() * 1.03     # limites derivados dos dados
    tr = np.linspace(x0, x1, 200)
    a2.plot(tr, tr / ic_ref, color=AZUL, linewidth=2.2)
    a2.axvspan(q.quantile(.05), q.quantile(.95), color="#BBD0E6", alpha=0.45,
               label="Faixa observada (P5–P95)")
    a2.axhline(1.0, color=CINZA, linewidth=0.9, linestyle=":")
    a2.axvline(ic_ref, color=CINZA, linewidth=0.9, linestyle=":")
    a2.plot([ic_ref], [1.0], "o", color=ESC, markersize=7, zorder=5)
    a2.annotate(f"referência\n{ic_ref:.0f} TR ↔ IC 1,00",
                xy=(ic_ref, 1.0), xytext=(ic_ref + 0.08 * (x1 - x0), 1.0 - 0.22 * (x1 - x0) / ic_ref), fontsize=7.8,
                color=ESC, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ESC, linewidth=0.9))

    a2.set_xlabel("Carga térmica estimada (TR equivalentes)")
    a2.set_ylabel("Índice de carga IC")
    a2.set_title("IC = carga estimada ÷ média da série")
    a2.legend(frameon=False, fontsize=7.8, loc="upper left")
    a2.set_xlim(x0, x1); a2.set_ylim(x0 / ic_ref, x1 / ic_ref)

    nota(fig, 0.5, -0.045,
             "A transformação é linear e preserva a ordenação. O IC remove a dependência da escala absoluta da vazão, "
             "mantendo íntegras as comparações relativas.",
             ha="center", fontsize=7.6, style="italic", color="#555555")
    salvar(fig, f"{OUT}/fig7_distribuicao_carga_IC.png")
    plt.close(fig)


fig_distribuicao()
print("ok: fig_distribuicao")
