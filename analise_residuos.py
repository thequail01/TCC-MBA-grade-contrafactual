"""
=============================================================================
ANÁLISE DE RESÍDUOS DO MODELO 2 ESTRATIFICADA POR CONFIGURAÇÃO
=============================================================================

Objeto: comportamento de ε = kW_real − kW_predito nos dois regimes de
operação N = 2 e N = 3 máquinas.

MOTIVAÇÃO
O instrumento proposto compara cenários de 2 e 3 máquinas. A métrica global
de erro não informa sobre a validade dessa comparação: um modelo pode
apresentar erro médio baixo e ainda assim superestimar sistematicamente uma
configuração e subestimar a outra, o que enviesaria o diferencial de consumo
apresentado ao operador. O viés condicional por configuração é, portanto,
requisito de validade mais pertinente que o erro global.

PROCEDIMENTO
Os resíduos são obtidos fora da amostra (out-of-fold). Reportam-se dois
esquemas, por responderem a perguntas distintas:
  (a) Janela expansiva: representa a condição de implantação, com treino
      acumulando o passado disponível. Os três cortes ficam em 55%, 70% e
      85% das semanas, com embargo de 24 h; NÃO são as mesmas partições do
      esquema B de comparacao_validacao.py.
  (b) Regime ausente (LORO): condição adversa, com extrapolação para regime
      de setpoint ausente do treino.

ESPECIFICAÇÃO DO MODELO. Para isolar o erro do estágio de potência, o modelo
aqui é alimentado pela carga MEDIDA (Q_TR_cal), e não pela carga estimada
pelo M1, com as variáveis Q_TR_cal, entalpia, hora, setpoint e número de
máquinas e as mesmas restrições de monotonicidade do M2. Os resíduos
descrevem, portanto, o erro do segundo estágio isoladamente, e não o da
cascata completa das Tabelas 6 a 8. Resíduo: ε = potência observada −
potência predita.

CUIDADO ESTATÍSTICO CENTRAL
Resíduos de séries temporais são autocorrelacionados. Testes que pressupõem
independência (t de Welch, Mann-Whitney, Levene) subestimam o erro-padrão e
produzem valores-p otimistas. Reporta-se, por isso, o valor-p ingênuo em
conjunto com intervalo de confiança por bootstrap em blocos, que preserva a
estrutura de dependência. A conclusão deve apoiar-se no segundo.

SAÍDA
  residuos_por_configuracao.csv               resíduos individuais
  figuras/fig8_residuos_configuracao.png      Figura 8 do artigo
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comum import EMBARGO_H, XGB_BASE, n_arvores as _n_arvores
from estilo import br, nota, salvar

warnings.filterwarnings("ignore")
OUT = "figuras"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

AZUL, LARANJA, CINZA, ESC = "#1F4E79", "#DD8452", "#9AA5B1", "#C44E52"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

FS = ["Q_TR_cal", "entalpia", "hour", "sp_chw", "n_chillers"]
MONO = {"Q_TR_cal": 1, "entalpia": 1, "hour": 0, "sp_chw": -1, "n_chillers": 1}

b = pd.read_csv("base_cag_v2.csv", index_col=0, parse_dates=True)
op = b[b["janela_analise"] == 1].copy().sort_index()
op["IC"] = op["Q_TR_cal"] / op["Q_TR_cal"].mean()
op["regime_sp"] = np.select([op.index < "2026-04-01", op.index < "2026-05-01"],
                            ["R1", "R2"], default="R3")
d = op[FS + ["kw_total", "IC", "regime_sp"]].dropna(subset=FS + ["kw_total"])


def n_arvores(Xtr, ytr):
    """Número de árvores pelo procedimento comum (comum.n_arvores)."""
    return _n_arvores(Xtr, ytr, MONO, FS)


def residuos(particoes, nome):
    """Ajusta em cada partição e devolve resíduos fora da amostra."""
    saida = []
    for rot, mtr, mte in particoes:
        dtr, dte = d[mtr], d[mte]
        if len(dte) < 10:
            continue
        imp = SimpleImputer(strategy="median").fit(dtr[FS])
        Xtr = pd.DataFrame(imp.transform(dtr[FS]), columns=FS, index=dtr.index)
        Xte = pd.DataFrame(imp.transform(dte[FS]), columns=FS, index=dte.index)
        m = xgb.XGBRegressor(n_estimators=n_arvores(Xtr, dtr["kw_total"]),
                             monotone_constraints=MONO, **XGB_BASE)
        m.fit(Xtr, dtr["kw_total"])
        pred = m.predict(Xte)
        r = dte[["kw_total", "n_chillers", "IC", "sp_chw", "regime_sp"]].copy()
        r["pred"] = pred
        r["res"] = dte["kw_total"].values - pred
        r["fold"] = rot
        r["esquema"] = nome
        saida.append(r)
    return pd.concat(saida)


# ---- esquema expansivo (condição de implantação) ----
semanas = pd.Series(d.index.to_period("W"), index=d.index)
u = sorted(semanas.unique())
cortes = [u[int(len(u) * f)] for f in (0.55, 0.70, 0.85)]
part_exp = []
for i, c in enumerate(cortes, 1):
    mtr = semanas < c
    lim = d.index[mtr].max() + pd.Timedelta(hours=EMBARGO_H)
    fim = cortes[i] if i < len(cortes) else None
    mte = (d.index > lim) & ((semanas < fim) if fim is not None else True)
    part_exp.append((f"F{i}", mtr, mte))

part_loro = [(f"testa_{r}", d["regime_sp"] != r, d["regime_sp"] == r) for r in ["R1", "R2", "R3"]]

R_exp = residuos(part_exp, "expansivo")
R_loro = residuos(part_loro, "LORO")
R = pd.concat([R_exp, R_loro])
R.to_csv("residuos_por_configuracao.csv")


# =============================================================================
# Ferramentas estatísticas
# =============================================================================
def bootstrap_blocos(x2, x3, n_boot=5000, bloco=8):
    """
    IC de 95% para a diferença de médias, por bootstrap em blocos móveis.
    Preserva a autocorrelação local, ao contrário do bootstrap i.i.d.
    Bloco de 8 h aproxima a duração de um dia útil na janela de análise.
    """
    def reamostra(x):
        n = len(x)
        nb = int(np.ceil(n / bloco))
        # inícios possíveis: 0 a n - bloco, inclusive (integers exclui o limite)
        ini = rng.integers(0, max(n - bloco + 1, 1), size=nb)
        return np.concatenate([x[i:i + bloco] for i in ini])[:n]
    dif = [reamostra(x3).mean() - reamostra(x2).mean() for _ in range(n_boot)]
    return np.percentile(dif, [2.5, 97.5]), np.std(dif)


def n_efetivo(x, max_lag=24):
    """Tamanho amostral efetivo sob autocorrelação (Bartlett)."""
    n = len(x)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den == 0:
        return n
    soma = sum((1 - k / n) * (xc[:-k] * xc[k:]).sum() / den for k in range(1, min(max_lag, n - 1)))
    # Autocorrelação líquida negativa produziria n efetivo maior que n, o que não
    # tem leitura amostral. O valor é limitado a n, e a dependência serial é
    # avaliada em separado pelo teste de Ljung-Box.
    return min(n / max(1 + 2 * soma, 1e-6), n)


def ljung_box(x, lags=12):
    n = len(x)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    q = n * (n + 2) * sum(((xc[:-k] * xc[k:]).sum() / den) ** 2 / (n - k)
                          for k in range(1, lags + 1))
    return q, 1 - stats.chi2.cdf(q, lags)


# =============================================================================
# Relatório
# =============================================================================
for esq in ["expansivo", "LORO"]:
    S = R[R["esquema"] == esq]
    e2 = S.loc[S["n_chillers"] == 2, "res"].values
    e3 = S.loc[S["n_chillers"] == 3, "res"].values

    print("=" * 78)
    print(f"ESQUEMA: {esq}   (n = {len(S)};  N=2: {len(e2)} h;  N=3: {len(e3)} h)")
    print("=" * 78)

    print("\n[1] ESTATÍSTICAS DESCRITIVAS DOS RESÍDUOS (kW)")
    tab = pd.DataFrame({
        "N = 2": [e2.mean(), np.median(e2), e2.std(ddof=1), stats.iqr(e2),
                  stats.skew(e2), stats.kurtosis(e2), np.percentile(e2, 5), np.percentile(e2, 95)],
        "N = 3": [e3.mean(), np.median(e3), e3.std(ddof=1), stats.iqr(e3),
                  stats.skew(e3), stats.kurtosis(e3), np.percentile(e3, 5), np.percentile(e3, 95)],
    }, index=["média", "mediana", "desvio-padrão", "amplitude interquartil",
              "assimetria", "curtose", "percentil 5", "percentil 95"])
    print(tab.round(2).to_string())

    print("\n[2] POSIÇÃO — há viés diferencial entre configurações?")
    t, p_t = stats.ttest_ind(e3, e2, equal_var=False)
    u_, p_u = stats.mannwhitneyu(e3, e2)
    ic, se_b = bootstrap_blocos(e2, e3)
    print(f"  diferença de médias (N=3 menos N=2) : {e3.mean() - e2.mean():+7.2f} kW")
    print(f"  t de Welch (p ingênuo)              : t = {t:+.3f}, p = {p_t:.4f}")
    print(f"  Mann-Whitney (p ingênuo)            : p = {p_u:.4f}")
    print(f"  intervalo 95% (bootstrap em blocos) : [{ic[0]:+.2f}; {ic[1]:+.2f}] kW  (ep {se_b:.2f})")
    print(f"  n efetivo sob autocorrelação        : N=2: {n_efetivo(e2):.0f} de {len(e2)}   "
          f"N=3: {n_efetivo(e3):.0f} de {len(e3)}")

    print("\n[3] ESCALA — a dispersão difere entre configurações?")
    lev = stats.levene(e2, e3, center="median")   # Brown-Forsythe
    print(f"  razão de desvios-padrão (N=3 / N=2) : {e3.std(ddof=1) / e2.std(ddof=1):.2f}")
    print(f"  Brown-Forsythe                      : W = {lev.statistic:.2f}, p = {lev.pvalue:.2e}")
    ks = stats.ks_2samp(e2, e3)
    print(f"  Kolmogorov-Smirnov (distribuições)  : D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")

    print("\n[4] AUTOCORRELAÇÃO DOS RESÍDUOS (Ljung-Box, 12 defasagens)")
    for rot, e in [("N = 2", e2), ("N = 3", e3)]:
        q, p = ljung_box(e)
        print(f"  {rot}: Q = {q:7.1f}, p = {p:.2e}  -> {'dependência serial presente' if p < 0.05 else 'sem evidência'}")

    print("\n[5] HETEROCEDASTICIDADE — |ε| explicado por configuração e carga")
    X = np.column_stack([np.ones(len(S)), (S["n_chillers"] == 3).astype(float), S["IC"]])
    y = S["res"].abs().values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    for nome, bb, ss in zip(["intercepto", "indicador N=3", "índice de carga"], beta, se):
        print(f"  {nome:16s}: {bb:+8.2f}  (ep {ss:5.2f}, t = {bb / ss:+6.2f})")

    print("\n[6] VIÉS POR REGIME DE SETPOINT (decomposição)")
    print(S.pivot_table(index="regime_sp", columns="n_chillers", values="res",
                        aggfunc=["mean", "std", "count"]).round(1).to_string())
    print()


# =============================================================================
# Figura
# =============================================================================
S = R[R["esquema"] == "expansivo"]
e2 = S.loc[S["n_chillers"] == 2, "res"].values
e3 = S.loc[S["n_chillers"] == 3, "res"].values

LIM_LO, LIM_HI = -200.0, 150.0   # faixa de exibicao; extremos sao contados e anotados

fig, ax = plt.subplots(1, 3, figsize=(12.4, 4.2))

# ---------------------------------------------------------------- painel 1
# Bins restritos a faixa de exibicao para que a massa central seja legivel.
bins = np.linspace(LIM_LO, LIM_HI, 40)
ax[0].hist(np.clip(e2, LIM_LO, LIM_HI), bins=bins, alpha=0.70, color=AZUL,
           label=f"N = 2 (n = {len(e2)})", density=True)
ax[0].hist(np.clip(e3, LIM_LO, LIM_HI), bins=bins, alpha=0.58, color=LARANJA,
           label=f"N = 3 (n = {len(e3)})", density=True)
ax[0].axvline(0, color="#333333", linewidth=1.1)
for e, c in [(e2, AZUL), (e3, LARANJA)]:
    ax[0].axvline(e.mean(), color=c, linewidth=1.4, linestyle="--", alpha=0.9)
ax[0].set_xlim(LIM_LO, LIM_HI)
ax[0].set_xlabel("Resíduo ε (kW)"); ax[0].set_ylabel("Densidade")
ax[0].set_title("Distribuição dos resíduos")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].text(0.02, 0.72, "tracejado: média", transform=ax[0].transAxes,
           fontsize=7.2, style="italic", color="#666666")

# ---------------------------------------------------------------- painel 2
def bigodes(e):
    q1, q3 = np.percentile(e, [25, 75])
    ii = q3 - q1
    dentro = e[(e >= q1 - 1.5 * ii) & (e <= q3 + 1.5 * ii)]
    return dentro.min(), dentro.max()

b2, b3 = bigodes(e2), bigodes(e3)
Y_LO = min(b2[0], b3[0]) - 18
Y_HI = max(b2[1], b3[1]) + 18
BANDA = (Y_HI - Y_LO) * 0.30          # faixa reservada aos rotulos, acima dos dados

bp = ax[1].boxplot([e2, e3], tick_labels=["N = 2", "N = 3"], patch_artist=True,
                   widths=0.46, showfliers=False,
                   medianprops=dict(color="white", linewidth=1.8),
                   whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
for patch, c in zip(bp["boxes"], [AZUL, LARANJA]):
    patch.set_facecolor(c); patch.set_edgecolor("none"); patch.set_alpha(0.85)
ax[1].axhline(0, color="#333333", linewidth=1.1)

for i, (e, c) in enumerate(zip([e2, e3], [AZUL, LARANJA]), start=1):
    ax[1].plot(i, e.mean(), marker="D", markersize=7, color=c,
               markeredgecolor="white", markeredgewidth=1.2, zorder=5,
               label="média" if i == 1 else None)

ax[1].set_ylim(Y_LO, Y_HI + BANDA)
ax[1].set_ylabel("Resíduo ε (kW)")
ax[1].set_title("Posição e dispersão")
ax[1].axhline(Y_HI, color="#DDDDDD", linewidth=0.8)

for i, (e, c) in enumerate(zip([e2, e3], [AZUL, LARANJA]), start=1):
    ax[1].text(i, Y_HI + BANDA * 0.58, f"σ = {br(e.std(ddof=1), 0)} kW", ha="center",
               fontsize=9.5, fontweight="bold", color=c)
    ax[1].text(i, Y_HI + BANDA * 0.18, f"média {br(e.mean(), 1, sinal=True)} kW", ha="center",
               fontsize=8, color="#555555")

fora2 = int((e2 < Y_LO).sum() + (e2 > Y_HI).sum())
fora3 = int((e3 < Y_LO).sum() + (e3 > Y_HI).sum())
ax[1].text(0.5, 0.02, f"◆ média  |  pontos além dos bigodes: {fora2} de {len(e2)} (N=2), {fora3} de {len(e3)} (N=3)",
           transform=ax[1].transAxes, ha="center", fontsize=7, color="#888888", style="italic")

# ---------------------------------------------------------------- painel 3
for lab, sub, c in [("N = 2", S[S["n_chillers"] == 2], AZUL),
                    ("N = 3", S[S["n_chillers"] == 3], LARANJA)]:
    ax[2].scatter(sub["pred"], np.clip(sub["res"], LIM_LO, LIM_HI), s=13, alpha=0.55,
                  color=c, label=lab, edgecolors="none")
ax[2].axhline(0, color="#333333", linewidth=1.1)
ax[2].set_ylim(LIM_LO, LIM_HI)
ax[2].set_xlabel("Potência predita (kW)"); ax[2].set_ylabel("Resíduo ε (kW)")
ax[2].set_title("Resíduo contra valor predito")
ax[2].legend(frameon=False, fontsize=8, loc="lower left")

nota(fig, 0.5, -0.09,
         "Resíduos fora da amostra, esquema de janela expansiva. As médias praticamente coincidentes indicam ausência de viés "
         "sistemático entre configurações; a dispersão substancialmente maior sob três máquinas indica incerteza distinta entre "
         "os cenários. Eixos limitados a [-200, +150] kW para legibilidade; valores extremos foram mantidos em todos os cálculos.",
         ha="center", fontsize=7.4, style="italic", color="#555555", wrap=True)
fig.tight_layout()
salvar(fig, f"{OUT}/fig8_residuos_configuracao.png")
plt.close(fig)
print("figura gerada: fig8_residuos_configuracao.png")
