"""
=============================================================================
COMPARACAO PAREADA 2 vs 3 MAQUINAS — Tabela 10
=============================================================================
Estratificacao simultanea por quintil de indice de carga e tercil de entalpia,
gerando 15 celulas potenciais. Retem-se as celulas em que ambas as
configuracoes ocorrem com pelo menos 5 horas cada, o que assegura comparacao
sob condicoes equivalentes de carga e clima.

O contraste e reportado por faixa de carga (media simples dos estratos de
entalpia dentro do quintil) e no agregado, com teste dos postos sinalizados
de Wilcoxon sobre os estratos pareados e intervalo de confianca por bootstrap
de estratos. A tendencia da penalidade com a carga e estimada por regressao
linear da diferenca de cada estrato sobre seu indice de carga medio.

SAIDA: tabela10_pareada.csv, tabela10_estratos.csv
=============================================================================
"""
import numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
MIN_N = 5

b = pd.read_csv("base_cag_v2.csv", index_col=0, parse_dates=True)
op = b[b["janela_analise"] == 1].copy()
op["IC"] = op["Q_TR_cal"] / op["Q_TR_cal"].mean()
d = op.dropna(subset=["Q_TR_cal", "kw_total"]).copy()
d["bq"] = pd.qcut(d["IC"], 5, labels=False)
d["bh"] = pd.qcut(d["entalpia"], 3, labels=False)

g = (d.groupby(["bq", "bh", "n_chillers"])
       .agg(kw=("kw_total", "mean"), ic=("IC", "mean"),
            ent=("entalpia", "mean"), n=("kw_total", "size")).reset_index())
p = g.pivot_table(index=["bq", "bh"], columns="n_chillers",
                  values=["kw", "ic", "ent", "n"]).dropna()
p.columns = [f"{a}{int(c)}" for a, c in p.columns]

print("=" * 96)
print("CELULAS DA ESTRATIFICACAO (5 quintis de carga x 3 tercis de entalpia)")
print("=" * 96)
todas = g.pivot_table(index=["bq", "bh"], columns="n_chillers", values="n")
todas.columns = [f"n{int(c)}" for c in todas.columns]
todas = todas.fillna(0).astype(int)
todas["retida"] = ((todas.get("n2", 0) >= MIN_N) & (todas.get("n3", 0) >= MIN_N))
print(todas.to_string())
print(f"\ncelulas com ambas as configuracoes: {int((todas[['n2','n3']] > 0).all(axis=1).sum())} de 15")
print(f"celulas retidas (>= {MIN_N} h em cada): {int(todas['retida'].sum())} de 15")

E = p[(p["n2"] >= MIN_N) & (p["n3"] >= MIN_N)].copy()
E["dif"] = E["kw3"] - E["kw2"]
E["ef2"] = E["ic2"] / E["kw2"]
E["ef3"] = E["ic3"] / E["kw3"]
E["var_ef"] = 100 * (E["ef3"] / E["ef2"] - 1)
E.to_csv("tabela10_estratos.csv")

print("\n" + "=" * 96)
print("ESTRATOS RETIDOS")
print("=" * 96)
print(E[["n2", "n3", "ic2", "ic3", "ent2", "ent3", "kw2", "kw3", "dif", "var_ef"]].round(2).to_string())

# --- agregacao por faixa de carga ---
R = E.groupby("bq").agg(ic=("ic2", "mean"), kw2=("kw2", "mean"), kw3=("kw3", "mean"),
                        var_ef=("var_ef", "mean"), n2=("n2", "sum"), n3=("n3", "sum"),
                        estratos=("dif", "size"))
R["dif"] = R["kw3"] - R["kw2"]
# rotulo pelo proprio quintil, e nao pela posicao: se algum quintil nao tiver
# estrato retido, os demais continuam com o rotulo correto
FAIXAS = {0: "Muito baixa", 1: "Baixa", 2: "Media", 3: "Alta", 4: "Muito alta"}
R.index = [FAIXAS[int(q)] for q in R.index]
R.to_csv("tabela10_pareada.csv")

print("\n" + "=" * 96)
print("TABELA 10 — contraste por faixa de carga")
print("=" * 96)
print(R[["estratos", "n2", "n3", "ic", "kw2", "kw3", "dif", "var_ef"]].round(2).to_string())

# --- testes sobre os estratos pareados ---
dif = E["dif"].values
w = stats.wilcoxon(dif, alternative="greater")
tt = stats.ttest_1samp(dif, 0, alternative="greater")
bs = np.array([RNG.choice(dif, len(dif), replace=True).mean() for _ in range(10000)])

print("\n" + "=" * 96)
print("TESTES SOBRE OS ESTRATOS PAREADOS")
print("=" * 96)
print(f"  n de estratos                 : {len(dif)}")
print(f"  estratos com diferenca > 0    : {(dif > 0).sum()} de {len(dif)}")
print(f"  diferenca media               : {dif.mean():+.1f} kW")
print(f"  diferenca mediana             : {np.median(dif):+.1f} kW")
print(f"  intervalo 95% (bootstrap)     : [{np.percentile(bs,2.5):+.1f} ; {np.percentile(bs,97.5):+.1f}] kW")
print(f"  Wilcoxon (unilateral)         : W={w.statistic:.1f}  p={w.pvalue:.5f}")
print(f"  t pareado (unilateral)        : t={tt.statistic:.2f}  p={tt.pvalue:.5f}")
print(f"  variacao media de eficiencia  : {E['var_ef'].mean():+.2f}%")

# --- tendencia da penalidade com a carga ---
sl = stats.linregress(E["ic2"], E["dif"])
print(f"\n  tendencia da penalidade com a carga:")
print(f"    inclinacao = {sl.slope:+.1f} kW por unidade de indice de carga   r={sl.rvalue:+.3f}  p={sl.pvalue:.4f}")

# --- robustez ao criterio de retencao ---
print("\n" + "=" * 96)
print("ROBUSTEZ AO CRITERIO DE RETENCAO")
print("=" * 96)
for mn in [3, 4, 5, 6, 8, 10]:
    S = p[(p["n2"] >= mn) & (p["n3"] >= mn)].copy()
    if len(S) < 3:
        continue
    dd = (S["kw3"] - S["kw2"]).values
    ve = (100 * ((S["ic3"] / S["kw3"]) / (S["ic2"] / S["kw2"]) - 1)).mean()
    print(f"  minimo {mn:>2} h: {len(dd):>2} estratos | positivos {(dd>0).sum():>2}/{len(dd):<2} | "
          f"media {dd.mean():+6.1f} kW | eficiencia {ve:+.2f}%")

print("\nArquivos: tabela10_pareada.csv, tabela10_estratos.csv")
