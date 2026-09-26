"""
=============================================================================
GRADE CONTRAFACTUAL — Tabela 11
=============================================================================
Estimativa de potencia para cada combinacao de setpoint e numero de maquinas,
sob entalpia mediana e hora fixa, para quatro percentis do indice de carga.

O modelo e o M2 com restricoes de monotonicidade, ajustado sobre a base
integral. As demais variaveis sao fixadas na mediana da janela, e o Q_TR_pred
correspondente a cada percentil de carga e obtido pela conversao do indice.

Com restricao +1 no numero de maquinas e -1 no setpoint, o SINAL desses
efeitos e imposto por construcao e so a magnitude e aprendida. A variante sem
restricao no numero de maquinas e os intervalos por bootstrap estao em
tabela11_grade_robusta.py.

SAIDA: tabela11_grade.csv
=============================================================================
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from comum import FS_M1, FS_M2, MONO_M1, MONO_M2, XGB_BASE as BASE, carregar, n_arvores

op = carregar()


def n_arv(Xtr, ytr, mono, cols):
    return n_arvores(Xtr, ytr, mono, cols, BASE, teto=800)


# --- M1 sobre a base integral ---
d1 = op[FS_M1 + ["Q_TR"]].dropna(subset=["Q_TR"])
imp1 = SimpleImputer(strategy="median").fit(d1[FS_M1])
X1 = pd.DataFrame(imp1.transform(d1[FS_M1]), columns=FS_M1, index=d1.index)
m1 = xgb.XGBRegressor(n_estimators=n_arv(X1, d1["Q_TR"], MONO_M1, FS_M1),
                      monotone_constraints={c: MONO_M1.get(c, 0) for c in FS_M1},
                      **BASE).fit(X1, d1["Q_TR"])
op["Q_TR_pred"] = m1.predict(pd.DataFrame(imp1.transform(op[FS_M1]),
                                          columns=FS_M1, index=op.index))

# --- M2 sobre a base integral ---
d2 = op[FS_M2 + ["kw_total"]].dropna(subset=["kw_total"])
imp2 = SimpleImputer(strategy="median").fit(d2[FS_M2])
X2 = pd.DataFrame(imp2.transform(d2[FS_M2]), columns=FS_M2, index=d2.index)
m2 = xgb.XGBRegressor(n_estimators=n_arv(X2, d2["kw_total"], MONO_M2, FS_M2),
                      monotone_constraints={c: MONO_M2.get(c, 0) for c in FS_M2},
                      **BASE).fit(X2, d2["kw_total"])
print(f"M1: {m1.n_estimators} arvores | M2: {m2.n_estimators} arvores")
print(f"R2 em treino (M2, base integral): {r2_score(d2['kw_total'], m2.predict(X2)):.3f}\n")

# --- grade ---
ent_med = op["entalpia"].median()
t_med = op["T_ext"].median()
hora = 15
print(f"condicoes fixas: entalpia={ent_med:.1f} kJ/kg, T_ext={t_med:.1f} C, hora={hora}h\n")

pcts = {"P25": 0.25, "P50": 0.50, "P75": 0.75, "P90": 0.90}
# converte percentil de IC em Q_TR_pred equivalente; o rotulo de cada linha e
# calculado a partir dos dados, e nao digitado
q_ref = op["Q_TR_cal"].mean()
linhas = []
for nome, q in pcts.items():
    ic = op["IC"].quantile(q)
    qtr = ic * q_ref
    rot = f"{ic:.2f} ({nome})".replace(".", ",")
    linha = {"faixa": rot, "IC": round(ic, 3)}
    for sp in [5.5, 6.5, 7.5]:
        for nc in [2, 3]:
            X = pd.DataFrame([{"Q_TR_pred": qtr, "n_chillers": nc, "entalpia": ent_med,
                               "hour": hora, "T_ext": t_med, "sp_chw": sp}])[FS_M2]
            linha[f"{sp}/{nc}"] = round(float(m2.predict(X)[0]))
    linhas.append(linha)

G = pd.DataFrame(linhas).set_index("faixa")
G.to_csv("tabela11_grade.csv")
print("=" * 84)
print("TABELA 11 — potencia estimada (kW) por configuracao")
print("=" * 84)
print(G.to_string())

print("\n" + "=" * 84)
print("ESTRUTURA DOS EFEITOS")
print("=" * 84)
for rot in G.index:
    c3 = [G.loc[rot, f"{sp}/3"] - G.loc[rot, f"{sp}/2"] for sp in [5.5, 6.5, 7.5]]
    sp_ef = [G.loc[rot, f"5.5/{n}"] - G.loc[rot, f"7.5/{n}"] for n in [2, 3]]
    print(f"  {rot}: 3a maquina custa {min(c3)} a {max(c3)} kW | "
          f"elevar SP 5,5->7,5 economiza {min(sp_ef)} a {max(sp_ef)} kW")
todos3 = [G.loc[r, f"{sp}/3"] - G.loc[r, f"{sp}/2"] for r in G.index for sp in [5.5, 6.5, 7.5]]
todosp = [G.loc[r, f"5.5/{n}"] - G.loc[r, f"7.5/{n}"] for r in G.index for n in [2, 3]]
print(f"\n  custo da 3a maquina, global : {min(todos3)} a {max(todos3)} kW")
print(f"  economia do setpoint, global: {min(todosp)} a {max(todosp)} kW")
print(f"  melhor configuracao em todas as linhas: 7,5 C com 2 maquinas "
      f"({all(G.loc[r,'7.5/2']==min(G.loc[r,[c for c in G.columns if '/' in c]]) for r in G.index)})")

print("\n" + "=" * 84)
print("COBERTURA EMPIRICA DE CADA CELULA DA GRADE (horas observadas)")
print("=" * 84)
op["spf"] = np.select([op.sp_chw < 6.0, op.sp_chw < 7.0], [5.5, 6.5], default=7.5)
print(pd.crosstab(op.spf, op.n_chillers).to_string())
