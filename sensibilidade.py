"""
=============================================================================
SENSIBILIDADE DE HIPERPARAMETROS, MONOTONICIDADE E IMPORTANCIA — Tabela 9
=============================================================================
Varredura unidimensional em torno da configuracao adotada, avaliada no
esquema E (treina marco+abril, testa maio), com o mesmo procedimento de
selecao do numero de arvores usado no pipeline principal.

Reporta R2 em teste e em treino; a diferenca diagnostica sobreajuste.

SAIDAS (impressas):
  varredura unidimensional     diagnostico de robustez citado no texto
  efeito das restricoes (M2)   restricoes do M2 ligadas e desligadas
  importancia das variaveis    Tabela 9 (ganho, frequencia e cobertura)

O efeito das restricoes do M1 sobre a cascata, em todos os esquemas, esta
em verificacoes_complementares.py.
=============================================================================
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from comum import (FS_M1, FS_M2, MONO_M1, MONO_M2, XGB_BASE as BASE, carregar,
                   n_arvores, purgar as purge)

op = carregar()


# --- Esquema E ---
tr = purge(op.index[op.index < "2026-05-01"], op.index[op.index >= "2026-05-01"])
te = op.index[op.index >= "2026-05-01"]

d1 = op.loc[tr, FS_M1 + ["Q_TR"]].dropna(subset=["Q_TR"])
imp1 = SimpleImputer(strategy="median").fit(d1[FS_M1])
X1 = pd.DataFrame(imp1.transform(d1[FS_M1]), columns=FS_M1, index=d1.index)
n1 = n_arvores(X1, d1["Q_TR"], MONO_M1, FS_M1, BASE)
m1 = xgb.XGBRegressor(n_estimators=n1,
                      monotone_constraints={c: MONO_M1.get(c, 0) for c in FS_M1},
                      **BASE).fit(X1, d1["Q_TR"])
allx = tr.union(te)
op.loc[allx, "Q_TR_pred"] = m1.predict(
    pd.DataFrame(imp1.transform(op.loc[allx, FS_M1]), columns=FS_M1, index=allx))

d2tr = op.loc[op.index.isin(tr), FS_M2 + ["kw_total"]].dropna(subset=["kw_total"])
d2te = op.loc[op.index.isin(te), FS_M2 + ["kw_total"]].dropna(subset=["kw_total"])
imp2 = SimpleImputer(strategy="median").fit(d2tr[FS_M2])
X2tr = pd.DataFrame(imp2.transform(d2tr[FS_M2]), columns=FS_M2, index=d2tr.index)
X2te = pd.DataFrame(imp2.transform(d2te[FS_M2]), columns=FS_M2, index=d2te.index)
y2tr, y2te = d2tr["kw_total"], d2te["kw_total"]
print(f"Esquema E: treino {len(X2tr)} h  teste {len(X2te)} h  (M1 com {n1} arvores)\n")


def avaliar(par, mono=MONO_M2, n_fixo=None):
    p = dict(BASE); p.update(par)
    n = n_fixo if n_fixo else n_arvores(X2tr, y2tr, mono, FS_M2, p)
    mv = {c: mono.get(c, 0) for c in FS_M2}
    m = xgb.XGBRegressor(n_estimators=n, monotone_constraints=mv, **p).fit(X2tr, y2tr)
    return r2_score(y2te, m.predict(X2te)), r2_score(y2tr, m.predict(X2tr)), n, m


base_te, base_tr, base_n, base_m = avaliar({})
print(f"CONFIGURACAO ADOTADA: R2 teste={base_te:.3f}  R2 treino={base_tr:.3f}  arvores={base_n}\n")

VARRE = {
    "max_depth": [1, 2, 3, 4, 5, 6],
    "reg_lambda": [0, 1, 5, 10, 25, 50],
    "min_child_weight": [1, 2, 4, 8, 16, 32],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
linhas = []
for nome, valores in VARRE.items():
    print(f"--- {nome} ---")
    res = []
    for v in valores:
        rt, rr, n, _ = avaliar({nome: v})
        res.append((v, rt, rr, n))
        print(f"   {v:>5}  teste={rt:.3f}  treino={rr:.3f}  arvores={n}")
    melhor = max(res, key=lambda x: x[1])
    linhas.append(dict(parametro=nome, faixa=f"{valores[0]} a {valores[-1]}",
                       otimo=melhor[0], minimo=min(r[1] for r in res),
                       maximo=max(r[1] for r in res),
                       treino_min=min(r[2] for r in res), treino_max=max(r[2] for r in res)))
    print()

print("--- learning_rate x n_estimators (produto ~constante) ---")
res = []
for lr, ne in [(0.02, 375), (0.05, 150), (0.08, 94), (0.1, 75), (0.2, 38), (0.3, 25)]:
    rt, rr, n, _ = avaliar({"learning_rate": lr}, n_fixo=ne)
    res.append((f"{lr}/{ne}", rt, rr))
    print(f"   {lr}/{ne:<4} teste={rt:.3f}  treino={rr:.3f}")
melhor = max(res, key=lambda x: x[1])
linhas.append(dict(parametro="learning_rate x n_estimators", faixa="0,02/375 a 0,3/25",
                   otimo=melhor[0], minimo=min(r[1] for r in res),
                   maximo=max(r[1] for r in res),
                   treino_min=min(r[2] for r in res), treino_max=max(r[2] for r in res)))

print("\n" + "=" * 78)
print("VARREDURA UNIDIMENSIONAL (diagnostico de robustez)")
print("=" * 78)
T = pd.DataFrame(linhas)
T["amplitude_teste"] = T.apply(lambda r: f"{r.minimo:.3f} a {r.maximo:.3f}", axis=1)
T["amplitude_treino"] = T.apply(lambda r: f"{r.treino_min:.3f} a {r.treino_max:.3f}", axis=1)
print(T[["parametro", "faixa", "otimo", "amplitude_teste", "amplitude_treino"]].to_string(index=False))

# --- Monotonicidade ---
print("\n" + "=" * 78)
print("EFEITO DAS RESTRICOES DE MONOTONICIDADE")
print("=" * 78)
com_te, com_tr, com_n, com_m = avaliar({})
sem_te, sem_tr, sem_n, sem_m = avaliar({}, mono={})
print(f"  com restricoes : teste={com_te:.3f}  treino={com_tr:.3f}  arvores={com_n}")
print(f"  sem restricoes : teste={sem_te:.3f}  treino={sem_tr:.3f}  arvores={sem_n}")

grade = np.arange(5.5, 7.51, 0.25)
base_row = X2te.median().to_frame().T
for rot, mod in [("com restricoes", com_m), ("sem restricoes", sem_m)]:
    G = pd.concat([base_row] * len(grade), ignore_index=True)
    G["sp_chw"] = grade
    p = mod.predict(G[FS_M2])
    inc = int((np.diff(p) > 0).sum())
    print(f"\n  varredura do setpoint, {rot}: {inc} de {len(grade)-1} incrementos elevam a potencia")
    print("   " + "  ".join(f"{s:.2f}:{v:.0f}" for s, v in zip(grade, p)))

# --- Importancia ---
print("\n" + "=" * 78)
print("TABELA 9 — IMPORTANCIA DE VARIAVEIS (modelo com restricoes, esquema E)")
print("=" * 78)
booster = com_m.get_booster()
tab = {}
for tipo in ["gain", "weight", "cover"]:
    sc = booster.get_score(importance_type=tipo)
    tot = sum(sc.values())
    tab[tipo] = {k: 100 * v / tot for k, v in sc.items()}
I = pd.DataFrame(tab).reindex(FS_M2).fillna(0).round(1)
I.columns = ["ganho_%", "frequencia_%", "cobertura_%"]
print(I.to_string())
