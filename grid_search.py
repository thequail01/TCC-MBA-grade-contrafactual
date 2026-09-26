"""
=============================================================================
GRID SEARCH RESTRITO — validacao da varredura unidimensional
=============================================================================
Objetivo: verificar se o otimo obtido por varredura sequencial (um parametro
por vez) e robusto quando os hiperparametros sao variados conjuntamente.

DESENHO. A selecao e feita pelo MESMO criterio de validacao interna usado
para escolher o numero de arvores: minimo da curva de perda media sobre
blocos disjuntos internos ao treino, com embargo nas fronteiras. O conjunto
de teste do esquema E nao participa da selecao em nenhuma etapa; ele e usado
apenas para reportar, ao final, o desempenho da configuracao escolhida.

Grade: 3 x 3 x 3 x 3 = 81 combinacoes.
=============================================================================
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from comum import (FS_M1, FS_M2, MONO_M1, MONO_M2, XGB_BASE, carregar,
                   curva_interna, purgar as purge)

op = carregar()

# configuracao adotada (a mesma de comum.XGB_BASE) e parametros fixos da grade
ADOTADA = dict(max_depth=3, min_child_weight=8, reg_lambda=5.0, subsample=0.9)
BASE = {k: v for k, v in XGB_BASE.items() if k not in ADOTADA}
assert {**BASE, **ADOTADA} == XGB_BASE


def cv_interna(Xtr, ytr, mono, cols, par, teto=800):
    """Retorna (score de validacao interna, numero de arvores no minimo).

    O score e o minimo da curva de perda media (RMSE) sobre blocos disjuntos
    internos ao treino (comum.curva_interna), com o mesmo teto de 800 arvores
    do pipeline principal. Nenhuma observacao de teste participa.
    """
    media, _ = curva_interna(Xtr, ytr, mono, cols, par, teto=teto)
    if media is None:
        return np.inf, 150
    return float(media.min()), int(np.argmin(media)) + 1


# --- Esquema E: treina marco+abril, testa maio ---
tr = purge(op.index[op.index < "2026-05-01"], op.index[op.index >= "2026-05-01"])
te = op.index[op.index >= "2026-05-01"]

d1 = op.loc[tr, FS_M1 + ["Q_TR"]].dropna(subset=["Q_TR"])
imp1 = SimpleImputer(strategy="median").fit(d1[FS_M1])
X1 = pd.DataFrame(imp1.transform(d1[FS_M1]), columns=FS_M1, index=d1.index)
par1 = dict(BASE); par1.update(ADOTADA)
_, n1 = cv_interna(X1, d1["Q_TR"], MONO_M1, FS_M1, par1)
m1 = xgb.XGBRegressor(n_estimators=n1,
                      monotone_constraints={c: MONO_M1.get(c, 0) for c in FS_M1},
                      **par1).fit(X1, d1["Q_TR"])
allx = tr.union(te)
op.loc[allx, "Q_TR_pred"] = m1.predict(
    pd.DataFrame(imp1.transform(op.loc[allx, FS_M1]), columns=FS_M1, index=allx))

d2tr = op.loc[op.index.isin(tr), FS_M2 + ["kw_total"]].dropna(subset=["kw_total"])
d2te = op.loc[op.index.isin(te), FS_M2 + ["kw_total"]].dropna(subset=["kw_total"])
imp2 = SimpleImputer(strategy="median").fit(d2tr[FS_M2])
X2tr = pd.DataFrame(imp2.transform(d2tr[FS_M2]), columns=FS_M2, index=d2tr.index)
X2te = pd.DataFrame(imp2.transform(d2te[FS_M2]), columns=FS_M2, index=d2te.index)
y2tr, y2te = d2tr["kw_total"], d2te["kw_total"]
print(f"Esquema E: treino {len(X2tr)} h, teste {len(X2te)} h\n")

GRADE = dict(max_depth=[2, 3, 4], min_child_weight=[4, 8, 16],
             reg_lambda=[1.0, 5.0, 10.0], subsample=[0.8, 0.9, 1.0])
chaves = list(GRADE)
combos = list(itertools.product(*[GRADE[k] for k in chaves]))
print(f"Avaliando {len(combos)} combinacoes por validacao interna...\n")

linhas = []
for i, valores in enumerate(combos, 1):
    cfg = dict(zip(chaves, valores))
    par = dict(BASE); par.update(cfg)
    score, n_arv = cv_interna(X2tr, y2tr, MONO_M2, FS_M2, par)
    mv = {c: MONO_M2.get(c, 0) for c in FS_M2}
    m = xgb.XGBRegressor(n_estimators=n_arv, monotone_constraints=mv, **par).fit(X2tr, y2tr)
    linhas.append(dict(**cfg, n_arv=n_arv, cv_rmse=score,
                       R2_teste=r2_score(y2te, m.predict(X2te)),
                       MAE_teste=mean_absolute_error(y2te, m.predict(X2te)),
                       R2_treino=r2_score(y2tr, m.predict(X2tr))))
    if i % 20 == 0:
        print(f"  {i}/{len(combos)}")

G = pd.DataFrame(linhas).sort_values("cv_rmse").reset_index(drop=True)
G.to_csv("grid_search_restrito.csv", index=False)

print("\n" + "=" * 92)
print("SELECAO POR VALIDACAO INTERNA (o teste nao participa)")
print("=" * 92)
print(G.head(8).round(4).to_string(index=False))

melhor = G.iloc[0]
ad = G[(G.max_depth == 3) & (G.min_child_weight == 8) &
       (G.reg_lambda == 5.0) & (G.subsample == 0.9)].iloc[0]

print("\n" + "=" * 92)
print("CONFIGURACAO ADOTADA vs SELECIONADA PELA GRADE")
print("=" * 92)
print(f"  adotada    : depth={int(ad.max_depth)} mcw={int(ad.min_child_weight)} "
      f"lambda={ad.reg_lambda} subsample={ad.subsample} | "
      f"cv_rmse={ad.cv_rmse:.3f} R2_teste={ad.R2_teste:.3f} MAE={ad.MAE_teste:.2f}")
print(f"  selecionada: depth={int(melhor.max_depth)} mcw={int(melhor.min_child_weight)} "
      f"lambda={melhor.reg_lambda} subsample={melhor.subsample} | "
      f"cv_rmse={melhor.cv_rmse:.3f} R2_teste={melhor.R2_teste:.3f} MAE={melhor.MAE_teste:.2f}")
print(f"\n  posicao da adotada no ranking por validacao interna: "
      f"{G.index[(G.max_depth==3)&(G.min_child_weight==8)&(G.reg_lambda==5.0)&(G.subsample==0.9)][0]+1} de {len(G)}")

print("\n" + "=" * 92)
print("DISPERSAO DA GRADE (diagnostico de robustez)")
print("=" * 92)
print(f"  R2 em teste : min={G.R2_teste.min():.3f}  mediana={G.R2_teste.median():.3f}  "
      f"max={G.R2_teste.max():.3f}  amplitude={G.R2_teste.max()-G.R2_teste.min():.3f}")
print(f"  desvio-padrao do R2 em teste entre as {len(G)} combinacoes: {G.R2_teste.std():.4f}")
print(f"  correlacao entre score de validacao interna e R2 em teste: "
      f"{G.cv_rmse.corr(G.R2_teste):+.3f}")

print("\n  efeito marginal medio de cada parametro sobre o R2 em teste:")
for k in chaves:
    print(f"    {k:18s} " + "  ".join(f"{v}={G[G[k]==v].R2_teste.mean():.3f}" for v in GRADE[k]))

print("\n  interacao: melhor valor de cada parametro condicionado aos demais")
for k in chaves:
    outros = [c for c in chaves if c != k]
    vencedores = G.loc[G.groupby(outros).R2_teste.idxmax(), k].value_counts().sort_index()
    print(f"    {k:18s} " + "  ".join(f"{v}:{vencedores.get(v,0)}x" for v in GRADE[k]))
