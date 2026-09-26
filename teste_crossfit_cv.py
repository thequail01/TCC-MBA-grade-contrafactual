"""
=============================================================================
teste_crossfit_cv.py — carga estimada dentro da amostra ou por cross-fitting?
=============================================================================
Nas treze partições, compara o M2 ajustado sobre a carga estimada pelo M1
nas próprias horas de treino (procedimento do artigo) com o M2 ajustado sobre
a carga estimada por cross-fitting interno ao treino (quatro blocos de dias,
embargo de 24 h). O teste recebe, nos dois casos, a predição do M1 ajustado
em todo o treino. Justifica a construção da cascata em comum.cascata.

SAÍDA: teste_crossfit_cv.csv
=============================================================================
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from comum import (carregar, particoes, preparar, ajustar_xgb, FS_M1, FS_M2,
                   MONO_M1, MONO_M2, XGB_BASE, EMBARGO_H)

op = carregar().dropna(subset=["Q_TR", "kw_total"])

def m1_fit(idx):
    X, y, _, _, imp = preparar(op, idx, idx[:1], FS_M1, "Q_TR")
    return ajustar_xgb(X, y, MONO_M1, FS_M1), imp

def pred(m, imp, idx):
    return pd.Series(m.predict(pd.DataFrame(imp.transform(op.loc[idx, FS_M1]),
                     columns=FS_M1, index=idx)), index=idx)

def crossfit(idx_tr, k=4):
    dias = np.array(sorted(idx_tr.normalize().unique()))
    s = pd.Series(np.nan, index=idx_tr)
    for bl in np.array_split(dias, k):
        te = idx_tr[idx_tr.normalize().isin(bl)]
        lo = pd.Timestamp(bl.min()) - pd.Timedelta(hours=EMBARGO_H)
        hi = pd.Timestamp(bl.max()) + pd.Timedelta(days=1, hours=EMBARGO_H)
        tr = idx_tr[(idx_tr < lo) | (idx_tr >= hi)]
        m, imp = m1_fit(tr)
        s[te] = pred(m, imp, te).values
    return s

linhas = []
for esq, rot, tr, te in particoes(op):
    tr, te = tr.intersection(op.index), te.intersection(op.index)
    m1, imp1 = m1_fit(tr)
    q_te = pred(m1, imp1, te)
    for modo in ["in-sample", "cross-fit"]:
        q_tr = pred(m1, imp1, tr) if modo == "in-sample" else crossfit(tr)
        d = op.copy(); d.loc[tr, "Q_TR_pred"] = q_tr; d.loc[te, "Q_TR_pred"] = q_te
        X2, y2, X2te, y2te, _ = preparar(d, tr, te, FS_M2, "kw_total")
        mx = ajustar_xgb(X2, y2, MONO_M2, FS_M2)
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(X2, y2)
        for nome, m in [("XGB", mx), ("Ridge", rg)]:
            p = m.predict(X2te)
            linhas.append(dict(esquema=esq[0], fold=rot, modo=modo, modelo=nome,
                               R2=r2_score(y2te, p), MAE=mean_absolute_error(y2te, p)))
    print(esq, rot, "ok", flush=True)

R = pd.DataFrame(linhas)
T = R.groupby(["modelo", "modo", "esquema"])[["R2", "MAE"]].mean().round(3).unstack("esquema")
print(T.to_string())
R.to_csv("teste_crossfit_cv.csv", index=False)
