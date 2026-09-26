"""
=============================================================================
tabela11_grade_robusta.py — grade contrafactual com cross-fitting, variante irrestrita
                             e intervalos por bootstrap de blocos diários
=============================================================================

Complementa tabela11_grade.py (grade de referência) em três pontos:

1. SENSIBILIDADE À CONSTRUÇÃO DA CARGA ESTIMADA. A referência ajusta o M2 sobre
   a carga estimada pelo M1 nas próprias horas de treino, como nas Tabelas 5 a 8.
   Comparam-se duas alternativas: cross-fitting (cada bloco de dias recebe
   estimativa de um M1 que não o viu, com embargo de 24 h) e carga medida.
   teste_crossfit_cv.py mostra que o cross-fitting prevê pior fora da amostra,
   razão pela qual a referência é mantida.

2. VARIANTE SEM RESTRIÇÃO EM n_chillers. Com +1 em n_chillers a grade não pode
   indicar que a terceira máquina economiza. Reajusta-se o M2 com restrição 0
   nessa variável para verificar se sinal e magnitude se mantêm sem a imposição.

3. INCERTEZA. Bootstrap de blocos diários (dias úteis reamostrados com
   reposição), reajustando a cascata a cada réplica com número de árvores
   fixado no valor da amostra original. Produz intervalos de 95% para o custo
   da terceira máquina e para a economia do setpoint.

Produz também uma estimativa da energia associada à terceira máquina nas horas
em que ela operou dentro da janela, como ordem de grandeza.

SAÍDAS: grade_robusta.csv, grade_bootstrap.csv
=============================================================================
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

from comum import (carregar, FS_M1, FS_M2, MONO_M1, MONO_M2, XGB_BASE,
                   EMBARGO_H, n_arvores)

B_REPLICAS = 200
# Variante de referência: a mesma das Tabelas 5 a 8 (M2 ajustado sobre a carga
# estimada pelo M1 nas próprias horas de treino). O teste_crossfit_cv.py mostra
# que essa variante prevê melhor fora da amostra que o cross-fitting.
REFERENCIA = "original (in-sample)"
K_BLOCOS = 5
PCTS = {"P25": 0.25, "P50": 0.50, "P75": 0.75, "P90": 0.90}
SPS, NCS, HORA = [5.5, 6.5, 7.5], [2, 3], 15

op = carregar()
op = op.dropna(subset=["Q_TR", "kw_total"])
q_ref = op["Q_TR_cal"].mean()
ent_med, t_med = op["entalpia"].median(), op["T_ext"].median()
# rótulos calculados a partir dos dados, como em tabela11_grade.py
q_grade = {f"{op['IC'].quantile(q):.2f} ({r})".replace(".", ","): op["IC"].quantile(q) * q_ref
           for r, q in PCTS.items()}


def ajustar(X, y, mono, cols, n):
    mv = {c: mono.get(c, 0) for c in cols}
    return xgb.XGBRegressor(n_estimators=n, monotone_constraints=mv,
                            **XGB_BASE).fit(X, y)


def matriz(d, cols, imp=None):
    imp = imp or SimpleImputer(strategy="median").fit(d[cols])
    return pd.DataFrame(imp.transform(d[cols]), columns=cols, index=d.index), imp


def q_pred_crossfit(d, n1):
    """Q_TR_pred fora do bloco: cada bloco de dias recebe predição de um M1
    ajustado nos demais, com embargo de 24 h nas fronteiras."""
    dias = np.array(sorted(d["_dia"].unique()))
    blocos = np.array_split(dias, K_BLOCOS)
    saida = pd.Series(np.nan, index=d.index)
    for bl in blocos:
        te = d["_dia"].isin(bl)
        lo = pd.Timestamp(bl.min()) - pd.Timedelta(hours=EMBARGO_H)
        hi = pd.Timestamp(bl.max()) + pd.Timedelta(days=1, hours=EMBARGO_H)
        tr = ~te & ((d["_t"] < lo) | (d["_t"] >= hi))
        X_tr, imp = matriz(d[tr], FS_M1)
        m1 = ajustar(X_tr, d.loc[tr, "Q_TR"], MONO_M1, FS_M1, n1)
        X_te, _ = matriz(d[te], FS_M1, imp)
        saida[te] = m1.predict(X_te)
    return saida


def grade(m2):
    linhas = []
    for rot, qtr in q_grade.items():
        lin = {"faixa": rot}
        for sp in SPS:
            for nc in NCS:
                X = pd.DataFrame([{"Q_TR_pred": qtr, "n_chillers": nc, "entalpia": ent_med,
                                   "hour": HORA, "T_ext": t_med, "sp_chw": sp}])[FS_M2]
                lin[f"{sp}/{nc}"] = float(m2.predict(X)[0])
        linhas.append(lin)
    return pd.DataFrame(linhas).set_index("faixa")


def efeitos(G):
    c3 = np.array([[G.loc[r, f"{sp}/3"] - G.loc[r, f"{sp}/2"] for sp in SPS] for r in G.index])
    esp = np.array([[G.loc[r, f"5.5/{nc}"] - G.loc[r, f"7.5/{nc}"] for nc in NCS] for r in G.index])
    return {"c3_media": c3.mean(), "c3_min": c3.min(), "c3_max": c3.max(),
            "c3_P90_sp5.5": c3[-1, 0],
            "sp_media": esp.mean(), "sp_min": esp.min(), "sp_max": esp.max(),
            "sp_por_C": esp.mean() / 2.0}


def cascata_completa(d, n1, n2, crossfit=True, mono2=MONO_M2, alvo_carga="pred"):
    d = d.copy()
    if alvo_carga == "obs":
        d["Q_TR_pred"] = d["Q_TR"]
    elif crossfit:
        d["Q_TR_pred"] = q_pred_crossfit(d, n1)
    else:
        X1, imp = matriz(d, FS_M1)
        d["Q_TR_pred"] = ajustar(X1, d["Q_TR"], MONO_M1, FS_M1, n1).predict(X1)
    d = d.dropna(subset=["Q_TR_pred"])
    X2, _ = matriz(d, FS_M2)
    return ajustar(X2, d["kw_total"], mono2, FS_M2, n2), d


# -----------------------------------------------------------------------------
# Número de árvores na amostra original (fixado depois nas réplicas)
# -----------------------------------------------------------------------------
op["_t"], op["_dia"] = op.index, op.index.normalize()
X1, _ = matriz(op, FS_M1)
n1 = n_arvores(X1, op["Q_TR"], MONO_M1, FS_M1)
# n2 é selecionado sobre a mesma construção de Q_TR_pred da referência,
# reproduzindo tabela11_grade.py quando a referência é a variante original.
op_ref = op.copy()
if REFERENCIA.startswith("original"):
    op_ref["Q_TR_pred"] = xgb.XGBRegressor(
        n_estimators=n1, monotone_constraints={c: MONO_M1.get(c, 0) for c in FS_M1},
        **XGB_BASE).fit(X1, op["Q_TR"]).predict(X1)
else:
    op_ref["Q_TR_pred"] = q_pred_crossfit(op, n1)
X2, _ = matriz(op_ref, FS_M2)
n2 = n_arvores(X2, op_ref["kw_total"], MONO_M2, FS_M2)
print(f"árvores: M1={n1}  M2={n2}\n")

MONO_M2_LIVRE = {**MONO_M2, "n_chillers": 0}
variantes = {
    "original (in-sample)": dict(crossfit=False),
    "original, n_chillers livre": dict(crossfit=False, mono2=MONO_M2_LIVRE),
    "cross-fitting": dict(crossfit=True),
    "cross-fitting, n_chillers livre": dict(crossfit=True, mono2=MONO_M2_LIVRE),
    "M2 sobre carga observada": dict(alvo_carga="obs"),
}
res, grades = {}, {}
for nome, kw in variantes.items():
    m2, dd = cascata_completa(op, n1, n2, **kw)
    grades[nome] = grade(m2)
    res[nome] = efeitos(grades[nome])
    if nome == REFERENCIA:
        m2_ref, d_ref = m2, dd

R = pd.DataFrame(res).T.round(1)
print("=" * 100)
print("EFEITOS DA GRADE POR VARIANTE (kW)")
print("=" * 100)
print(R.to_string())
print("\nGrade de referência:")
print(grades[REFERENCIA].round(0).to_string())
print("\nGrade com cross-fitting e n_chillers livre:")
print(grades["cross-fitting, n_chillers livre"].round(0).to_string())

# -----------------------------------------------------------------------------
# Energia associada à terceira máquina nas horas em que operou (ordem de grandeza)
# -----------------------------------------------------------------------------
h3 = d_ref[d_ref["n_chillers"] == 3]
X3, _ = matriz(h3, FS_M2)
X2c = X3.copy(); X2c["n_chillers"] = 2
dkw = m2_ref.predict(X3) - m2_ref.predict(X2c)
print(f"\nHoras com 3 máquinas na janela: {len(h3)}")
print(f"Diferença estimada média nessas horas: {dkw.mean():.1f} kW")
print(f"Energia associada no período (janela 12h-20h): {dkw.sum()/1000:.1f} MWh")

# -----------------------------------------------------------------------------
# Bootstrap de blocos diários
# -----------------------------------------------------------------------------
rng = np.random.default_rng(42)
dias = op["_dia"].unique()
boot, falhas = [], 0
for b in range(B_REPLICAS):
    esc = rng.choice(dias, size=len(dias), replace=True)
    partes = []
    for j, dia in enumerate(esc):
        p = op[op["_dia"] == dia].copy()
        # desloca cada réplica de dia para uma posição artificial própria,
        # preservando a ordem e evitando colisão de índices
        desloc = pd.Timedelta(days=j) - (dia - dias.min())
        p.index = p.index + desloc
        p["_t"], p["_dia"] = p.index, p.index.normalize()
        partes.append(p)
    db = pd.concat(partes).sort_index()
    try:
        m2b, _ = cascata_completa(db, n1, n2, **variantes[REFERENCIA])
        boot.append(efeitos(grade(m2b)))
    except Exception:  # réplica degenerada; contabilizada e reportada abaixo
        falhas += 1
        continue
    if (b + 1) % 50 == 0:
        print(f"  réplicas: {b+1}")

BT = pd.DataFrame(boot)
BT.to_csv("grade_bootstrap.csv", index=False)
print("\n" + "=" * 100)
print(f"BOOTSTRAP DE BLOCOS DIÁRIOS ({len(BT)} réplicas válidas, {falhas} descartadas): intervalos de 95%")
print("=" * 100)
for c in ["c3_media", "c3_P90_sp5.5", "sp_media", "sp_por_C"]:
    print(f"  {c:14s}: {res[REFERENCIA][c]:7.1f}  [{BT[c].quantile(.025):6.1f} ; {BT[c].quantile(.975):6.1f}]")
print(f"  P(custo médio da 3ª máquina > economia média do setpoint) = {(BT['c3_media'] > BT['sp_media']).mean():.3f}")

pd.concat({k: v for k, v in grades.items()}).round(1).to_csv("grade_robusta.csv")
