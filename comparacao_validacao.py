"""
=============================================================================
comparacao_validacao.py — cinco esquemas de validação (Tabelas 5 a 8)
=============================================================================

MOTIVAÇÃO
A série cobre três regimes operacionais de setpoint:
    R1 = março  -> 5,5 °C
    R2 = abril  -> setpoint variável (5,5 a 7,5 °C)
    R3 = maio   -> 7,5 °C predominante
Com regimes diferentes mês a mês, o esquema de validação determina o que a
métrica significa. Este script executa cinco esquemas sobre exatamente os
mesmos dados e modelos, para tornar essa escolha explícita e verificável.

ESQUEMAS (construídos em comum.particoes)
  A_expansivo_AM    janela expansiva, abril e maio (dois meses de histórico)
  B_expansivo_MAM   janela expansiva, março a maio
  C_deslizante      janela deslizante de seis semanas
  D_LORO            regime ausente (Leave-One-Regime-Out)
  E_holdout_final   treina março e abril, testa maio inteiro

PROCEDIMENTOS COMUNS A TODAS AS PARTIÇÕES
  - O M1 é ajustado apenas no treino da partição, e a carga estimada que
    alimenta o M2 é gerada dentro da própria partição (comum.cascata), sem
    reaproveitamento entre partições.
  - A imputação usa a mediana do treino da partição (comum.preparar).
  - O número de árvores do XGBoost é escolhido por validação interna ao
    treino (comum.n_arvores); o teste não participa da seleção.

SAÍDAS
  comparacao_validacao.csv   uma linha por partição, alvo e modelo
  dispersao_arvores.csv      contagens de árvores de cada divisão interna

USO
    python comparacao_validacao.py
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comum import (FS_M1, FS_M2, MONO_M1, MONO_M2, ajustar_xgb, carregar,
                   particoes, persistencia, preparar, ridge)

SAIDA = "comparacao_validacao.csv"
DISPERSAO = []          # contagens de árvores por divisão interna


def linha(esquema, fold, alvo, modelo, n_tr, n_te, y, p):
    return dict(esquema=esquema, fold=fold, alvo=alvo, modelo=modelo,
                n_tr=n_tr, n_te=n_te, R2=r2_score(y, p),
                MAE=mean_absolute_error(y, p))


def avaliar_particao(op, tr, te, esquema, fold):
    """Executa M1 -> carga estimada -> M2 para um único par (treino, teste)."""
    if len(tr) < 30 or len(te) < 10:
        return []
    out = []

    # ---- M1: carga térmica ----
    X1tr, y1tr, X1te, y1te, imp1 = preparar(op, tr, te, FS_M1, "Q_TR")
    m1 = ajustar_xgb(X1tr, y1tr, MONO_M1, FS_M1, registro=DISPERSAO,
                     rotulo=f"{esquema}|{fold}|M1")
    out.append(linha(esquema, fold, "M1_Q_TR", "XGB_mono", len(X1tr), len(X1te),
                     y1te, m1.predict(X1te)))
    out.append(linha(esquema, fold, "M1_Q_TR", "Persistencia", len(X1tr), len(X1te),
                     y1te, persistencia(op["Q_TR"], y1te.index, y1tr.mean())))

    # ---- carga estimada gerada dentro desta partição ----
    todos = tr.union(te)
    o2 = op.loc[todos].copy()
    o2["Q_TR_pred"] = m1.predict(pd.DataFrame(imp1.transform(op.loc[todos, FS_M1]),
                                              columns=FS_M1, index=todos))

    # ---- M2: potência elétrica ----
    X2tr, y2tr, X2te, y2te, _ = preparar(o2, tr, te, FS_M2, "kw_total")
    candidatos = {
        "XGB_mono": ajustar_xgb(X2tr, y2tr, MONO_M2, FS_M2, registro=DISPERSAO,
                                rotulo=f"{esquema}|{fold}|M2"),
        "Ridge": ridge().fit(X2tr, y2tr),
        "LinReg": make_pipeline(StandardScaler(), LinearRegression()).fit(X2tr, y2tr),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1).fit(X2tr, y2tr),
    }
    for nome, m in candidatos.items():
        out.append(linha(esquema, fold, "M2_kW", nome, len(X2tr), len(X2te),
                         y2te, m.predict(X2te)))
    out.append(linha(esquema, fold, "M2_kW", "Persistencia", len(X2tr), len(X2te),
                     y2te, persistencia(op["kw_total"], y2te.index, y2tr.mean())))
    out.append(linha(esquema, fold, "M2_kW", "Media", len(X2tr), len(X2te),
                     y2te, np.full(len(y2te), y2tr.mean())))
    return out


def main():
    op = carregar()
    res = []
    for esquema, fold, tr, te in particoes(op):
        res += avaliar_particao(op, tr, te, esquema, fold)

    R = pd.DataFrame(res)
    R.to_csv(SAIDA, index=False)

    for alvo in ["M1_Q_TR", "M2_kW"]:
        print(f"\n{'=' * 86}\n{alvo}: R² médio por esquema de validação\n{'=' * 86}")
        print(R[R.alvo == alvo].pivot_table(index="modelo", columns="esquema",
                                            values="R2", aggfunc="mean").round(3).to_string())
        print(f"\n{alvo}: MAE médio por esquema")
        print(R[R.alvo == alvo].pivot_table(index="modelo", columns="esquema",
                                            values="MAE", aggfunc="mean").round(1).to_string())

    DP = pd.DataFrame(DISPERSAO)
    if not DP.empty:
        DP.to_csv("dispersao_arvores.csv", index=False)
        m2 = DP[DP["fold"].str.endswith("M2")]
        print(f"\n{'=' * 86}\nDispersão das contagens de árvores (M2)\n{'=' * 86}")
        print(m2[["fold", "contagens", "escolhido", "amplitude", "cv_pct"]]
              .round(1).to_string(index=False))
        print(f"\nCoeficiente de variação médio: {m2['cv_pct'].mean():.0f}% | "
              f"amplitude máxima: {int(m2['amplitude'].max())} árvores")

    d = R[(R.esquema == "D_LORO") & (R.alvo == "M2_kW")]
    print(f"\n{'=' * 86}\nD_LORO detalhado (M2): regime deixado de fora\n{'=' * 86}")
    print(d.pivot_table(index="fold", columns="modelo", values="R2").round(3).to_string())
    print("\nMAE (kW):")
    print(d.pivot_table(index="fold", columns="modelo", values="MAE").round(1).to_string())
    print(f"\nArquivos: {SAIDA}, dispersao_arvores.csv")


if __name__ == "__main__":
    main()
