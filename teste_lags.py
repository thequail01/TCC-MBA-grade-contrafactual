"""
=============================================================================
teste_lags.py — quanto do desempenho vem das defasagens da carga?
=============================================================================

PERGUNTA. O primeiro modelo combina variáveis climáticas e de calendário com
defasagens intradiárias da própria carga. Quanto de seu poder explicativo
provém de cada grupo?

POR QUE IMPORTA. O título e a formulação do trabalho apoiam-se na entalpia
como variável climática central. Se a maior parte do desempenho do primeiro
modelo provier da persistência da carga, e não do clima, isso precisa ser
declarado, e delimita o horizonte de aplicação: com defasagens, o instrumento
opera para a hora seguinte a partir do estado corrente da planta, e não em
horizonte de um dia à frente apoiado apenas em previsão meteorológica.

PROCEDIMENTO. Dois conjuntos de variáveis para o primeiro modelo, avaliados
sobre os mesmos cinco esquemas de validação e com o mesmo procedimento de
seleção do número de árvores:

    COM_DEFASAGENS  clima + calendário + Q_TR_L1, Q_TR_L2, Q_TR_R3  (adotado)
    SEM_DEFASAGENS  clima + calendário apenas

Mede-se também o efeito propagado: o segundo modelo é ajustado sobre a carga
estimada por cada versão do primeiro, isolando o impacto na cascata.

SAÍDA. Tabelas de R² por esquema para M1 e M2. Alimenta a primeira limitação
declarada no artigo (faixa de 0,10 a 0,17 sem defasagens).

USO
    python teste_lags.py
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from comum import (FS_M2, MONO_M1, MONO_M2, ajustar_xgb, carregar, particoes,
                   preparar)

CLIMA = ["entalpia", "T_ext", "UR_ext", "hour", "day_of_week", "month"]
DEFASAGENS = ["Q_TR_L1", "Q_TR_L2", "Q_TR_R3"]

CONJUNTOS = {
    "COM_DEFASAGENS": CLIMA + DEFASAGENS,
    "SEM_DEFASAGENS": CLIMA,
}


def avaliar(op, idx_tr, idx_te, esquema):
    """Ajusta M1 nas duas versões e propaga cada uma para o M2."""
    linhas = []
    for nome, cols in CONJUNTOS.items():
        # ---- primeiro modelo ----
        X_tr, y_tr, X_te, y_te, imp = preparar(op, idx_tr, idx_te, cols, "Q_TR")
        m1 = ajustar_xgb(X_tr, y_tr, MONO_M1, cols)
        pred1 = m1.predict(X_te)
        linhas.append(dict(esquema=esquema, conjunto=nome, alvo="M1",
                           R2=r2_score(y_te, pred1),
                           MAE=mean_absolute_error(y_te, pred1)))

        # ---- propagação para o segundo modelo ----
        todos = idx_tr.union(idx_te)
        X_todos = pd.DataFrame(imp.transform(op.loc[todos, cols]),
                               columns=cols, index=todos)
        op2 = op.loc[todos].copy()
        op2["Q_TR_pred"] = m1.predict(X_todos)

        X2_tr, y2_tr, X2_te, y2_te, _ = preparar(
            op2, todos.intersection(idx_tr), todos.intersection(idx_te),
            FS_M2, "kw_total")
        m2 = ajustar_xgb(X2_tr, y2_tr, MONO_M2, FS_M2)
        pred2 = m2.predict(X2_te)
        linhas.append(dict(esquema=esquema, conjunto=nome, alvo="M2",
                           R2=r2_score(y2_te, pred2),
                           MAE=mean_absolute_error(y2_te, pred2)))
    return linhas


def main():
    op = carregar()
    resultados = []
    for esquema, rotulo, idx_tr, idx_te in particoes(op):
        if len(idx_tr) < 30 or len(idx_te) < 10:
            continue
        resultados += avaliar(op, idx_tr, idx_te, esquema[0])

    R = pd.DataFrame(resultados)
    for alvo in ["M1", "M2"]:
        piv = (R[R.alvo == alvo]
               .pivot_table(index="conjunto", columns="esquema", values="R2")
               .round(3))
        piv.loc["diferença"] = (piv.loc["COM_DEFASAGENS"]
                                - piv.loc["SEM_DEFASAGENS"]).round(3)
        print(f"\n{'=' * 64}\n{alvo} — coeficiente de determinação por esquema\n{'=' * 64}")
        print(piv.to_string())

    print(f"\n{'=' * 64}\nMédia sobre os cinco esquemas\n{'=' * 64}")
    print(R.groupby(["alvo", "conjunto"])[["R2", "MAE"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
