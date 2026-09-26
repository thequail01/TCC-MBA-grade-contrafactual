"""
=============================================================================
verificacoes_complementares.py — quatro checagens de robustez metodológica
=============================================================================
Nenhuma destas variantes altera os resultados do artigo; cada uma responde a
uma pergunta que a banca pode levantar, sobre as mesmas treze partições de
comparacao_validacao.py.

1. RESTRIÇÕES DO M1. Efeito das restrições de monotonicidade do primeiro
   modelo sobre a cascata, mantidas as do segundo: R² do M1 em treino e em
   teste e R² do M2 com cada versão da carga estimada.

2. MÊS COMO VARIÁVEL DO M1. Com três meses de dados, a variável month
   coincide com o regime de setpoint e, no esquema por regime ausente e no
   holdout final, assume no teste um valor nunca visto no treino. Reajusta-se
   a cascata sem ela.

3. RIDGE COM VALIDAÇÃO TEMPORAL. O RidgeCV do artigo escolhe a regularização
   por validação cruzada generalizada (leave-one-out eficiente), que não
   respeita a ordem temporal. Compara-se com TimeSeriesSplit de três divisões.

4. PERSISTÊNCIA NA PRIMEIRA HORA. No artigo, a hora anterior é buscada só
   dentro da janela, e às 12h a persistência recebe a média do treino; as
   defasagens do M1 também não existem nessa hora. Duas leituras:
     (a) persistência com a observação das 11h da base completa, informação
         disponível ao operador, mas fora do conjunto usado pelo modelo;
     (b) modelo e persistência avaliados só das 13h às 20h, horas em que
         ambos dispõem da hora anterior dentro da janela (mesma informação).

SAÍDAS: verificacoes_complementares.csv e resumo impresso por esquema
USO   : python verificacoes_complementares.py
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from comum import (ARQUIVO_BASE, FS_M1, FS_M2, MONO_M1, MONO_M2, ajustar_xgb,
                   carregar, particoes, persistencia, preparar, ridge)

VARIANTES_M1 = {
    "M1_padrao": (FS_M1, MONO_M1),
    "M1_sem_restricoes": (FS_M1, {}),
    "M1_sem_mes": ([c for c in FS_M1 if c != "month"], MONO_M1),
}


def metricas(chk, esquema, fold, variante, alvo, y, p, extra=None):
    d = dict(verificacao=chk, esquema=esquema, fold=fold, variante=variante,
             alvo=alvo, R2=r2_score(y, p), MAE=mean_absolute_error(y, p))
    d.update(extra or {})
    return d


def main():
    op = carregar()
    base = pd.read_csv(ARQUIVO_BASE, index_col=0, parse_dates=True)
    linhas = []
    for esquema, fold, tr, te in particoes(op):
        if len(tr) < 30 or len(te) < 10:
            continue
        # ---- 1 e 2: variantes do M1 propagadas ao M2 ----
        m2_padrao = None
        for nome, (cols, mono) in VARIANTES_M1.items():
            X1tr, y1tr, X1te, y1te, imp = preparar(op, tr, te, cols, "Q_TR")
            m1 = ajustar_xgb(X1tr, y1tr, mono, cols)
            chk = "mes" if nome == "M1_sem_mes" else "restricoes_M1"
            linhas.append(metricas(chk, esquema, fold, nome, "M1", y1te, m1.predict(X1te),
                                   {"R2_treino": r2_score(y1tr, m1.predict(X1tr))}))
            todos = tr.union(te)
            o2 = op.loc[todos].copy()
            o2["Q_TR_pred"] = m1.predict(pd.DataFrame(imp.transform(op.loc[todos, cols]),
                                                      columns=cols, index=todos))
            X2tr, y2tr, X2te, y2te, _ = preparar(o2, tr, te, FS_M2, "kw_total")
            m2 = ajustar_xgb(X2tr, y2tr, MONO_M2, FS_M2)
            linhas.append(metricas(chk, esquema, fold, nome, "M2", y2te, m2.predict(X2te),
                                   {"R2_treino": r2_score(y2tr, m2.predict(X2tr))}))
            if nome == "M1_padrao":
                m2_padrao = (X2tr, y2tr, X2te, y2te)
                pred_padrao = {"M1": pd.Series(m1.predict(X1te), index=y1te.index),
                               "M2": pd.Series(m2.predict(X2te), index=y2te.index)}

        # ---- 3: Ridge com escolha temporal do parâmetro ----
        X2tr, y2tr, X2te, y2te = m2_padrao
        for nome, cv in [("Ridge_LOO_artigo", None),
                         ("Ridge_TimeSeriesSplit", TimeSeriesSplit(n_splits=3))]:
            m = ridge(cv=cv).fit(X2tr, y2tr)
            linhas.append(metricas("ridge", esquema, fold, nome, "M2", y2te, m.predict(X2te),
                                   {"alpha": m[-1].alpha_}))

        # ---- 4: persistência com a hora anterior real ----
        for alvo, col, y_tr, y_te in [("M1", "Q_TR", op.loc[tr, "Q_TR"].dropna(),
                                       op.loc[te, "Q_TR"].dropna()),
                                      ("M2", "kw_total", y2tr, y2te)]:
            for nome, completa in [("persist_janela_artigo", None),
                                   ("persist_hora_real", base[col])]:
                p = persistencia(op[col], y_te.index, y_tr.mean(), completa)
                linhas.append(metricas("persistencia", esquema, fold, nome, alvo, y_te, p))
            # mesma informação: só as horas com hora anterior dentro da janela
            h = y_te.index.hour >= 13
            p = persistencia(op[col], y_te.index, y_tr.mean())
            linhas.append(metricas("persistencia", esquema, fold, "persist_13a20h", alvo,
                                   y_te[h], p[h]))
            linhas.append(metricas("persistencia", esquema, fold, "XGB_13a20h", alvo,
                                   y_te[h], pred_padrao[alvo].reindex(y_te.index)[h]))
        print(f"  {esquema} {fold} ok", flush=True)

    R = pd.DataFrame(linhas)
    R.to_csv("verificacoes_complementares.csv", index=False)

    def tabela(chk, alvo, valor="R2"):
        s = R[(R.verificacao == chk) & (R.alvo == alvo)]
        return s.pivot_table(index="variante", columns="esquema", values=valor).round(3)

    print("\n1. RESTRIÇÕES DO M1 (R² médio por esquema)")
    print("   M1 em teste:\n" + tabela("restricoes_M1", "M1").to_string())
    print("   M1 em treino:\n" + tabela("restricoes_M1", "M1", "R2_treino").to_string())
    print("   M2 (cascata) em teste:\n" + tabela("restricoes_M1", "M2").to_string())
    print("\n2. M1 SEM A VARIÁVEL month (compare com M1_padrao acima)")
    print("   M1 em teste:\n" + tabela("mes", "M1").to_string())
    print("   M2 (cascata) em teste:\n" + tabela("mes", "M2").to_string())
    print("\n3. RIDGE: validação generalizada (artigo) contra temporal")
    print("   R²:\n" + tabela("ridge", "M2").to_string())
    print("   MAE (kW):\n" + tabela("ridge", "M2", "MAE").round(1).to_string())
    print("\n4. PERSISTÊNCIA NA PRIMEIRA HORA (compare também com XGB_13a20h)")
    for alvo in ["M1", "M2"]:
        print(f"   {alvo} R²:\n" + tabela("persistencia", alvo).to_string())
        print(f"   {alvo} MAE:\n" + tabela("persistencia", alvo, "MAE").round(1).to_string())


if __name__ == "__main__":
    main()
