"""
=============================================================================
acerto_nch.py — por que a formulação condicional é necessária
=============================================================================

PERGUNTA. O instrumento estima o custo de cada configuração, tomando o número
de máquinas como dado. A alternativa seria um modelo preditivo, que
antecipasse essa variável a partir do clima e da hora. Qual acurácia esse
modelo precisaria alcançar para ser preferível?

PROCEDIMENTO. Ajusta-se a cascata no esquema E (treina março e abril, testa
maio) e, no conjunto de teste, perturba-se a variável de número de máquinas
com taxa de erro controlada: com probabilidade (1 − acurácia), a configuração
é trocada de duas para três ou vice-versa. Repete-se 200 vezes por nível de
acurácia, para estimar média e dispersão do efeito.

INTERPRETAÇÃO. O ponto de referência é a linha de base de persistência no
mesmo esquema. A acurácia mínima exigida é aquela em que o desempenho do
modelo perturbado iguala o da persistência: abaixo dela, um modelo preditivo
seria pior que a linha de base ingênua, e a formulação condicional deixa de
ser uma escolha para tornar-se a única alternativa defensável.

SAÍDA. Tabela de R² e MAE por nível de acurácia. Alimenta a quinta conclusão
do artigo (acurácia mínima da ordem de 97%). Rode depois de
comparacao_validacao.py, de onde vem a referência de persistência.

USO
    python acerto_nch.py
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from comum import (EMBARGO_H, FS_M2, MONO_M2, ajustar_xgb, carregar, cascata,
                   preparar, purgar)

NIVEIS_ACURACIA = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
N_REPETICOES = 200
SEMENTE = 42


def r2_persistencia_e(arquivo="comparacao_validacao.csv", recurso=0.599):
    """R² da persistência no esquema E, lido da saída de comparacao_validacao.py.

    Serve de referência para o limiar de acurácia mínima. Se o arquivo ainda
    não existir, usa o valor publicado (0,599) e avisa.
    """
    try:
        R = pd.read_csv(arquivo)
        sel = R[(R.esquema == "E_holdout_final") & (R.alvo == "M2_kW")
                & (R.modelo == "Persistencia")]
        return float(sel["R2"].mean())
    except (FileNotFoundError, KeyError):
        print(f"aviso: {arquivo} ausente; usando R² de persistência = {recurso}")
        return recurso


R2_PERSISTENCIA_E = r2_persistencia_e()


def main():
    rng = np.random.default_rng(SEMENTE)
    op = carregar()

    # --- esquema E: treina março e abril, testa maio ---
    idx_te = op.index[op.index >= "2026-05-01"]
    idx_tr = purgar(op.index[op.index < "2026-05-01"], idx_te, EMBARGO_H)

    _, q_tr_pred = cascata(op, idx_tr, idx_te)
    op = op.copy()
    op["Q_TR_pred"] = q_tr_pred

    X_tr, y_tr, X_te, y_te, _ = preparar(op, idx_tr, idx_te, FS_M2, "kw_total")
    modelo = ajustar_xgb(X_tr, y_tr, MONO_M2, FS_M2)

    r2_ref = r2_score(y_te, modelo.predict(X_te))
    mae_ref = mean_absolute_error(y_te, modelo.predict(X_te))
    print(f"Esquema E: treino {len(X_tr)} h, teste {len(X_te)} h")
    print(f"Configuração conhecida sem erro: R² = {r2_ref:.3f}, "
          f"MAE = {mae_ref:.1f} kW")
    print(f"Linha de base de persistência:   R² = {R2_PERSISTENCIA_E:.3f}\n")

    linhas = []
    for acuracia in NIVEIS_ACURACIA:
        r2s, maes = [], []
        for _ in range(N_REPETICOES):
            X_perturbado = X_te.copy()
            trocar = rng.random(len(X_perturbado)) > acuracia
            # 2 e 3 são as únicas configurações da janela; 5 − n alterna entre elas
            X_perturbado.loc[trocar, "n_chillers"] = (
                5 - X_perturbado.loc[trocar, "n_chillers"])
            pred = modelo.predict(X_perturbado)
            r2s.append(r2_score(y_te, pred))
            maes.append(mean_absolute_error(y_te, pred))
        linhas.append(dict(acuracia=acuracia, R2=np.mean(r2s),
                           R2_dp=np.std(r2s), MAE=np.mean(maes)))

    T = pd.DataFrame(linhas)
    print(f"{'acurácia':>9} {'R² médio':>10} {'desvio':>8} {'MAE médio':>11}")
    for _, r in T.iterrows():
        print(f"{r.acuracia * 100:>8.0f}% {r.R2:>10.3f} {r.R2_dp:>8.3f} "
              f"{r.MAE:>10.1f}")

    # --- limiar de acurácia mínima, por interpolação linear entre níveis ---
    acima = T[T.R2 >= R2_PERSISTENCIA_E]
    abaixo = T[T.R2 < R2_PERSISTENCIA_E]
    if len(acima) and len(abaixo):
        a, b = acima.iloc[-1], abaixo.iloc[0]
        limiar = b.acuracia + (a.acuracia - b.acuracia) * \
            (R2_PERSISTENCIA_E - b.R2) / (a.R2 - b.R2)
        print(f"\nAcurácia mínima para superar a persistência: "
              f"{limiar * 100:.0f}%")

    inclinacao = (T.R2.iloc[0] - T.R2.iloc[-1]) / \
        ((T.acuracia.iloc[0] - T.acuracia.iloc[-1]) * 100)
    print(f"Degradação média: {inclinacao:.4f} de R² por ponto percentual "
          f"de acurácia perdido")


if __name__ == "__main__":
    main()
