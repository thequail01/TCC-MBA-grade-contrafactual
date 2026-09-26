"""
=============================================================================
sensibilidade_janela.py — a escolha do recorte horário altera as conclusões?
=============================================================================
A janela adotada é 12h-20h em dias úteis, com potência acima de 50 kW e ao
menos duas máquinas em operação. Este script reexecuta o pipeline completo
(M1 -> M2, cinco esquemas) para janelas alternativas, usando exatamente as
mesmas partições e o mesmo procedimento de comparacao_validacao.py.

Horários no relógio dos servidores de automação.

SAÍDA: sensibilidade_janela.csv
=============================================================================
"""
import pandas as pd

from comparacao_validacao import avaliar_particao
from comum import ARQUIVO_BASE, carregar, particoes

JANELAS = [(12, 20), (12, 21), (12, 22), (11, 21), (11, 22), (10, 22)]


def rodar(op):
    res = []
    for esquema, fold, tr, te in particoes(op):
        res += avaliar_particao(op, tr, te, esquema, fold)
    return pd.DataFrame(res)


def main():
    resumo = []
    for h0, h1 in JANELAS:
        op = carregar(h_ini=h0, h_fim=h1)
        R = rodar(op)
        m1 = R[R.alvo == "M1_Q_TR"]
        m2 = R[R.alvo == "M2_kW"]
        media = lambda d, mod, col="R2": d[d.modelo == mod][col].mean()
        linha = dict(janela=f"{h0}h-{h1}h", n=len(op),
                     dias=op.index.normalize().nunique(),
                     pct_3ch=round(100 * (op.n_chillers == 3).mean(), 1),
                     kw_med=round(op.kw_total.mean()),
                     IC_cv=round(100 * op.IC.std() / op.IC.mean(), 1),
                     M1_XGB=round(media(m1, "XGB_mono"), 3),
                     M2_XGB=round(media(m2, "XGB_mono"), 3),
                     M2_Ridge=round(media(m2, "Ridge"), 3),
                     M2_Persist=round(media(m2, "Persistencia"), 3),
                     MAE_XGB=round(media(m2, "XGB_mono", "MAE"), 1),
                     LORO_XGB=round(media(m2[m2.esquema == "D_LORO"], "XGB_mono"), 3))
        resumo.append(linha)
        print(f"  {linha['janela']}: n={linha['n']}  M2_XGB={linha['M2_XGB']}")

    S = pd.DataFrame(resumo)
    S.to_csv("sensibilidade_janela.csv", index=False)
    print("\n" + "=" * 110)
    print("SENSIBILIDADE DA JANELA DE ANÁLISE (R² médio sobre as treze partições)")
    print("=" * 110)
    print(S.to_string(index=False))

    print("\n" + "=" * 110)
    print("PERFIL DAS HORAS ADICIONAIS (dias úteis, potência > 50 kW)")
    print("=" * 110)
    bruta = pd.read_csv(ARQUIVO_BASE, index_col=0, parse_dates=True)
    b = bruta[(bruta.day_of_week <= 4) & (bruta.kw_total > 50)].copy()
    b["IC"] = b["Q_TR_cal"] / bruta[bruta.janela_analise == 1]["Q_TR_cal"].mean()
    p = b[b.hour.between(10, 23)].groupby("hour").agg(
        n=("kw_total", "size"), kw=("kw_total", "mean"), IC=("IC", "mean"),
        nch=("n_chillers", "mean"), sp=("sp_chw", "mean")).round(2)
    p["ef"] = (p.IC / p.kw * 1000).round(3)
    print(p.to_string())


if __name__ == "__main__":
    main()
