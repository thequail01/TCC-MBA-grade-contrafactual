"""
=============================================================================
comum.py — definições compartilhadas pelo pipeline
=============================================================================

Este módulo concentra o que era duplicado entre os scripts de modelagem:
o carregamento da base, os conjuntos de variáveis, as restrições de
monotonicidade, a seleção do número de árvores e a construção das partições
de validação. Os demais scripts importam daqui em vez de reimplementar.

MOTIVO DA EXTRAÇÃO
A seleção do número de árvores é o ponto metodológico mais delicado do
trabalho, e reimplementá-la em cada script abriria espaço para divergências
silenciosas entre seções do artigo. Com uma única implementação, qualquer
alteração se propaga a todos os resultados de forma consistente.

CONVENÇÕES
- Horários no relógio dos servidores de automação; os registros climáticos
  do INMET foram convertidos a esse referencial em consolidar_base.py.
- A janela de análise compreende dias úteis, das 12h às 20h, com potência
  acima de 50 kW e ao menos duas máquinas em operação.
- O embargo entre treino e teste é de 24 h, necessário porque as variáveis
  defasadas criam dependência entre observações próximas à fronteira. É
  aplicado aos esquemas de partições contíguas (A, B, C e E) e dispensado no
  esquema por regime ausente (D), ver particoes().

USO
    from comum import carregar, FS_M1, FS_M2, MONO_M1, MONO_M2, ajustar_xgb
    op = carregar()
=============================================================================
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Constantes do domínio
# -----------------------------------------------------------------------------
ARQUIVO_BASE = "base_cag_v2.csv"

CP_AGUA = 4.187        # kJ/(kg·K)
RHO_AGUA = 1000.0      # kg/m³
CONV_TR = 3.517        # kW por TR
EMBARGO_H = 24         # horas de embargo entre treino e teste

HORA_INI, HORA_FIM = 12, 20    # janela de análise, relógio do servidor
KW_MINIMO = 50                 # limiar de operação ativa
N_CHILLERS_MIN = 2             # configuração mínima considerada

# -----------------------------------------------------------------------------
# Conjuntos de variáveis
#
# M1 estima o índice de carga térmica. Exclui deliberadamente as variáveis de
# decisão do operador (setpoint e número de máquinas), de modo a aprender a
# demanda da edificação e não a resposta das máquinas. Inclui defasagens
# intradiárias da própria carga, o que caracteriza o modelo como
# autorregressivo com covariáveis climáticas.
#
# M2 estima a potência elétrica a partir da carga estimada por M1 e das duas
# variáveis de decisão, que são avaliadas exaustivamente em grade.
# -----------------------------------------------------------------------------
FS_M1 = ["entalpia", "T_ext", "UR_ext", "hour", "day_of_week", "month",
         "Q_TR_L1", "Q_TR_L2", "Q_TR_R3"]

FS_M2 = ["Q_TR_pred", "n_chillers", "entalpia", "hour", "T_ext", "sp_chw"]

# -----------------------------------------------------------------------------
# Restrições de monotonicidade
#
# +1 obriga resposta não decrescente na variável, −1 não crescente, 0 deixa
# livre. A restrição é ESTRUTURAL: atua na seleção das divisões e nos valores
# das folhas durante o treinamento, garantindo monotonicidade por construção
# e não apenas favorecendo-a, o que é requisito para o uso contrafactual.
#
# A restrição sobre n_chillers deriva de regularidade observada nesta
# instalação, associada ao desequilíbrio hidráulico do CH-03, e não constitui
# relação termodinâmica universal. O achado correspondente é sustentado pela
# comparação pareada (tabela10_comparacao_pareada.py), evidência medida e
# independente do modelo. Como a restrição fixa o SINAL do efeito da terceira
# máquina, a grade contrafactual só aprende sua magnitude; a variante sem essa
# restrição é avaliada em tabela11_grade_robusta.py.
# -----------------------------------------------------------------------------
MONO_M1 = {"entalpia": 1, "T_ext": 1, "UR_ext": 0, "hour": 0, "day_of_week": 0,
           "month": 0, "Q_TR_L1": 1, "Q_TR_L2": 1, "Q_TR_R3": 1}

MONO_M2 = {"Q_TR_pred": 1, "n_chillers": 1, "entalpia": 1, "hour": 0,
           "T_ext": 1, "sp_chw": -1}

# -----------------------------------------------------------------------------
# Hiperparâmetros do XGBoost
#
# Fixados a priori a partir de critérios estruturais (árvores rasas com
# regularização), e não selecionados por desempenho. As varreduras em
# sensibilidade.py e grid_search.py constituem diagnóstico de robustez.
# n_estimators não aparece aqui: é determinado por validação interna ao
# treino (ver n_arvores).
# -----------------------------------------------------------------------------
XGB_BASE = dict(learning_rate=0.05, max_depth=3, subsample=0.9,
                colsample_bytree=0.9, min_child_weight=8, reg_lambda=5.0,
                tree_method="hist", random_state=42, verbosity=0)


# =============================================================================
# Dados
# =============================================================================
def carregar(arquivo: str = ARQUIVO_BASE, h_ini: int = HORA_INI,
             h_fim: int = HORA_FIM) -> pd.DataFrame:
    """Carrega a base, aplica a janela de análise e calcula as variáveis derivadas.

    As defasagens da carga são calculadas POR DIA (`groupby` na data), de modo
    que a primeira hora de cada dia não herda valores do dia anterior nem do
    período de planta fechada. Isso torna estruturalmente impossível o
    vazamento de informação entre dias pela via das variáveis defasadas.

    Retorna o DataFrame indexado por hora, restrito à janela, com as colunas
    Q_TR_L1, Q_TR_L2, Q_TR_R3, IC e regime acrescentadas.
    """
    b = pd.read_csv(arquivo, index_col=0, parse_dates=True)
    janela = ((b["day_of_week"] <= 4)
              & b["hour"].between(h_ini, h_fim)
              & (b["kw_total"] > KW_MINIMO)
              & (b["n_chillers"] >= N_CHILLERS_MIN))
    op = b[janela].copy().sort_index()

    # A base já traz a janela marcada em recalibrar_vazao.py; as duas
    # definições precisam coincidir para que todos os scripts vejam as
    # mesmas horas.
    if "janela_analise" in b.columns and (h_ini, h_fim) == (HORA_INI, HORA_FIM):
        marcada = b.index[b["janela_analise"] == 1]
        if not marcada.sort_values().equals(op.index):
            warnings.warn("janela recalculada difere da coluna janela_analise "
                          f"({len(op)} contra {len(marcada)} horas)")

    dia = op.index.normalize()
    g = op.groupby(dia)["Q_TR"]
    op["Q_TR_L1"] = g.shift(1)
    op["Q_TR_L2"] = g.shift(2)
    op["Q_TR_R3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Índice de carga: carga normalizada pela média da série. A constante da
    # calibração de vazão cancela nesta razão, de modo que o índice depende
    # apenas do expoente, e não do fator de escala.
    op["IC"] = op["Q_TR_cal"] / op["Q_TR_cal"].mean()

    op["regime"] = np.select(
        [op.index < "2026-04-01", op.index < "2026-05-01"],
        ["R1_marco_SP5.5", "R2_abril_SPvar"], default="R3_maio_SP7.5")
    return op


# =============================================================================
# Validação
# =============================================================================
def purgar(treino: pd.DatetimeIndex, teste: pd.DatetimeIndex,
           horas: int = EMBARGO_H) -> pd.DatetimeIndex:
    """Remove do treino as horas a menos de `horas` de distância do teste.

    Sem o embargo, uma observação de treino imediatamente anterior ao teste
    compartilharia com ele as variáveis defasadas, o que constituiria
    vazamento. Aplicado aos esquemas cujas partições são contíguas no tempo.
    """
    lo, hi = teste.min(), teste.max()
    return treino[(treino < lo - pd.Timedelta(hours=horas))
                  | (treino > hi + pd.Timedelta(hours=horas))]


def curva_interna(X_tr: pd.DataFrame, y_tr: pd.Series, mono: dict, cols: list,
                  parametros: dict | None = None, n_div: int = 3,
                  teto: int = 800, embargo_h: int = EMBARGO_H):
    """Curva de perda média (RMSE por rodada de reforço) em validação interna ao treino.

    PROCEDIMENTO. O treino é dividido em n_div+1 blocos contíguos. Cada um dos
    n_div ÚLTIMOS blocos serve uma vez como validação, com o modelo ajustado
    apenas sobre o que o precede. As fronteiras internas recebem o mesmo
    embargo aplicado no nível externo, garantindo consistência entre os dois
    níveis do aninhamento. O conjunto de teste não participa de nenhuma etapa.

    Retorna (curva_media, contagens), em que contagens é o número de árvores no
    mínimo de cada curva individual; dispersão elevada entre elas indica que o
    número ótimo de árvores depende do regime. Retorna (None, []) quando
    nenhuma divisão reúne observações suficientes após o embargo interno.
    """
    par = dict(XGB_BASE if parametros is None else parametros)
    mv = {c: mono.get(c, 0) for c in cols}
    n = len(X_tr)
    bordas = [int(n * (i + 1) / (n_div + 1)) for i in range(n_div)]

    curvas = []
    for k, corte in enumerate(bordas):
        fim = bordas[k + 1] if k + 1 < len(bordas) else n
        idx_val = X_tr.index[corte:fim]
        if len(idx_val) < 15:
            continue
        idx_tr = X_tr.index[:corte]
        idx_tr = idx_tr[idx_tr < idx_val.min() - pd.Timedelta(hours=embargo_h)]
        if len(idx_tr) < 30:
            continue
        sonda = xgb.XGBRegressor(n_estimators=teto, monotone_constraints=mv,
                                 eval_metric="rmse", **par)
        sonda.fit(X_tr.loc[idx_tr], y_tr.loc[idx_tr],
                  eval_set=[(X_tr.loc[idx_val], y_tr.loc[idx_val])], verbose=False)
        curvas.append(np.array(sonda.evals_result()["validation_0"]["rmse"]))

    if not curvas:
        return None, []
    L = min(len(c) for c in curvas)
    media = np.vstack([c[:L] for c in curvas]).mean(axis=0)
    return media, [int(np.argmin(c)) + 1 for c in curvas]


def n_arvores(X_tr: pd.DataFrame, y_tr: pd.Series, mono: dict, cols: list,
              parametros: dict | None = None, n_div: int = 3,
              teto: int = 800, embargo_h: int = EMBARGO_H,
              registro: list | None = None, rotulo: str = "") -> int:
    """Número de árvores no mínimo da curva de perda média (ver curva_interna).

    POR QUE NÃO PARADA ANTECIPADA EM RECORTE ÚNICO. A alternativa de usar os
    últimos 20% do treino como validação mostrou-se sensível ao recorte:
    quando o trecho coincide com transição de regime operacional, interrompe o
    treinamento prematuramente.

    POR QUE NÃO A REGRA DO ERRO-PADRÃO UNITÁRIO. A curva de perda é
    assimétrica neste conjunto, caindo abruptamente à esquerda do mínimo e
    apresentando platô à direita. O erro-padrão no mínimo excede a variação na
    região de platô, de modo que a regra recua a seleção para subajuste.

    Devolve 150 (valor de recurso) quando curva_interna não produz curva.
    Se `registro` for uma lista, acrescenta a ela um dicionário com as
    contagens individuais, para o relatório de dispersão.
    """
    media, contagens = curva_interna(X_tr, y_tr, mono, cols, parametros,
                                     n_div, teto, embargo_h)
    n = int(np.argmin(media)) + 1 if media is not None else 150
    if registro is not None and contagens:
        registro.append(dict(fold=rotulo, contagens=contagens, escolhido=n,
                             amplitude=max(contagens) - min(contagens),
                             cv_pct=100 * np.std(contagens) / np.mean(contagens)))
    return n


def ajustar_xgb(X_tr: pd.DataFrame, y_tr: pd.Series, mono: dict, cols: list,
                parametros: dict | None = None, teto: int = 800,
                registro: list | None = None, rotulo: str = "",
                n_fixo: int | None = None) -> xgb.XGBRegressor:
    """Ajusta XGBoost monotônico com o número de árvores dado por n_arvores.

    `n_fixo` dispensa a seleção e usa o número de árvores informado (usado em
    réplicas de bootstrap e em varreduras com produto taxa × árvores fixo).
    """
    par = dict(XGB_BASE if parametros is None else parametros)
    n = n_fixo if n_fixo else n_arvores(X_tr, y_tr, mono, cols, par, teto=teto,
                                        registro=registro, rotulo=rotulo)
    mv = {c: mono.get(c, 0) for c in cols}
    return xgb.XGBRegressor(n_estimators=n, monotone_constraints=mv,
                            **par).fit(X_tr, y_tr)


def preparar(op: pd.DataFrame, idx_tr: pd.DatetimeIndex,
             idx_te: pd.DatetimeIndex, cols: list, alvo: str):
    """Monta as matrizes de treino e teste com imputação ajustada SÓ no treino.

    A mediana usada na imputação é estimada exclusivamente sobre o conjunto de
    treino de cada partição; usar a mediana global constituiria vazamento.

    Retorna (X_treino, y_treino, X_teste, y_teste, imputador).
    """
    d_tr = op.loc[idx_tr, cols + [alvo]].dropna(subset=[alvo])
    d_te = op.loc[idx_te, cols + [alvo]].dropna(subset=[alvo])
    imp = SimpleImputer(strategy="median").fit(d_tr[cols])
    X_tr = pd.DataFrame(imp.transform(d_tr[cols]), columns=cols, index=d_tr.index)
    X_te = pd.DataFrame(imp.transform(d_te[cols]), columns=cols, index=d_te.index)
    return X_tr, d_tr[alvo], X_te, d_te[alvo], imp


def cascata(op: pd.DataFrame, idx_tr: pd.DatetimeIndex, idx_te: pd.DatetimeIndex):
    """Executa M1 e devolve a carga estimada para treino e teste da partição.

    A carga estimada é gerada DENTRO de cada partição, a partir de um M1
    ajustado exclusivamente com o conjunto de treino dela, e nunca
    reaproveitada entre partições. Reaproveitar produziria vazamento: as
    partições iniciais receberiam predições de um M1 que processou suas
    próprias horas de teste.

    Nas horas de treino, a carga estimada é a predição do M1 DENTRO da amostra.
    A alternativa de cross-fitting foi avaliada em teste_crossfit_cv.py.

    Retorna (modelo_M1, série de Q_TR_pred indexada por treino ∪ teste).
    """
    X_tr, y_tr, _, _, imp = preparar(op, idx_tr, idx_te, FS_M1, "Q_TR")
    m1 = ajustar_xgb(X_tr, y_tr, MONO_M1, FS_M1)
    todos = idx_tr.union(idx_te)
    X_todos = pd.DataFrame(imp.transform(op.loc[todos, FS_M1]),
                           columns=FS_M1, index=todos)
    return m1, pd.Series(m1.predict(X_todos), index=todos)


def particoes(op: pd.DataFrame) -> list:
    """Constrói as treze partições dos cinco esquemas de validação.

    A  expansivo, abril e maio    treino cresce; três folds de duas semanas
    B  expansivo, março a maio    idem, com março incluído no treino
    C  janela deslizante          treino de seis semanas, só o passado recente
    D  regime ausente (LORO)      treina em dois regimes, testa no terceiro
    E  holdout final              treina março e abril, testa maio inteiro

    O embargo é aplicado a A, B, C e E. No esquema D é dispensado: as
    fronteiras coincidem com mudanças de mês, e as defasagens são calculadas
    dentro do próprio dia, de modo que nenhuma observação de treino
    compartilha informação com o dia de teste adjacente.

    Retorna lista de (esquema, rótulo, índice_treino, índice_teste).
    """
    saida = []
    sem = op.index.to_period("W")
    semanas = sorted(sem.unique())

    for esquema, inicio in [("A_expansivo_AM", "2026-03-30"),
                            ("B_expansivo_MAM", "2026-03-01")]:
        sub = op.index[op.index >= inicio]
        sw = sorted(sub.to_period("W").unique())
        n_te = 2
        for i, corte in enumerate(range(len(sw) - 3 * n_te,
                                        len(sw) - n_te + 1, n_te), start=1):
            tr = sub[sub.to_period("W").isin(sw[:corte])]
            te = sub[sub.to_period("W").isin(sw[corte:corte + n_te])]
            saida.append((esquema, f"F{i}", purgar(tr, te), te))

    JANELA = 6  # semanas de treino
    for i, corte in enumerate(range(len(semanas) - 6, len(semanas) - 1, 2), start=1):
        tr = op.index[sem.isin(semanas[max(0, corte - JANELA):corte])]
        te = op.index[sem.isin(semanas[corte:corte + 2])]
        saida.append(("C_deslizante", f"F{i}", purgar(tr, te), te))

    for r in sorted(op["regime"].unique()):
        te = op.index[op["regime"] == r]
        tr = op.index[op["regime"] != r]
        saida.append(("D_LORO", f"testa_{r}", tr, te))

    tr = op.index[op.index < "2026-05-01"]
    te = op.index[op.index >= "2026-05-01"]
    saida.append(("E_holdout_final", "maio", purgar(tr, te), te))
    return saida


# =============================================================================
# Linhas de base e modelos de referência
# =============================================================================
def persistencia(serie_janela: pd.Series, idx_te: pd.DatetimeIndex,
                 recurso: float, serie_completa: pd.Series | None = None) -> np.ndarray:
    """Predição por persistência: o valor da hora corrente é o da hora anterior.

    Por padrão (serie_completa=None), a hora anterior é procurada apenas dentro
    da janela de análise. Na primeira hora de cada dia (12h) ela não existe, e
    a predição recebe `recurso` (a média do treino). É o procedimento que gerou
    os resultados do artigo.

    Com `serie_completa` (a série de 24 h da base), a hora anterior é buscada
    fora da janela quando necessário, o que corresponde à informação de fato
    disponível ao operador às 12h. Variante avaliada em
    verificacoes_complementares.py.
    """
    fonte = serie_janela if serie_completa is None else serie_completa
    prev = fonte.reindex(idx_te.shift(-1, freq="h")).values
    return np.nan_to_num(prev, nan=recurso)


def ridge(alphas=None, cv=None):
    """Regressão Ridge padronizada, com o parâmetro escolhido por RidgeCV.

    Com cv=None (padrão do scikit-learn e procedimento do artigo), a escolha é
    feita por validação cruzada generalizada, isto é, leave-one-out eficiente
    sobre o conjunto de treino, que não respeita a ordem temporal. Passar
    cv=TimeSeriesSplit(...) torna a escolha temporal; a comparação entre as
    duas está em verificacoes_complementares.py.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    alphas = np.logspace(-2, 3, 30) if alphas is None else alphas
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=cv))
