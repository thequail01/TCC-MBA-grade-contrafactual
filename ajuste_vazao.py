"""
=============================================================================
AJUSTE DA CALIBRACAO DE VAZAO — V = K · ΔP^N
=============================================================================
Este script documenta e reproduz o ajuste que produziu os coeficientes
K = 27,5 e N = 0,549 usados em recalibrar_vazao.py.

FUNDAMENTO FISICO
Em circuito hidraulico de resistencia fixa, a perda de carga varia com o
quadrado da vazao:
        ΔP = R · Q²        (R = coeficiente de resistencia do circuito)
Isolando a vazao:
        Q = (ΔP / R)^0,5 = K · ΔP^0,5
A forma geral e Q = K · ΔP^N, com N teoricamente igual a 0,5. Estimar N em vez
de impo-lo permite VERIFICAR a premissa em vez de assumi-la (WANG, 2014).

LINEARIZACAO
Aplicando logaritmo a ambos os lados:
        ln Q = ln K + N · ln ΔP
que e uma reta em escala log-log. Uma regressao de minimos quadrados de ln Q
sobre ln ΔP fornece:
        inclinacao  = N
        intercepto  = ln K   ->  K = exp(intercepto)

DADOS DO AJUSTE ORIGINAL
Export complementar do sistema de automacao, com vazao medida e pressao
diferencial simultaneas entre 21/05 e 04/06/2026. Filtro: vazao > 50 m³/h e
ΔP > 10 kPa. Restaram 1.162 observacoes em operacao.

USO
  python ajuste_vazao.py                      -> demonstracao com a base atual
  python ajuste_vazao.py <arquivo_bruto.csv>  -> reajuste sobre o export bruto
=============================================================================
"""
import sys
import numpy as np
import pandas as pd

K_PUB, N_PUB = 27.5, 0.549          # coeficientes publicados no trabalho
CP, RHO, CONV = 4.187, 1000.0, 3.517


def ajustar(dp, q):
    """Regressao log-log de Q sobre ΔP. Retorna K, N e diagnosticos."""
    m = (dp > 10) & (q > 50) & np.isfinite(dp) & np.isfinite(q)
    x, y = np.log(dp[m]), np.log(q[m])
    N, lnK = np.polyfit(x, y, 1)
    r_loglog = np.corrcoef(x, y)[0, 1]
    q_aj = np.exp(lnK) * dp[m] ** N
    r2 = 1 - ((q[m] - q_aj) ** 2).sum() / ((q[m] - q[m].mean()) ** 2).sum()
    r_sqrt = np.corrcoef(np.sqrt(dp[m]), q[m])[0, 1]
    return dict(K=float(np.exp(lnK)), N=float(N), n=int(m.sum()),
                r_loglog=float(r_loglog), R2=float(r2), r_vs_sqrtdp=float(r_sqrt))


if len(sys.argv) > 1:
    # --- reajuste sobre o export bruto -------------------------------------
    bruto = pd.read_csv(sys.argv[1])
    col_dp = [c for c in bruto.columns if "dp" in c.lower() or "press" in c.lower()][0]
    col_q = [c for c in bruto.columns if "vaz" in c.lower() or "flow" in c.lower()][0]
    print(f"colunas usadas: ΔP='{col_dp}'  vazao='{col_q}'")
    r = ajustar(bruto[col_dp].to_numpy(float), bruto[col_q].to_numpy(float))
    print(f"\nAJUSTE: Q = {r['K']:.2f} · ΔP^{r['N']:.3f}")
    print(f"  n = {r['n']} | r log-log = {r['r_loglog']:.3f} | "
          f"R² na escala original = {r['R2']:.3f} | r contra √ΔP = {r['r_vs_sqrtdp']:.3f}")
    print(f"\n  expoente ajustado {r['N']:.3f} contra teorico 0,500: "
          f"desvio de {100*(r['N']/0.5-1):+.1f}%")
    sys.exit()

# --- demonstracao: a formula reproduz a vazao estimada da base --------------
b = pd.read_csv("base_cag_v2.csv", index_col=0, parse_dates=True)
op = b[b["janela_analise"] == 1].copy()

print("=" * 78)
print("1. A FORMULA, PASSO A PASSO")
print("=" * 78)
print(f"  Q = {K_PUB} · ΔP^{N_PUB}\n")
for dp in [60, 70, 85, 100, 130]:
    q = K_PUB * dp ** N_PUB
    print(f"  ΔP = {dp:>3} kPa  ->  ΔP^{N_PUB} = {dp**N_PUB:6.3f}  ->  "
          f"Q = {K_PUB} × {dp**N_PUB:6.3f} = {q:6.1f} m³/h")

print("\n" + "=" * 78)
print("2. VERIFICACAO: a formula reproduz V_est da base?")
print("=" * 78)
rec = K_PUB * op["dp_bags1"].clip(lower=1) ** N_PUB
print(f"  erro absoluto maximo: {(rec - op['V_est']).abs().max():.6f} m³/h  -> "
      f"{'reproduz exatamente' if (rec-op['V_est']).abs().max() < 1e-6 else 'DIVERGE'}")
print(f"\n  media por mes:")
for m, s in op.groupby(op.index.to_period("M")):
    print(f"    {m}: ΔP = {s['dp_bags1'].mean():5.1f} kPa  ->  "
          f"V_est = {s['V_est'].mean():5.1f} m³/h")

print("\n" + "=" * 78)
print("3. O EXPOENTE CONTRA A TEORIA")
print("=" * 78)
print(f"  ajustado : {N_PUB}")
print(f"  teorico  : 0,500  (perda de carga proporcional ao quadrado da vazao)")
print(f"  desvio   : {100*(N_PUB/0.5-1):+.1f}%")
print("  Como o expoente foi ESTIMADO e nao imposto, sua proximidade de 0,5")
print("  constitui verificacao independente da premissa de curva de sistema.")

print("\n" + "=" * 78)
print("4. O QUE CADA COEFICIENTE AFETA")
print("=" * 78)
print("  K escala a vazao proporcionalmente. Como o indice de carga e normalizado")
print("  pela media da serie, K CANCELA e nao afeta nenhuma analise normalizada.")
print("  Ele afeta apenas os valores absolutos em TR e o COP.")
print("  N altera a FORMA da relacao e, portanto, o indice de carga.\n")
ic_pub = (op["dp_bags1"] ** N_PUB * op["dt_chw"])
ic_pub = ic_pub / ic_pub.mean()
for k in [10.0, 27.5, 50.0]:
    v = k * op["dp_bags1"] ** N_PUB
    q = (v * RHO / 3600 * CP * op["dt_chw"]) / CONV
    ic = q / q.mean()
    print(f"  K = {k:>5}: Q_TR medio = {q.mean():6.1f} TR | "
          f"IC identico ao publicado: {np.allclose(ic, ic_pub)}")

print("\n" + "=" * 78)
print("5. SENSIBILIDADE AO EXPOENTE (a analise depende de N?)")
print("=" * 78)
alvo = op.dropna(subset=["kw_total"]).copy()
print(f"  {'N':>6} {'Q_TR medio':>11} {'corr com o publicado':>22} "
      f"{'penalidade 3 maq (kW)':>23}")
for n in [0.35, 0.45, 0.50, 0.549, 0.65, 0.80, 1.00]:
    v = K_PUB * alvo["dp_bags1"].clip(lower=1) ** n
    q = (v * RHO / 3600 * CP * alvo["dt_chw"]) / CONV
    ic = q / q.mean()
    d = alvo.assign(IC=ic)
    d["bq"] = pd.qcut(d["IC"], 5, labels=False)
    d["bh"] = pd.qcut(d["entalpia"], 3, labels=False)
    g = (d.groupby(["bq", "bh", "n_chillers"])
           .agg(kw=("kw_total", "mean"), nn=("kw_total", "size")).reset_index())
    p = g.pivot_table(index=["bq", "bh"], columns="n_chillers",
                      values=["kw", "nn"]).dropna()
    p.columns = [f"{a}{int(c)}" for a, c in p.columns]
    E = p[(p["nn2"] >= 5) & (p["nn3"] >= 5)]
    dif = (E["kw3"] - E["kw2"]).mean()
    print(f"  {n:>6} {q.mean():>11.1f} {np.corrcoef(ic, ic_pub[alvo.index])[0,1]:>22.4f} "
          f"{dif:>18.1f} ({len(E)} estratos)")
