"""
=============================================================================
executar_tudo.py — roda o pipeline de resultados na ordem correta
=============================================================================
Pressupõe base_cag_v2.csv no diretório corrente. As etapas de preparação da
base (consolidar_base.py e recalibrar_vazao.py) dependem dos exports brutos
do sistema de automação, que não são distribuídos, e ficam de fora.

USO
    python executar_tudo.py            # resultados do artigo (alguns minutos)
    python executar_tudo.py --completo # inclui ensaios de robustez
=============================================================================
"""
import subprocess
import sys
import time

ARTIGO = [
    "comparacao_validacao.py",        # Tabelas 5 a 8 (gera o CSV lido depois)
    "tabela10_comparacao_pareada.py", # Tabela 10
    "tabela11_grade.py",              # Tabela 11 (gera o CSV lido pelas figuras)
    "sensibilidade.py",               # Tabela 9 e efeito das restrições do M2
    "acerto_nch.py",                  # quinta conclusão
    "analise_residuos.py",            # Figura 8 e diagnóstico dos resíduos
    "gerar_eda.py",                   # Figuras 1, 3, 4, 5, 6 e 7
    "gerar_figuras.py",               # Figura 2 e figuras complementares
]
ROBUSTEZ = [
    "tabela11_grade_robusta.py",      # grade sem restrição e intervalos
    "teste_crossfit_cv.py",
    "teste_lags.py",
    "grid_search.py",
    "sensibilidade_janela.py",
    "verificacoes_complementares.py",
    "ajuste_vazao.py",
]

etapas = ARTIGO + (ROBUSTEZ if "--completo" in sys.argv else [])
for script in etapas:
    t0 = time.time()
    with open(f"log_{script[:-3]}.txt", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, script], stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if r.returncode == 0 else f"FALHOU (código {r.returncode})"
    print(f"{script:34s} {status:22s} {time.time() - t0:6.0f} s")
    if r.returncode != 0:
        sys.exit(f"Interrompido. Veja log_{script[:-3]}.txt")
print("\nConcluído. Saídas no diretório corrente e figuras em figuras/.")
