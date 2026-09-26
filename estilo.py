"""
=============================================================================
estilo.py — convenções gráficas comuns às figuras do artigo
=============================================================================

SEPARADOR DECIMAL. Os eixos numéricos passam a usar vírgula, como exige o
texto em português. salvar() troca o formatador de todo eixo numérico da
figura antes de gravá-la; eixos de datas e de categorias não são alterados.
Para rótulos escritos no gráfico (anotações, legendas), use br().

NOTAS DE RODAPÉ. As figuras traziam uma nota explicativa em itálico abaixo
dos painéis, que era recortada ao inserir a imagem no Word, pois a legenda e
a fonte ficam no próprio documento. Com NOTAS_NA_FIGURA = False (padrão),
nota() não desenha nada; mude para True para recuperá-las.
=============================================================================
"""
from matplotlib.ticker import ScalarFormatter

NOTAS_NA_FIGURA = False


def br(valor: float, casas: int = 2, sinal: bool = False) -> str:
    """Formata número com vírgula decimal: br(0.614) -> '0,61'; br(4.5, 1, True) -> '+4,5'."""
    txt = f"{valor:{'+' if sinal else ''}.{casas}f}"
    return txt.replace("-", "−").replace(".", ",")


class _Virgula(ScalarFormatter):
    """ScalarFormatter que devolve os rótulos com vírgula decimal."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def get_offset(self):
        return super().get_offset().replace(".", ",")


def virgula(fig) -> None:
    """Aplica vírgula decimal a todos os eixos numéricos da figura."""
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if type(eixo.get_major_formatter()) is ScalarFormatter:
                eixo.set_major_formatter(_Virgula())


def nota(fig, x: float, y: float, texto: str, **kw) -> None:
    """Nota explicativa no rodapé da figura, desenhada só se NOTAS_NA_FIGURA."""
    if NOTAS_NA_FIGURA:
        fig.text(x, y, texto, **kw)


def salvar(fig, caminho: str, **kw) -> None:
    """Aplica a vírgula decimal e grava a figura (300 dpi, fundo branco)."""
    virgula(fig)
    kw.setdefault("bbox_inches", "tight")
    kw.setdefault("facecolor", "white")
    fig.savefig(caminho, **kw)
