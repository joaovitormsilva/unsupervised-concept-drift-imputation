"""
CD-diagram (Demšar, 2006) com teste de Friedman + Wilcoxon pareado + correção
de Holm, comparando os métodos de imputação pareados por paciente.

Adaptado de https://github.com/hfawaz/cd-diagram (Fawaz et al., GPL-3.0):
graph_ranks e form_cliques são do repositório original (só sem os prints e
sem forçar Arial). wilcoxon_holm foi reescrito para (a) parear os métodos
explicitamente por paciente (o original depende da ordem das linhas),
(b) aceitar métricas de erro (menor = melhor) e (c) não encerrar o programa
quando o Friedman não rejeita H0 (o diagrama é desenhado com todos ligados).

Uso:
    python cd_diagram.py                      # resultados com defaults do river
    python cd_diagram.py --suffix _tuned_mod  # qualquer batch_results{suffix}_S*.csv
Saída: Plots/<rodada>/cd_diagrams/{métrica}_{cenário}.png (+ pdf/ ao lado)
       Plots/<rodada>/cd_diagrams/wilcoxon_holm.csv (todas as comparações)
"""
import argparse
import operator
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

import math  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402
import networkx  # noqa: E402

from boxplots import BATCH_RESULTS_DIR, METHOD_LABELS, METRICS, SCENARIOS, out_dir, save_figure  # noqa: E402

ALPHA = 0.05


# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)



def wilcoxon_holm(scores: pd.DataFrame, alpha: float = ALPHA, lower_is_better: bool = True):
    """scores: linhas = pacientes, colunas = métodos (um valor da métrica por célula).
    Devolve (p_values, average_ranks, friedman_p). p_values = [(m1, m2, p, significativo)]."""
    scores = scores.dropna()
    methods = list(scores.columns)
    friedman_p = friedmanchisquare(*(scores[m].values for m in methods))[1]

    p_values = []
    for i in range(len(methods) - 1):
        for j in range(i + 1, len(methods)):
            p = wilcoxon(scores[methods[i]], scores[methods[j]], zero_method="pratt")[1]
            p_values.append((methods[i], methods[j], p, False))

    # Holm só se o Friedman rejeitar H0 (sem diferença global, nenhum par é significativo)
    p_values.sort(key=operator.itemgetter(2))
    if friedman_p < alpha:
        k = len(p_values)
        for i in range(k):
            if p_values[i][2] <= alpha / (k - i):
                p_values[i] = (*p_values[i][:3], True)
            else:
                break

    # rank 1 = melhor; o diagrama é desenhado com reverse=True (melhor à direita)
    ranks = scores.rank(axis=1, ascending=lower_is_better)
    average_ranks = ranks.mean().sort_values(ascending=False)
    return p_values, average_ranks, friedman_p


def draw_cd_diagram(scores: pd.DataFrame, title: str, folder: str, name: str, alpha: float = ALPHA):
    p_values, average_ranks, friedman_p = wilcoxon_holm(scores, alpha)
    graph_ranks(average_ranks.values, np.array(average_ranks.index), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=True)
    note = "" if friedman_p < alpha else " (Friedman n.s.)"
    plt.title(f"{title}{note}", fontdict={"size": 18}, y=0.9, x=0.5)
    save_figure(folder, name)
    plt.close("all")
    return p_values, average_ranks, friedman_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="ex.: _tuned, _mod, _tuned_mod")
    parser.add_argument("--metrics", nargs="+", default=["mae", "rmse"])
    args = parser.parse_args()

    folder = out_dir(args.suffix, "cd_diagrams")
    rows = []
    for s in SCENARIOS:
        df = pd.read_csv(os.path.join(BATCH_RESULTS_DIR, f"batch_results{args.suffix}_{s}.csv"))
        df["method"] = df["detector"].map(METHOD_LABELS)
        for metric in args.metrics:
            scores = df.pivot(index="patient", columns="method", values=metric)
            p_values, avg, fp = draw_cd_diagram(scores, f"{METRICS[metric].split(' ')[0]} — {s}", folder, f"{metric}_{s}")
            print(f"{s} {metric}: Friedman p={fp:.2e} | ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in avg[::-1].items()))
            for m1, m2, p, sig in p_values:
                rows.append({"scenario": s, "metric": metric, "friedman_p": fp, "method_1": m1,
                             "method_2": m2, "rank_1": avg[m1], "rank_2": avg[m2],
                             "p_wilcoxon": p, "significativo_holm": sig})
    out_csv = os.path.join(folder, "wilcoxon_holm.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Comparações salvas em {out_csv}")


if __name__ == "__main__":
    main()
