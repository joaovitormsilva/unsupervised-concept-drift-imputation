"""
Boxplots da distribuição das métricas por paciente (um ponto = um paciente),
por cenário e método de imputação. Estilo replicado da Fig. 3a do artigo do
Afonso (SBBD 2026) / notebook Imputation/Analysis/analysis_plots.ipynb.

Uso:
    python boxplots.py            # detectores com defaults do river
    python boxplots.py --tuned    # detectores com parâmetros do sweep
    python boxplots.py --suffix _tuned_mod   # qualquer batch_results{suffix}_S*.csv
Saída: Plots/<rodada>/boxplots/{mae,rmse,error_ratio}.png (+ pdf/ ao lado)
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(PLOTS_DIR)
BATCH_RESULTS_DIR = os.path.join(ANALYSIS_DIR, "batch_results")

# Uma pasta por rodada do batch (sufixo de batch_results{sufixo}_S*.csv),
# numeradas na ordem em que foram feitas.
RUN_DIRS = {
    "": "1_hora_minuto_defaults",
    "_tuned": "2_hora_minuto_ajustados",
    "_mod": "3_minuto_do_dia_defaults",
    "_tuned_mod": "4_minuto_do_dia_ajustados",
    "_tuned_mod_30p": "5_minuto_do_dia_ajustados_30p",
}


def out_dir(suffix: str, kind: str) -> str:
    """Pasta de saída de um tipo de gráfico (boxplots, cd_diagrams) para a rodada; cria pdf/ dentro."""
    path = os.path.join(PLOTS_DIR, RUN_DIRS[suffix], kind)
    os.makedirs(os.path.join(path, "pdf"), exist_ok=True)
    return path


def save_figure(path_png_dir: str, name: str):
    """Salva name.png na pasta e name.pdf na subpasta pdf/."""
    plt.savefig(os.path.join(path_png_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(path_png_dir, "pdf", f"{name}.pdf"), bbox_inches="tight")

SCENARIOS = ["S1", "S2", "S3"]
METHOD_LABELS = {
    "Media": "Média",
    "Sem detector (HT)": "HT",
    "PH": "HT+PH",
    "KSWIN": "HT+KSWIN",
    "ADWIN": "HT+ADWIN",
}
METHOD_ORDER = list(METHOD_LABELS.values())
METRICS = {
    "mae": "MAE (bpm)",
    "rmse": "RMSE (bpm)",
    "error_ratio": "RMSE / MAE",
}


def load_results(suffix: str) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(os.path.join(BATCH_RESULTS_DIR, f"batch_results{suffix}_{s}.csv")) for s in SCENARIOS],
        ignore_index=True,
    )
    df["method"] = df["detector"].map(METHOD_LABELS)
    return df


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, suffix: str):
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df,
        x="scenario",
        y=metric,
        hue="method",
        order=SCENARIOS,
        hue_order=METHOD_ORDER,
        palette="colorblind",
        showmeans=True,
        meanprops={"marker": "x", "markerfacecolor": "white", "markeredgecolor": "white", "markersize": 6},
    )
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.legend(
        title="Método de imputação",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(METHOD_ORDER),
        frameon=False,
    )
    plt.tight_layout()
    save_figure(out_dir(suffix, "boxplots"), metric)
    plt.close()


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="atalho para --suffix _tuned")
    parser.add_argument("--suffix", default="", help="ex.: _tuned, _mod, _tuned_mod")
    args = parser.parse_args()
    suffix = "_tuned" if args.tuned else args.suffix
    df = load_results(suffix)
    for metric, ylabel in METRICS.items():
        plot_metric(df, metric, ylabel, suffix)
        print(f"{metric} salvo em {out_dir(suffix, 'boxplots')}")


if __name__ == "__main__":
    main()
