"""
Boxplots da distribuição das métricas por paciente (um ponto = um paciente),
por cenário e método de imputação. Estilo replicado da Fig. 3a do artigo do
Afonso (SBBD 2026) / notebook Imputation/Analysis/analysis_plots.ipynb.

Uso:
    python boxplots.py            # detectores com defaults do river
    python boxplots.py --tuned    # detectores com parâmetros do sweep
Saída: Plots/boxplots/boxplot[_tuned]_{mae,rmse,error_ratio}.{png,pdf}
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(PLOTS_DIR)
BATCH_RESULTS_DIR = os.path.join(ANALYSIS_DIR, "batch_results")
OUT_DIR = os.path.join(PLOTS_DIR, "boxplots")

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
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT_DIR, f"boxplot{suffix}_{metric}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    os.makedirs(OUT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true")
    suffix = "_tuned" if parser.parse_args().tuned else ""
    df = load_results(suffix)
    for metric, ylabel in METRICS.items():
        plot_metric(df, metric, ylabel, suffix)
        print(f"boxplot{suffix}_{metric} salvo em {OUT_DIR}")


if __name__ == "__main__":
    main()
