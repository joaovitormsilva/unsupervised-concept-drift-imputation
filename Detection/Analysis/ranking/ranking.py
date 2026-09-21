"""
Compara detectores de drift por RANKING de RMSE dentro de cada paciente, em vez
de média simples de RMSE entre pacientes.

Por que ranking e não média simples?
A escala de FC (e portanto a magnitude do RMSE) varia muito entre pacientes.
Um paciente com RMSE naturalmente mais alto domina a média bruta e pode
distorcer a comparação entre detectores. Rankear os detectores dentro de cada
paciente (1 = menor RMSE = melhor, ..., N = maior RMSE = pior) e só depois
tirar a média dos ranks neutraliza essa diferença de escala entre pacientes.
Essa é a mesma abordagem usada em análises tipo Critical-Difference-Diagram.

Uso:
    python ranking.py [caminho_para_csv]

Por padrão, usa Detection/Analysis/batch_results/batch_results_S1.csv.
"""
import argparse
import os

import pandas as pd

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ANALYSIS_DIR, "batch_results", "batch_results_S1.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranking_S1.csv")


def compute_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula rank médio (e desvio padrão) de RMSE por detector, através dos pacientes.

    Rank 1 = menor RMSE (melhor) dentro do paciente; rank N = maior RMSE (pior).
    Empates recebem o rank médio (method='average').
    """
    df = results_df.copy()
    df["rank"] = df.groupby("patient")["rmse"].rank(method="average", ascending=True)

    summary = df.groupby("detector").agg(
        rank_medio=("rank", "mean"),
        rank_std=("rank", "std"),
        rmse_medio=("rmse", "mean"),
    )
    summary = summary.sort_values("rank_medio")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Caminho do CSV de resultados (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Caminho para salvar a tabela agregada (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_df = pd.read_csv(args.csv_path)

    n_patients = results_df["patient"].nunique()
    print(f"Carregado {args.csv_path} | {n_patients} pacientes | {results_df['detector'].nunique()} detectores\n")

    summary = compute_ranking(results_df)

    print("Ranking de detectores por RMSE (rank 1 = melhor dentro de cada paciente):")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    summary.to_csv(args.output)
    print(f"\nTabela agregada salva em {args.output}")


if __name__ == "__main__":
    main()
