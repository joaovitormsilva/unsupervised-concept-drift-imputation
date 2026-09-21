"""
Roda a imputação (com e sem detecção não supervisionada de drift) para vários
pacientes em paralelo (um processo por paciente), para um único cenário.

Uso:
    python run_batch.py --scenario S1 --patients 5
    python run_batch.py --scenario S1 --patients all --workers 6
"""
import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from river import preprocessing, tree, drift

from missing_simulation import simulate_patient
from eval_uns_cdd_imp import eval_oml_uns_cdd_imp_horizon

FEATURES = ["hour", "minute"]
SPLIT = 0
HORIZON = 1000
GRACE_PERIOD = 0

PREPARED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prepared")
SIMULATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "simulated")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "Analysis")
BATCH_RESULTS_DIR = os.path.join(ANALYSIS_DIR, "batch_results")
RANKING_DIR = os.path.join(ANALYSIS_DIR, "ranking")


def build_base_model():
    return preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()


def build_detectors():
    return {
        "Sem detector (HT)": None,
        "ADWIN": drift.ADWIN(),
        "KSWIN": drift.KSWIN(),
        "PH": drift.PageHinkley(),
    }


def load_or_simulate(file_id: str, scenario: str) -> pd.DataFrame:
    sim_path = os.path.join(SIMULATED_DIR, f"{file_id}_hr_{scenario}_30.csv")
    if os.path.exists(sim_path):
        return pd.read_csv(sim_path)

    df, _ = simulate_patient(file_id, scenario)
    os.makedirs(SIMULATED_DIR, exist_ok=True)
    df.to_csv(sim_path, index=False)
    return df


def process_patient(file_id: str, scenario: str) -> list[dict]:
    t0 = time.time()
    try:
        df = load_or_simulate(file_id, scenario)

        first_valid_idx = df["heartrate"].first_valid_index()
        df = df.loc[first_valid_idx:].reset_index(drop=True)

        df["datetime"] = pd.to_datetime(df["datetime"])
        for feat in FEATURES:
            df[feat] = df["datetime"].dt.__getattribute__(feat)
        df = df.drop(columns=["datetime"])

        train = df.iloc[:SPLIT].copy()
        test = df.iloc[SPLIT:].copy()

        rows = []
        for name, cdd in build_detectors().items():
            model = build_base_model()
            _, df_true, drifts = eval_oml_uns_cdd_imp_horizon(
                model=model,
                cdd=cdd,
                train=train,
                test=test,
                imp_column="heartrate",
                target_column="target",
                horizon=HORIZON,
                include_remainder=True,
                metric=root_mean_squared_error,
                oml_grace_period=GRACE_PERIOD,
                tqdm_flag=False,
            )
            rmse = root_mean_squared_error(df_true["target"], df_true["Prediction"])
            rows.append({
                "patient": file_id,
                "scenario": scenario,
                "detector": name,
                "rmse": rmse,
                "n_drifts": len(drifts),
            })

        elapsed = time.time() - t0
        print(f"[{file_id}] concluído em {elapsed:.1f}s", flush=True)
        return rows

    except Exception:
        print(f"[{file_id}] ERRO:\n{traceback.format_exc()}", flush=True)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--patients", default="5", help="'all' ou um número de pacientes (ordem de patients.csv)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    patients_df = pd.read_csv(os.path.join(PREPARED_DIR, "patients.csv"))
    patient_ids = patients_df["patient"].tolist()

    if args.patients != "all":
        patient_ids = patient_ids[: int(args.patients)]

    print(f"Processando {len(patient_ids)} pacientes | cenário={args.scenario} | workers={args.workers}")
    t0 = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_patient, pid, args.scenario): pid
            for pid in patient_ids
        }
        for future in as_completed(futures):
            all_rows.extend(future.result())

    elapsed = time.time() - t0
    print(f"\nTempo total: {elapsed / 60:.1f} min para {len(patient_ids)} pacientes")

    results_df = pd.DataFrame(all_rows)
    os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(BATCH_RESULTS_DIR, f"batch_results_{args.scenario}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Resultados salvos em {out_path}")

    if not results_df.empty:
        summary = results_df.groupby("detector")[["rmse", "n_drifts"]].mean()
        print("\nMédia por detector:")
        print(summary)

        # Ranking de RMSE por paciente (rank 1 = menor RMSE = melhor). A média
        # simples de RMSE entre pacientes pode ser distorcida por diferenças
        # de escala de FC entre pacientes; o rank médio neutraliza isso
        # (mesma lógica de Analysis/ranking.py).
        results_df["rank"] = results_df.groupby("patient")["rmse"].rank(
            method="average", ascending=True
        )
        rank_summary = results_df.groupby("detector").agg(
            rank_medio=("rank", "mean"),
            rank_std=("rank", "std"),
            rmse_medio=("rmse", "mean"),
        ).sort_values("rank_medio")
        print("\nRanking por detector (rank 1 = melhor RMSE dentro de cada paciente):")
        print(rank_summary)

        os.makedirs(RANKING_DIR, exist_ok=True)
        rank_out_path = os.path.join(RANKING_DIR, f"ranking_{args.scenario}.csv")
        rank_summary.to_csv(rank_out_path)
        print(f"Ranking salvo em {rank_out_path}")


if __name__ == "__main__":
    main()
