"""
Roda a imputação (com e sem detecção não supervisionada de drift) para vários
pacientes em paralelo (um processo por paciente), para um único cenário.

Cada paciente concluído é salvo em Analysis/batch_results/checkpoints/<cenário>/;
ao relançar, pacientes com checkpoint são pulados. Use --fresh para recomeçar.

Uso:
    python run_batch.py --scenario S1 --patients 5
    python run_batch.py --scenario S1 --patients all --workers 6
    python run_batch.py --scenario S1 --patients all --fresh
    python run_batch.py --scenario S1 --patients all --tuned   # parâmetros do sweep
"""
import argparse
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from river import preprocessing, tree, drift, dummy, stats

from missing_simulation import simulate_patient
from eval_uns_cdd_imp import eval_oml_uns_cdd_imp_horizon

FEATURES = ["hour", "minute"]
SPLIT = 0
GRACE_PERIOD = 0

PREPARED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prepared")
SIMULATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "simulated")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "Analysis")
BATCH_RESULTS_DIR = os.path.join(ANALYSIS_DIR, "batch_results")
CHECKPOINT_DIR = os.path.join(BATCH_RESULTS_DIR, "checkpoints")
RANKING_DIR = os.path.join(ANALYSIS_DIR, "ranking")


def build_base_model():
    return preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()


def build_mean_model():
    return dummy.StatisticRegressor(statistic=stats.Mean())


DETECTORS = {"ADWIN": drift.ADWIN, "KSWIN": drift.KSWIN, "PH": drift.PageHinkley}
SWEEP_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Parameters", "results")


def tuned_params(scenario: str) -> dict[str, dict]:
    """Melhores parâmetros por detector no sweep (Parameters/run_sweep.py) para o cenário:
    menor rank médio de RMSE entre as configurações de cada detector."""
    ranking = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, f"sweep_ranking_{scenario}.csv"))
    sweep = pd.read_csv(os.path.join(SWEEP_RESULTS_DIR, f"sweep_{scenario}.csv"))
    best = {}
    for det in DETECTORS:
        cands = ranking[(ranking["detector"] == det) & (ranking["config_id"] != "HT")]
        config_id = cands.sort_values("rank_medio").iloc[0]["config_id"]
        row = sweep[sweep["config_id"] == config_id].iloc[0]
        params = {c[2:]: row[c] for c in sweep.columns if c.startswith("p_") and pd.notna(row[c])}
        best[det] = {k: int(v) if k in ("window_size", "seed") else float(v) for k, v in params.items()}
    return best


def build_configs(scenario: str | None = None, tuned: bool = False):
    """name -> (model_builder, cdd, params). cdd=None significa sem detecção de drift.
    tuned=True usa os melhores parâmetros do sweep para o cenário; senão, defaults do river."""
    params = tuned_params(scenario) if tuned else {det: {} for det in DETECTORS}
    configs = {
        "Media": (build_mean_model, None, {}),
        "Sem detector (HT)": (build_base_model, None, {}),
    }
    for det, cls in DETECTORS.items():
        configs[det] = (build_base_model, cls(**params[det]), params[det])
    return configs


def load_or_simulate(file_id: str, scenario: str) -> pd.DataFrame:
    sim_path = os.path.join(SIMULATED_DIR, f"{file_id}_hr_{scenario}_30.csv")
    if os.path.exists(sim_path):
        return pd.read_csv(sim_path)

    df, _ = simulate_patient(file_id, scenario)
    os.makedirs(SIMULATED_DIR, exist_ok=True)
    df.to_csv(sim_path, index=False)
    return df


def prepare_stream(file_id: str, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega a série com ausência simulada e devolve (train, test) prontos para o loop online."""
    df = load_or_simulate(file_id, scenario)

    first_valid_idx = df["heartrate"].first_valid_index()
    df = df.loc[first_valid_idx:].reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    for feat in FEATURES:
        df[feat] = df["datetime"].dt.__getattribute__(feat)
    df = df.drop(columns=["datetime"])

    return df.iloc[:SPLIT].copy(), df.iloc[SPLIT:].copy()


def run_config(model, cdd, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Roda um imputador (com ou sem detector) e devolve as métricas sobre os valores imputados."""
    _, df_true, drifts = eval_oml_uns_cdd_imp_horizon(
        model=model,
        cdd=cdd,
        train=train,
        test=test,
        imp_column="heartrate",
        target_column="target",
        horizon=len(test),
        include_remainder=True,
        metric=root_mean_squared_error,
        oml_grace_period=GRACE_PERIOD,
        tqdm_flag=False,
    )
    mae = mean_absolute_error(df_true["target"], df_true["Prediction"])
    rmse = root_mean_squared_error(df_true["target"], df_true["Prediction"])
    err = df_true["target"] - df_true["Prediction"]
    return {
        "mae": mae,
        "rmse": rmse,
        "error_ratio": rmse / mae if mae != 0 else float("inf"),
        "n_drifts": len(drifts),
        # somas brutas: permitem calcular MAE/RMSE agregados sobre todos os pontos depois
        "n_imputed": len(err),
        "sae": err.abs().sum(),
        "sse": (err ** 2).sum(),
    }


def process_patient(file_id: str, scenario: str, tuned: bool = False) -> list[dict]:
    t0 = time.time()
    cpu0 = time.process_time()
    try:
        train, test = prepare_stream(file_id, scenario)

        rows = []
        for name, (model_builder, cdd, params) in build_configs(scenario, tuned).items():
            metrics = run_config(model_builder(), cdd, train, test)
            rows.append({
                "patient": file_id, "scenario": scenario, "detector": name,
                "params": ";".join(f"{k}={v}" for k, v in params.items()), **metrics,
            })

        # CPU time não conta o tempo em que a máquina ficou suspensa; wall time conta.
        print(
            f"[{file_id}] concluído em {time.time() - t0:.1f}s (cpu {time.process_time() - cpu0:.1f}s)",
            flush=True,
        )
        return rows

    except Exception:
        print(f"[{file_id}] ERRO:\n{traceback.format_exc()}", flush=True)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--patients", default="5", help="'all' ou um número de pacientes (ordem de patients.csv)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh", action="store_true", help="apaga os checkpoints do cenário e recomeça do zero")
    parser.add_argument("--tuned", action="store_true", help="detectores com os melhores parâmetros do sweep (saída com sufixo _tuned)")
    args = parser.parse_args()

    patients_df = pd.read_csv(os.path.join(PREPARED_DIR, "patients.csv"))
    patient_ids = patients_df["patient"].tolist()

    if args.patients != "all":
        patient_ids = patient_ids[: int(args.patients)]

    suffix = "_tuned" if args.tuned else ""
    ckpt_dir = os.path.join(CHECKPOINT_DIR + suffix, args.scenario)
    if args.fresh and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    def ckpt_path(pid):
        return os.path.join(ckpt_dir, f"{pid}.csv")

    pending = [p for p in patient_ids if not os.path.exists(ckpt_path(p))]
    print(
        f"Cenário={args.scenario} | {len(patient_ids)} pacientes | "
        f"{len(patient_ids) - len(pending)} já com checkpoint | {len(pending)} a processar | workers={args.workers}",
        flush=True,
    )
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_patient, pid, args.scenario, args.tuned): pid
            for pid in pending
        }
        for future in as_completed(futures):
            rows = future.result()
            if rows:
                pid = futures[future]
                tmp = ckpt_path(pid) + ".tmp"
                pd.DataFrame(rows).to_csv(tmp, index=False)
                os.replace(tmp, ckpt_path(pid))

    elapsed = time.time() - t0
    print(f"\nTempo total: {elapsed / 60:.1f} min para {len(pending)} pacientes processados nesta execução")

    done = [p for p in patient_ids if os.path.exists(ckpt_path(p))]
    missing = sorted(set(patient_ids) - set(done))
    if missing:
        print(f"ATENÇÃO: {len(missing)} paciente(s) sem resultado (erro): {missing}")

    results_df = pd.concat([pd.read_csv(ckpt_path(p)) for p in done], ignore_index=True) if done else pd.DataFrame()
    os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(BATCH_RESULTS_DIR, f"batch_results{suffix}_{args.scenario}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Resultados salvos em {out_path}")

    if not results_df.empty:
        summary = results_df.groupby("detector")[["mae", "rmse", "error_ratio", "n_drifts"]].mean()
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
            mae_medio=("mae", "mean"),
            rmse_medio=("rmse", "mean"),
            error_ratio_medio=("error_ratio", "mean"),
        ).sort_values("rank_medio")
        print("\nRanking por detector (rank 1 = melhor RMSE dentro de cada paciente):")
        print(rank_summary)

        os.makedirs(RANKING_DIR, exist_ok=True)
        rank_out_path = os.path.join(RANKING_DIR, f"ranking{suffix}_{args.scenario}.csv")
        rank_summary.to_csv(rank_out_path)
        print(f"Ranking salvo em {rank_out_path}")


if __name__ == "__main__":
    main()
