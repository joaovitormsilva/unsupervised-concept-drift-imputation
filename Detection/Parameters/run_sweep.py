"""
Sweep de hiperparâmetros dos detectores (ADWIN, KSWIN, PageHinkley) com a HT
default como imputador, em 10 pacientes sorteados por estrato de tamanho.

Decisões (usuário, 2026-09-27): ajustar nos 10 e avaliar nos 30 (os 10 entram
otimistas, limitação a registrar), um ótimo por cenário (S1, S2, S3),
critério = rank médio por RMSE dentro de cada paciente.

Cada tarefa (cenário, paciente, configuração) é uma execução independente,
salva em checkpoints/<cenário>/; ao relançar, tarefas prontas são puladas.

Uso:
    python run_sweep.py --workers 6
    python run_sweep.py --scenarios S1 --workers 4
    python run_sweep.py --summary-only        # só refaz os resumos a partir dos checkpoints
"""
import argparse
import itertools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from river import drift

PARAMS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PARAMS_DIR))

from run_batch import BATCH_RESULTS_DIR, build_base_model, prepare_stream, run_config  # noqa: E402

RESULTS_DIR = os.path.join(PARAMS_DIR, "results")
CHECKPOINT_DIR = os.path.join(PARAMS_DIR, "checkpoints")
PATIENTS_FILE = os.path.join(PARAMS_DIR, "sweep_patients.csv")

SEED = 1
N_PER_STRATUM = {"pequeno": 3, "medio": 4, "grande": 3}

# Defaults do river incluídos em cada grade. KSWIN recebe seed fixa porque
# amostra aleatoriamente a janela de referência.
GRIDS = {
    "ADWIN": (drift.ADWIN, {"delta": [1e-5, 1e-4, 0.002, 0.01, 0.05]}),
    "KSWIN": (drift.KSWIN, {
        "alpha": [1e-5, 1e-4, 1e-3, 0.005, 0.01],
        "window_size": [100, 300, 1000],
        "seed": [SEED],
    }),
    "PH": (drift.PageHinkley, {
        "threshold": [25, 50, 100, 200, 500],
        "delta": [0.005, 0.5, 2, 5],
    }),
}


def build_sweep_configs() -> list[tuple[str, str, dict]]:
    """Lista de (detector, config_id, params). config_id é seguro para nome de arquivo."""
    configs = [("Sem detector (HT)", "HT", {})]
    for det, (_, grid) in GRIDS.items():
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            config_id = det + "_" + "_".join(f"{k}={v}" for k, v in params.items() if k != "seed")
            configs.append((det, config_id, params))
    return configs


def select_patients() -> pd.DataFrame:
    """Sorteia pacientes por tercil de tamanho (nº de pontos imputados em S1). Resultado fica salvo."""
    if os.path.exists(PATIENTS_FILE):
        return pd.read_csv(PATIENTS_FILE)
    s1 = pd.read_csv(os.path.join(BATCH_RESULTS_DIR, "batch_results_S1.csv"))
    sizes = s1[s1["detector"] == "Media"][["patient", "n_imputed"]].sort_values("n_imputed")
    sizes["estrato"] = pd.qcut(sizes["n_imputed"], 3, labels=list(N_PER_STRATUM))
    rng = np.random.default_rng(SEED)
    picked = pd.concat([
        g.iloc[rng.choice(len(g), N_PER_STRATUM[str(s)], replace=False)]
        for s, g in sizes.groupby("estrato", observed=True)
    ])
    picked.to_csv(PATIENTS_FILE, index=False)
    return picked


def ckpt_path(scenario: str, patient: str, config_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, scenario, f"{patient}__{config_id}.csv")


def run_task(scenario: str, patient: str, detector: str, config_id: str, params: dict) -> dict | None:
    t0 = time.process_time()
    try:
        train, test = prepare_stream(patient, scenario)
        cdd = GRIDS[detector][0](**params) if detector in GRIDS else None
        metrics = run_config(build_base_model(), cdd, train, test)
        row = {
            "scenario": scenario, "patient": patient, "detector": detector,
            "config_id": config_id, **{f"p_{k}": v for k, v in params.items()}, **metrics,
            "cpu_s": time.process_time() - t0,
        }
        path = ckpt_path(scenario, patient, config_id)
        pd.DataFrame([row]).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        return row
    except Exception:
        print(f"[{scenario} {patient} {config_id}] ERRO:\n{traceback.format_exc()}", flush=True)
        return None


def summarize(scenario: str):
    ckpt_dir = os.path.join(CHECKPOINT_DIR, scenario)
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".csv")] if os.path.isdir(ckpt_dir) else []
    if not files:
        return
    df = pd.concat([pd.read_csv(os.path.join(ckpt_dir, f)) for f in files], ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"sweep_{scenario}.csv"), index=False)

    # Rank de RMSE entre as configurações do mesmo detector, dentro de cada
    # paciente (rank 1 = melhor). A HT sem detector entra em todos os grupos
    # como referência.
    ht = df[df["detector"] == "Sem detector (HT)"]
    parts = []
    for det in GRIDS:
        g = pd.concat([df[df["detector"] == det], ht.assign(detector=det)], ignore_index=True)
        g["rank"] = g.groupby("patient")["rmse"].rank(method="average")
        parts.append(g)
    ranked = pd.concat(parts, ignore_index=True)
    summary = ranked.groupby(["detector", "config_id"]).agg(
        n_pacientes=("patient", "nunique"),
        rank_medio=("rank", "mean"),
        rmse_medio=("rmse", "mean"),
        rmse_mediana=("rmse", "median"),
        mae_medio=("mae", "mean"),
        error_ratio_medio=("error_ratio", "mean"),
        n_drifts_medio=("n_drifts", "mean"),
    ).reset_index().sort_values(["detector", "rank_medio"])
    summary.to_csv(os.path.join(RESULTS_DIR, f"sweep_ranking_{scenario}.csv"), index=False)

    print(f"\n=== {scenario}: top 3 por detector (rank médio de RMSE) ===")
    print(summary.groupby("detector").head(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    patients = select_patients()
    configs = build_sweep_configs()
    size = dict(zip(patients["patient"], patients["n_imputed"]))

    if not args.summary_only:
        for s in args.scenarios:
            os.makedirs(os.path.join(CHECKPOINT_DIR, s), exist_ok=True)
        tasks = [
            (s, p, det, cid, params)
            for s in args.scenarios for p in patients["patient"] for det, cid, params in configs
            if not os.path.exists(ckpt_path(s, p, cid))
        ]
        # maiores primeiro: melhor balanceamento entre workers
        tasks.sort(key=lambda t: -size[t[1]])
        total = len(args.scenarios) * len(patients) * len(configs)
        print(
            f"Pacientes: {', '.join(patients['patient'])}\n"
            f"{len(configs)} configs × {len(patients)} pacientes × {len(args.scenarios)} cenários = {total} tarefas | "
            f"{total - len(tasks)} prontas | {len(tasks)} a rodar | workers={args.workers}",
            flush=True,
        )
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_task, *t) for t in tasks]
            for f in as_completed(futures):
                done += 1
                row = f.result()
                if row and done % 10 == 0:
                    print(f"{done}/{len(tasks)} tarefas | {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"\nTempo total: {(time.time() - t0) / 60:.1f} min", flush=True)

    for s in args.scenarios:
        summarize(s)


if __name__ == "__main__":
    main()
