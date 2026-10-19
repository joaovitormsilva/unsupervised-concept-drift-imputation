"""
Rodada 9 sem rodar nada de novo: junta o melhor PH do sweep v3 (thresholds altos,
rank médio por MAE) com Média, HT, ADWIN e KSWIN da rodada 8 (_tuned_mod_30p_v2_mae,
grade do orientador escolhida por MAE).

O sweep roda cada configuração completa em cada paciente, então as linhas do
melhor PH são exatamente o que o run_batch.py produziria para ele.

Saída: Analysis/batch_results/batch_results_tuned_mod_30p_v3_mae_S{1,2,3}.csv

Uso:
    python consolidate_ph_v3.py
"""
import os
import sys

import numpy as np
import pandas as pd

PARAMS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PARAMS_DIR))

from run_batch import BATCH_RESULTS_DIR  # noqa: E402

RESULTS_DIR = os.path.join(PARAMS_DIR, "results")
BASE_SUFFIX = "_tuned_mod_30p_v2_mae"
OUT_SUFFIX = "_tuned_mod_30p_v3_mae"


def main():
    for s in ["S1", "S2", "S3"]:
        base = pd.read_csv(os.path.join(BATCH_RESULTS_DIR, f"batch_results{BASE_SUFFIX}_{s}.csv"))
        ranking = pd.read_csv(os.path.join(RESULTS_DIR, f"sweep_ranking_mod_30p_v3_mae_{s}.csv"))
        sweep = pd.read_csv(os.path.join(RESULTS_DIR, f"sweep_mod_30p_v3_mae_{s}.csv"))
        best = ranking[(ranking["detector"] == "PH") & (ranking["config_id"] != "HT")].iloc[0]["config_id"]

        # sanidade: a HT do sweep tem que bater com a HT da rodada 8
        ht_sweep = sweep[sweep["config_id"] == "HT"].set_index("patient")["mae"]
        ht_base = base[base["detector"] == "Sem detector (HT)"].set_index("patient")["mae"]
        assert np.allclose(ht_sweep.sort_index(), ht_base.sort_index()), f"{s}: HT do sweep difere da rodada 8"

        ph = sweep[sweep["config_id"] == best].copy()
        pcols = [c for c in ph.columns if c.startswith("p_")]
        ph["params"] = ph[pcols].apply(lambda r: ";".join(f"{c[2:]}={r[c]}" for c in pcols), axis=1)
        ph = ph[base.columns]

        out = pd.concat([base[base["detector"] != "PH"], ph], ignore_index=True)
        out = out.sort_values(["patient", "detector"], key=lambda c: c if c.name == "patient" else c.map(
            {d: i for i, d in enumerate(base["detector"].drop_duplicates())})).reset_index(drop=True)
        path = os.path.join(BATCH_RESULTS_DIR, f"batch_results{OUT_SUFFIX}_{s}.csv")
        out.to_csv(path, index=False)
        print(f"{s}: PH = {best} | {len(out)} linhas -> {path}")


if __name__ == "__main__":
    main()
