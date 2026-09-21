"""
Smoke test: paciente AV2GF3B, cenário S1, imputação online com e sem
detecção não supervisionada de concept drift (ADWIN / KSWIN / Page-Hinkley
aplicados diretamente na FC observada, com reset completo do imputador ao
detectar drift).
"""
import os
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from river import preprocessing, tree, drift

from missing_simulation import simulate_patient
from eval_uns_cdd_imp import eval_oml_uns_cdd_imp_horizon

FILE_ID = "AV2GF3B"
SCENARIO = "S1"
FEATURES = ["hour", "minute"]
SPLIT = 0
HORIZON = 1000
GRACE_PERIOD = 0

DETECTORS = {
    "Sem detector (HT)": None,
    "ADWIN": drift.ADWIN(),
    "KSWIN": drift.KSWIN(),
    "PH": drift.PageHinkley(),
}


def build_base_model():
    return preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()


def main():
    sim_path = os.path.join(os.path.dirname(__file__), "..", "data", "simulated", f"{FILE_ID}_hr_{SCENARIO}_30.csv")
    if not os.path.exists(sim_path):
        df, _ = simulate_patient(FILE_ID, SCENARIO)
    else:
        df = pd.read_csv(sim_path)

    first_valid_idx = df["heartrate"].first_valid_index()
    df = df.loc[first_valid_idx:].reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    for feat in FEATURES:
        df[feat] = df["datetime"].dt.__getattribute__(feat)
    df = df.drop(columns=["datetime"])

    train = df.iloc[:SPLIT].copy()
    test = df.iloc[SPLIT:].copy()

    results = []
    for name, cdd in DETECTORS.items():
        model = build_base_model()
        df_eval, df_true, drifts = eval_oml_uns_cdd_imp_horizon(
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
            tqdm_flag=True,
        )
        rmse = root_mean_squared_error(df_true["target"], df_true["Prediction"])
        results.append({"detector": name, "rmse": rmse, "n_drifts": len(drifts)})
        print(f"{name}: RMSE={rmse:.3f} | drifts detectados={len(drifts)}")

    results_df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(__file__), "Analysis", "smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"smoke_test_{FILE_ID}_{SCENARIO}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
