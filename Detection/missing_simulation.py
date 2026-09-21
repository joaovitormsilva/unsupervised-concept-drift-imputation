"""
Simulação de ausência (MCAR/MAR/MNAR) por segmento, replicando exatamente a
lógica de `Missing Simulation/mmd_simulation_ppl.py` do projeto irmão
(supervised-concept-drift-analysis), para um único paciente/cenário por vez.

Escopo do smoke test: paciente AV2GF3B, cenário S1, mr=30 (mesmos valores
usados para gerar data/prepared/summary_S1_30.csv).
"""
import os
import pandas as pd
import numpy as np

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMAR import uMAR
from mdatagen.univariate.uMNAR import uMNAR

MR_F = 30
SEED = 1

SPLITS_RATIO = {
    "S1": [0.34, 0.33, 0.33],
    "S2": [0.25, 0.25, 0.25, 0.25],
    "S3": [0.2, 0.2, 0.2, 0.2, 0.2],
}
MECHS = {
    "S1": ["MAR", "MCAR", "MNAR"],
    "S2": ["MAR", "MNAR", "MAR", "MNAR"],
    "S3": ["MNAR", "MAR", "MNAR", "MAR", "MNAR"],
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "COVID-19-Wearables")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "simulated")


def simulate_mm(mech, X_split, mr_f, mnar_t=1):
    if mech == "MCAR":
        generator = uMCAR(
            X=X_split.set_index("datetime"),
            y=X_split.heartrate.to_numpy(),
            missing_rate=mr_f,
            x_miss="heartrate",
            seed=SEED,
        )
        return generator.random().reset_index()

    if mech == "MAR":
        X_split["time"] = pd.to_datetime(X_split["datetime"]).dt.time
        generator = uMAR(
            X=X_split,
            y=X_split.heartrate.to_numpy(),
            missing_rate=mr_f,
            x_miss="heartrate",
            x_obs="time",
            seed=SEED,
        )
        return generator.lowest().reset_index()

    if mech == "MNAR":
        generator = uMNAR(
            X=X_split.reset_index(drop=True),
            y=X_split.heartrate.to_numpy(),
            threshold=mnar_t,
            missing_rate=mr_f,
            x_miss="heartrate",
            seed=SEED,
        )
        return generator.run(deterministic=False).reset_index()

    raise ValueError(f"Mechanism {mech} not recognized")


def simulate_patient(file_id: str, scenario: str, mr_f: int = MR_F) -> tuple[pd.DataFrame, list[int]]:
    """Reproduz o pipeline de simulação de ausência para um paciente/cenário.

    Retorna o dataframe simulado (datetime, heartrate com NaN, target = valor
    original) e a lista de índices de corte dos segmentos (deve bater com
    data/prepared/summary_{scenario}_30.csv).
    """
    splits_ratio = SPLITS_RATIO[scenario]
    mechs = MECHS[scenario]

    file_path = os.path.join(RAW_DIR, f"{file_id}_hr.csv")
    data = pd.read_csv(file_path)
    data["target"] = data["heartrate"].astype(float)
    data = data.reset_index(drop=True)
    data["row_id"] = data.index

    X = data[["datetime", "heartrate", "row_id"]]
    X = X[~X["heartrate"].isna()].copy()

    split_sizes = [int(len(X) * ratio) for ratio in splits_ratio]
    indexes = np.cumsum([0] + split_sizes).tolist()

    split_mr = mr_f / len(splits_ratio)
    mms = {
        f"{mech}_{i}": X.iloc[indexes[i] : indexes[i + 1]]
        for i, mech in enumerate(mechs)
    }

    for mech_key, X_split in mms.items():
        mech = mech_key.split("_")[0]
        generated = simulate_mm(mech, X_split.copy(), split_mr)
        data.loc[data["row_id"].isin(generated["row_id"]), "heartrate"] = generated["heartrate"].to_numpy()

    data = data.drop(columns="row_id")[["datetime", "heartrate", "target"]]
    return data, indexes


if __name__ == "__main__":
    file_id = "AV2GF3B"
    scenario = "S1"

    df, idx_list = simulate_patient(file_id, scenario)

    print(f"Índices de corte gerados: {idx_list}")
    expected = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "prepared", f"summary_{scenario}_30.csv")
    )
    expected_row = expected[expected["file_id"] == file_id].iloc[0]
    print(f"Índices esperados (summary_{scenario}_30.csv): "
          f"{[int(expected_row[c]) for c in expected.columns if c.startswith('index_')]}")

    n_missing = df["heartrate"].isna().sum()
    print(f"Total de linhas: {len(df)} | Valores ausentes gerados: {n_missing} "
          f"({100 * n_missing / len(df):.1f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{file_id}_hr_{scenario}_30.csv")
    df.to_csv(out_path, index=False)
    print(f"Salvo em {out_path}")
