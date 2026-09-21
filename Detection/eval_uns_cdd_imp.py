import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from spotriver.evaluation.eval_bml import ResourceMonitor, evaluate_model, gen_sliding_window, gen_horizon_shifted_window
from river import stream as river_stream
from typing import Tuple

def eval_oml_uns_cdd_imp_horizon(
    model: object,
    cdd: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    imp_column: str,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
    oml_grace_period: int = None,
    tqdm_flag: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    if oml_grace_period is None:
        oml_grace_period = horizon
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]

    model_current = model.clone()
    cdd_current = deepcopy(cdd)

    eval_data = []
    series_preds = []
    series_diffs = []
    drifts = []


    # Fit the model on the train data, i.e., initial Training on Train Data.
    # This is performed on a limited subset only (oml_grace_period).
    # No predictions are made here, only the model is fitted.
    # Memory and runtime are measured for the model fitting
    if(not train.empty):
        train_X = train.loc[:, ~train.columns.isin([target_column, imp_column])]
        train_y = train[imp_column]
        train_X = train_X.tail(oml_grace_period)
        train_y = train_y.tail(oml_grace_period)
        rm = ResourceMonitor()
        with rm:
            try:
                # for xi, yi in tqdm(river_stream.iter_pandas(train_X, train_y), desc="Initial training on train data", total=len(train_X)):
                for xi, yi in river_stream.iter_pandas(train_X, train_y):
                    if ~np.isnan(yi):
                        model.learn_one(xi, yi)
            except Exception as e:
                print(f"train_X data: {train_X}")
                print(f"train_y data: {train_y}")
                print(f"An error occurred while fitting the model: {e}")

        # Create empty lists to collect data
        
        # Measure the costs of the initial training:
        # Add the evaluation of the model (memory and time, not predictions) on the train data to the eval_data list
        # A metric must not be passed to the evaluate_model function, because no predictions are made here
        # If a metric is passed, it will be ignored, because no predictions are passed to the evaluation function
        # So, metric=None and metric=mean_absolute_error will both work
        # Return res_dict = {"Metric": score, "Memory (MB)": memory, "CompTime (s)": r_time}
        eval_data.append(
            evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)
        )

    # Test Data Evaluation
    # A sliding window of length horizon is used to evaluate the model on the test data

    for i, new_df in tqdm(enumerate(gen_sliding_window(test, horizon)), desc="Evaluating data points", total=len(test)//horizon, disable = not tqdm_flag):
    # for i, new_df in enumerate(gen_sliding_window(test, horizon)):
        preds = []
        nan_indexes = new_df[new_df[imp_column].isna()].index
        # if len(nan_indexes) == 0:
        #     continue
        test_X = new_df.loc[:, ~new_df.columns.isin([target_column, imp_column])]
        test_y = new_df[imp_column]
        rm = ResourceMonitor()
        with rm:
            try:
                for xi, yi in river_stream.iter_pandas(test_X, test_y):

                    if np.isnan(yi):
                        pred = model_current.predict_one(xi)
                        preds.append(round(pred,0))

                    else:
                        if cdd_current is not None:
                            cdd_current.update(yi)

                            if cdd_current.drift_detected:
                                cdd_current= deepcopy(cdd)
                                model_current = model.clone()
                                drifts.append(i)

                        model_current.learn_one(xi, yi)

            except Exception as e:
                print(f"test_X data: {test_X}")
                print(f"test_y data: {test_y}")
                print(f"An error occurred while predicting: {e}")
        
        preds = pd.Series(preds)
        diffs = new_df.loc[nan_indexes, target_column].values - preds

        # Collect data in lists
        eval_data.append(
            evaluate_model(
                y_true=new_df.loc[nan_indexes, target_column].values, 
                y_pred=preds, 
                memory=rm.memory, 
                r_time=rm.r_time, 
                metric=metric
            )
        )
        series_preds.extend(preds)
        series_diffs.extend(diffs)

    # Create DataFrames from the collected data
    df_eval = pd.DataFrame(eval_data)
    
    df_true = pd.DataFrame(test[target_column][test[imp_column].isna()])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs

    return df_eval, df_true, drifts