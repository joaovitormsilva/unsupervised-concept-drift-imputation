# Rodada 9: PH com threshold alto (critério MAE)

- **Entrada do imputador:** minuto do dia. **Detectores:** recebem a FC observada.
  **Imputador:** `StandardScaler | HoeffdingTreeRegressor` default, reset completo no drift.
- **30 pacientes**, cenários S1–S3, avaliação por paciente.
- **PH:** melhor configuração da grade v3 (threshold ∈ {1k…50k} × δ ∈ {0.005, 0.5, 2, 5}
  × min_instances ∈ {30, 1k, 5k, 10k}), escolhida pelo rank médio de MAE nos 30
  pacientes: `threshold=50000, delta=5, min_instances=5000` nos três cenários.
- **ADWIN e KSWIN:** iguais à rodada 8 (grade do orientador nos 30 pacientes,
  critério MAE): ADWIN `min_window_length=50000`; KSWIN `window_size=30000`.
  É a rodada 6 com escolha por MAE; a única diferença para a rodada 6 é o KSWIN
  do S3 (`stat_size` 400 → 200). Média e HT sem detector iguais em todas as rodadas.
- As linhas do PH vêm do sweep (`Detection/Parameters/consolidate_ph_v3.py`), que
  roda cada configuração completa em cada paciente.

## Arquivos

- `boxplots/`: MAE, RMSE e error_ratio por cenário (uma caixa = 30 pacientes).
- `cd_diagrams/`: Friedman + Wilcoxon pareado por paciente + Holm (α = 0,05), só MAE;
  `wilcoxon_holm.csv` tem todos os pares.
- `dados/batch_results_*_S{1,2,3}.csv`: métricas por paciente e método.
- `dados/resumo_mae.csv`: mediana, média, quartis do MAE e mediana de resets.

## Resultado em uma frase

Com ~16–18 resets por paciente, o PH fica pior que a HT sem detector nos três
cenários (MAE mediana 33,5 / 62,1 / 67,3 contra 13,2 / 18,7 / 18,0). A HT sem
detector tem o melhor rank em todos os cenários.
