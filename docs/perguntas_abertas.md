# Perguntas abertas

Decisões pendentes do TCC. Cada item diz o contexto, as opções e o que depende
dele. Ao resolver, mover para "Resolvidas" com a data e a decisão.

## Abertas

### 1. Métrica principal: RMSE ou MAE?
- **Contexto:** o artigo do Afonso (SBBD 2026) reporta "RMSE", mas o pipeline
  dele usa `HORIZON = 1` e tira a média do RMSE por janela. Com um ponto por
  janela, o RMSE vira o erro absoluto, então o valor reportado é o MAE.
  Verificado: o `mean_rmse` dele bate com o nosso MAE para HT e Média nos 180
  pares paciente×cenário (diferença máxima 0,005).
- **Hoje:** calculamos e plotamos MAE, RMSE e error_ratio (RMSE/MAE). O RMSE
  segue como métrica mantida até a definição com o orientador.
- **Depende disso:** qual boxplot é o principal no texto e qual é comparável à
  Fig. 3a do Afonso (o de MAE).
- **Com quem:** orientador.

### 2. Adaptar a HT para não gerar predições fisiologicamente impossíveis
- **Contexto:** a `HoeffdingTreeRegressor` default usa modelo linear nas folhas
  (`leaf_prediction="adaptive"`) e extrapola. Exemplo: AJMQUVV, S1, HT sem
  detector: 977 de 89.748 pontos imputados abaixo de 30 bpm, mínimo −1724 bpm
  (FC real entre 61 e 164). MAE = 32, RMSE = 172. O pipeline do Afonso tem o
  mesmo comportamento (há um `if pred > 300` comentado no código dele).
- **Opções:** (a) limitar as predições a uma faixa fisiológica, ex. 30–220 bpm;
  (b) `leaf_prediction="mean"` na HT; (c) manter como está e discutir.
- **Decisão até agora:** adaptar a HT, mas não agora.
- **Depende disso:** exige rodar o batch de novo (S1–S3, 30 pacientes, ~3h).
  Pergunta a responder depois: a vantagem do PH continua depois da adaptação?

### 3. Hiperparâmetros dos detectores: defaults ou ajustados?
- **Contexto:** usamos os defaults do river. O Afonso ajustou os detectores
  por cenário, com subconjuntos de 1, 3 e 5 pacientes.
- **Opções:** manter defaults (mais simples, sem ajuste com dados de teste) ou
  fazer uma análise de sensibilidade (`Detection/Parameters/`).
- **Com quem:** orientador.
- **Em andamento (2026-09-27):** sweep em `Detection/Parameters/run_sweep.py`.
  10 pacientes sorteados por tercil de tamanho (seed 1, lista em
  `sweep_patients.csv`), S1–S3, 40 configurações + HT sem detector,
  critério = rank médio de RMSE por paciente. Decisão do usuário: ajustar nos
  10 e avaliar nos 30, então os 10 entram otimistas (registrar como
  limitação). Ressalva: o ótimo por RMSE pode mudar depois de adaptar a HT
  (item 2).

- **2026-10-07:** com minuto do dia, o ótimo dos 10 pacientes não
  generalizou para os 30 (ex.: ADWIN S2). Decisão do usuário: refazer o sweep
  com os 30 pacientes (`run_sweep.py --features mod --patients all`, saídas
  `_30p`) e usar esses parâmetros na rodada 5. Parâmetros escolhidos e
  avaliados nos mesmos 30 pacientes: resultado otimista, registrar como
  limitação no texto.

- **2026-10-11 — grade do orientador (v2):** KSWIN `window_size` ∈
  {5k, 10k, 20k, 30k} × `stat_size` ∈ {200, 400, 800, 1000} × α ∈ {1e-5,
  1e-4, 1e-3}; ADWIN `min_window_length` ∈ {10k, 15k, 20k, 30k, 50k} (δ
  default); PH igual. Janelas na escala de dias (os detectores atuais disparam
  a cada 1–2 h). 30 pacientes, minuto do dia, HT sem correção (decisão do
  usuário). Saídas `_30p_v2`; rodada 6. Hipótese a checar nos resultados: com
  janelas grandes KSWIN/ADWIN podem virar reset periódico (n_drifts ≈
  pontos ÷ janela).
  - **Resultado (2026-10-13, 776 min na VM):** hipótese confirmada. KSWIN
    dispara toda vez que a janela enche (n_drifts ÷ (pontos/janela) ≈ 0,97–1,00,
    α indiferente); ADWIN dispara a cada ~2×`min_window_length` (razão ≈ 0,4).
    Com 5–22 resets por paciente, ADWIN e KSWIN ficaram piores que a HT sem
    detector em todos os cenários (ex. S1 mediana RMSE: HT 20,9, ADWIN 30,7,
    KSWIN 79,5); o melhor de cada grade é o maior valor (borda da grade).
    PH inalterado. Rodada 6 em `Plots/6_minuto_do_dia_ajustados_30p_grade_orientador/`.

- **2026-10-16 — PH com thresholds altos (v3):** o PH ajustado reseta a cada
  ~46–68 pontos observados (Afonso: nesse regime seria mais simples imputar os
  valores mais recentes). Grade: threshold ∈ {1k, 2k, 5k, 10k, 20k, 50k} ×
  δ ∈ {0.005, 0.5, 2, 5} × `min_instances` ∈ {30, 1k, 5k, 10k} (96 configs).
  30 pacientes, minuto do dia, HT sem correção, **rank médio por MAE**
  (Afonso pediu foco só no MAE). Saídas `_30p_v3_mae`.
  - **Resultado (2026-10-18, 1369 min na VM):** nenhuma config v3 chega perto
    da HT sem detector nem do PH da v1. Melhor v3 = threshold=50000, δ=5
    (~23 resets/paciente): MAE mediana 31,9 / 61,8 / 62,1 (S1/S2/S3) contra
    HT 13,2 / 18,7 / 18,0 e PH v1 10,9 / 14,1 / 16,5. `min_instances` quase
    não muda nada. Curva MAE × nº de resets: resets muito frequentes
    (~8–13 mil, a cada ~50 pontos) e nenhum reset são bons; qualquer coisa
    no meio (dezenas a milhares de resets) é ruim, pior entre ~500 e 2000.
- **2026-10-16 — MAE como critério em todos os sweeps (decisão do usuário).**
  Os sweeps de 30 pacientes (v1 e v2) foram re-ranqueados por MAE a partir dos
  checkpoints (saídas `_30p_mae`, `_30p_v2_mae`). Muda só o S3: PH threshold
  50 → 100 (v1 e v2) e KSWIN stat_size 400 → 200 (v2). Rodadas 7 e 8 em
  `Plots/7_*` e `Plots/8_*`, CD-diagram só de MAE.

### 4. Ausência descartada no início de cada série
- **Contexto:** o pipeline corta tudo antes do primeiro valor observado
  (`first_valid_index()`), o que descarta ~2% dos ausentes simulados. O
  pipeline do Afonso faz o mesmo, então os resultados seguem comparáveis.
- **Prioridade:** baixa; registrar como limitação no texto.

## Resolvidas

- **2026-10-06 — KSWIN sempre com `seed=1`.** Todas as rodadas (o KSWIN é o
  único componente aleatório; Média, HT, ADWIN e PH são determinísticos). As
  rodadas 1 e 3 tiveram só o KSWIN refeito; o `run_batch.py` agora refaz
  apenas os métodos cujos parâmetros mudaram em relação ao checkpoint.
- **2026-10-06 — Sweep refeito com minuto do dia.** `run_sweep.py --features mod`
  (mesmos 10 pacientes e mesma grade); a rodada 4 usa esses parâmetros.

- **2026-10-05 — CD-diagram incluído.** `Detection/Analysis/Plots/cd_diagram.py`,
  adaptado de hfawaz/cd-diagram (Friedman + Wilcoxon pareado por paciente +
  Holm, α = 0,05), para MAE e RMSE por cenário.
- **2026-10-05 — Entrada do imputador = minuto do dia.** `x = {minute_of_day}`
  (hora*60 + minuto) em vez de `{hour, minute}`. Rodadas com sufixo `_mod`
  (defaults e ajustados); as rodadas antigas foram mantidas para comparação.

- **2026-09-27 — Avaliação por paciente.** Cada paciente é um fluxo
  independente com modelo e detector próprios. Métricas por paciente,
  distribuição mostrada em boxplots; sem métrica global agregada. Confirmado
  com o orientador.
