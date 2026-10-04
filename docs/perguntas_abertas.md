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

### 6. KSWIN sem seed no batch principal
- **Contexto:** o `KSWIN` sorteia a janela de referência, e o `run_batch.py`
  usa `seed=None`. Os resultados de KSWIN no batch não são reproduzíveis
  exatamente. O sweep já usa `seed=1`.
- **Opções:** fixar `seed=1` no batch principal na próxima rodada (junto com a
  adaptação da HT ou os parâmetros ajustados).

### 4. Incluir CD-diagram (Wilcoxon + Holm)?
- **Contexto:** o artigo do Afonso usa boxplot (distribuição) + CD-diagram
  (significância estatística das diferenças entre métodos, pareado por
  paciente). Hoje só temos os boxplots.
- **Depende disso:** se o texto vai afirmar que um método é melhor que outro.

### 5. Ausência descartada no início de cada série
- **Contexto:** o pipeline corta tudo antes do primeiro valor observado
  (`first_valid_index()`), o que descarta ~2% dos ausentes simulados. O
  pipeline do Afonso faz o mesmo, então os resultados seguem comparáveis.
- **Prioridade:** baixa; registrar como limitação no texto.

## Resolvidas

- **2026-09-27 — Avaliação por paciente.** Cada paciente é um fluxo
  independente com modelo e detector próprios. Métricas por paciente,
  distribuição mostrada em boxplots; sem métrica global agregada. Confirmado
  com o orientador.
