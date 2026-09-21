# CLAUDE.md — instruções para agentes neste projeto

TCC: avaliação de detectores **não supervisionados** de concept drift (ADWIN,
KSWIN, Page-Hinkley) aplicados diretamente à frequência cardíaca (FC)
observada de wearables, disparando reset completo de um imputador online
(river) quando um drift é detectado. Comparação contra baseline sem detector.

Trabalho irmão/precursor (abordagem supervisionada, mesmo dataset, chamado
apenas de "o projeto irmão" ou pelo nome do autor — nunca "código do amigo"):
https://github.com/afonsoMatheus/supervised-concept-drift-analysis

## Ambiente

- Venv em `.venv/` (Python 3.14). Ative com `.venv/bin/python`, não há `pip`
  no Python de sistema.
- `requirements.txt` está atualizado, mas **atenção**: `mdatagen` NÃO deve
  vir do PyPI — o pacote publicado lá (0.2.0) não tem o parâmetro `seed` em
  `uMAR`/`uMNAR` (só `uMCAR` tem). O projeto irmão usa uma versão instalada
  direto do GitHub (`main` branch) que já tem `seed` nos três. Instale com:
  `pip install git+https://github.com/ArthurMangussi/pymdatagen.git@main`
- `setuptools` precisa ser `<81` nesse venv, porque `mdatagen` ainda importa
  `pkg_resources` (removido de versões recentes do setuptools).

## Decisões de design já travadas (não reabrir sem discutir com o usuário)

1. **Sinal do detector**: ADWIN/KSWIN/PageHinkley são alimentados com o valor
   de FC observado (`yi`), não com hora/minuto nem com erro de predição. Isso
   é o que torna a detecção "não supervisionada" (o projeto irmão alimenta
   esses mesmos detectores com o erro de um modelo supervisionado).
2. **Reação ao drift**: reset completo do imputador (`model.clone()`), não
   retreino em janela deslizante. Decisão explícita do usuário.
3. **Dados com ausência simulada**: regenerados localmente via
   `Detection/missing_simulation.py`, replicando exatamente a lógica de
   `Missing Simulation/mmd_simulation_ppl.py` do projeto irmão (mesmo
   `SEED=1`, mesmos splits/mecanismos por cenário). Validado que os índices de
   corte batem com `data/prepared/summary_S{1,2,3}_30.csv`.
4. **Modelo imputador base**: `preprocessing.StandardScaler() |
   tree.HoeffdingTreeRegressor()`, igual ao baseline "HT" do projeto irmão.

## Armadilhas já encontradas (não repetir)

- **`river` 0.26+**: `detector.update(x)` não retorna mais bool — ele muda
  estado interno. Sempre checar `detector.drift_detected` depois de chamar
  `.update()`.
- **`ResourceMonitor` do spotriver é caro por chamada.** Como o loop de
  avaliação usa `horizon` como tamanho do lote medido (não o tamanho do passo
  de aprendizado — `learn_one`/`predict_one` continuam ponto a ponto dentro do
  lote), rodar com `horizon=1` mede recursos a cada linha e fica ~15x mais
  lento sem ganho nenhum de precisão nas métricas que usamos (RMSE, n_drifts).
  Use `HORIZON=1000` (já configurado em `run_smoke_test.py`/`run_batch.py`).
- **Tamanho de série varia MUITO entre pacientes** (75 mil a 1,4 milhão de
  linhas). Não assuma que todos os pacientes têm custo parecido — o paciente
  `ASFODQR` sozinho já levou ~19 min mesmo com o horizon otimizado.
- **Média simples de RMSE entre pacientes pode distorcer conclusões** —
  pacientes com FC mais volátil têm RMSE naturalmente maior e dominam a
  média. Sempre olhar também o ranking por paciente (`Detection/Analysis/ranking/`)
  e, se possível, resultados individuais por paciente antes de concluir algo.

## Estrutura de pastas

```
data/
  raw/                 dados brutos por paciente (COVID-19-Wearables), não versionado
  prepared/            patients.csv + summary_S{1,2,3}_30.csv (índices de corte, do projeto irmão)
  simulated/           cache dos CSVs com ausência já simulada (gerado por missing_simulation.py)
Detection/
  missing_simulation.py   gera ausência (MCAR/MAR/MNAR) por paciente/cenário
  eval_uns_cdd_imp.py      núcleo do loop online (detecção + imputação), não gera ausência
  run_smoke_test.py        roda 1 paciente/1 cenário, todos os detectores
  run_batch.py             roda vários pacientes em paralelo (ProcessPoolExecutor)
  Analysis/
    smoke_test/             resultados do smoke test
    batch_results/          resultados agregados por lote (batch_results_<cenário>.csv)
    ranking/                ranking.py + ranking_<cenário>.csv
    individual_patients/    aprofundamento em pacientes específicos
    Plots/                  gráficos (CD-diagrams etc, inspirados nos notebooks do projeto irmão)
  Parameters/               análise de sensibilidade de hiperparâmetro (futuro)
docs/
  conceitos.md          explicação didática de RMSE/ADWIN/KSWIN/PageHinkley
```

## Como rodar

```bash
cd Detection
../.venv/bin/python run_smoke_test.py                              # 1 paciente (AV2GF3B, S1)
../.venv/bin/python run_batch.py --scenario S1 --patients 5         # lote pequeno
../.venv/bin/python run_batch.py --scenario S1 --patients all --workers 4  # todos os 30
```

## Worktrees

Git worktrees deste repositório (e de outros projetos do usuário dentro de
`HeadQuarter/`) devem ficar centralizados em `HeadQuarter/worktrees/<nome-do-
projeto>/<branch>/`, não espalhados dentro da pasta de cada projeto. Isso dá
visão única de todas as worktrees ativas, independente do projeto.

## Comunicação com o usuário

- Não assumir decisões metodológicas por conta própria — perguntar quando
  houver ambiguidade real de design (sinal usado, estratégia de reset,
  escopo de um experimento, etc). Isso é uma preferência explícita do
  usuário, reforçada várias vezes ao longo do projeto.
- O trabalho irmão deve ser chamado pelo nome do autor ("Afonso"), nunca
  "código do amigo".
