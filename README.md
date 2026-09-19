# Avaliação de métodos não supervisionados de concept drift para imputação online em fluxos de wearables

TCC que avalia se detectores **não supervisionados** de concept drift (ADWIN, KSWIN, Page-Hinkley aplicados diretamente sobre a série observada, sem usar o erro de um modelo supervisionado) melhoram a imputação online de valores ausentes em séries de frequência cardíaca de wearables, sob cenários com mudança de mecanismo de ausência (MCAR/MAR/MNAR).

Trabalho irmão/precursor (abordagem supervisionada, mesma base de dados): https://github.com/afonsoMatheus/supervised-concept-drift-analysis

## Fonte dos dados

Dataset original: Mishra et al. 2020, "Pre-symptomatic detection of COVID-19 from smartwatch data", *Nature Biomedical Engineering* (DOI: 10.1038/s41551-020-00640-6).

- Dados brutos (FC, passos, sono, de-identificados): https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip
- Código dos algoritmos originais do artigo: https://github.com/mwgrassgreen/WearableDetection (RHR-Diff, CuSum) e https://github.com/gireeshkbogu/AnomalyDetect (HROS-AD)

## Estrutura

```
data/
  raw/                  dados brutos por paciente baixados do bucket acima (não versionado)
  prepared/             saídas já prontas da etapa de simulação de ausência (reaproveitadas do
                         projeto irmão): patients.csv (30 pacientes elegíveis) e
                         summary_S1_30.csv / S2 / S3 (índices de corte dos segmentos MAR/MCAR/MNAR
                         para os cenários de 3, 4 e 5 segmentos, mr=30%)

Detection/               pipeline principal: aplica os detectores não supervisionados sobre o
                         fluxo observado e conduz a imputação online (test-then-train, estilo river)
  Analysis/               notebooks e plots de avaliação final (RMSE, CD diagrams)
  Parameters/             análise de sensibilidade de hiperparâmetros por detector (ADWIN/KS/PH)

notebooks/                exploração ad-hoc
```

## Pipeline planejado

1. **Carregamento**: ler `data/prepared/patients.csv` (lista de pacientes) e `summary_S{1,2,3}_30.csv`
   (índices de corte por cenário) para saber como fatiar cada série bruta em segmentos.
2. **Dados brutos**: localizar em `data/raw/` a série de FC por paciente (`file_id` dos summary_S*),
   já com a ausência simulada aplicada pelo pipeline de "Missing Simulation" do projeto irmão.
3. **Detecção não supervisionada**: para cada paciente/cenário, rodar ADWIN, KSWIN e Page-Hinkley
   monitorando diretamente os valores observados de FC (sem depender do erro de um modelo preditivo),
   disparando reset/retreino do imputador quando um drift é sinalizado.
4. **Imputação online**: loop `predict_one`/`learn_one` (river) — mesmo padrão test-then-train do
   projeto irmão — com um regressor online (ex. HoeffdingTreeRegressor) como imputador.
5. **Avaliação**: RMSE por paciente/cenário/detector, comparando contra os baselines `Mean` e `HT`
   sem detecção de drift, e (se fizer sentido) contra os resultados do artigo supervisionado irmão.
6. **Análise**: hiperparâmetros por detector em `Detection/Parameters/`, resultados finais e
   comparação estatística (CD diagrams) em `Detection/Analysis/`.
