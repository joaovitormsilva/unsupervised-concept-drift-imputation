# Análise individual de 3 pacientes — Cenário S1

## 1. Seleção dos pacientes

RMSE médio entre os 4 métodos (Sem detector, ADWIN, KSWIN, PH), por paciente:

| patient | rmse médio (4 métodos) |
|---|---|
| ATHKM6V | 18.71 |
| APGIB2T | 19.20 |
| A1ZJ41O | 21.30 |
| A7EM0B6 | 24.87 |
| ASFODQR | 38.27 |

Pacientes escolhidos para aprofundar (cobrindo melhor caso, caso típico e pior caso):

- **ATHKM6V** — menor RMSE médio (melhor caso).
- **A1ZJ41O** — RMSE médio na mediana dos 5 pacientes (caso típico).
- **ASFODQR** — maior RMSE médio, e disparado o pior caso (mais de 2x o do melhor).

Os outros dois (APGIB2T, A7EM0B6) ficaram entre esses extremos e foram deixados de fora para manter o escopo enxuto, conforme pedido.

## 2. Tamanho da série e escala real de FC

| patient | linhas no bruto (`*_hr.csv`) | n (simulado, sem NaN) | FC min | FC max | FC média | FC desvio padrão |
|---|---|---|---|---|---|---|
| ATHKM6V | 459.588 | 459.587 | 37.00 | 170.00 | 63.30 | **9.49** |
| A1ZJ41O | 798.813 | 798.812 | 46.00 | 183.00 | 74.37 | **12.04** |
| ASFODQR | 1.426.028 | 1.426.027 | 31.00 | 220.00 | 74.56 | **23.57** |

Achado chave: ASFODQR não é só o paciente com a série mais longa (~3,1x mais linhas que ATHKM6V) — é também o paciente com a FC naturalmente mais volátil (desvio padrão 23.57, quase 2,5x o de ATHKM6V, e faixa de valores muito mais ampla: 31–220 bpm vs 37–170 bpm). Isso por si só já explica boa parte de por que o RMSE bruto de ASFODQR (30–48) é estruturalmente maior que o de ATHKM6V (14–29): não é necessariamente que o imputador funcione "pior" nesse paciente, é que os alvos que ele tenta acertar variam muito mais.

## 3. Tabelas por paciente: RMSE bruto vs. RMSE normalizado (RMSE / desvio padrão da FC real)

### ATHKM6V (melhor caso, std FC = 9.49)

| detector | rmse | n_drifts | rmse_normalizado |
|---|---|---|---|
| Sem detector (HT) | 14.994 | 0 | 1.580 |
| ADWIN | 14.592 | 3.897 | **1.538** (melhor) |
| KSWIN | 29.402 | 3.501 | 3.098 (pior, muito destoante) |
| PH | 15.832 | 7.608 | 1.668 |

### A1ZJ41O (caso típico, std FC = 12.04)

| detector | rmse | n_drifts | rmse_normalizado |
|---|---|---|---|
| Sem detector (HT) | 21.966 | 0 | 1.825 |
| ADWIN | 27.649 | 7.094 | 2.297 (pior) |
| KSWIN | 17.358 | 6.475 | **1.442** (melhor) |
| PH | 18.239 | 14.061 | 1.515 |

### ASFODQR (pior caso, std FC = 23.57)

| detector | rmse | n_drifts | rmse_normalizado |
|---|---|---|---|
| Sem detector (HT) | 34.387 | 0 | 1.459 |
| ADWIN | 38.989 | 12.319 | 1.654 |
| KSWIN | 47.629 | 11.160 | 2.020 (pior) |
| PH | 32.090 | 28.515 | **1.361** (melhor) |

## 4. Resumo dos achados

**Os 3 pacientes NÃO se comportam de forma parecida — o "detector vencedor" muda a cada paciente.** Olhando o RMSE normalizado (que corrige apenas a diferença de escala entre pacientes, mas não muda o ranking *dentro* de cada paciente, já que dividir todos os detectores de um mesmo paciente pelo mesmo desvio padrão preserva a ordem):

- No melhor caso (ATHKM6V), quem vence é o **ADWIN**, com margem mínima sobre o baseline sem detector; **KSWIN** é dramaticamente pior aqui (rmse_norm 3.10, o triplo dos demais) — parece ter tido um episódio de reset mal calibrado nesse paciente específico.
- No caso típico (A1ZJ41O), quem vence é o **KSWIN**, seguido de perto pelo **PH**; agora é o **ADWIN** que fica em último lugar, pior até que não usar detector nenhum.
- No pior caso / paciente mais volátil (ASFODQR), quem vence é o **PH**, seguido do baseline sem detector; **KSWIN** volta a ser o pior.

Ou seja: em nenhum dos 3 pacientes o mesmo detector aparece como vencedor duas vezes, e cada detector (ADWIN, KSWIN, PH) já foi tanto o melhor quanto o pior colocado dependendo do paciente. Isso é uma evidência bastante direta de que a média simples entre os 5 pacientes é instável: o "campeão" no agregado depende muito de qual paciente pesa mais na conta, e não existe um comportamento consistente entre detectores e pacientes neste cenário S1.

**A normalização por desvio padrão não muda o ranking dentro de cada paciente** (isso é esperado matematicamente — dividir pela mesma constante não reordena os valores), mas muda a leitura *entre* pacientes: o RMSE bruto de ASFODQR (32–48) parece "catastroficamente pior" que o de ATHKM6V (14–29) à primeira vista, mas quando normalizado pela volatilidade natural da FC de cada paciente, os valores ficam na mesma ordem de grandeza (rmse_norm entre 1.3 e 3.1 nos três pacientes). Isso confirma a suspeita inicial: parte relevante do motivo de ASFODQR (e, em menor grau, A7EM0B6, não aprofundado aqui) dominarem a média simples de RMSE é a escala/variabilidade absoluta da FC desse paciente, não necessariamente um desempenho pior do imputador.

**Sobre o Page-Hinkley "vencer" na média global**: aqui há um indício de por que isso acontece. PH só venceu no paciente mais volátil e com a série mais longa (ASFODQR), justamente o paciente cujo RMSE bruto tem a maior magnitude absoluta e portanto o maior peso numa média simples não ponderada. Nos outros dois pacientes (o melhor e o típico), PH ficou em posição intermediária (nem o melhor nem o pior), enquanto KSWIN e ADWIN se revezaram na liderança. Isso sugere que a vantagem aparente do PH na média geral pode estar sendo puxada desproporcionalmente por seu bom desempenho num único paciente atípico (mais volátil, mais longo), apesar do número de resets do PH ser sistematicamente de 2 a 4x maior que os de ADWIN/KSWIN em todos os pacientes — um comportamento de reset muito mais agressivo que nem sempre se traduz em RMSE pior, mas também não se traduz em vantagem consistente.

**Conclusão prática**: os resultados por paciente individual mostram alta variância de comportamento entre detectores, sem um vencedor único e estável nos 3 casos analisados (melhor, típico, pior). Isso reforça que qualquer ranking feito só pela média simples entre pacientes deve ser interpretado com cautela, e que uma análise por paciente (ou por peso/variabilidade de FC) é necessária para não tirar conclusões enviesadas por 1-2 pacientes com FC mais extrema.
