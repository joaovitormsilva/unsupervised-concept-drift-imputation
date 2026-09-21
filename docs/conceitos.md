# Conceitos-chave do projeto: RMSE e detectores de concept drift

Este documento explica, de forma didática e conectada ao seu pipeline (`Detection/eval_uns_cdd_imp.py`
e `Detection/run_batch.py`), os quatro conceitos que sustentam a avaliação do seu TCC: **RMSE**,
**ADWIN**, **KSWIN** e **Page-Hinkley**. A ideia não é substituir a teoria formal, mas dar a intuição
certa para você conseguir discutir e defender os resultados que já apareceram nos seus experimentos.

---

## 1. RMSE (Root Mean Squared Error)

### O que é

RMSE mede o quão longe, em média, as suas imputações ficaram do valor real de frequência cardíaca
(FC) que foi escondido pela simulação de ausência. No seu código, isso acontece literalmente em
`run_batch.py`:

```python
rmse = root_mean_squared_error(df_true["target"], df_true["Prediction"])
```

Ou seja: `target` é o valor de FC verdadeiro (que existia antes de você simular o "buraco" nos dados),
e `Prediction` é o valor que o modelo (`HoeffdingTreeRegressor`) imputou no lugar do dado ausente.

### Como é calculado

Para cada ponto ausente $i$, você tem o erro $e_i = y_i - \hat{y}_i$ (real menos previsto). O RMSE é:

```
RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)² )
```

Em palavras: eleva cada erro ao quadrado, tira a média desses quadrados, e depois tira a raiz
quadrada para voltar à unidade original. Como a FC é medida em **batimentos por minuto (bpm)**, o
RMSE também é expresso em **bpm** — essa é uma das vantagens do RMSE sobre o MSE (erro quadrático
médio sem a raiz): você consegue interpretar o número diretamente ("meu erro típico de imputação é
de X bpm"), em vez de olhar para uma unidade ao quadrado sem significado físico intuitivo.

### Por que é usado aqui

No seu problema, você não está fazendo classificação (drift sim/não é só um meio, não o fim) — o
objetivo final é a **qualidade da imputação**, que é uma tarefa de regressão (prever um valor
contínuo de FC). RMSE é a métrica padrão para regressão porque:

- Está na mesma escala da variável de interesse (bpm), facilitando interpretação clínica/prática.
- É diferenciável e amplamente comparável entre estudos de imputação de séries temporais.
- Penaliza mais os erros grandes que os pequenos (por causa do quadrado) — o que faz sentido aqui,
  porque um erro de imputação de 40 bpm é muito mais perigoso/enganoso num contexto de saúde do que
  quatro erros de 10 bpm cada.

### Interpretando o número

- **RMSE mais baixo** = as imputações estão, em média, mais próximas do valor real de FC → melhor
  qualidade de imputação.
- **RMSE mais alto** = o modelo está "chutando" valores mais distantes da FC real, seja porque não
  teve tempo de aprender o padrão do paciente, seja porque um reset (drift detectado) o jogou de
  volta à estaca zero bem na hora errada.

### Limitação importante para o seu TCC

Duas armadilhas valem a pena registrar explicitamente:

1. **Sensibilidade a erros grandes (efeito do quadrado).** Como o erro é elevado ao quadrado antes
   de tirar a média, um único ponto com erro grande (por exemplo, o modelo imputa 60 bpm onde o real
   era 140 bpm, num momento de exercício físico) pesa muito mais no RMSE final do que vários pontos
   com erro pequeno. Isso significa que o RMSE de um detector pode "parecer ruim" principalmente por
   causa de poucos picos de erro, não porque ele erra sistematicamente todo o tempo.

2. **Comparar RMSE médio entre pacientes com escalas de FC diferentes pode inverter conclusões.**
   Cada paciente tem sua própria faixa fisiológica de FC (repouso, esforço, variabilidade natural).
   Se o Paciente A tem FC variando pouco (ex.: 60–90 bpm) e o Paciente B tem FC muito mais volátil
   (ex.: 50–160 bpm, por ser mais ativo), o RMSE "natural" de qualquer modelo tende a ser maior para
   o Paciente B só pela escala/variância dos dados — não porque a imputação seja pior. Quando você
   tira a **média do RMSE entre vários pacientes**, um paciente com FC muito volátil pode dominar a
   média e mudar o ranking dos detectores. **Isso é exatamente o que já foi observado no projeto**:
   com um único paciente, um detector parecia o pior; ao agregar a média de RMSE de cinco pacientes,
   o ranking se inverteu. Isso não é necessariamente um erro de implementação — é um efeito conhecido
   de agregar métricas em escalas diferentes. Para o TCC, vale reportar tanto a média quanto a
   dispersão (ou o RMSE por paciente individualmente), e considerar normalizar o erro (por exemplo,
   RMSE relativo ao desvio-padrão de FC do próprio paciente) como análise complementar.

---

## 2. ADWIN (Adaptive Windowing)

### Intuição

Pense no ADWIN como alguém que mantém uma "memória recente" da FC observada — uma janela de valores —
e que está constantemente cortando essa janela ao meio para comparar o passado próximo com o passado
mais distante. Se a média dos dois pedaços for "estatisticamente diferente demais", o ADWIN decide
que houve uma mudança (drift) e descarta a parte mais antiga da janela.

Mais tecnicamente: o ADWIN mantém uma janela de tamanho variável (por isso "adaptativa") com todas as
observações mais recentes. A cada novo valor, ele testa várias formas de dividir a janela em duas
sub-janelas ($W_0$, mais antiga, e $W_1$, mais recente) e verifica se a diferença entre as médias das
duas é grande o suficiente, dado o tamanho das janelas e um nível de confiança, para não ser só
ruído estatístico. Se for, ele "corta" a janela: descarta $W_0$ e mantém só $W_1$ — como se dissesse
"o que eu vi antes já não representa mais o presente, vou esquecer aquilo".

### Parâmetros principais

- **`delta`** (no `river`, `drift.ADWIN(delta=0.002)` por padrão): é o parâmetro de confiança
  estatística. Ele controla o quão "rigoroso" o teste é antes de declarar drift. Um `delta` menor
  significa que o ADWIN exige mais evidência (menos falsos positivos, mas reage mais devagar a
  mudanças reais); um `delta` maior o torna mais sensível (detecta drift mais rápido, mas com mais
  risco de disparar por flutuação normal do sinal). No seu `run_batch.py`, o ADWIN é instanciado com
  os parâmetros padrão do `river` (`drift.ADWIN()`), então está usando `delta=0.002`.

### Por que é "não supervisionado" aqui

Em muitos usos de detecção de drift, o detector observa o **erro de predição** de um modelo
supervisionado (por exemplo, "o modelo errou muito nas últimas N previsões, algo mudou"). Isso exige
rótulo/verdade disponível em tempo real para calcular o erro.

No seu pipeline, o ADWIN é alimentado diretamente com o **valor observado de FC** (`cdd_current.update(yi)`,
onde `yi` é o próprio valor de FC medido, não um erro de predição). Isso significa que o detector está
monitorando mudanças na **distribuição/média do próprio sinal fisiológico**, sem depender de saber se
o modelo de imputação acertou ou errou. É exatamente esse desacoplamento entre "o que está sendo
monitorado" (o sinal bruto) e "o que está sendo avaliado depois" (a qualidade da imputação, via RMSE)
que caracteriza a abordagem como detecção **não supervisionada** de drift: você não precisa de um
modelo de referência nem do valor real por trás do "buraco" para decidir que algo mudou — só precisa
observar o comportamento do próprio dado de FC ao longo do tempo.

---

## 3. KSWIN (Kolmogorov-Smirnov Windowing)

### Intuição

Enquanto o ADWIN essencialmente compara **médias** entre duas janelas, o KSWIN vai um passo além: ele
compara as **distribuições inteiras** de duas janelas do stream usando o teste estatístico de
Kolmogorov-Smirnov (KS). O teste KS mede a maior distância vertical entre as funções de distribuição
acumulada (CDF) de duas amostras — ou seja, não olha só "a média mudou?", mas "a forma da distribuição
mudou?" (variância, assimetria, presença de novos padrões, etc.).

Na prática: o KSWIN mantém uma janela deslizante com as observações mais recentes. Periodicamente, ele
separa um pedaço menor e mais recente dessa janela (`stat_size`) e compara com o restante (a parte
mais antiga da janela, `window_size - stat_size`), rodando o teste KS entre as duas amostras. Se a
estatística do teste ultrapassar um limiar baseado em `alpha`, ele sinaliza drift.

### Parâmetros principais

- **`alpha`** (padrão `0.005` no `river`): o nível de significância do teste KS — quanto menor, mais
  evidência estatística é exigida para declarar drift (menos sensível); quanto maior, mais fácil
  disparar.
- **`window_size`** (padrão `100`): o tamanho total da janela de observações recentes mantida em
  memória.
- **`stat_size`** (padrão `30`): o tamanho da sub-amostra mais recente que é comparada contra o
  restante da janela no teste KS. Precisa ser menor que `window_size`.

No `run_batch.py`, o KSWIN também é instanciado com os defaults (`drift.KSWIN()`).

### Por que é sensível a mudanças de distribuição, não só de média

Imagine que a FC de um paciente passa de "estável em torno de 70 bpm" para "oscilando bastante entre
60 e 100 bpm, mas ainda com média em torno de 75 bpm". Um detector baseado só em média (como o ADWIN,
de forma simplificada) pode demorar a perceber essa mudança, porque a média não se moveu muito. Já o
KSWIN, ao comparar as distribuições completas via teste KS, consegue captar que a **forma** da
distribuição mudou (maior variância, por exemplo), mesmo que a média esteja parecida. Isso é
particularmente relevante para FC, que pode mudar de regime (repouso → atividade física leve →
esforço) de formas que alteram mais a variabilidade do sinal do que necessariamente seu nível médio
no curto prazo.

---

## 4. Page-Hinkley (PH)

### Intuição

O Page-Hinkley é o mais "impaciente" dos três detectores usados no projeto — e isso é visível nos
seus resultados. A ideia é acumular, ponto a ponto, o quanto cada nova observação se desvia da média
observada até então (com uma pequena tolerância `delta`), e ir somando esse desvio ao longo do tempo
(um **CUSUM**, soma cumulativa). Quando essa soma acumulada cresce demais em relação ao seu próprio
mínimo histórico, o PH interpreta isso como evidência de que a média do processo mudou de patamar.

Formalmente, o PH mantém uma média móvel $\bar{x}_T$ das observações e calcula, a cada novo ponto
$x_T$:

```
m_T = Σ (x_t − x̄_t − delta),  para t = 1..T
```

e compara com o mínimo acumulado até agora:

```
M_T = min(m_1, m_2, ..., m_T)

PH_T = m_T − M_T
```

Se `PH_T > threshold`, o detector dispara drift. Intuitivamente: `m_T` só cresce quando os valores
recentes estão consistentemente **acima** da média histórica (mais o termo de tolerância `delta`);
`M_T` guarda o "menor acumulado já visto"; e a diferença entre os dois mede o quão longe o processo
já se afastou desse ponto de referência. É um detector desenhado para achar **mudanças de nível
sustentadas**, e ele é bem mais simples/leve computacionalmente que ADWIN e KSWIN — o que também
contribui para ele reagir com mais facilidade.

### Parâmetros principais (defaults do `river.drift.PageHinkley()`)

- **`delta`** (padrão `0.005`): a margem de tolerância — o quanto uma observação pode se desviar da
  média sem contar como "evidência de mudança". Quanto menor, mais sensível.
- **`threshold`** (padrão `50`): o limiar que a soma acumulada `PH_T` precisa ultrapassar para disparar
  o drift. Quanto menor, mais fácil disparar.
- **`min_instances`** (padrão `30`): número mínimo de observações que precisam ter sido vistas antes
  de o detector começar a poder disparar (evita disparos espúrios logo no início, quando a média ainda
  está sendo estabelecida).

### Por que ele dispara tanto mais que os outros

No lote de 5 pacientes que você já rodou, o PH disparou em média **13.161 drifts**, contra **6.116**
do ADWIN e **5.486** do KSWIN — mais que o dobro. Isso é consistente com a natureza do algoritmo:

- O PH acumula qualquer desvio sustentado acima da média (mesmo pequeno) — ele não exige que a
  diferença entre duas janelas inteiras seja estatisticamente robusta como o ADWIN ou o KSWIN fazem.
  Basta uma sequência de valores um pouco acima do "normal" (por exemplo, um paciente que fica mais
  tempo com FC levemente elevada) para a soma cumulativa ultrapassar o `threshold=50`.
- Com os parâmetros padrão do `river`, o PH também é conhecido por ser mais sensível a variações no
  próprio ritmo natural da FC (que sobe e desce constantemente ao longo do dia), interpretando parte
  dessa variabilidade normal como mudança de regime.
- Diferente do ADWIN (que reavalia várias formas de dividir a janela com garantias estatísticas mais
  fortes) e do KSWIN (que exige uma diferença de distribuição detectável pelo teste KS num tamanho de
  amostra fixo), o PH é conceitualmente mais simples e "gatilho fácil" por padrão — o que na prática
  se traduz em muito mais resets do modelo de imputação.

---

## 5. Como tudo se encaixa no pipeline — e o trade-off que explica seus resultados

O laço principal do seu código (`eval_uns_cdd_imp.py`, dentro do loop de teste) resume o fluxo:

```python
if cdd_current is not None:
    cdd_current.update(yi)              # detector observa a FC real (yi)
    if cdd_current.drift_detected:      # detector aponta drift
        cdd_current = deepcopy(cdd)     # detector é reiniciado do zero
        model_current = model.clone()   # modelo de imputação é reiniciado do zero
        drifts.append(i)
model_current.learn_one(xi, yi)         # modelo (novo ou não) aprende com o ponto atual
```

Ou seja, o encadeamento é: **detector observa a FC → aponta drift → reseta o detector e o
imputador → o imputador reaprende do zero a partir daquele ponto em diante.** O `HoeffdingTreeRegressor`
resetado perde toda a árvore de decisão que havia construído até ali (todos os padrões aprendidos
sobre horário do dia, tendência da FC, etc.) e começa a aprender de novo, ponto a ponto, com um
modelo "virgem".

### O trade-off central

Cada detector (ADWIN, KSWIN, PH) tem, no fundo, um "botão de sensibilidade" (`delta`/`alpha`/`threshold`)
que define um equilíbrio entre dois erros opostos:

- **Detector pouco sensível** (reage devagar): quando o padrão de FC realmente muda (ex.: paciente
  dorme, depois acorda e faz exercício), o modelo demora a perceber e continua tentando imputar com
  base num padrão antigo que não vale mais → RMSE alto durante a "mudança real" que não foi capturada.
- **Detector muito sensível** (reage rápido, como o PH): qualquer flutuação natural da FC (que é um
  sinal ruidoso por natureza) já é tratada como "mudança real", disparando resets constantes. Cada
  reset joga fora tudo que o modelo já tinha aprendido, forçando-o a reconstruir conhecimento do zero
  repetidamente — o que teoricamente deveria prejudicar o desempenho, porque o modelo nunca tem tempo
  de "amadurecer".

### O padrão contraintuitivo observado no seu TCC

Intuitivamente, seria de se esperar que resetar demais (como o PH faz) fosse sempre pior — afinal, o
modelo perde conhecimento acumulado com frequência. E foi isso que você observou **com um único
paciente**: o PH era o pior detector.

Mas, ao rodar o lote com **cinco pacientes**, o resultado se inverteu: o PH passou a ser o **melhor**
detector, apesar de continuar disparando muito mais resets (13.161 contra 6.116 do ADWIN e 5.486 do
KSWIN). Algumas hipóteses plausíveis para investigar no seu TCC (não são conclusões definitivas, mas
linhas de raciocínio válidas para a discussão):

1. **Heterogeneidade entre pacientes.** Como você mesmo notou na seção de RMSE, a média entre
   pacientes pode inverter rankings observados individualmente. É possível que, para a maioria dos
   cinco pacientes (ou para os que têm FC mais instável/com mudanças de regime mais frequentes —
   repouso/atividade), resetar com mais frequência seja realmente vantajoso porque o "conhecimento
   antigo" descartado era mesmo obsoleto com mais frequência do que no paciente testado isoladamente.
2. **`HoeffdingTreeRegressor` recomeça rápido.** Modelos incrementais como a Hoeffding Tree conseguem
   reconstruir uma estrutura útil com relativamente poucos exemplos, especialmente se o padrão de FC
   por horário do dia (suas features são `hour` e `minute`) se repete de forma parecida todos os dias.
   Se o "custo" de resetar é baixo (o modelo se recupera rápido), a vantagem de "esquecer" um padrão
   desatualizado rapidamente pode superar o custo do reset — principalmente em pacientes com FC mais
   dinâmica.
3. **O paciente único testado pode não ser representativo.** Um único caso pode ter uma dinâmica de FC
   mais estável, onde resets frequentes do PH realmente atrapalham (porque não havia muita mudança real
   para justificar tantos resets). Isso reforça a importância de reportar resultados por paciente
   individualmente, além da média agregada, e de discutir explicitamente essa não-generalização de
   conclusões tiradas de N=1 para o conjunto completo — um ponto metodológico relevante para a seção de
   limitações do TCC.

Em suma: **não existe um detector universalmente "mais sensível é melhor" ou "mais sensível é pior"** —
o resultado depende de quão bem a frequência de reset do detector está alinhada com a frequência real
de mudanças de regime na FC de cada paciente. Esse é, provavelmente, um dos achados mais interessantes
para a discussão do seu TCC: a superioridade de um detector depende do perfil de variabilidade da
população estudada, não apenas de sua sensibilidade "em teoria".
