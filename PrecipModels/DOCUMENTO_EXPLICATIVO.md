# Documento Explicativo — Framework PrecipModels

> **Público-alvo:** Pessoas sem experiência prévia em IA generativa.
> **Objetivo:** Explicar, do zero, a teoria e o código por trás de cada arquivo do projeto.
> **Idioma:** Português

---

## Índice

1. [Introdução Geral](#1-introdução-geral)
2. [Fundamentos Matemáticos](#2-fundamentos-matemáticos)
3. [data_utils.py — Carregamento dos Dados](#3-data_utilspy--carregamento-e-preparação-dos-dados)
4. [base_model.py — O Contrato de Interface](#4-base_modelpy--o-contrato-de-interface)
5. [metrics.py — Como Avaliamos Qualidade](#5-metricspy--como-avaliamos-qualidade)
6. [train.py — O Loop de Treinamento](#6-trainpy--o-loop-de-treinamento)
7. [Os Modelos — do mais simples ao mais complexo](#7-os-modelos)
   - [7.1 copula.py — Baseline Estatístico](#71-copulapy--baseline-estatístico-cópula-gaussiana)
   - [7.2 hurdle_simple.py — Melhor Modelo](#72-hurdle_simplepy--modelo-hurdle-melhor-do-projeto)
   - [7.3 vae.py — Autoencoder Variacional](#73-vaepy--autoencoder-variacional-vae)
   - [7.4 hurdle_vae.py — Dois VAEs Separados](#74-hurdle_vaepy--dois-vaes-separados)
   - [7.5 real_nvp.py — Fluxo Normalizante](#75-real_nvppy--fluxo-normalizante-realnvp)
   - [7.6 flow_match.py — Flow Matching](#76-flow_matchpy--flow-matching-trajetórias-retas)
   - [7.7 ldm.py — Latent Diffusion Model](#77-ldmpy--latent-diffusion-model)
   - [7.8 hurdle_temporal.py — Hurdle com GRU](#78-hurdle_temporalpy--hurdle-com-contexto-temporal-gru)
   - [7.9 latent_flow.py — Flow Matching com Transformer](#79-latent_flowpy--flow-matching-com-transformer)
   - [7.10 hurdle_flow.py — Hurdle + Fluxo Normalizante Condicional](#710-hurdle_flowpy--hurdle--fluxo-normalizante-condicional)
   - [7.11 hurdle_vae_cond.py — CVAE com Máscara de Ocorrência](#711-hurdle_vae_condpy--cvae-com-máscara-de-ocorrência)
   - [7.12 hurdle_vae_cond_nll.py — CVAE + NLL Log-Normal](#712-hurdle_vae_cond_nllpy--cvae--nll-log-normal)
   - [7.13 flow_match_film.py — Flow Matching com FiLM](#713-flow_match_filmpy--flow-matching-com-film)
   - [7.14 glow.py — GLOW (Fluxo com Mixing Completo)](#714-glowpy--glow-fluxo-normalizante-com-mixing-completo)
8. [Condicionamento por Mês (Modelos _mc)](#8-condicionamento-por-mês-modelos-_mc)
   - [8.0 Por que condicionar no mês?](#80-por-que-condicionar-no-mês-do-ano)
   - [8.1 conditioning.py — Infraestrutura Compartilhada](#81-conditioningpy--infraestrutura-compartilhada)
   - [8.2 hurdle_simple_mc.py — HurdleSimpleMC](#82-hurdle_simple_mcpy--hurdlesimplemc)
   - [8.3 vae_mc.py — VAEModelMC](#83-vae_mcpy--vaemodelmc)
   - [8.4 real_nvp_mc.py — RealNVPMC](#84-real_nvp_mcpy--realnvpmc)
   - [8.5 glow_mc.py — GlowMC](#85-glow_mcpy--glowmc)
   - [8.6 flow_match_mc.py — FlowMatchingMC](#86-flow_match_mcpy--flowmatchingmc)
   - [8.7 flow_match_film_mc.py — FlowMatchingFilmMC](#87-flow_match_film_mcpy--flowmatchingfilmmc)
9. [compare.py — Pipeline de Comparação](#9-comparepy--a-pipeline-de-comparação)
10. [Exemplos de Uso Completos](#10-exemplos-de-uso-completos)
11. [Resultados e Conclusões](#11-resultados-e-conclusões)

---

## 1. Introdução Geral

### O que é precipitação e por que modelá-la é difícil?

Precipitação é chuva. Simples assim. Mas modelar matematicamente a quantidade de chuva que vai cair amanhã (ou gerar dados sintéticos de chuva que sejam realistas) é um dos problemas mais difíceis da hidrometeorolgia.

Por que é difícil?

1. **Massa pontual em zero:** Na maioria dos dias, não chove. Isso cria uma distribuição que tem uma "pilha" enorme no zero — não é uma curva contínua normal.
2. **Cauda pesada:** Quando chove, pode chuver muito ou pouquíssimo. A distribuição não é simétrica — tem uma cauda longa para valores altos (eventos extremos).
3. **Correlação espacial:** Se está chovendo em São Paulo, é provável que também esteja chovendo em Campinas. As estações não são independentes entre si.
4. **Dependência temporal:** Se ontem choveu, hoje é mais provável que chova novamente (autocorrelação temporal).

### O que é um modelo generativo?

Imagine que você quer ensinar uma máquina a escrever textos que pareçam escritos por humanos. Você mostra milhares de textos reais, e a máquina aprende os padrões — vocabulário, gramática, estilo. Depois, ela consegue gerar textos novos que nunca existiram, mas que "parecem" humanos.

Um **modelo generativo de precipitação** faz a mesma coisa:
- Você mostra dados históricos de chuva de ~15 estações meteorológicas
- O modelo aprende os padrões estatísticos (frequência de chuva, correlação entre estações, eventos extremos...)
- Depois, o modelo gera novos dias de chuva que nunca aconteceram, mas que têm as mesmas propriedades estatísticas dos dados reais

**Analogia concreta:** Pense numa máquina de dados climáticos. Você alimenta com 10 anos de medições reais, e ela consegue gerar mais 100 anos de dados climáticos artificiais — que serão usados para planejar hidrelétricas, estimar riscos de enchente, etc.

### Visão Geral do Framework PrecipModels

O projeto implementa e compara **9 abordagens diferentes** para esse problema, desde métodos estatísticos clássicos até redes neurais modernas:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXO DO FRAMEWORK                            │
│                                                                   │
│  DADOS BRUTOS (CSV)                                               │
│       ↓                                                           │
│  data_utils.py    → limpa, normaliza, divide treino/teste         │
│       ↓                                                           │
│  train.py         → loop de treinamento para cada modelo          │
│       ├── copula.py      (ajuste analítico, sem gradiente)         │
│       ├── hurdle_simple.py (MLP + Log-Normal + Cópula)            │
│       ├── vae.py          (Autoencoder Variacional)               │
│       ├── hurdle_vae.py   (dois VAEs)                             │
│       ├── real_nvp.py     (Fluxo Normalizante)                    │
│       ├── flow_match.py   (Flow Matching simples)                 │
│       ├── ldm.py          (Latent Diffusion Model)                │
│       ├── hurdle_temporal.py (Hurdle + GRU temporal)              │
│       └── latent_flow.py  (Flow Matching + Transformer)           │
│       ↓                                                           │
│  metrics.py       → avalia qualidade das amostras geradas         │
│       ↓                                                           │
│  compare.py       → compara todos os modelos, gera gráficos       │
│       ↓                                                           │
│  outputs/         → model.pt, config.json, metrics.json, plots    │
└─────────────────────────────────────────────────────────────────┘
```

### Dados Utilizados

Os dados vêm do INMET (Instituto Nacional de Meteorologia), com precipitação diária de aproximadamente 15 estações meteorológicas brasileiras. Cada linha do CSV representa um dia, cada coluna representa uma estação.

---

## 2. Fundamentos Matemáticos

Esta seção explica os conceitos matemáticos usados no projeto de forma acessível, com analogias antes das fórmulas.

### 2.1 O que é uma distribuição de probabilidade?

Imagine que você joga um dado 1000 vezes e anota os resultados. Se você fizer um histograma (gráfico de barras com frequências), vai ver que cada face aparece ~1/6 do tempo. Isso é uma **distribuição de probabilidade uniforme**.

Agora imagine medir a precipitação diária durante 10 anos:
- Na maioria dos dias (digamos 70%): chuva = 0 mm
- Alguns dias: chuva entre 0.1 e 10 mm
- Poucos dias: chuva entre 10 e 50 mm
- Rarissimamente: chuva > 100 mm

Esse perfil de frequências é a **distribuição de probabilidade da precipitação**. O modelo generativo tenta aprender exatamente esse perfil.

### 2.2 Por que precipitação não é Gaussiana (curva em sino)?

A distribuição Normal (Gaussiana, "curva em sino") é simétrica — valores abaixo da média são tão comuns quanto acima. Mas precipitação é fundamentalmente assimétrica:
- Há um limite inferior em **zero** (não existe chuva negativa)
- Há dias com **zero chuva** (massa pontual — a distribuição não é contínua)
- A cauda direita é **pesada** (eventos extremos são mais comuns do que uma Gaussiana preveria)

Por isso, usamos a **distribuição Log-Normal**: se X é Log-Normal, então log(X) é Normal. Aplicar o logaritmo "achata" a cauda pesada e torna os dados mais simétricos.

**Fórmula:** Se X ~ Log-Normal(μ, σ), então:
```
P(X = x) ∝ (1/x) · exp(-(log(x) - μ)² / (2σ²))
```

### 2.3 O que é uma Cópula?

Imagine duas cidades próximas, A e B. Sabemos que:
- A tem chuva 40% dos dias
- B tem chuva 45% dos dias
- **Mas quando chove em A, tem 85% de chance de também chover em B**

A terceira informação é a **dependência** (correlação) entre as duas. Uma **cópula** é uma ferramenta matemática que separa as distribuições individuais (marginais) da estrutura de dependência entre elas.

Em termos simples: a cópula responde "dado que aconteceu X na estação A, qual a probabilidade de acontecer Y na estação B?"

A **Cópula Gaussiana** usa a matriz de correlação da distribuição Normal multivariada para modelar essa dependência.

### 2.4 O que é KL Divergence?

**KL Divergence** (divergência de Kullback-Leibler) mede o quanto duas distribuições de probabilidade são diferentes. É como a "distância" entre elas, mas assimétrica.

**Analogia:** Imagine que P é a distribuição real da precipitação, e Q é a que seu modelo aprendeu. KL(P||Q) mede o quão diferente Q é de P — quanto "informação extra" você perde ao usar Q em vez de P.

**Fórmula simplificada:**
```
KL(P||Q) = Σ P(x) · log(P(x) / Q(x))
```

KL = 0 significa que as distribuições são idênticas. KL > 0 significa que há diferença.

### 2.5 O que é o ELBO (Evidence Lower Bound)?

No VAE, queremos maximizar a probabilidade dos dados p(x). Mas isso é matematicamente intratável diretamente. O **ELBO** é um limite inferior (lower bound) que podemos otimizar:

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
         ↑ reconstrução      ↑ regularização
```

Maximizar o ELBO é equivalente a:
- Garantir que o decoder reconstrói bem os dados (primeiro termo)
- Garantir que o espaço latente não "colapsa" para um ponto específico (segundo termo)

**Analogia:** Imagine que você quer comprimir e descomprimir uma imagem. O ELBO equilibra: (1) a qualidade da descompressão, e (2) a regularidade do código comprimido.

### 2.6 O que é um Espaço Latente?

**Espaço latente** é uma representação comprimida dos dados em dimensão menor. Pense assim:

- Seus dados têm 15 dimensões (15 estações)
- O VAE comprime para 128 dimensões latentes
- Cada ponto no espaço latente representa um "padrão climático" abstrato
- Para gerar dados novos: você sorteia um ponto no espaço latente (z ~ N(0,I)) e o decodifica

**Vantagem:** Operar no espaço latente é como trabalhar em uma linguagem mais concisa — o modelo não precisa aprender detalhes irrelevantes, só os padrões essenciais.

---

## 3. data_utils.py — Carregamento e Preparação dos Dados

### O que este arquivo faz?

`data_utils.py` é responsável por transformar o arquivo CSV bruto de medições meteorológicas em arrays NumPy prontos para serem consumidos pelos modelos.

### Fonte dos dados

Os dados vêm de duas fontes:
- **INMET** (`inmet_relevant_data.csv`): Dados de barragens BTG com estações do INMET
- **SABESP** (`dayprecip.dat`): Dados de precipitação diária da SABESP

O CSV contém:
- `cd_inmet_station`: código da estação
- `dt`: data (YYYY-MM-DD)
- `hr`: hora (formato HH:MM)
- `precipitacao`: precipitação em mm (vírgula como separador decimal, -9999 para dados faltantes)

### O que é normalização e por que fazemos?

As redes neurais aprendem melhor quando os dados estão em escalas parecidas. Se uma variável vai de 0 a 1000 e outra de 0 a 0.001, a rede vai ter dificuldade. Normalização coloca tudo na mesma escala.

O projeto oferece dois modos:

**`scale_only`** (padrão para modelos hurdle):
```python
data_norm = data_raw / std
# mu = zeros  →  não subtrai a média
```

**Por que usar scale_only?** Modelos hurdle precisam que o zero continue sendo zero (dia seco). Se subtrairmos a média, o zero deixaria de ser zero e a máscara de dias secos ficaria errada.

**`standardize`** (padrão para VAE, RealNVP, Flow Matching):
```python
data_norm = (data_raw - mu) / std
# z-score completo
```

**Por que usar standardize para VAE?** O VAE opera no espaço contínuo completo — não precisa preservar o zero. Dados centralizados em zero ajudam o encoder a aprender melhor.

### Função `load_data()` em detalhe

```python
def load_data(
    data_path: str = SABESP_DATA_PATH,
    normalization_mode: str = "scale_only",
    dropna: bool = True,
    missing_strategy: str = "drop_any",
):
```

**Parâmetros:**
- `data_path`: caminho para o CSV
- `normalization_mode`: `"scale_only"` ou `"standardize"`
- `dropna`: se True, aplica tratamento de dados faltantes
- `missing_strategy`:
  - `"drop_any"`: remove qualquer linha com NaN (comportamento conservador)
  - `"impute_station_median"`: imputa valores faltantes pela mediana da estação (perde menos dados)

**Retornos:**
```python
data_norm,      # (N, S) array normalizado — entrada para o modelo
data_raw,       # (N, S) array em mm/dia — para calcular métricas
mu,             # (1, S) média subtraída (zeros se scale_only)
std,            # (1, S) desvio padrão usado na normalização
station_names   # lista de nomes das estações
```

**Exemplo de valores reais:**
```
Dados carregados: (3287, 15) | estações: 15 | modo: scale_only
→ N = 3287 dias (~9 anos)
→ S = 15 estações
→ std médio ≈ 8.5 mm/dia
```

### Função `temporal_holdout_split()`

Explicada em detalhe na seção 6 (train.py), mas o conceito central é:

```python
# ❌ ERRADO — split aleatório vaza informação do futuro para o treino
indices = shuffle(range(N))
train_idx = indices[:int(0.8*N)]
eval_idx  = indices[int(0.8*N):]

# ✓ CORRETO — split temporal preserva a ordem cronológica
train_data = data[:int(0.8*N)]  # primeiros 80% dos dias
eval_data  = data[int(0.8*N):]  # últimos 20% dos dias
```

**Analogia:** Para prever o futuro, você treina o modelo no passado e testa no futuro. Se embaralhar os dados, o modelo "vê" o futuro durante o treino — é como estudar para uma prova olhando as respostas.

### Função `denormalize()`

```python
def denormalize(data_norm, mu, std):
    return data_norm * std + mu
```

Simples: inverte a normalização para recuperar mm/dia.

### Exemplo de uso prático

```python
from data_utils import load_data, temporal_holdout_split

# Carrega os dados
data_norm, data_raw, mu, std, station_names = load_data(
    normalization_mode="scale_only"
)
print(f"Forma dos dados: {data_norm.shape}")  # ex: (3287, 15)
print(f"Estações: {station_names}")

# Divide em treino e avaliação
train_raw, eval_raw = temporal_holdout_split(data_raw, holdout_ratio=0.2)
print(f"Treino: {train_raw.shape}, Avaliação: {eval_raw.shape}")
# Treino: (2629, 15), Avaliação: (658, 15)
```

---

## 4. base_model.py — O Contrato de Interface

### O que é uma classe abstrata?

Uma **classe abstrata** define um "contrato" — ela diz "todo modelo DEVE implementar esses métodos". É como uma especificação técnica que garante que todos os modelos são intercambiáveis.

**Analogia:** Imagine que você tem uma tomada elétrica de padrão universal. Todo aparelho que quiser se conectar à tomada DEVE ter o conector certo — não importa se é um ventilador, geladeira ou televisão. A classe abstrata é essa "tomada universal" para modelos.

### A classe BaseModel

```python
class BaseModel(ABC, nn.Module):
    """
    Classe base para todos os modelos de precipitação.
    """

    @abstractmethod
    def sample(self, n, steps=None, method=None) -> Tensor:
        """Gera n amostras sintéticas."""
        ...

    @abstractmethod
    def loss(self, x, beta=1.0) -> dict:
        """Calcula a loss para um batch."""
        ...

    def count_parameters(self) -> int:
        """Conta parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, data, **kwargs):
        """Ajuste analítico (apenas para modelos estatísticos como Cópula)."""
        pass
```

### Método `sample(n)`

Gera `n` dias sintéticos de precipitação. Todos os modelos retornam um tensor de shape `(n, S)` onde `S` é o número de estações.

**Parâmetros opcionais:**
- `steps`: número de passos de integração (para Flow Matching, LDM)
- `method`: solver a usar (ex: `'euler'`, `'heun'`, `'ddpm'`, `'ddim'`)

### Método `loss(x, beta)`

Calcula a perda de treinamento para um batch de dados. Retorna sempre um dicionário com pelo menos a chave `'total'` (escalar Tensor com gradiente).

**Por que retornar um dicionário?** Para facilitar o logging de sub-losses (ex: reconstrução separada do KL no VAE).

```python
# Exemplo de retorno do VAE:
{'total': tensor(1.234), 'recons': tensor(0.8), 'kl': tensor(0.434)}

# Exemplo de retorno do Hurdle:
{'total': tensor(0.567), 'bce': tensor(0.3), 'nll': tensor(0.267)}
```

**O parâmetro `beta`:** Controla o peso do termo KL nos VAEs. Começa em 0 e vai crescendo até 1 (KL annealing — explicado em detalhe na seção 6).

### Por que essa abstração é poderosa?

O `train.py` nunca precisa saber qual modelo está treinando:

```python
# Este código funciona para QUALQUER modelo
loss_dict = model.loss(batch, beta=beta)  # chama o método correto
loss_dict['total'].backward()             # gradiente flui automaticamente
```

Se você adicionar um modelo novo (implementando `sample` e `loss`), o `train.py` funciona automaticamente — sem nenhuma mudança.

---

## 5. metrics.py — Como Avaliamos Qualidade

### O que é "qualidade" num modelo generativo?

Para modelos de classificação, usamos acurácia: "o modelo acertou ou errou?". Mas para modelos generativos, não existe uma resposta "certa" — o modelo gera dados sintéticos e queremos saber se eles têm as mesmas propriedades estatísticas dos dados reais.

**Analogia:** Imagine que você pediu para uma IA pintar quadros no estilo Van Gogh. Como avaliar se é bom? Não existe "a pintura certa" — você compara: as cores têm a mesma distribuição? As pinceladas têm o mesmo padrão? A proporção de azul para amarelo é similar?

No projeto, fazemos o mesmo com precipitação: comparamos as propriedades estatísticas dos dados reais com as dos dados gerados.

### Threshold de dia úmido

```python
WET_DAY_THRESHOLD_MM = 0.1  # mm
```

Um dia é considerado "chuvoso" se precipitação > 0.1 mm. Esse limiar evita que gotinhas de orvalho sejam contadas como chuva.

### Métrica 1: Distância Wasserstein

**Analogia intuitiva:** Imagine dois montes de terra (histogramas) — um representando a distribuição real da chuva, outro a gerada pelo modelo. A distância Wasserstein é o custo mínimo de "mover" a terra de um monte para formar o outro. É como medir o trabalho necessário para transformar uma distribuição na outra.

**Por que é melhor que a diferença simples?** A Wasserstein respeita a geometria — mover terra para um local próximo custa menos do que para um local distante.

```python
def wasserstein_per_station(real, generated):
    """
    Calcula Wasserstein-1 por estação.
    real:      (N_real, S) em mm/dia
    generated: (N_gen, S) em mm/dia
    """
    for i in range(S):
        w[i] = stats.wasserstein_distance(real[:, i], generated[:, i])
    return {'per_station': w, 'mean': float(w.mean())}
```

**Interpretação:** Se `mean_wasserstein = 1.6`, significa que, em média, as distribuições de chuva por estação diferem por ~1.6 mm/dia.

### Métrica 2: Correlation RMSE

**Motivação:** A precipitação em São Paulo e Campinas é correlacionada — quando uma região tem chuva, a outra tende a ter também. Um bom modelo deve preservar essas correlações espaciais.

**Como funciona:**
1. Calcula a matriz de correlação dos dados reais (S × S)
2. Calcula a matriz de correlação dos dados gerados (S × S)
3. Compara as duas matrizes com RMSE (Root Mean Square Error)

```python
def correlation_rmse(real, generated):
    cx = np.corrcoef(real, rowvar=False)    # matriz real (S×S)
    cy = np.corrcoef(generated, rowvar=False)  # matriz gerada (S×S)
    diff = cx - cy
    return float(np.sqrt(np.nanmean(diff ** 2)))
```

**Interpretação:** RMSE = 0 significa correlações idênticas. RMSE = 0.5 significa diferenças moderadas nas correlações.

### Métrica 3: Wet Day Frequency Error

**Pergunta:** O modelo sabe que chove aproximadamente X% dos dias?

Se na realidade chove em 40% dos dias mas o modelo gera chuva em 70% dos dias, algo está errado.

```python
def wet_day_frequency_error(real, generated):
    p_real = (real > WET_DAY_THRESHOLD_MM).mean(axis=0)   # (S,) frequência real
    p_gen  = (generated > WET_DAY_THRESHOLD_MM).mean(axis=0)  # (S,) frequência gerada
    err = np.abs(p_real - p_gen)
    return {'per_station': err, 'mean': float(err.mean())}
```

**Interpretação:** `wet_day_freq_error = 0.08` significa que a frequência de dias chuvosos difere em ~8% entre dados reais e gerados.

### Métrica 4: Extreme Quantile Error

**Motivação:** Para engenharia hidráulica, o que mais importa são os eventos extremos. Uma enchente de 100 anos não acontece todo dia, mas quando acontece é catastrófica. O modelo deve saber que eventos extremos existem.

```python
def extreme_quantile_error(real, generated, quantiles=(0.90, 0.95, 0.99)):
    for q in quantiles:
        qr = np.nanquantile(real, q, axis=0)      # quantil real
        qg = np.nanquantile(generated, q, axis=0) # quantil gerado
        err = np.abs(qr - qg)                      # diferença absoluta
```

**Interpretação:** `extreme_q99_mean = 15 mm` significa que, no percentil 99%, os dados gerados diferem dos reais em ~15 mm.

### Métrica 5: Energy Score

**Motivação:** As métricas acima são **univariadas** (olham uma estação de cada vez). O Energy Score é **multivariado** — avalia se o modelo captura as relações entre TODAS as estações simultaneamente.

**Fórmula:**
```
ES = E||Y - x|| - 0.5 · E||Y - Y'||
```
onde Y, Y' são amostras da distribuição gerada e x é uma observação real.

**Intuição:** O primeiro termo mede o quão próximos os dados gerados estão dos reais. O segundo termo penaliza se o modelo gera sempre a mesma coisa (colapso — baixa diversidade).

### Métricas Temporais (Informativas)

Estas métricas não entram no composite score, mas são informativas para entender o comportamento temporal:

**`wet_spell_length_error`:** Erro no comprimento médio de sequências chuvosas. Se na realidade a chuva dura em média 2.3 dias consecutivos e o modelo gera 1.1 dias, algo está errado na modelagem temporal.

**`dry_spell_length_error`:** Mesmo conceito para períodos secos.

**`lag1_autocorr_error`:** Erro na autocorrelação lag-1. Mede se o modelo captura que "dia chuvoso tende a ser seguido por dia chuvoso".

```python
# Autocorrelação lag-1: correlação entre x[t] e x[t-1]
ac_real = float(np.corrcoef(r[:-1], r[1:])[0, 1])
ac_gen  = float(np.corrcoef(g[:-1], g[1:])[0, 1])
err = abs(ac_real - ac_gen)
```

### O Composite Score (Pontuação Composta)

`compare.py` combina todas as métricas em um único número para ranking. A lógica é:

1. Para cada métrica, colete o valor de todos os modelos
2. Normalize para [0, 1] com min-max scaling (0 = melhor, 1 = pior)
3. Tire a média das métricas normalizadas

```python
QUALITY_METRICS = [
    ("mean_wasserstein",       "Wasserstein Médio",     True),
    ("corr_rmse",              "Corr RMSE",             True),
    ("wet_day_freq_error_mean","Wet Day Freq Err",      True),
    ("extreme_q90_mean",       "Quantil 90% Err",       True),
    ("extreme_q95_mean",       "Quantil 95% Err",       True),
    ("extreme_q99_mean",       "Quantil 99% Err",       True),
    ("energy_score",           "Energy Score",          True),
]
```

**Interpretação:** Composite score = 0 seria o modelo perfeito (melhor em tudo). Composite score = 1 seria o pior em tudo. Na prática, scores abaixo de 0.3 são considerados bons.

### Exemplo: Interpretando um Relatório de Comparação

```
================================
  MÉTRICAS DE AVALIAÇÃO
================================
  Wasserstein médio:        1.603    ← diferença média de 1.6 mm/dia na distribuição
  Correlation RMSE:         0.180    ← correlações espaciais relativamente preservadas
  Wet day freq error:       0.080    ← frequência de chuva certa com ±8% de erro
  Quantil 90% erro:         2.140    ← percentil 90% erra por ~2.1 mm
  Quantil 95% erro:         3.820    ← percentil 95% erra por ~3.8 mm
  Quantil 99% erro:        12.340    ← percentil 99% erra por ~12 mm (eventos extremos mais difíceis)
  Energy score:             4.230    ← distribuição multivariada com diferença moderada
================================
```

---

## 6. train.py — O Loop de Treinamento

### O que é descida de gradiente?

**Analogia:** Imagine que você está numa montanha coberta de neblina e quer chegar ao vale mais fundo (o mínimo da função de perda). Você não consegue ver o vale diretamente, mas pode sentir o terreno ao redor. O gradiente é a "inclinação" naquele ponto — e você dá um passinho na direção mais descendente. Repita isso muitas vezes e você eventualmente chega ao vale.

Matematicamente:
```
parâmetros ← parâmetros - lr · ∇(loss)
                               ↑ gradiente da loss em relação aos parâmetros
```

`lr` é a **taxa de aprendizado** (learning rate) — o tamanho do passo. Muito grande: você "pula" o vale. Muito pequeno: demora uma eternidade.

### KL Annealing (Aquecimento Gradual do KL)

Para modelos VAE, existe um problema chamado **posterior collapse**: se o termo KL for muito forte no início do treinamento, o encoder aprende a ignorar a entrada e simplesmente mapeia tudo para N(0,I). O decoder não aprende nada útil.

A solução é começar com KL=0 e aumentar gradualmente:

```python
def get_beta(epoch: int, kl_warmup: int) -> float:
    """
    KL annealing: rampa linear 0 → 1 ao longo de kl_warmup épocas.
    """
    if kl_warmup <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_warmup)
```

**Visualização da rampa:**
```
beta
1.0 |         ┌──────────────
0.5 |      /
0.0 |─────/
    └─────────────────────── época
      0   warmup
```

Se `kl_warmup = 100`, nas primeiras 100 épocas, `beta` vai de 0 a 1 linearmente.

### Split Temporal vs Split Aleatório

```python
def temporal_holdout_split(data_raw, holdout_ratio=0.2):
    """
    Split temporal (sem embaralhar):
      - treino = parte inicial (80% padrão)
      - avaliação = parte final (20% padrão)
    """
    n_total = data_raw.shape[0]
    n_eval = int(round(n_total * holdout_ratio))
    return data_raw[:-n_eval], data_raw[-n_eval:]
```

**Por que temporal?** Séries temporais têm autocorrelação — ontem influencia hoje. Se você sortear aleatoriamente, informação do "futuro" vaza para o treino, tornando a avaliação otimista demais.

**Exemplo concreto:** Com N=3287 dias e holdout=0.2:
- Treino: dias 1 a 2629 (anos 1 a 7.2)
- Avaliação: dias 2630 a 3287 (últimos 1.8 anos)

### MODEL_DEFAULTS e ARCH_DEFAULTS

O arquivo define configurações padrão para cada modelo:

```python
MODEL_DEFAULTS = {
    "hurdle_simple": {
        "normalization_mode": "scale_only",   # preserva zero
        "max_epochs": 500,                     # número de passagens pelo dataset
        "latent_size": 0,                      # sem espaço latente
        "lr": 0.001,                           # taxa de aprendizado
        "batch_size": 128,                     # amostras por passo de gradiente
        "kl_warmup": 0,                        # sem KL (não é VAE)
    },
    "vae": {
        "normalization_mode": "standardize",
        "max_epochs": 500,
        "latent_size": 128,
        "lr": 0.0003,
        "batch_size": 128,
        "kl_warmup": 100,                      # 100 épocas de warmup do KL
    },
    "ldm": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,                    # Estágio 1 (VAE)
        "ldm_epochs": 1000,                    # Estágio 2 (DDPM)
        "lr": 0.005,
        "ldm_lr": 0.001,
        ...
    },
}

ARCH_DEFAULTS = {
    "hurdle_simple": {"hidden_occ": 32, "hidden_amt": 64},
    "real_nvp":      {"hidden_size": 256, "n_coupling": 12},
    "flow_match":    {"hidden_size": 256, "n_layers": 4, ...},
    ...
}
```

### O Loop de Treinamento Principal

```python
def train_neural_model(model, train_norm, max_epochs, lr, batch_size, kl_warmup, device, ...):

    # Prepara o DataLoader (embaralha os dados a cada época)
    t_data = torch.FloatTensor(train_norm).to(device)
    dataset = TensorDataset(t_data, t_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Otimizador Adamax (versão robusta do Adam)
    optimizer = optim.Adamax(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        model.train()

        beta = get_beta(epoch, kl_warmup)  # aquecimento do KL

        for x_batch, _ in loader:
            optimizer.zero_grad()                           # zera gradientes
            loss_dict = model.loss(x_batch, beta=beta)     # calcula loss
            loss_dict['total'].backward()                   # calcula gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabilidade
            optimizer.step()                               # atualiza parâmetros
```

**Gradient clipping** (`clip_grad_norm_`): limita o tamanho máximo do gradiente a 1.0. Sem isso, gradientes explosivos podem fazer os parâmetros "explodir" para infinito.

### Salvamento de Artefatos

Após o treinamento, três arquivos são salvos em `outputs/<nome_modelo>/`:

- **`config.json`**: todos os hiperparâmetros usados (para reprodutibilidade)
- **`model.pt`**: pesos da rede neural (ou `copula.pkl` para cópula)
- **`metrics.json`**: resultado de todas as métricas no conjunto de avaliação

```python
# Salva modelo + estado do otimizador (para retomar treinamento com --resume)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, model_path)
```

### Treinamento em Dois Estágios (LDM)

O modelo LDM é especial: precisa de dois estágios sequenciais:

```python
if model_name == "ldm":
    # Estágio 1: treina VAE (encoder + decoder)
    model.set_stage("vae")
    train_neural_model(..., max_epochs=1000, lr=0.005)

    # Estágio 2: congela VAE, treina DDPM no espaço latente
    model.set_stage("ldm")
    train_neural_model(..., max_epochs=1000, lr=0.001)
```

**Por que dois estágios?** Primeiro o VAE aprende a comprimir os dados em um espaço latente de boa qualidade. Só então o DDPM aprende a gerar amostras nesse espaço. Se treinasse tudo junto, os dois estágios interferiram negativamente.

### Exemplos de Comandos

```bash
# Treinar o melhor modelo (hurdle_simple)
python train.py --model hurdle_simple

# Treinar VAE com configuração customizada
python train.py --model vae --max_epochs 1000 --latent_size 64 --lr 0.0001

# Treinar LDM (demorado — dois estágios)
python train.py --model ldm

# Retomar treinamento de onde parou
python train.py --model vae --resume

# Usar GPU (se disponível)
python train.py --model hurdle_simple --device cuda
```

---

## 7. Os Modelos

Os modelos estão ordenados da abordagem mais simples para a mais complexa. Isso ajuda a entender como cada novo modelo tenta superar as limitações do anterior.

---

## 7.1 copula.py — Baseline Estatístico (Cópula Gaussiana)

### A Pergunta

> "O estado-da-arte estatístico clássico da hidrologia consegue bons cenários de precipitação?"

### Intuição

Antes de qualquer rede neural, a hidrologia usa cópulas gaussianas. É uma abordagem matemática elegante que não precisa de treinamento iterativo — o ajuste é analítico, calculado diretamente dos dados.

**Ideia central em três passos:**

1. **Para cada estação:** ajusta uma distribuição "mista" — massa em zero (dias secos) + Log-Normal (dias chuvosos)
2. **Entre estações:** captura a correlação usando uma cópula gaussiana (baseada na matriz de correlação das transformações normalizadas)
3. **Para gerar:** sorteia amostras gaussianas correlacionadas e converte de volta para precipitação

### Teoria: Distribuição Mista

Para cada estação i:
```
P(X_i = 0) = p_dry_i          (probabilidade de dia seco)
P(X_i | X_i > 0) = LogNormal(μ_log_i, σ_log_i)
```

**Estimação direta dos dados:**
```python
p_dry[i] = (data[:, i] == 0).mean()           # fração de dias secos
wet = data[:, i][data[:, i] > 0]              # apenas dias chuvosos
mu_log[i] = np.mean(np.log(wet + 1e-8))       # média do log
sigma_log[i] = np.std(np.log(wet + 1e-8))     # desvio do log
```

### Teoria: Normal-Score Transform (Transformada de Ranque Normal)

Para capturar correlações espaciais, usamos a transformada normal-score:

1. Para cada estação, calcule os ranques dos dados: `rank[i] / (N+1)` → uniformes em (0,1)
2. Aplique a inversa da Normal: `Φ⁻¹(rank)` → scores normais
3. Estime a correlação entre esses scores → matriz de correlação C

**Por que usar ranques?** A transformada de ranque remove a influência da distribuição marginal, capturando só a dependência. É não-paramétrica e robusta.

```python
# Normal-score transform
for i in range(S):
    ranks = stats.rankdata(data[:, i]) / (N + 1)  # uniformes
    ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
    z_scores[:, i] = stats.norm.ppf(ranks)         # scores normais

# Correlação empírica dos scores normais
corr_matrix = np.corrcoef(z_scores, rowvar=False)
```

### Amostragem Passo a Passo

```
Passo 1: z_iid ~ N(0, I)          → amostra gaussiana independente (n, S)
Passo 2: z_corr = z_iid @ L.T      → aplica Cholesky → correlacionada
Passo 3: u = Φ(z_corr)             → transforma em uniformes (0,1)
Passo 4: x = InvCDF_misto(u)       → aplica CDF inversa mista por estação
```

**Decomposição de Cholesky:** Para amostrar de MVN(0, C), usamos L tal que `C = L @ L.T`. Então `z_corr = z_iid @ L.T` tem a correlação certa.

```python
chol = cholesky(corr_matrix, lower=True)

z_iid = np.random.randn(n, S)
z_corr = z_iid @ chol.T    # (n, S) — correlacionado

u = stats.norm.cdf(z_corr) # (n, S) — uniformes marginais

# CDF inversa mista
for i in range(S):
    p0 = p_dry[i]
    wet_mask = u[:, i] > p0
    u_wet = (u[wet_mask, i] - p0) / (1.0 - p0)  # rescala para parte contínua
    log_x = stats.norm.ppf(u_wet) * sigma_log[i] + mu_log[i]
    out[wet_mask, i] = np.exp(log_x)
```

### Por que `count_parameters() = 0`?

A cópula não usa redes neurais — não há pesos para otimizar. O "ajuste" é a estimação analítica de `p_dry`, `mu_log`, `sigma_log` e `corr_matrix` diretamente dos dados. Por isso `count_parameters()` retorna 0.

```python
def count_parameters(self) -> int:
    return 0  # sem parâmetros treináveis
```

### Resultados

| Composite | Wasserstein | Wet Freq Err |
|-----------|-------------|--------------|
| 0.736     | 5.891       | 0.319        |

É o pior modelo do projeto — serve como linha de base (*baseline*) para saber se as redes neurais valem a pena.

### Quando Usar a Cópula

- Quando você quer um resultado rápido e interpretável
- Como baseline para comparar com métodos mais sofisticados
- Quando não há dados suficientes para treinar uma rede neural

### Perguntas Frequentes

**P: Por que a cópula performa pior que redes neurais?**
R: A cópula assume que a dependência espacial é Gaussiana (elíptica) e que é constante ao longo do tempo. A realidade tem dependências mais complexas e não-lineares que redes neurais capturam melhor.

**P: Se não tem gradiente, como o train.py lida com isso?**
R: O train.py detecta `model_name == "copula"` e chama `model.fit(train_raw)` em vez do loop de gradient descent. Depois salva o modelo com `pickle` em `copula.pkl` em vez de `model.pt`.

---

## 7.2 hurdle_simple.py — Modelo Hurdle (Melhor do Projeto)

### A Pergunta

> "Uma rede neural simples com distribuições paramétricas supera a cópula?"

### O que é um "Hurdle Model"?

Um hurdle model separa o problema em duas etapas:
1. **Primeira etapa (hurdle = barreira):** O dia tem chuva ou não? (variável binária)
2. **Segunda etapa:** Dado que há chuva, quanto choveu? (variável contínua positiva)

**Analogia:** Para modelar o consumo de álcool de uma pessoa:
- Primeiro: ela bebe? (0% não bebe, 100% bebe às vezes)
- Segundo, dado que bebe: quanto ela bebe? (distribuição contínua)

Essa separação faz sentido física: o mecanismo que determina "se vai chover" é diferente do que determina "quanto vai chover".

### Arquitetura

```
Dados Brutos
    ↓ fit_copulas() — estima correlações espaciais
    ↓
┌─────────────────────────────────────────────────────────┐
│  PARTE 1: Ocorrência                                     │
│                                                           │
│  dummy_input (zeros) → OccurrenceMLP → sigmoid → p_rain  │
│  OccurrenceMLP: Linear(S→32) → ReLU → Linear(32→32) → ReLU → Linear(32→S)  │
│                                                           │
│  Cópula Occ: z_occ ~ MVN(0, C_occ) → u_occ = Φ(z_occ)  │
│  occur[i] = (u_occ[i] < p_rain[i])                       │
│                                                           │
│  PARTE 2: Quantidade                                      │
│                                                           │
│  occ_mask → AmountMLP → (mu_log, sigma_log)              │
│  AmountMLP: Linear(S→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2S) │
│                                                           │
│  Cópula Amt: z_amt ~ MVN(0, C_amt) → u_amt = Φ(z_amt)  │
│  log_amount[i] = mu_log[i] + sigma_log[i] * Φ⁻¹(u_amt[i])   │
│  amount[i] = exp(log_amount[i])                          │
│                                                           │
│  SAÍDA: precip = occur * amount                          │
└─────────────────────────────────────────────────────────┘
```

### Detalhe: OccurrenceMLP

```python
class _OccurrenceMLP(nn.Module):
    def __init__(self, input_size, hidden=32):
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),  # uma saída por estação
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # probabilidade ∈ [0,1]
```

**Observação importante:** A entrada é sempre `torch.zeros_like(x)` — a rede não recebe os dados reais como entrada, apenas produz parâmetros por estação. É como um vetor de parâmetros aprendidos diretamente.

**Por que sigmoid?** Garante que a saída está em [0,1] — interpretável como probabilidade.

### Detalhe: AmountMLP

```python
class _AmountMLP(nn.Module):
    def __init__(self, input_size, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size * 2),  # mu E sigma por estação
        )

    def forward(self, occ_mask):
        out = self.net(occ_mask)
        mu    = out[:, :S]                         # média do log
        sigma = F.softplus(out[:, S:]) + 1e-4     # desvio padrão (positivo!)
        return mu, sigma
```

**Softplus:** `softplus(x) = log(1 + exp(x))` — garante que sigma > 0 (desvio padrão não pode ser negativo).

### `fit_copulas()` — Por que chamado antes do treinamento?

A cópula estima a estrutura de correlação espacial dos dados reais. Essa estrutura é fixa — não muda durante o treinamento. Por isso é calculada uma vez antes:

```python
# No train.py, antes do loop de gradiente:
model.fit_copulas(train_raw)  # estima C_occ e C_amt
```

**Por que duas cópulas?** Uma para a correlação de **ocorrência** (se chove em A, provavelmente chove em B) e outra para a correlação de **quantidade** (se choveu muito em A, provavelmente choveu muito em B). As duas correlações podem ser diferentes.

### A Função de Perda

```python
def loss(self, x, beta=1.0):
    occ_target = (x > 0).float()  # 1 se chuvoso, 0 se seco

    # Parte 1: Binary Cross-Entropy para ocorrência
    dummy_input = torch.zeros_like(x)
    p_rain = self.occ_mlp(dummy_input)
    bce = F.binary_cross_entropy(p_rain, occ_target, reduction='mean')

    # Parte 2: NLL Log-Normal para quantidade (apenas dias chuvosos)
    mu, sigma = self.amt_mlp(occ_target)
    wet = x > 0
    log_x = torch.log(x[wet].clamp(min=1e-8))
    nll = 0.5 * mean(
        ((log_x - mu[wet]) / sigma[wet])²   # erro padronizado
        + 2 * log(sigma[wet])               # penalidade por sigma grande
        + log(x[wet])                       # jacobiano da transformação log
    )

    return {'total': bce + nll, 'bce': bce, 'nll': nll}
```

**Por que BCE (Binary Cross-Entropy)?** É a perda padrão para classificação binária. Mede o quão bem a probabilidade prevista `p_rain` corresponde à ocorrência real (0 ou 1).

**Por que NLL Log-Normal?** O NLL (Negative Log-Likelihood) é a perda correta quando se assume que os dados seguem uma distribuição específica (aqui, Log-Normal). Minimizar NLL é equivalente a maximizar a verossimilhança — encontrar os parâmetros (mu, sigma) que tornam os dados observados mais prováveis.

**O termo do Jacobiano** (`+ log(x[wet])`): quando transformamos x para log(x) no cálculo do NLL, precisamos incluir o Jacobiano da transformação para que a integral da probabilidade continue sendo 1.

### Resultados (Melhor Modelo do Projeto)

| Composite | Wasserstein | Wet Freq Err | Parâmetros |
|-----------|-------------|--------------|------------|
| **0.140** | 1.603       | 0.080        | ~8.000     |

Apesar da simplicidade (apenas ~8.000 parâmetros), supera todos os modelos mais complexos.

### Por que o Hurdle Simples é o Melhor?

1. **Indução correta:** Separa o problema em dois sub-problemas físicamente corretos
2. **Distribuição correta:** Usa Log-Normal, que é conhecidamente boa para precipitação
3. **Poucos parâmetros:** Com dados escassos (~3000 dias), modelos simples geralmente ganham
4. **Correlação espacial via cópula:** Usa a estrutura de correlação real dos dados, não precisa aprendê-la

### Perguntas Frequentes

**P: Por que o input da OccurrenceMLP é zeros e não os dados reais?**
R: O modelo aprendeu que a probabilidade de chuva é estacionária (a mesma em qualquer dia). Uma versão mais avançada seria o `hurdle_temporal.py`, que usa dados do passado como contexto.

**P: A cópula é o mesmo objeto do copula.py?**
R: Não exatamente. O `copula.py` usa a cópula como modelo completo. O `hurdle_simple.py` usa apenas a cópula como mecanismo de correlação espacial — as distribuições marginais são aprendidas pelas redes neurais.

---

## 7.3 vae.py — Autoencoder Variacional (VAE)

### A Pergunta

> "Um espaço latente contínuo melhora sobre distribuições paramétricas fixas?"

### Intuição

**Analogia:** Imagine um músico que ouviu milhares de músicas. Internamente, ele formou um "espaço mental" de música — um continuum de possibilidades. Para compor uma música nova, ele simplesmente "navega" por esse espaço mental e transcreve o que imagina.

O VAE funciona assim:
- **Encoder:** "ouve" os dados reais e os mapeia para um ponto no espaço mental (latente)
- **Decoder:** a partir de um ponto no espaço latente, "compõe" dados sintéticos

A diferença crucial do autoencoder clássico: o espaço latente é **probabilístico** — cada ponto real é mapeado para uma **distribuição** (não um único ponto), e o decoder é treinado para ser robusto a essa incerteza.

### Arquitetura

```
ENCODER:
  input (S=15)
  → Linear(15 → 512) → LeakyReLU
  → Linear(512 → 256) → LeakyReLU
  → Linear(256 → 128)
  → fc_mu(128 → 128)     ← média do latente
  → fc_logvar(128 → 128) ← log-variância do latente

DECODER:
  z (128)
  → Linear(128 → 256) → LeakyReLU
  → Linear(256 → 512) → LeakyReLU
  → Linear(512 → 15)
  → ReLU                  ← garante não-negatividade
```

(com `latent_size=128`, como nos experimentos com melhor resultado)

### O Truque de Reparametrização

Este é o insight central do VAE. O problema: queremos amostrar `z ~ N(mu, sigma)`, mas isso não permite backpropagation (amostrar não é diferenciável).

**Solução:** escreva `z = mu + sigma · ε` onde `ε ~ N(0,1)`.

Agora:
- `ε` é amostrado externamente (sem gradiente)
- `z` é uma função diferenciável de `mu` e `sigma`
- O gradiente pode fluir de `z` para `mu` e `sigma`!

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)    # σ = exp(log_σ²/2)
    eps = torch.randn_like(std)      # ε ~ N(0,I) — amostrado externamente
    return mu + eps * std            # z = μ + σ·ε — diferenciável em μ e σ
```

**Visualização:**
```
        ε ~ N(0,I)  ←── não-diferenciável (ok! ε é constante)
             |
             ↓
  z = μ + σ · ε    ←── diferenciável em μ e σ
  ↑          ↑
  Encoder(x)  Encoder(x)
```

### A Loss ELBO

```python
def loss(self, x, beta=1.0):
    x_hat, mu, logvar = self.forward(x)

    # Termo 1: Reconstrução (quão bem o decoder recupera os dados?)
    recons = F.mse_loss(x_hat, x, reduction='mean')

    # Termo 2: KL divergence (quão próximo o latente está de N(0,I)?)
    kl = -0.5 * mean(1 + logvar - mu² - exp(logvar))

    total = recons + beta * kl
    return {'total': total, 'recons': recons, 'kl': kl}
```

**Interpretação do KL:**
- `KL(q(z|x) || p(z))` onde `q(z|x) = N(mu, exp(logvar))` e `p(z) = N(0,I)`
- Fórmula fechada para gaussianas: `KL = -0.5 · Σ(1 + logvar - mu² - exp(logvar))`
- KL = 0 quando `mu = 0` e `logvar = 0` (espaço latente colapsa para prior N(0,I))

**Por que minimizar KL?** Para que o espaço latente seja organizado e contínuo. Se cada dado é mapeado para um ponto isolado (sem regularização KL), gerar novos dados seria impossível — qualquer ponto entre dois dados reais seria um "buraco" no espaço.

### ReLU na Saída do Decoder

```python
if output_activation == "relu":
    decoder_layers.append(nn.ReLU())
```

**Por que ReLU?** Precipitação ≥ 0. Sem a ReLU, o decoder poderia gerar valores negativos, o que não faz sentido físico. A ReLU trunca em zero: `ReLU(x) = max(0, x)`.

### Geração de Amostras

```python
def sample(self, n):
    z = torch.randn(n, self.latent_size)   # z ~ N(0,I)
    return self.decoder(z)                  # mapeia para espaço de dados
```

A beleza do VAE: graças ao KL, o espaço latente é organizado como N(0,I). Para gerar, basta sortear pontos desse prior e passar pelo decoder.

### Resultados

| Composite | Wasserstein | Wet Freq Err | Parâmetros |
|-----------|-------------|--------------|------------|
| 0.443     | 2.569       | 0.246        | ~270.000   |

Piora em relação ao hurdle_simple porque:
1. Muito mais parâmetros para poucos dados
2. Não separa explicitamente ocorrência de quantidade
3. A ReLU corta gradientes para dias secos (não aprende bem a massa em zero)

### Perguntas Frequentes

**P: Por que usar MSE e não NLL para a reconstrução?**
R: MSE é o NLL de uma distribuição gaussiana. Para dados de precipitação (que têm massa em zero e cauda pesada), o MSE não é ideal — mas é simples e funciona razoavelmente. O hurdle_simple usa NLL Log-Normal que é mais correto.

**P: Por que `logvar` em vez de `sigma` diretamente?**
R: É mais numericamente estável. Sigma deve ser positivo — usar `exp(0.5 * logvar)` garante isso sem precisar de restrições.

---

## 7.4 hurdle_vae.py — Dois VAEs Separados

### A Pergunta

> "Separar ocorrência de quantidade em VAEs diferentes ajuda?"

### Ideia

Combina a ideia do hurdle model (separar ocorrência de quantidade) com a capacidade expressiva do VAE (espaço latente aprendido). Em vez de usar distribuições paramétricas fixas (Log-Normal), cada estágio tem seu próprio VAE.

### Arquitetura

```
OccurrenceVAE (latent_size=32):
  Input:  o ∈ {0,1}^S (máscara binária)
  Encoder: (S) → (max(64,S)) → (32, mu) + (32, logvar)
  Decoder: (32) → (max(64,S)) → (S) [sem ativação]
  Loss:    BCE_with_logits + beta * KL
  Sample:  z_o ~ N(0,I) → decoder → sigmoid → Bernoulli

AmountVAE (latent_size=64):
  Input:  [x_masked | occ_mask] ∈ R^{2S} (concat quantidade+ocorrência)
  Encoder: (2S) → (max(128,2S)) → (64, mu) + (64, logvar)
  Decoder: (64) → (max(128,2S)) → (S) → ReLU
  Loss:    MSE apenas dias úmidos + beta * KL
  Sample:  z_a ~ N(0,I) → decoder → ReLU
```

### A Loss Combinada

```python
def loss(self, x, beta=1.0):
    # Ocorrência
    occ_target = (x > 0).float()
    p_hat, mu_o, logvar_o = self.occ_vae(occ_target)
    bce = F.binary_cross_entropy_with_logits(p_hat, occ_target)
    kl_occ = -0.5 * mean(1 + logvar_o - mu_o² - exp(logvar_o))

    # Quantidade (condicionada na ocorrência)
    x_masked = x * occ_target                     # zera dias secos
    amt_input = torch.cat([x_masked, occ_target], dim=1)  # (B, 2S)
    x_hat_amt, mu_a, logvar_a = self.amt_vae(amt_input)
    kl_amt = -0.5 * mean(1 + logvar_a - mu_a² - exp(logvar_a))

    # MSE apenas nas posições chuvosas
    mse_wet = F.mse_loss(x_hat_amt[wet_mask], x[wet_mask])

    total = bce + mse_wet + beta * (kl_occ + kl_amt)
```

### Amostragem em Dois Passos

```python
def sample(self, n):
    # Passo 1: ocorrência
    z_o = torch.randn(n, 32)                    # latente de ocorrência
    p_rain = torch.sigmoid(occ_vae.decode(z_o)) # probabilidades
    occ_sim = torch.bernoulli(p_rain)           # amostras binárias

    # Passo 2: quantidade
    zeros = torch.zeros(n, S)
    amt_input = torch.cat([zeros, occ_sim], dim=1)  # condicionado na ocorrência
    z_a = torch.randn(n, 64)                        # latente de quantidade
    a_raw = amt_vae.decode(z_a)

    return F.relu(a_raw) * occ_sim   # zera onde não chove
```

### Por que Performa Pior que hurdle_simple?

| Composite | Wasserstein | Wet Freq Err | Parâmetros |
|-----------|-------------|--------------|------------|
| 0.504     | 2.829       | 0.270        | ~150.000   |

1. **Mais parâmetros, menos dados:** O trade-off da complexidade — modelos mais expressivos precisam de mais dados para generalizar
2. **Distribuição implícita vs explícita:** O VAE aprende uma distribuição implícita; hurdle_simple usa Log-Normal que sabemos ser boa para precipitação
3. **MSE vs NLL Log-Normal:** A loss do AmountVAE usa MSE que não é ideal para distribuições assimétricas

### Perguntas Frequentes

**P: Por que o AmountVAE recebe [amount | occ_mask] concatenados?**
R: Para que a rede saiba onde há chuva e onde não há — o contexto da ocorrência ajuda a gerar quantidades condicionalmente corretas.

---

## 7.5 real_nvp.py — Fluxo Normalizante (RealNVP)

### A Pergunta

> "Um fluxo normalizante com likelihood exata supera o VAE?"

### O que é um Fluxo Normalizante?

**Analogia:** Imagine que você tem uma argila (distribuição gaussiana simples) e quer transformá-la numa escultura complexa (distribuição de precipitação). Um fluxo normalizante é uma sequência de transformações invertíveis que molda a argila.

A chave: cada transformação é **invertível** e tem **Jacobiano calculável**. Isso permite calcular a likelihood exata dos dados, sem aproximações.

Matematicamente:
```
x (dados) = f₁(f₂(... fₙ(z) ...))   onde z ~ N(0,I)
z = fₙ⁻¹(... f₂⁻¹(f₁⁻¹(x)) ...)   (inversão)

log p(x) = log p(z) + Σᵢ log|det ∂fᵢ/∂zᵢ|
            ↑ simples   ↑ log-determinante Jacobiano
```

**Diferença do VAE:** O VAE usa ELBO (limite inferior aproximado). O RealNVP usa a likelihood EXATA.

### As Camadas de Acoplamento Afim

A transformação central do RealNVP é a **affine coupling layer**:

```
Dado x = (x₁, x₂) particionado por uma máscara binária:

  y₁ = x₁   (parte fixa — não transformada)
  s, t = MLP(x₁)   (rede que processa a parte fixa)
  y₂ = x₂ · exp(tanh(s)) + t   (transforma a outra parte)

Log-det Jacobiano: sum(tanh(s)) — simples de calcular!
```

**Por que `tanh(s)` no expoente?** Para evitar overflow numérico. `tanh` limita s ao intervalo (-1, 1), então `exp(tanh(s))` fica em (0.37, 2.72) — nunca explode.

```python
class _CouplingLayer(nn.Module):
    def forward(self, x):
        x1 = x[:, mask]       # parte fixa
        x2 = x[:, ~mask]      # parte transformada

        st = self.net(x1)     # MLP processa parte fixa
        s, t = st[:, :n_free], st[:, n_free:]
        s = torch.tanh(s)     # limita para evitar overflow

        y2 = x2 * torch.exp(s) + t     # transformação afim
        log_det = s.sum(dim=-1)         # log-det = soma de s

        y = x.clone()
        y[:, ~mask] = y2
        return y, log_det

    def inverse(self, y):
        x1 = y[:, mask]
        y2 = y[:, ~mask]
        st = self.net(x1)
        s, t = ...
        x2 = (y2 - t) * torch.exp(-s)  # inversa simples!
        ...
```

### Por que Alternamos Máscaras?

Com 12 camadas de acoplamento alternando máscara "pares/ímpares":

```
Camada 1: fixa posições pares  (0,2,4...), transforma ímpares (1,3,5...)
Camada 2: fixa posições ímpares, transforma pares
Camada 3: fixa pares novamente...
```

Cada dimensão é transformada em metade das camadas. Após 12 camadas, cada dimensão já foi influenciada por todas as outras — o fluxo captura a estrutura completa dos dados.

### A Loss (NLL Exata)

```python
def loss(self, x, beta=1.0):
    log_p = self.log_prob(x)    # likelihood exata
    nll = -log_p.mean()         # negativa pois minimizamos
    return {'total': nll, 'nll': nll}

def log_prob(self, x):
    z = x
    log_det_total = 0

    # Escala global aprendida
    z = z * torch.exp(self.log_scale)
    log_det_total += self.log_scale.sum()

    # Passa por todas as camadas
    for layer in self.layers:
        z, log_det = layer(z)
        log_det_total += log_det

    # log p(z) sob N(0,I)
    log_pz = -0.5 * (z² + log(2π)).sum(dim=-1)

    return log_pz + log_det_total
```

### Geração

```python
def sample(self, n):
    z = torch.randn(n, S)        # z ~ N(0,I) — simples!

    # Inverte as camadas em ordem reversa
    for layer in reversed(self.layers):
        z = layer.inverse(z)

    # Inverte a escala global
    z = z * torch.exp(-self.log_scale)

    return z   # x no espaço de dados
```

### Resultados

| Composite | Wasserstein | Wet Freq Err | Parâmetros |
|-----------|-------------|--------------|------------|
| 0.717     | 3.399       | 0.618        | ~1.500.000 |

Apesar da likelihood exata, não supera o hurdle_simple. Problemas:

1. **Muitos parâmetros:** 12 camadas × MLP com 256 unidades
2. **Sem indução de estrutura:** Não sabe que há massa em zero (dias secos)
3. **Transformações limitadas:** As transformações afins são simples — podem não capturar a complexidade de precipitação

### Perguntas Frequentes

**P: ELBO vs NLL exata — qual é melhor na prática?**
R: Depende! NLL exata é teoricamente superior, mas na prática, o VAE com ELBO pode ser mais flexível porque o encoder pode usar qualquer arquitetura. RealNVP é limitado às transformações invertíveis.

**P: Por que 12 camadas de acoplamento?**
R: Mais camadas = mais capacidade expressiva. 12 é um valor empírico comum na literatura. Com 15 dimensões, 12 camadas garantem cobertura suficiente.

---

## 7.6 flow_match.py — Flow Matching (Trajetórias Retas)

### A Pergunta

> "Flow Matching com trajetórias retas supera o fluxo normalizante?"

### Intuição

**Comparação com difusão (DDPM):** A difusão destroça os dados aos poucos adicionando ruído gradualmente (como um barulho que vai crescendo). O processo reverso (geração) é uma curva sinusoidal complicada de denoising.

**Flow Matching:** Em vez de curvas, usa **trajetórias retas**. Imagina uma bola rolando de um ponto de ruído (z₀ ~ N(0,I)) até um ponto de dado real (z₁). Flow Matching ensina a rede a seguir essa reta.

**Por que retas são melhores?**
- Mais fáceis de integrar numericamente (menos passos necessários)
- Loss mais simples (apenas MSE — sem ELBO, sem log-det)
- Geração mais rápida

### A Trajetória Reta (OT-CFM)

```
z_t = (1 - t) · z₀ + t · z₁,   t ∈ [0, 1]

velocidade alvo = dz_t/dt = z₁ - z₀   (constante!)
```

**Por que a velocidade é constante?** Para uma trajetória reta, a "velocidade" (derivada da posição em relação ao tempo) é sempre o mesmo vetor (z₁ - z₀). A rede aprende a prever essa velocidade.

### SinusoidalEmbedding — Como o Modelo "Sabe" em que Ponto do Tempo Está

A rede precisa saber o valor de `t` para prever a velocidade correta. Mas `t` é um escalar — como passá-lo para uma rede que espera vetores?

**Embedding sinusoidal** (mesma ideia do Transformer original):

```python
class SinusoidalEmbedding(nn.Module):
    def forward(self, t):  # t: (B,)
        freqs = exp(-log(10000) * arange(dim//2) / (dim//2))
        args = t * freqs              # (B, dim//2)
        emb = cat([sin(args), cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)         # passa por MLP para mais expressividade
```

**Por que seno e cosseno?** Frequências diferentes de sin/cos capturam diferentes "escalas de tempo". É como um relógio com múltiplos ponteiros — horas, minutos, segundos.

### A Loss

```python
def loss(self, x, beta=1.0):
    B = x.shape[0]
    z_0 = torch.randn_like(x)    # ruído aleatório
    z_1 = x                       # dados reais

    t = torch.rand(B)             # tempo aleatório em [0, 1]
    t_exp = t.unsqueeze(-1)       # para broadcasting

    z_t = (1 - t_exp) * z_0 + t_exp * z_1    # trajetória reta
    target = z_1 - z_0                         # velocidade constante

    t_emb = self.t_embed(t)                   # embedding do tempo
    v_pred = self.velocity(z_t, t_emb)        # velocidade predita

    fm_loss = F.mse_loss(v_pred, target)      # MSE simples
    return {'total': fm_loss}
```

**Elegância:** A loss é simplesmente MSE entre a velocidade prevista e a velocidade alvo. Sem KL, sem ELBO, sem log-det. Muito mais simples de implementar e de treinar.

### Integração (Geração)

```python
def sample(self, n, steps=50, method='euler'):
    z = torch.randn(n, S)    # z₀ ~ N(0,I)
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i * dt
        t_tensor = full(n, t_val)
        t_emb = self.t_embed(t_tensor)
        v = self.velocity(z, t_emb)    # velocidade predita

        if method == 'euler':
            z = z + v * dt             # passo Euler simples

        elif method == 'heun':         # 2ª ordem (mais preciso)
            z_tmp = z + v * dt         # predictor Euler
            v_next = self.velocity(z_tmp, t_emb_next)  # velocidade no ponto futuro
            v_avg = (v + v_next) / 2   # corretor: média
            z = z + v_avg * dt

    return z   # z₁ ≈ dados reais
```

**Euler vs Heun:**
- **Euler:** 1 avaliação da rede por passo, mais rápido, menos preciso
- **Heun:** 2 avaliações por passo, mais lento, mais preciso (2ª ordem Runge-Kutta)

### Resultados

| Composite | Wasserstein | Wet Freq Err | Parâmetros |
|-----------|-------------|--------------|------------|
| 0.724     | 3.600       | 0.496        | ~480.000   |

Similar ao RealNVP — não supera hurdle_simple. Mesmas limitações: sem indução de estrutura para a massa em zero.

### Perguntas Frequentes

**P: Quantos passos de integração são necessários?**
R: Para trajetórias retas (OT-CFM), poucas dezenas de passos geralmente bastam (50 no default). Difusão clássica precisa de centenas. Isso é uma das vantagens do Flow Matching.

**P: O beta é ignorado aqui — por quê?**
R: Beta controla o peso do KL, que é um conceito do VAE. Flow Matching não tem KL — é puramente uma perda de regressão (MSE). O parâmetro beta existe na interface por compatibilidade, mas não é usado.

---

## 7.7 ldm.py — Latent Diffusion Model

### A Pergunta

> "Um modelo de difusão no espaço latente de um VAE supera o hurdle_simple?"

### Ideia

**DDPM no espaço de dados:** O diffusion model original (DDPM) destroça os dados reais gradualmente adicionando ruído e aprende a reverter esse processo. Opera diretamente no espaço de dados (15 dimensões de precipitação).

**LDM:** Em vez de operar nos dados brutos, primeiro comprime para um espaço latente (VAE com 128 dimensões), e então treina o DDPM nesse espaço. Mais eficiente e pode gerar representações de melhor qualidade.

```
TREINO:
  Estágio 1 (VAE):   x → Encoder → z (latente) → Decoder → x̂  [MSE + KL]
  Estágio 2 (DDPM):  z → add_noise(z, t) → z_t → Denoiser → ε_pred  [MSE]

GERAÇÃO:
  z_T ~ N(0,I)  →  Reverse DDPM (T → 0)  →  z_0  →  VAE Decoder  →  x
```

### O Schedule de Ruído (Cosine Schedule)

O DDPM define um schedule que controla quanta chuva é adicionada a cada timestep:

```python
@staticmethod
def _cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal 2021)"""
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = cos(((x / T + s) / (1 + s)) * π/2)²
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.02)
```

**Por que cosine?** O schedule linear (betas igualmente espaçados) destroça demais o sinal no início (quando os dados já parecem razoavelmente ruidosos) e muito pouco no final. O schedule coseno é mais suave, resultando em melhor qualidade de geração.

**Visualização:**
```
ᾱ_t (quantidade de sinal restante)
1.0 |\ ← cosine (suave)
    | \___
0.5 |     \____
    |          \____
0.0 |               \_______ ← linear (cai rápido demais no início)
    └─────────────────────── timestep T
```

### `set_stage()` — Congelando e Descongelando Parâmetros

```python
def set_stage(self, stage: str):
    vae_params = (list(encoder.parameters()) + list(fc_mu.parameters())
                  + list(fc_logvar.parameters()) + list(decoder.parameters()))
    ldm_params = list(denoiser.parameters())

    if stage == "vae":
        for p in vae_params: p.requires_grad_(True)   # treina VAE
        for p in ldm_params:  p.requires_grad_(False)  # congela denoiser
    else:
        for p in vae_params: p.requires_grad_(False)  # congela VAE
        for p in ldm_params:  p.requires_grad_(True)   # treina denoiser
```

**Por que congelar?** No estágio LDM, queremos treinar apenas o denoiser — se o VAE continuar mudando, o espaço latente se move e o denoiser nunca converge.

### A Loss do DDPM (Estágio 2)

```python
def _ddpm_loss(self, x):
    # 1. Codifica ao latente determinístico (sem amostrar — usa apenas mu)
    with torch.no_grad():
        mu, _ = self._encode(x)
    z0 = mu   # (B, latent_size)

    # 2. Sorteia timestep aleatório
    t_int = torch.randint(0, T, (B,))
    t_float = t_int.float() / (T - 1)

    # 3. Adiciona ruído ao latente
    noise = torch.randn_like(z0)
    sqrt_a = sqrt_alphas_cumprod[t_int]
    sqrt_1a = sqrt_one_minus_alphas_cumprod[t_int]
    z_noisy = sqrt_a * z0 + sqrt_1a * noise   # z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε

    # 4. Treina denoiser a prever o ruído adicionado
    noise_pred = self.denoiser(z_noisy, t_float)
    diff_loss = F.mse_loss(noise_pred, noise)
```

**Por que usar `mu` (sem amostrar)?** No estágio LDM, o VAE está congelado. Para ter latentes determinísticos e estáveis como "ponto de partida" para o ruído, usamos apenas `mu` do encoder (sem o truque de reparametrização).

### Amostragem: DDPM vs DDIM

**DDPM (estocástico):** A cada passo reverso, adiciona ruído aleatório:
```
z_{t-1} = coef₁ · x0_pred + coef₂ · z_t + σ_t · ε,  ε ~ N(0,I)
```

**DDIM (determinístico):** Sem ruído — segue uma ODE determinística:
```
z_{t-1} = √ᾱ_{t-1} · x0_pred + √(1-ᾱ_{t-1}) · ε_pred
```

DDIM é mais rápido (pode usar menos timesteps com subsampling) e produz resultados mais consistentes. DDPM é mais estocástico (mais diversidade).

### Subsampling de Timesteps

```python
# Subconjunto de índices igualmente espaçados
indices = np.round(np.linspace(0, ldm_timesteps - 1, num_steps)).astype(int)
```

Se o modelo foi treinado com 100 timesteps mas você quer gerar em 20 passos, escolhe 20 timesteps igualmente espaçados. Esse "DDIM-style subsampling" permite geração mais rápida sem retraining.

### Resultados

| Composite | Wasserstein | Parâmetros  |
|-----------|-------------|-------------|
| ~0.5*     | **1.287**   | ~500.000    |

*Não comparado diretamente no PrecipModels — medido em VAE_Tests/best_v2_ldm.

O LDM tem o melhor Wasserstein dos modelos testados, mas o composite score é prejudicado por outras métricas.

### Perguntas Frequentes

**P: Por que treinar em dois estágios em vez de treinar tudo junto?**
R: Se você treinar VAE e DDPM juntos, o VAE não terá tempo de convergir para um bom espaço latente antes do DDPM começar a tentar aprender nele. O resultado é instável e inferior.

**P: Qual o efeito do cosine schedule na qualidade?**
R: O cosine schedule evita que o modelo "desperdice" capacidade em timesteps onde o sinal já está completamente destruído (t próximo de T). Isso melhora a qualidade geral das amostras geradas.

---

## 7.8 hurdle_temporal.py — Hurdle com Contexto Temporal (GRU)

### A Pergunta

> "Adicionar contexto temporal ao hurdle_simple captura melhor a autocorrelação?"

### Motivação

O `hurdle_simple.py` trata cada dia como independente. Mas na realidade, a precipitação tem persistência temporal: se ontem choveu muito, hoje é mais provável que chova novamente (frentes frias passam ao longo de vários dias).

O `hurdle_temporal.py` adiciona um **codificador GRU** que processa os últimos 30 dias e produz um vetor de contexto que influencia as previsões de ocorrência e quantidade.

### O que é um GRU?

GRU (Gated Recurrent Unit) é uma rede neural projetada para processar sequências. Diferente de um MLP que trata cada entrada independentemente, o GRU tem "memória" — o que viu antes influencia como processa o que vê agora.

**Analogia:** Um leitor que lê um livro. Ao processar a palavra 1000, ele lembra o contexto das páginas anteriores — não começa do zero a cada palavra.

**Arquitetura simplificada do GRU:**
```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

onde:
  z_t = sigmoid(Wz · [h_{t-1}, x_t])   (gate de atualização)
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t]) (candidato)
  r_t = sigmoid(Wr · [h_{t-1}, x_t])   (gate de reset)
```

O gate de atualização (z_t) controla quanto da memória anterior é mantido vs quanto da entrada nova é incorporada.

### _GRUContextEncoder

```python
class _GRUContextEncoder(nn.Module):
    def __init__(self, n_stations, window_size, hidden_dim, context_dim):
        self.gru = nn.GRU(
            input_size=n_stations,    # S dimensões por timestep
            hidden_size=hidden_dim,   # dimensão do estado oculto
            num_layers=2,             # 2 camadas GRU empilhadas
            batch_first=True,         # dimensão do batch vem primeiro
            dropout=0.1,              # regularização
        )
        self.proj = nn.Linear(hidden_dim, context_dim)

    def forward(self, window):  # window: (B, 30, S)
        _, h_n = self.gru(window)   # h_n: (2, B, hidden_dim) — estados finais
        h_last = h_n[-1]             # (B, hidden_dim) — última camada
        return self.proj(h_last)     # (B, context_dim) — vetor de contexto
```

**Fluxo:** 30 dias de precipitação → GRU processa sequencialmente → estado final h_n → Linear → vetor de contexto de 32 dimensões.

### Estratégia de Treinamento: Janelas Aleatórias

O DataLoader padrão fornece amostras i.i.d. (sem ordem temporal). Como usar contexto temporal num DataLoader bagunçado?

**Estratégia do projeto:**
1. Armazena todos os dados de treino no modelo via `fit_temporal()`
2. Pré-constrói todas as janelas de 30 dias como array numpy
3. Durante cada batch do treinamento, amostra B janelas aleatórias do histórico

```python
def fit_temporal(self, data_norm):
    self._windows_np = np.stack(
        [data_norm[i : i + W] for i in range(N - W)]
    ).astype(np.float32)

def _sample_random_windows(self, B, device):
    idx = np.random.randint(0, len(self._windows_np), size=B)
    return torch.tensor(self._windows_np[idx], ...)
```

**É uma aproximação?** Sim. A loss calculada é `E[loss | contexto aleatório]`, não a loss condicional exata `E[loss | contexto do dia anterior]`. Mas permite que o GRU aprenda a extrair estatísticas climáticas relevantes.

### A Loss

```python
def loss(self, x, beta=1.0):
    # Contexto temporal (janelas aleatórias)
    windows = self._sample_random_windows(B, device)  # (B, W, S)
    context = self.context_encoder(windows)            # (B, context_dim)

    # Ocorrência condicionada no contexto
    p_rain = self.occ_mlp(context)                    # (B, S) em [0,1]
    bce = F.binary_cross_entropy(p_rain, occ_target)

    # Quantidade condicionada no contexto
    mu, sigma = self.amt_mlp(context)
    nll = ...  # mesma fórmula do hurdle_simple

    return {'total': bce + nll, 'bce': bce, 'nll': nll}
```

**Diferença do hurdle_simple:** Em vez de `self.occ_mlp(zeros)`, agora é `self.occ_mlp(context)` — o contexto temporal influencia as previsões.

### Pré-ajustes Necessários (Antes do Treinamento)

```python
# No train.py, antes do loop de gradiente:
model.fit_copulas(train_raw)      # estima matrizes de correlação (igual ao hurdle_simple)
model.fit_temporal(train_norm)    # armazena dados para janelas aleatórias
```

### Perguntas Frequentes

**P: O contexto temporal ajuda a capturar autocorrelação?**
R: Teoricamente sim — o GRU pode aprender que "30 dias de chuva → maior probabilidade de chuva hoje". Na prática, os resultados do projeto mostram desempenho similar ao hurdle_simple, provavelmente porque o conjunto de dados é pequeno demais para o GRU aprender bem.

**P: Por que janela de 30 dias?**
R: 30 dias é um mês — captura a memória de curto prazo relevante para eventos climáticos. Janelas maiores podem capturar sazonalidade, mas também aumentam o custo computacional.

---

## 7.9 latent_flow.py — Flow Matching com Transformer

### A Pergunta

> "Flow Matching com contexto temporal Transformer supera o flow_match simples?"

### Diferenças em Relação ao flow_match.py

| Aspecto | flow_match.py | latent_flow.py |
|---------|--------------|----------------|
| Arquitetura | MLP simples | Transformer + AdaLN |
| Contexto temporal | Nenhum | Janela 30 dias |
| Condicionamento sazonal | Nenhum | Mês/dia-do-ano (cíclico) |
| Normalização interna | Nenhuma (usa scale_only do train.py) | log1p + padronização |
| EMA | Não | Sim |
| Complexidade | Baixa | Alta |

### _WindowEncoder (Transformer para Contexto)

```python
class _WindowEncoder(nn.Module):
    def __init__(self, n_stations, window_size, hidden_dim, output_dim, n_heads=4):
        self.input_proj = nn.Linear(n_stations, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, window):  # (B, 30, S)
        h = self.input_proj(window) + self.pos_embed  # projeta + adiciona posição
        h = self.transformer(h)                        # atenção sobre 30 dias
        h = h.mean(dim=1)                             # pooling: (B, hidden_dim)
        return self.output_proj(h)                    # (B, output_dim)
```

**Por que Transformer em vez de GRU?** O Transformer pode capturar dependências de longo alcance na janela de 30 dias sem o problema de "gradiente que desaparece" do GRU. A atenção permite que cada dia da janela "preste atenção" em todos os outros.

### _AdaLN (Feature-wise Linear Modulation)

```python
class _AdaLNBlock(nn.Module):
    def forward(self, x, cond):
        scale, shift, gate = self.adaLN(cond).chunk(3, dim=-1)

        # Normaliza x e aplica scale/shift condicionais
        h = self.norm1(x) * (1 + scale) + shift
        h = self.linear1(h)
        h = F.gelu(h)
        h = self.linear2(h)

        return x + gate * h   # skip connection com gate condicional
```

**O que é AdaLN?** Adaptive Layer Norm — uma forma de condicionamento onde o contexto (tempo t + janela histórica) controla os parâmetros da normalização (scale e shift) e a força da conexão skip (gate).

**Analogia:** É como "sintonizar" a rede para diferentes condições. Em vez de ter parâmetros fixos, a rede aprende a se adaptar ao contexto atual.

### Normalização Interna (`_PrecipTransform`)

O latent_flow.py usa sua própria normalização interna, além da scale_only do train.py:

```python
class _PrecipTransform:
    def fit(self, data):
        # log1p remove a cauda pesada da precipitação
        transformed = np.log1p(np.clip(data, 0, 300))
        self.station_mean = np.nanmean(transformed, axis=0)
        self.station_std  = np.nanstd(transformed, axis=0)

    def forward(self, data):
        x = np.log1p(np.clip(data, 0, 300))
        return (x - self.station_mean) / self.station_std

    def inverse(self, x):
        denorm = x * self.station_std + self.station_mean
        return np.expm1(np.clip(denorm, -10, 10))  # inversa do log1p
```

**Por que log1p?** `log1p(x) = log(1+x)` é contínuo em x=0 (diferente de log(x) que vai para -∞). Aplica o mesmo efeito que Log-Normal — achata a cauda pesada e torna a distribuição mais próxima de gaussiana, facilitando o aprendizado do flow.

### `fit_flow()` — Inicialização Necessária

```python
model.fit_flow(train_raw, std_scale=std)
```

Isso:
1. Ajusta a transformação interna (log1p + padronização)
2. Armazena os dados de treino
3. Pré-constrói todas as janelas e encodings sazonais
4. Inicializa o EMA

**Por que necessário no compare.py?** Ao carregar um modelo salvo do disco, o estado interno (transformação, janelas) não é salvo no `model.pt` — precisa ser recomputado.

### EMA (Exponential Moving Average)

```python
class _EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {n: p.clone() for n, p in model.named_parameters()}

    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.data, 1 - decay)  # média exponencial
```

**O que é EMA?** Em vez de usar os pesos mais recentes do modelo, mantém uma média ponderada dos pesos ao longo do treinamento. Os pesos mais recentes têm mais peso, mas todos os passos anteriores contribuem.

**Por que usar?** Estabiliza o treinamento — os pesos EMA são mais suaves e frequentemente têm melhor generalização do que os pesos "crus" do último passo de gradiente.

### Perguntas Frequentes

**P: O Transformer é melhor que o GRU para contexto temporal?**
R: Teoricamente sim para janelas longas (onde o Transformer pode prestar atenção em posições distantes). Na prática, os resultados dependem muito da quantidade de dados disponíveis.

**P: Por que amostragem "midpoint" por padrão?**
R: O método midpoint (Runge-Kutta de 2ª ordem) é mais preciso que Euler com o mesmo número de passos, sem precisar de 2 avaliações da rede como o Heun. É um bom equilíbrio entre velocidade e qualidade.

---

## 7.10 hurdle_flow.py — Hurdle + Fluxo Normalizante Condicional

### A Pergunta

> "E se usarmos um fluxo normalizante para modelar as quantidades de chuva, em vez de uma distribuição Log-Normal paramétrica fixa? O fluxo poderia capturar formas mais complexas sem assumir uma família de distribuições específica?"

O `hurdle_simple.py` assume que a quantidade de chuva segue uma Log-Normal — uma escolha bem motivada fisicamente, mas ainda assim uma suposição. O `hurdle_flow.py` substitui essa suposição por um **fluxo normalizante condicional**: o modelo aprende a distribuição que melhor se encaixa nos dados, condicionando na informação de quais estações estão chovendo.

### Separação Hurdle: Dois Componentes

```
Entrada: x (B, S) — precipitação de S estações

Componente 1 — Ocorrência:
    _OccurrenceMLP(zeros) → p_rain por estação (idêntico ao hurdle_simple)

Componente 2 — Quantidade:
    _ConditionalCouplingLayer × 8 camadas (RealNVP condicionado na máscara)
```

### O Truque Fundamental: Dias Secos como Ruído

O fluxo normalizante precisa mapear **todos** os dados, mas dias secos têm `x = 0`. Se tentarmos mapear zero direto para o espaço latente, o Jacobiano explode.

**Solução:** Durante o treino, dias secos (`x = 0`) são substituídos por amostras de `N(0, 1)`:

```python
# hurdle_flow.py — loss(), linha ~171
y_in = torch.where(
    occ_target > 0,       # onde choveu: usa log1p(x)
    torch.log1p(x),
    torch.randn_like(x)   # onde não choveu: ruído gaussiano
)
```

Por que isso funciona? O fluxo **é condicionado na máscara de ocorrência** (`occ_target`). Assim, durante a geração, ele sabe exatamente quais dimensões são "dias secos" e as mascara com zero. O ruído injetado no treino é ignorado porque o fluxo aprende que a máscara determina o comportamento daquela dimensão.

**Analogia:** É como ensinar um chef de cozinha que certos pratos do menu estão "indisponíveis hoje" (máscara = 0). Para não travar a cozinha, você preenche esses pedidos com um prato genérico durante o treino — mas na hora de servir de verdade, o chef simplesmente não serve nada naquelas posições.

### Como o Fluxo é Condicionado

A diferença fundamental do RealNVP padrão está na `_ConditionalCouplingLayer`:

```python
class _ConditionalCouplingLayer(nn.Module):
    # RealNVP padrão:      MLP(x1)          → (s, t)
    # Aqui (condicional):  MLP(cat([x1, occ_target])) → (s, t)

    def forward(self, x: Tensor, cond: Tensor):
        x1 = x[:, self.mask.bool()]   # metade "fixa" (não transformada)
        x2 = x[:, ~self.mask.bool()]  # metade "livre" (será transformada)

        mlp_in = torch.cat([x1, cond], dim=1)  # ← condição concatenada!
        st = self.net(mlp_in)
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)  # limita s para evitar overflow

        y2 = x2 * torch.exp(s) + t   # transformação afim

        # Log-determinante: só conta dimensões com chuva
        cond_free = cond[:, ~self.mask.bool()]
        log_det = (s * cond_free).sum(dim=-1)  # ← mascarado!
```

O `log_det` mascarado é crucial: o fluxo só "cobra" a likelihood das dimensões chuvosas. Estações secas não contribuem para a perda de quantidade.

### Loss: BCE + NLL Exato do Fluxo

```
loss_total = loss_ocorrência + loss_quantidade

loss_ocorrência = BCE(p_rain_predita, occ_target)

loss_quantidade = -log_prob_flow(y_in | occ_target)
                = -(log_p(z) + Σ log|det J_i| × occ_target)
```

A grande vantagem: **NLL exato**. O fluxo normalizante permite calcular a probabilidade exata dos dados (sem aproximação variacional como no VAE). A desvantagem: calcular o Jacobiano para cada camada tem custo computacional maior.

### Geração: Fluxo Inverso

```python
def sample(self, n):
    # 1. Ocorrência (idêntico ao hurdle_simple)
    p_rain = occ_mlp(zeros)
    occ_sim = Bernoulli(p_rain)  # (n, S)

    # 2. Quantidade via fluxo inverso (Latente → Dados)
    z = N(0, I)  # ruído latente
    for layer in reversed(self.layers):
        z = layer.inverse(z, cond=occ_sim)  # desencadeia o fluxo

    # 3. Desfaz o log1p
    a_raw = expm1(z)  # volta à escala original
    precip = relu(a_raw) * occ_sim  # zera dias secos
```

### FAQ

**P: Por que 8 camadas de acoplamento?**
R: Mais camadas = fluxo mais expressivo (pode modelar distribuições mais complexas), mas também mais lento. Com `input_size=15` dimensões, 8 camadas de acoplamento alternadas (par/ímpar) são suficientes para cada dimensão ser transformada em média ~4 vezes.

**P: Por que `tanh(s)` em vez de simplesmente usar `s` bruto?**
R: A transformação afim usa `exp(s)` como fator de escala. Se `s` fosse ilimitado, `exp(s)` poderia explodir numericamente. `tanh` limita `s` ao intervalo `(-1, 1)`, garantindo que `exp(s)` fique entre `e^{-1} ≈ 0.37` e `e^1 ≈ 2.72` — transformações razoáveis.

**P: Qual a diferença em relação ao `real_nvp.py` regular?**
R: O RealNVP padrão modela `p(x)` diretamente (todos os dados juntos). O HurdleFlow modela separadamente `p(ocorrência)` e `p(quantidade | ocorrência)`, com o fluxo sendo condicionado na máscara de ocorrência. Isso respeita a estrutura física do problema (dias secos vs. úmidos).

---

## 7.11 hurdle_vae_cond.py — CVAE com Máscara de Ocorrência

### A Pergunta

> "E se o VAE responsável pela quantidade de chuva souber, desde o início, em quais estações está chovendo? Isso ajudaria a organizar melhor o espaço latente e gerar quantidades mais realistas?"

O `hurdle_vae.py` usa dois VAEs independentes: um para ocorrência (binária) e um para quantidade. O problema é que o VAE de quantidade não tem acesso à informação de ocorrência durante o encoding — ele precisa "descobrir sozinho" que certos valores são zero porque não choveu.

O `hurdle_vae_cond.py` corrige isso com um **CVAE (Conditional VAE)**: a máscara de ocorrência é injetada tanto no encoder quanto no decoder do VAE de quantidade.

### CVAE (Conditional VAE): A Ideia Geral

Um VAE padrão aprende `p(x)`. Um CVAE aprende `p(x | c)` — a distribuição de `x` dado uma condição `c`.

```
VAE padrão:   encoder(x)       → z → decoder(z)       → x_hat
CVAE:         encoder(x, c)    → z → decoder(z, c)    → x_hat
```

No nosso caso:
- `x` = quantidade de chuva (em espaço log1p)
- `c` = máscara de ocorrência (0 = seco, 1 = chuvoso)

### Arquitetura: `_ConditionalMiniVAE`

```python
class _ConditionalMiniVAE(nn.Module):
    def __init__(self, input_size, cond_size, latent_size):
        # Encoder: recebe (x + máscara) concatenados
        enc_in = input_size + cond_size  # S + S = 2S dimensões
        self.encoder = MLP(enc_in → h → h)
        self.fc_mu    = Linear(h → latent)
        self.fc_logvar = Linear(h → latent)

        # Decoder: recebe (z + máscara) concatenados
        dec_in = latent_size + cond_size  # Z + S dimensões
        self.decoder = MLP(dec_in → h → h → S)  # saída: S quantidades
```

### Por que injetar a máscara nos dois lados?

**No encoder:** A máscara ajuda o encoder a organizar o espaço latente. "Saber que hoje choveu em SP mas não em Campinas" ajuda o encoder a encontrar uma representação latente `z` mais informativa.

**No decoder:** A máscara guia a geração. "Dado que vai chover em SP, qual quantidade gerar?" é uma pergunta muito mais específica do que "Qual quantidade gerar?" sem informação de ocorrência.

**Analogia:** É como um tradutor (encoder/decoder) que, além do texto original, recebe o "tom" da conversa (formal/informal). O tom ajuda a codificar (entender o contexto) e a decodificar (escolher as palavras certas na tradução).

### Fluxo de Dados no Treino

```python
def loss(self, x):
    occ_target = (x > 0).float()  # máscara de ocorrência (B, S)

    # --- VAE de Ocorrência (sem condicionamento) ---
    p_hat, mu_o, logvar_o = self.occ_vae(occ_target)
    bce   = BCE(p_hat, occ_target)
    kl_occ = KL(mu_o, logvar_o)

    # --- CVAE de Quantidade ---
    x_log = log1p(x * occ_target)  # espaço logarítmico, zeros secos = 0

    # Encoder recebe: cat([quantidade_log, máscara])
    # Decoder recebe: cat([z, máscara])
    x_hat_log, mu_a, logvar_a = self.amt_vae(x_log, occ_target)

    kl_amt = KL(mu_a, logvar_a)

    # MSE apenas nos pixels chuvosos (ignora dias secos)
    mse_wet = MSE(x_hat_log[wet], x_log[wet])

    total = bce + mse_wet + beta * (kl_occ + kl_amt)
```

### Geração em Dois Passos

```python
def sample(self, n):
    # Passo 1: Ocorrência
    z_o = N(0, I)  → OccurrenceVAE.decode(z_o) → sigmoid → Bernoulli → occ_sim

    # Passo 2: Quantidade (condicionada na ocorrência gerada)
    z_a = N(0, I)  → AmountVAE.decode(z_a, occ_sim) → ReLU → a_log
    a_raw = expm1(a_log)  # inverte log1p

    precip = a_raw * occ_sim  # zera onde não choveu
```

A elegância: o `occ_sim` gerado no passo 1 é **automaticamente consistente** com o `occ_sim` passado ao CVAE no passo 2. Não é possível gerar "quantidade sem ocorrência" porque a ocorrência é injetada como condição.

### Diferença em Relação ao `hurdle_vae.py`

| Aspecto | `hurdle_vae.py` | `hurdle_vae_cond.py` |
|---------|-----------------|----------------------|
| VAE de ocorrência | `_MiniVAE(occ)` | `_MiniVAE(occ)` (igual) |
| VAE de quantidade | `_MiniVAE(amount)` | `_ConditionalMiniVAE(amount, occ_mask)` |
| Informação de ocorrência na quantidade | ✗ (independente) | ✓ (encoder + decoder) |
| Loss de quantidade | MSE wet | MSE wet (igual) |

### FAQ

**P: O que é o parâmetro `beta` no CVAE?**
R: Controla o balanço entre reconstrução (MSE) e regularização (KL). Com `beta = 0`, o modelo foca apenas em reconstruir os dados (pode causar posterior collapse). Com `beta = 1`, equilibra os dois objetivos. O `train.py` aumenta beta gradualmente de 0 para 1 durante o treino (KL annealing).

**P: Por que usar `log1p(x)` em vez de `x` direto?**
R: Precipitação tem distribuição de cauda pesada (a maioria dos dias tem 0-5mm, mas alguns dias têm >100mm). Trabalhar no espaço `log1p` comprime essa cauda, tornando a distribuição mais próxima de gaussiana e o treino do VAE mais estável.

**P: Por que `output_activation="relu"` no decoder de quantidade?**
R: O decoder emite valores no espaço `log1p`, que são sempre ≥ 0 (já que `log1p(x) ≥ 0` para `x ≥ 0`). O ReLU garante que o decoder não emita valores negativos nesse espaço.

---

## 7.12 hurdle_vae_cond_nll.py — CVAE + NLL Log-Normal

### A Pergunta

> "E se, em vez de MSE entre os valores log1p reconstruídos, o CVAE minimizasse diretamente a NLL de uma distribuição Log-Normal explícita? O MSE assume implicitamente que os erros são gaussianos — a NLL Log-Normal é mais honesta sobre a incerteza."

O `hurdle_vae_cond.py` usa MSE como loss de reconstrução. O MSE é equivalente a maximizar a log-likelihood sob uma distribuição gaussiana com variância constante. O `hurdle_vae_cond_nll.py` muda isso: o decoder emite **dois parâmetros** `(μ, σ)` em vez de uma reconstrução direta, e a loss é a NLL exata de uma distribuição Log-Normal.

### A Diferença Chave: Decoder Probabilístico

```python
# hurdle_vae_cond.py — decoder emite uma reconstrução
self.decoder = MLP(dec_in → h → h → S)     # saída: S valores

# hurdle_vae_cond_nll.py — decoder emite (μ, σ) para cada dimensão
self.decoder = MLP(dec_in → h → h → S*2)   # saída: S médias + S desvios-padrão
```

No decode:
```python
def decode(self, z, c):
    zc = cat([z, c], dim=1)
    out = self.decoder(zc)
    mu    = out[:, :S]                            # médias
    sigma = softplus(out[:, S:]) + 1e-4           # desvios (sempre > 0)
    return mu, sigma
```

### NLL Log-Normal com Jacobiano log1p

A loss de quantidade minimiza a NLL de uma Log-Normal no espaço `log1p(x)`:

```python
# Na prática: trabalhamos em y = log1p(x) (espaço logarítmico)
# O decoder emite (mu, sigma) para y ~ N(mu, sigma)
# Mas queremos a NLL para x, não para y
# O Jacobiano da transformação y = log1p(x) é dy/dx = 1/(1+x)
# Portanto: -log p(x) = -log p(y) + log|dy/dx|^{-1}
#                     = NLL_gaussiana(y; mu, sigma) + log(1+x)
#                     = NLL_gaussiana(y; mu, sigma) + y   ← (pois y = log1p(x))

nll_wet = 0.5 * mean(
    ((y - mu) / sigma)^2      # distância ao centro em unidades de sigma
    + 2 * log(sigma)           # penalidade por sigma grande
    + y                        # jacobiano log1p (= log(1+x))
)
```

**Por que incluir o Jacobiano?** Sem o Jacobiano, estaríamos minimizando a NLL de `y = log1p(x)`, mas `y` não é o que queremos modelar — queremos modelar `x`. O Jacobiano corrige para a mudança de variável. É a diferença entre "a curva gaussiana no espaço log" e "a curva Log-Normal no espaço original".

### Comparação com `hurdle_simple.py`

Curiosamente, `hurdle_simple.py` e `hurdle_vae_cond_nll.py` usam a mesma Log-Normal como distribuição base para a quantidade. A diferença é:

| Aspecto | `hurdle_simple` | `hurdle_vae_cond_nll` |
|---------|-----------------|------------------------|
| Parâmetros (μ, σ) | MLP direto (sem encoder) | Decoder de CVAE (via espaço latente) |
| Encoder | Não tem | `_ConditionalMiniVAE.encoder` |
| Espaço latente | Não tem | Latente de dim 64 |
| KL Regularização | Não tem | KL(q(z\|x,c) ‖ N(0,I)) |
| Condicionamento na ocorrência | ✗ | ✓ (encoder + decoder) |

O `hurdle_simple` é mais direto: a MLP aprende (μ, σ) direto dos dados. O `hurdle_vae_cond_nll` passa pelo VAE, que tem um espaço latente intermediário e regularização KL — isso pode ajudar a suavizar a geração, mas também introduz mais parâmetros.

### Geração

```python
def sample(self, n):
    # Passo 1: Ocorrência (idêntico ao hurdle_vae_cond)
    z_o = N(0, I) → decode_occ → sigmoid → Bernoulli → occ_sim

    # Passo 2: Quantidade — agora AMOSTRAMOS de (μ, σ)
    z_a = N(0, I)
    mu, sigma = amt_vae.decode(z_a, occ_sim)  # μ e σ no espaço log1p

    noise  = N(0, I)                   # ruído de geração
    log_a  = mu + sigma * noise        # amostrar y ~ N(mu, sigma)
    a_raw  = expm1(log_a.clamp(max=20)) # converter de log1p para original

    precip = a_raw * occ_sim
```

Diferença importante em relação ao `hurdle_vae_cond`: aqui **geramos amostras da distribuição** `N(μ, σ)`, em vez de usar `μ` diretamente. Isso produz mais variabilidade nas quantidades geradas — mais realista.

### FAQ

**P: NLL é sempre melhor que MSE?**
R: Não necessariamente. A NLL mede a log-probabilidade dos dados sob o modelo — se o modelo estiver bem calibrado (σ correto), a NLL é melhor. Mas se σ for mal estimado, a NLL pode ser pior. Com poucos dados (~3.000 dias), o MSE pode ser mais estável.

**P: Por que `softplus` em vez de `exp` para σ?**
R: `exp(x)` cresce muito rápido e pode gerar gradientes explosivos. `softplus(x) = log(1 + exp(x))` é sempre positivo mas cresce linearmente para valores grandes — mais estável numericamente.

---

## 7.13 flow_match_film.py — Flow Matching com FiLM

### A Pergunta

> "FiLM (Feature-wise Linear Modulation) na rede de velocidade melhora o Flow Matching, mesmo sem condicionamento extra? Arquitetura residual com FiLM é mais expressiva que MLP simples?"

O `flow_match.py` usa uma MLP simples: `cat([x_t, t_emb]) → Linear+ReLU × N → velocidade`. O `flow_match_film.py` substitui isso por uma **arquitetura residual** onde o embedding de tempo modula cada camada via FiLM.

### O que é FiLM?

**FiLM** (Feature-wise Linear Modulation) é uma técnica de condicionamento que, em vez de simplesmente concatenar a condição ao input, usa a condição para gerar **parâmetros de escala e translação** que modulam as ativações da rede:

```
Sem FiLM (concatenação):    h = ReLU(W * cat([x, c]) + b)
Com FiLM:                   h = base(x)
                            scale, shift = Proj(c).split(2)
                            h_modulado = h * (1 + scale) + shift
```

**Analogia:** Pense numa receita de bolo (a rede base) onde você pode ajustar "mais doce" ou "mais salgado" a cada etapa. FiLM é como um chef que, a cada passo da receita, ajusta os temperos com base na ocasião (a condição `c`). Concatenação é como adicionar os ingredientes da ocasião junto com os ingredientes normais — mais rígido.

### Arquitetura: `_FiLMLayer`

```python
class _FiLMLayer(nn.Module):
    def __init__(self, hidden_dim, t_embed_dim):
        self.linear    = Linear(hidden_dim, hidden_dim)
        self.film_proj = Linear(t_embed_dim, hidden_dim * 2)  # → (scale, shift)

        # INICIALIZAÇÃO ZERO: começa como flow_match puro
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, t_embed):
        h = SiLU(self.linear(x))          # transformação base (sem condição)

        film_params = self.film_proj(t_embed)
        scale, shift = film_params.chunk(2, dim=-1)

        h_film = h * (1 + scale) + shift  # modulação FiLM

        return x + h_film                  # conexão RESIDUAL
```

### Truque de Inicialização com Zeros

No início do treino, `film_proj` é inicializado com **pesos e bias zero**:
- `scale = 0` → `h * (1 + 0) = h` (sem escala)
- `shift = 0` → `h + 0 = h` (sem translação)
- Resultado: `_FiLMLayer` se comporta como identidade + conexão residual = `x + SiLU(Linear(x))`

Isso significa que o modelo começa como um Flow Matching simples (sem FiLM) e **gradualmente aprende a usar o tempo** para modular as camadas. Isso estabiliza o início do treino.

### Arquitetura Completa da Rede de Velocidade

```
z_t (15 dim)  ──── input_proj ────► hidden (256 dim)
                                          │
t_embed (64 dim) ──────────────────► _FiLMLayer × 4 (residual)
                                          │
                                    output_proj ──► velocidade (15 dim)
```

Comparação com `flow_match.py`:

| | `flow_match.py` | `flow_match_film.py` |
|-|-----------------|----------------------|
| Arquitetura | MLP linear (Linear+ReLU) | Residual + FiLM |
| Como t é usado | Concatenado ao input | Modula cada camada (FiLM) |
| Conexões residuais | ✗ | ✓ (`x + h_film`) |
| Ativação | ReLU | SiLU (Swish) |
| Parâmetros (aprox.) | menor | maior |

### Loss e Sampling: Idênticos ao flow_match.py

O treinamento é exatamente o mesmo (MSE entre velocidade predita e alvo):
```
z_0 ~ N(0, I),   z_1 = x_dado
t ~ Uniform(0, 1)
z_t = (1-t)*z_0 + t*z_1
loss = ||v_θ(z_t, t) - (z_1 - z_0)||²
```

O sampling também suporta Euler e Heun (idêntico ao `flow_match.py`).

### FAQ

**P: FiLM é melhor que simplesmente concatenar o embedding de tempo?**
R: Depende. FiLM é especialmente poderoso quando a condição precisa influenciar o processamento em múltiplas escalas (como em síntese de imagem). Para dados 1D de precipitação (~15 dimensões), a diferença prática pode ser pequena. O benefício teórico é que FiLM permite interações multiplicativas entre a condição e as features, enquanto concatenação só permite adições lineares.

**P: Por que SiLU (Swish) em vez de ReLU?**
R: SiLU = `x * sigmoid(x)` é suave em zero (ReLU tem uma descontinuidade na derivada em x=0) e tem gradientes não-nulos para x < 0. Para redes com conexões residuais e FiLM, SiLU tende a treinar mais suavemente.

**P: `output_proj` também é inicializado com zeros. Por quê?**
R: Para que a velocidade inicial seja próxima de zero no começo do treino. Uma velocidade inicial muito grande poderia criar trajetórias caóticas e loss explosiva nos primeiros passos.

---

## 7.14 glow.py — GLOW (Fluxo Normalizante com Mixing Completo)

### A Pergunta

> "O que é GLOW e em que difere do RealNVP? O RealNVP usa máscaras fixas para decidir quais dimensões transformar — será que uma mistura completa de dimensões a cada passo (como GLOW) ajuda?"

O RealNVP (seção 7.5) alterna entre transformar as dimensões pares e ímpares — um padrão fixo. O GLOW adiciona uma camada de **mistura completa**: em cada bloco, uma matriz invertível `W` reorganiza **todas** as dimensões antes do acoplamento afim.

### Os Três Componentes de Cada Bloco GLOW

```
Entrada x
    │
    ▼
┌─────────────┐
│  ActNorm    │  ← Normalização aprendida por ativação
└─────────────┘
    │
    ▼
┌─────────────────────┐
│ InvertibleLinearLU  │  ← Rotação completa: y = x @ W
└─────────────────────┘
    │
    ▼
┌──────────────────┐
│ AffineCouplingGLOW│  ← Acoplamento afim (como RealNVP)
└──────────────────┘
    │
    ▼
Saída y
```

### 1. ActNorm — Normalização por Ativação

```python
class ActNorm(nn.Module):
    def __init__(self, input_size):
        self.loc       = Parameter(zeros(1, S))   # bias aprendido
        self.log_scale = Parameter(zeros(1, S))   # escala aprendida

    def forward(self, x):
        # INICIALIZAÇÃO: na primeira iteração, calibra como batch norm
        if not self.initialized:
            self.loc.copy_(-x.mean(dim=0))
            self.log_scale.copy_(-log(x.std(dim=0) + 1e-6))

        y = x * exp(log_scale) + loc
        log_det = log_scale.sum()  # Σ log|escala por dimensão|
        return y, log_det
```

**O que faz:** Normaliza as ativações por canal (dimensão), de forma análoga ao Batch Normalization, mas com parâmetros aprendidos e determinísticos na inferência. Isso evita que as camadas de acoplamento recebam ativações muito grandes ou pequenas.

**Diferença do BatchNorm:** O BatchNorm usa estatísticas do batch atual (varia entre batches). O ActNorm inicializa com o primeiro batch, mas depois usa parâmetros fixos aprendidos — completamente determinístico na geração.

### 2. InvertibleLinearLU — Rotação Completa via Decomposição LU

```python
class InvertibleLinearLU(nn.Module):
    # W = P @ L @ U  (decomposição LU com pivoteamento)
    # P: permutação fixa (não aprendida)
    # L: triangular inferior, diagonal = 1 (aprendida)
    # U: triangular superior, diagonal = sign_s * exp(log_s) (aprendida)

    def forward(self, x):
        W = assemble_W()    # reconstrói W a partir de P, L, U
        y = x @ W           # mistura linear completa de TODAS as dimensões
        log_det = log_s.sum()  # O(D) em vez de O(D³)!
```

**Por que LU?** A operação `y = x @ W` (multiplicação por matriz) é facilmente invertível se `W` for invertível. Mas calcular `log|det(W)|` naïvamente custa `O(D³)`. Com decomposição LU, `log|det(W)| = Σ log|diag(U)|` — custo `O(D)`.

**Por que é poderoso:** Cada passo do GLOW mistura TODAS as 15 dimensões com uma rotação aprendida. O RealNVP só mistura metade (par/ímpar). Depois de `n_steps` blocos, as dimensões se "conversam" muito mais umas com as outras.

**Analogia:** RealNVP é como um mixer de música que só processa canais pares ou ímpares de cada vez. GLOW é como um mixer que aplica uma equalização em todos os canais simultaneamente, mas de forma que você ainda consiga desfazer.

### 3. AffineCouplingGLOW — Acoplamento Afim Simples

```python
class AffineCouplingGLOW(nn.Module):
    def forward(self, x):
        x1, x2 = x[:, :n_half], x[:, n_half:]  # divide ao meio
        st = MLP(x1)                             # MLP só vê a primeira metade
        s, t = st.chunk(2, dim=-1)
        s = tanh(s)                              # limita s
        y2 = x2 * exp(s) + t                    # transforma a segunda metade
        log_det = s.sum(dim=-1)
        return cat([x1, y2]), log_det
```

Esta camada é simples (sem condicionamento de mês). A inteligência de mistura está no `InvertibleLinearLU` acima dela.

### Diferença vs. RealNVP

| | `real_nvp.py` | `glow.py` |
|-|---------------|-----------|
| Mistura de dimensões | Máscara fixa (par/ímpar) | W invertível completo (LU) |
| Normalização por ativação | ✗ | ActNorm ✓ |
| Log-det | `s.sum()` | `s.sum() + log_s.sum() + log_scale.sum()` |
| Parâmetros por bloco | 1 camada (coupling) | 3 camadas (ActNorm + LU + coupling) |
| Capacidade de expressão | Menor | Maior |

### Loss e Sampling

```python
def loss(self, x):
    z = x
    log_det_total = 0
    for layer in self.layers:  # layers = [ActNorm, InvLinearLU, AffCoupling] × n_steps
        z, log_det = layer(z)
        log_det_total += log_det
    log_pz = -0.5 * (z² + log(2π)).sum(dim=-1)
    nll = -(log_pz + log_det_total).mean()

def sample(self, n):
    z = N(0, I)
    for layer in reversed(self.layers):
        z = layer.inverse(z)  # ActNorm, InvLinear e AffCoupling têm inverse()
    return z
```

### FAQ

**P: Por que P (permutação) é fixa e não aprendida?**
R: A permutação `P` é inicializada como a permutação LU de uma matriz ortogonal aleatória. Como é uma permutação de coordenadas (log-det = 0), não contribui para a likelihood. Aprendê-la não ajudaria, então é mantida fixa para reduzir parâmetros.

**P: `ActNorm` tem `initialized = False` — o que acontece após salvar e carregar o modelo?**
R: Ao carregar um modelo treinado, o `initialized` volta para `False`. Isso significa que o próximo forward reinicializará o ActNorm com os dados daquele batch. Para evitar isso em avaliação, chame `model.eval()` antes — em modo de avaliação, o PyTorch não reaplica a inicialização.

**P: GLOW é mais lento que RealNVP?**
R: Sim, marginalmente. O `InvertibleLinearLU.inverse()` requer `torch.linalg.inv(W)` — uma inversão de matriz `O(D³)`. Para `D = 15` dimensões, isso é rápido, mas para imagens de alta resolução seria gargalo.

---

## 8. Condicionamento por Mês (Modelos _mc)

### 8.0 Por que Condicionar no Mês do Ano?

Imagine que você pediu ao modelo uma amostra de precipitação. Ele gera um vetor de 15 valores. Mas **qual dia do ano** é esse? Um modelo incondicional mistura todos os meses — ele gera "um dia qualquer de qualquer época do ano". Isso é problemático porque:

**Sazonalidade é forte no Brasil:** O regime de chuvas é completamente diferente entre o verão (dezembro–março) e o inverno (junho–agosto). Um dia de julho em Brasília tem ~80% de chance de ser completamente seco. Um dia de dezembro tem ~60% de chance de chuva.

```
Distribuição aprendida por modelo INCONDICIONAL:
    p(x) = Σ_{m=1}^{12} p(x | mês=m) × p(mês=m)
         = uma mistura de 12 distribuições mensais

Distribuição aprendida por modelo CONDICIONAL:
    p(x | mês=m) — especifica para o mês m
```

**Analogia:** Você pergunta para um meteorologista "vai chover amanhã?". Se ele não souber em qual mês estamos, vai dar uma resposta média para o ano todo. Se ele souber que é julho, vai dizer "muito improvável". A condição (mês) transforma uma previsão vaga em uma previsão específica.

### O que é `nn.Embedding`?

`nn.Embedding` é uma **tabela de lookup** (dicionário) onde cada número inteiro (categoria) é mapeado para um vetor de números reais (embedding):

```
Mês 0  (janeiro)  → [0.3, -0.7, 1.2, 0.5, -0.1, 0.8]    ← vetor aprendido
Mês 1  (fevereiro) → [0.2, -0.6, 1.1, 0.4, -0.2, 0.7]
...
Mês 11 (dezembro)  → [0.4, -0.8, 1.3, 0.6, -0.3, 0.9]
```

```python
# Criando um embedding de 12 meses com vetores de 6 dimensões:
embedding = nn.Embedding(num_embeddings=12, embedding_dim=6)

# Usando:
meses = torch.LongTensor([0, 5, 11])  # janeiro, junho, dezembro
vetores = embedding(meses)             # → Tensor(3, 6)
```

**Por que 6 dimensões para 12 meses?** Poderia ser 12 (one-hot), mas vetores de dimensão menor são aprendidos pelo gradiente e podem capturar **relações entre meses**. Por exemplo, o modelo pode aprender que janeiro e fevereiro são similares (ambos verão), e que são opostos a julho (inverno). Essas relações emergem do treino.

**Analogia:** É como um dicionário onde cada país tem um cartão de visita com informações resumidas (idioma, fuso, clima). Em vez de dizer "você está na França", você passa o cartão da França — um resumo de 6 números que o modelo aprende a usar.

### `set_cond_distribution()`: Compatibilidade Retroativa

Todos os modelos `_mc` têm um método `set_cond_distribution()`:

```python
def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
    """
    Armazena as probabilidades empíricas dos meses no treino.
    Quando sample(n, cond=None), sorteia meses dessa distribuição.
    """
    # Conta quantas vezes cada mês aparece no conjunto de treino
    arr = cond_arrays["month"]   # ex: [0, 0, 1, 0, 2, ...]  (N dias)
    counts = np.bincount(arr, minlength=12)  # ex: [250, 220, 280, ..., 260]
    self._cond_probs["month"] = counts / counts.sum()  # normaliza para prob
```

Sem isso, `sample(n)` precisaria receber sempre um `cond` explícito (ex: `{"month": tensor([0]*1000)}`). Com `set_cond_distribution`, o modelo pode ser usado como um modelo incondicional (`sample(n)`) e ainda assim respeitar a distribuição sazonal dos dados — mantendo compatibilidade com o pipeline de comparação (`compare.py`).

---

### 8.1 conditioning.py — Infraestrutura Compartilhada

Todos os modelos `_mc` importam de `conditioning.py`. Este arquivo centraliza a lógica de embedding para facilitar extensões futuras.

#### `ConditioningBlock`: Múltiplos Condicionadores

```python
# Configuração atual: apenas mês
DEFAULT_CATEGORICALS = [("month", 12, 6)]
# Futuramente: mês + fase ENSO + estação do ano
# DEFAULT_CATEGORICALS = [("month", 12, 6), ("enso", 3, 4), ("season", 4, 3)]

class ConditioningBlock(nn.Module):
    def __init__(self, categoricals: list[tuple[str, int, int]]):
        # Cria uma tabela nn.Embedding para CADA condicionador
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(n_classes, embed_dim)
            for name, n_classes, embed_dim in categoricals
        })
        self.total_dim = sum(embed_dim for _, _, embed_dim in categoricals)
        # Para DEFAULT_CATEGORICALS: total_dim = 6

    def forward(self, cond: dict) -> Tensor:
        # cond = {"month": LongTensor(B,)}
        # Retorna: concatenação de todos os embeddings → (B, total_dim)
        parts = [self.embeddings[name](cond[name]) for name, _, _ in self.categoricals]
        return torch.cat(parts, dim=1)   # (B, 6)
```

#### Design para Extensibilidade

A grande vantagem desta arquitetura: para adicionar um novo condicionador (ex: fase ENSO: La Niña / Neutro / El Niño), você:
1. Atualiza `DEFAULT_CATEGORICALS` para incluir `("enso", 3, 4)`
2. Garante que `data_utils.py` retorne os labels ENSO
3. **Não modifica nenhum modelo** — eles usam `self.cond_block.total_dim` automaticamente

```python
# ANTES: total_dim = 6  (só mês)
block = ConditioningBlock([("month", 12, 6)])
c_emb = block({"month": tensor})   # → (B, 6)

# DEPOIS: total_dim = 10  (mês + ENSO)
block = ConditioningBlock([("month", 12, 6), ("enso", 3, 4)])
c_emb = block({"month": t_month, "enso": t_enso})   # → (B, 10)
# ← Os modelos recebem c_emb e usam self.cond_block.total_dim — sem mudanças!
```

#### `sample_cond()`: Sortear Condições

```python
def sample_cond(self, n: int, probs: dict[str, np.ndarray]) -> dict:
    """
    Sorteia n condições da distribuição empírica armazenada.

    probs = {"month": array([0.085, 0.079, 0.088, ...])}
            ←  8.5% dos dias de treino são janeiro, etc.
    """
    return {
        name: torch.LongTensor(
            np.random.choice(n_classes, size=n, p=probs[name])
        )
        for name, n_classes, _ in self.categoricals
    }
```

---

### 8.2 hurdle_simple_mc.py — HurdleSimpleMC

**Base:** `hurdle_simple.py` (melhor modelo do projeto)

**Mudança:** O embedding mensal `c_emb` é concatenado à entrada das duas MLPs.

#### Como o Embedding é Injetado

```python
class _OccurrenceMLP(nn.Module):
    # ANTES (hurdle_simple): MLP(zeros(S))        → p_rain (S,)
    # AGORA (mc):            MLP(cat([zeros, c_emb])) → p_rain (S,)
    def __init__(self, input_size, embed_dim, hidden=32):
        self.net = nn.Sequential(
            nn.Linear(input_size + embed_dim, hidden),  # ← entrada maior!
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x, c_emb):
        return sigmoid(self.net(cat([x, c_emb], dim=1)))

class _AmountMLP(nn.Module):
    # ANTES: MLP(occ_mask)        → (mu, sigma)
    # AGORA: MLP(cat([occ_mask, c_emb])) → (mu, sigma)
    def forward(self, occ_mask, c_emb):
        out = self.net(cat([occ_mask, c_emb], dim=1))
        mu    = out[:, :S]
        sigma = softplus(out[:, S:]) + 1e-4
        return mu, sigma
```

#### Loss Condicional

```python
def loss(self, x, beta=1.0, cond=None):
    # Se cond=None: sorteia meses da distribuição empírica do treino
    if cond is None:
        cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs)
    cond = {k: v.to(x.device) for k, v in cond.items()}
    c_emb = self.cond_block(cond)  # (B, 6)

    # BCE: ocorrência condicionada no mês
    p_rain = self.occ_mlp(zeros_like(x), c_emb)
    bce = BCE(p_rain, occ_target)

    # NLL Log-Normal: quantidade condicionada no mês + ocorrência
    mu, sigma = self.amt_mlp(occ_target, c_emb)
    nll = NLL_lognormal(x[wet], mu[wet], sigma[wet])

    return bce + nll
```

#### Exemplo: Comparar Chuva em Janeiro vs. Julho

```python
# Gerar 1000 amostras de JANEIRO (mês 0)
cond_jan = {"month": torch.zeros(1000, dtype=torch.long)}  # 1000 × mês 0
samples_jan = model.sample(1000, cond=cond_jan)

# Gerar 1000 amostras de JULHO (mês 6)
cond_jul = {"month": torch.full((1000,), 6, dtype=torch.long)}
samples_jul = model.sample(1000, cond=cond_jul)

# Comparar frequência de dias com chuva:
wet_jan = (samples_jan > 0).float().mean()
wet_jul = (samples_jul > 0).float().mean()
print(f"Janeiro: {wet_jan:.1%} dias com chuva")  # ex: 65%
print(f"Julho:   {wet_jul:.1%} dias com chuva")  # ex: 22%
```

#### FAQ

**P: O hurdle_simple_mc inclui cópulas gaussianas?**
R: Sim! O `fit_copulas(data_raw)` é herdado do `hurdle_simple` — as matrizes de correlação espacial são estimadas e usadas no `sample()`. A única diferença é que `p_rain` e `(mu, sigma)` agora dependem do mês.

**P: E se eu não chamar `set_cond_distribution`?**
R: O `self._cond_probs` estará vazio, e `sample_cond()` lançará um erro. Sempre chame `set_cond_distribution` antes de usar `sample(n)` sem `cond` explícito.

---

### 8.3 vae_mc.py — VAEModelMC

**Base:** `vae.py`

**Mudança:** `c_emb` concatenado ao encoder (durante a compressão) **e** ao decoder (durante a geração).

#### Por que nos Dois Lados?

**No encoder:** "Sabendo que é janeiro, comprime os dados de forma diferente." O espaço latente pode organizar as amostras por mês — cluster de janeiro vs. cluster de julho.

**No decoder:** "Ao descomprimir, use o contexto do mês para reconstruir corretamente." Sem isso, o decoder teria que "adivinhar" o mês a partir de `z` — informação que o encoder tem, mas que não é necessariamente preservada inteiramente em `z` após a regularização KL.

```python
class VAEModelMC(BaseModel):
    def __init__(self, input_size, latent_size, ...):
        E = cond_block.total_dim  # 6

        # Encoder recebe (input_size + E) dimensões
        self.encoder = MLP(input_size + E → latent*4 → latent*2 → latent)
        self.fc_mu    = Linear(latent → latent)
        self.fc_logvar = Linear(latent → latent)

        # Decoder recebe (latent_size + E) dimensões
        self.decoder = MLP(latent + E → latent*2 → latent*4 → input_size)

    def encode(self, x, c_emb):
        h = self.encoder(cat([x, c_emb], dim=1))  # ← mês + dados
        return fc_mu(h), fc_logvar(h)

    def loss(self, x, beta=1.0, cond=None):
        c_emb = self.cond_block(cond)
        mu, logvar = self.encode(x, c_emb)
        z = reparameterize(mu, logvar)

        x_hat = self.decoder(cat([z, c_emb], dim=1))  # ← mês + latente

        recons = MSE(x_hat, x)
        kl = KL(mu, logvar)
        return recons + beta * kl

    def sample(self, n, cond=None):
        c_emb = cond_block(cond)
        z = N(0, I)
        return self.decoder(cat([z, c_emb], dim=1))
```

#### Diagrama do Fluxo

```
Treino:
    x (15) ──cat──► encoder ──► mu, logvar
    c (6)  ──/                       │
                              reparametrize
                                     │
                                     z (128)
                                     │
    c (6)  ──cat──► decoder ──► x_hat (15)
    z ─────────/

Geração:
    c (6)  ──cat──► decoder ──► sample (15)
    z~N(0,I)──/
```

---

### 8.4 real_nvp_mc.py — RealNVPMC

**Base:** `real_nvp.py`

**Mudança:** Cada camada de acoplamento `_CouplingLayerCond` recebe `cat([x1, c_emb])` no MLP interno.

#### Acoplamento Afim Condicionado

```python
class _CouplingLayerCond(nn.Module):
    # RealNVP padrão:  MLP(x1)              → (s, t)
    # RealNVPMC:       MLP(cat([x1, c_emb])) → (s, t)

    def __init__(self, input_size, embed_dim, mask, hidden=256):
        n_active = mask.sum()
        n_free   = input_size - n_active

        # Agora MLP recebe x1 (n_active) + embedding (embed_dim)
        self.net = MLP(n_active + embed_dim → hidden → hidden → n_free*2)

    def forward(self, x, c_emb):
        x1, x2 = x[:, mask], x[:, ~mask]
        st = self.net(cat([x1, c_emb], dim=-1))  # ← c_emb aqui!
        s, t = st.split(n_free, dim=-1)
        s = tanh(s)
        y2 = x2 * exp(s) + t
        log_det = s.sum(dim=-1)
        ...
```

#### Likelihood Condicional Exata

A grande vantagem do fluxo normalizante condicional: **a likelihood é exata**:

```
log p(x | mês=m) = log p(z) + Σ_i log|det J_i(x | c_m)|
```

O condicionamento `c_m` **não** quebra a invertibilidade do fluxo — cada camada ainda é bijetiva dado o condicionamento. Você pode calcular:
1. `log p(x | janeiro)` — quão provável é esta amostra em janeiro?
2. `log p(x | julho)` — quão provável é em julho?
3. Comparar: se a razão for grande, `x` é muito mais um "dia de janeiro"

#### Diagrama de Fluxo

```
Treino (Data → Latente):
    x ──────────────────────────────────────────────────────► z
         scale → coupling(x, c) → coupling(x, c) → ...
    c ────────/──────────────/──────────────/

Geração (Latente → Data):
    z ──────────────────────────────────────────────────────► x
         coupling⁻¹(z, c) → coupling⁻¹(z, c) → ... → scale⁻¹
    c ────────────/────────────────/
```

---

### 8.5 glow_mc.py — GlowMC

**Base:** `glow.py`

**Mudança:** Apenas as camadas `_AffineCouplingCond` recebem `c_emb`. `ActNorm` e `InvertibleLinearLU` **não** recebem.

#### Por que não Condicionar o ActNorm e o InvertibleLinearLU?

```python
# glow_mc.py — log_prob condicional
for layer in self.layers:
    if isinstance(layer, _AffineCouplingCond):
        z, log_det = layer(z, c_emb)  # ← recebe mês
    else:
        z, log_det = layer(z)          # ActNorm e InvLinear: sem condição
```

- **ActNorm**: É uma normalização de ativação — garante que os valores tenham média ~0 e variância ~1 antes de cada acoplamento. Isso é uma operação de "limpeza" que não depende do conteúdo semântico (mês). Condicionar o ActNorm mudaria a normalização por mês — uma complicação desnecessária.

- **InvertibleLinearLU**: É uma rotação aprendida para misturar todas as dimensões. Essa rotação captura **correlações espaciais** entre as 15 estações — algo que não muda com o mês. O que muda com o mês são as *distribuições marginais*, capturadas pelo acoplamento afim.

- **`_AffineCouplingCond`**: É onde a distribuição das quantidades é modelada. Faz sentido que a forma dessa distribuição dependa do mês.

#### `_AffineCouplingCond`: Simples e Eficaz

```python
class _AffineCouplingCond(nn.Module):
    def __init__(self, input_size, embed_dim, hidden=128):
        n_half = input_size // 2
        n_out  = input_size - n_half
        # MLP recebe: primeira metade de x + embedding mensal
        self.net = MLP(n_half + embed_dim → hidden → hidden → n_out*2)

    def forward(self, x, c_emb):
        x1, x2 = x[:, :n_half], x[:, n_half:]
        st = self.net(cat([x1, c_emb], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = tanh(s)
        y2 = x2 * exp(s) + t
        log_det = s.sum(dim=-1)
        return cat([x1, y2]), log_det
```

---

### 8.6 flow_match_mc.py — FlowMatchingMC

**Base:** `flow_match.py`

**Mudança:** A rede de velocidade recebe `cat([x_t, t_emb, c_emb])`.

#### Rede de Velocidade Condicional

```python
class _VelocityMLPCond(nn.Module):
    def __init__(self, data_dim, t_embed_dim, cond_dim, hidden, n_layers):
        in_dim = data_dim + t_embed_dim + cond_dim  # D + T + E = 15 + 64 + 6 = 85
        # MLP simples (sem FiLM, sem residual)
        layers = [Linear(in_dim, hidden), ReLU()]
        for _ in range(n_layers - 1):
            layers += [Linear(hidden, hidden), ReLU()]
        layers.append(Linear(hidden, data_dim))
        self.net = Sequential(*layers)

    def forward(self, x_t, t_embed, c_emb):
        inp = cat([x_t, t_embed, c_emb], dim=-1)  # concatenação simples
        return self.net(inp)
```

O embedding de tempo `t_emb` (64 dim) e o embedding mensal `c_emb` (6 dim) entram juntos com `x_t` (15 dim) como input da rede de velocidade.

#### Loss e Sampling Condicionais

```python
def loss(self, x, beta=1.0, cond=None):
    c_emb = cond_block(cond)   # (B, 6)

    z_0 = N(0, I)
    t = Uniform(0, 1)
    z_t = (1-t)*z_0 + t*x
    target = x - z_0

    t_emb = t_embed(t)
    v_pred = velocity(z_t, t_emb, c_emb)   # ← c_emb aqui!

    return MSE(v_pred, target)

def sample(self, n, cond=None):
    c_emb = cond_block(cond)
    z = N(0, I)
    for i in range(num_steps):
        t = i * dt
        v = velocity(z, t_embed(t), c_emb)   # ← c_emb em cada passo!
        z = z + v * dt
    return z
```

**Importante:** o `c_emb` é calculado uma vez e reutilizado em **todos os passos** da integração ODE. O mês não muda ao longo da trajetória.

---

### 8.7 flow_match_film_mc.py — FlowMatchingFilmMC

**Base:** `flow_match_film.py` (FiLM) + condicionamento mensal

**Mudança:** Em vez de concatenar `c_emb` ao input, `c_emb` e `t_emb` são **combinados** para gerar os parâmetros FiLM.

#### Diferença Chave: FiLM vs. Concatenação

```python
# flow_match_mc.py — concatenação simples:
inp = cat([x_t, t_emb, c_emb], dim=-1)   # (B, 15+64+6)
v = MLP(inp)

# flow_match_film_mc.py — FiLM com mês E tempo juntos:
cond_emb = cat([t_emb, c_emb], dim=-1)   # (B, 64+6 = 70)
h = input_proj(x_t)                        # (B, 256) — só os dados
for layer in hidden_layers:
    h = FiLMCondLayer(h, cond_emb)         # modula h com tempo+mês
v = output_proj(h)
```

#### `_FiLMCondLayer`: Modulação com Vetor de Contexto Combinado

```python
class _FiLMCondLayer(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        # cond_dim = t_embed_dim + c_embed_dim = 64 + 6 = 70
        self.linear    = Linear(hidden_dim, hidden_dim)
        self.film_proj = Linear(cond_dim, hidden_dim * 2)  # (scale, shift)

        # Inicialização zero: começa incondicional
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, cond_emb):
        h = SiLU(self.linear(x))              # transformação base
        film_params = self.film_proj(cond_emb) # escala e translação
        scale, shift = film_params.chunk(2, dim=-1)
        h_film = h * (1 + scale) + shift       # modulação conjunta tempo+mês
        return x + h_film                       # residual
```

#### Diagrama Comparativo

```
flow_match_mc  (concatenação):
    x_t ──────────────────────── MLP ──► velocidade
    t_emb ─ cat ─ → input ─ /
    c_emb ─ /

flow_match_film_mc  (FiLM):
    x_t ──── input_proj ──► h ──┬── FiLMLayer ──► h' ──► output_proj ──► velocidade
                                 │       ↑
    t_emb ─ cat ─ → cond_emb ──────────/
    c_emb ─ /                   ↑ (em CADA camada)
```

A vantagem do FiLM: o contexto `(t_emb, c_emb)` modula as features em **cada camada**, permitindo interações mais ricas. A concatenação injeta o contexto apenas na entrada — a rede precisa "lembrar" dessa informação ao longo de todas as camadas.

#### Inicialização Zero: Começa Incondicional

Com `film_proj` inicializado em zero:
- `scale = 0`, `shift = 0` → sem modulação
- O modelo começa equivalente ao `flow_match_film.py` (sem condicionamento de mês)
- Gradualmente aprende a usar o mês à medida que os gradientes chegam ao `film_proj`

Isso estabiliza o início do treino: o modelo primeiro aprende a gerar precipitação razoável (ignora o mês), depois refina para capturar sazonalidade.

#### FAQ

**P: Qual é melhor — concatenação (flow_match_mc) ou FiLM (flow_match_film_mc)?**
R: Para dados de baixa dimensão como precipitação (15 estações), a diferença prática é pequena. FiLM é teoricamente mais expressivo e tem vantagem para condicionamento complexo ou dados de alta dimensão. A concatenação é mais simples e interpretável.

**P: Por que combinar `t_emb` e `c_emb` antes do FiLM, em vez de ter dois FiLMs separados?**
R: Combiná-los permite que a rede capture interações entre tempo e mês — por exemplo, "em julho, no começo da trajetória (t pequeno), fazer X". Dois FiLMs separados não capturariam essa interação multiplicativa.

**P: Como treinar um modelo `_mc`?**
R: O `train.py` detecta automaticamente modelos com `set_cond_distribution` e injeta o mês durante o treino. Basta:
```bash
python train.py --model hurdle_simple_mc
python train.py --model vae_mc
python train.py --model flow_match_film_mc
# etc.
```

---

## 9. compare.py — A Pipeline de Comparação

### O que este arquivo faz?

`compare.py` automatiza todo o ciclo de comparação:
1. Opcionalmente treina todos os modelos
2. Carrega cada modelo do disco
3. Gera amostras e calcula métricas
4. Produz o composite score
5. Gera visualizações comparativas

### Como Modelos São Carregados

O carregamento é diferente por tipo de modelo:

```python
# Modelos neurais padrão (VAE, RealNVP, Flow Matching, etc.)
model = get_model(model_name, **kwargs)
checkpoint = torch.load("outputs/<model>/model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Cópula (salva com pickle)
with open("outputs/copula/copula.pkl", "rb") as f:
    model = pickle.load(f)

# Modelos com pré-ajustes especiais
if model_name == "latent_flow":
    model.fit_flow(data_raw, std_scale=std)          # restaura transformação

if model_name == "hurdle_temporal":
    model.fit_temporal(data_raw / std_scale)          # restaura janelas
```

### O Composite Score

```python
QUALITY_METRICS = [
    ("mean_wasserstein",       True),   # menor = melhor
    ("corr_rmse",              True),
    ("wet_day_freq_error_mean",True),
    ("extreme_q90_mean",       True),
    ("extreme_q95_mean",       True),
    ("extreme_q99_mean",       True),
    ("energy_score",           True),
]

# Normalização min-max por métrica
for metric_key, lower_is_better in QUALITY_METRICS:
    vals = [all_metrics[m][metric_key] for m in model_names]
    min_v, max_v = min(vals), max(vals)
    normalized = (v - min_v) / (max_v - min_v)   # 0 = melhor, 1 = pior
    composite += normalized / len(QUALITY_METRICS)
```

**Limitação do composite score:** A normalização min-max é relativa ao conjunto de modelos comparados. Se você adicionar um modelo muito bom ou muito ruim, os scores dos outros modelos mudam. Não é uma escala absoluta.

### Visualizações Geradas

1. **`comparison_quality.png`:** Bar charts por métrica — barras verdes = melhores, vermelhas = piores
2. **`radar.png`:** Spider/radar plot — cada eixo é uma métrica normalizada. Ideal: área pequena e circular
3. **`station_samples_comparison.png`:** Boxplots por estação para cada modelo — compara a distribuição amostrada com a real
4. **`overall_correlation_chart.png`:** Matrizes de correlação de cada modelo vs realidade

### Comandos Especiais

```bash
# Comparar apenas subconjunto de modelos (mais rápido)
python compare.py --models hurdle_simple copula vae

# Pular treinamento e usar modelos existentes
python compare.py --skip_training

# Testar variantes de solver/steps
python compare.py --solver_grid --models flow_match ldm

# Comparação rápida (poucas épocas, para debug)
python compare.py --max_epochs 50
```

---

## 10. Exemplos de Uso Completos

### 10.1 Treinar um Modelo do Zero

```bash
cd PrecipModels/

# Modelo mais simples e rápido (recomendado para começar)
python train.py --model copula

# Melhor modelo (~2 minutos)
python train.py --model hurdle_simple

# VAE padrão
python train.py --model vae

# LDM (demorado — 2 estágios de 1000 épocas cada)
python train.py --model ldm

# Com configuração customizada
python train.py --model vae \
    --max_epochs 1000 \
    --latent_size 64 \
    --lr 0.0001 \
    --name vae_experimento_1

# Com GPU (se disponível)
python train.py --model hurdle_simple --device cuda

# Retomar treinamento de onde parou
python train.py --model vae --resume
```

Saídas em `outputs/<nome_modelo>/`:
- `config.json` — hiperparâmetros usados
- `model.pt` — pesos do modelo
- `metrics.json` — métricas no conjunto de validação
- `training_loss.png` — curva de perda
- `training_history.json` — histórico completo

### 10.2 Comparar Todos os Modelos

```bash
# Treina todos e compara
python compare.py

# Compara apenas os modelos já treinados (skip training)
python compare.py --skip_training

# Compara subconjunto
python compare.py --models hurdle_simple copula vae real_nvp --skip_training
```

Saídas em `outputs/comparison/`:
- `composite_scores.json` — ranking final
- `comparison_quality.png` — barras por métrica
- `radar.png` — spider plot
- `comparison_report.txt` — tabela texto

### 10.3 Carregar e Usar um Modelo Treinado

```python
import torch
import pickle
import numpy as np
from models import get_model
from data_utils import load_data

# Carrega dados (para denormalizar e calcular métricas)
data_norm, data_raw, mu, std, station_names = load_data()

# ── Exemplo 1: Hurdle Simple ──────────────────────────────
model = get_model("hurdle_simple", input_size=data_norm.shape[1])
checkpoint = torch.load("outputs/hurdle_simple/model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Carrega cópulas (necessário para hurdle_simple)
model.fit_copulas(data_raw)  # usa dados de treino!

# Gera 1000 amostras
model.eval()
with torch.no_grad():
    samples = model.sample(1000)
print(f"Amostras geradas: {samples.shape}")  # (1000, 15)
print(f"Média por estação: {samples.mean(dim=0)}")

# ── Exemplo 2: Cópula ──────────────────────────────────────
with open("outputs/copula/copula.pkl", "rb") as f:
    copula = pickle.load(f)

samples_copula = copula.sample(1000)
print(f"Cópula — min: {samples_copula.min():.3f}, max: {samples_copula.max():.3f}")

# ── Exemplo 3: VAE ────────────────────────────────────────
vae = get_model("vae", input_size=data_norm.shape[1], latent_size=128)
checkpoint = torch.load("outputs/vae/model.pt")
vae.load_state_dict(checkpoint['model_state_dict'])

samples_vae = vae.sample(1000)
# Denormalizar (VAE usa standardize)
import json
with open("outputs/vae/config.json") as f:
    config = json.load(f)
# mu e std do treino seriam necessários para denormalizar corretamente
```

### 10.4 Adicionar um Novo Modelo ao Framework

Para adicionar um novo modelo, você precisa:

1. **Criar o arquivo do modelo** em `models/meu_modelo.py`:

```python
"""
meu_modelo.py — Descrição do novo modelo
"""
import torch
import torch.nn as nn
from base_model import BaseModel

class MeuModelo(BaseModel):
    def __init__(self, input_size=15, **kwargs):
        super().__init__()
        self.input_size = input_size
        # ... define camadas da rede
        self.rede = nn.Linear(input_size, input_size)

    def loss(self, x, beta=1.0):
        saida = self.rede(x)
        total = nn.functional.mse_loss(saida, x)
        return {'total': total}

    def sample(self, n, steps=None, method=None):
        device = next(self.parameters()).device
        z = torch.randn(n, self.input_size, device=device)
        with torch.no_grad():
            return self.rede(z)
```

2. **Registrar no `models/__init__.py`:**

```python
from .meu_modelo import MeuModelo

MODEL_NAMES = [..., "meu_modelo"]

def get_model(name, **kwargs):
    models = {
        ...,
        "meu_modelo": MeuModelo,
    }
    return models[name](**kwargs)
```

3. **Adicionar defaults no `train.py`:**

```python
MODEL_DEFAULTS["meu_modelo"] = {
    "normalization_mode": "scale_only",
    "max_epochs": 500,
    "latent_size": 0,
    "lr": 0.001,
    "batch_size": 128,
    "kl_warmup": 0,
}

ARCH_DEFAULTS["meu_modelo"] = {}  # parâmetros extras de arquitetura
```

4. **Adicionar ao argparse do `train.py`:**

```python
parser.add_argument("--model", choices=[..., "meu_modelo"], ...)
```

5. **Treinar e comparar:**

```bash
python train.py --model meu_modelo
python compare.py --models meu_modelo hurdle_simple copula
```

### 10.5 Interpretar as Saídas do Training

```json
{
  "mean_wasserstein": 1.603,
  "corr_rmse": 0.180,
  "wet_day_freq_error_mean": 0.080,
  "extreme_q90_mean": 2.140,
  "energy_score": 4.230,
  "wet_spell_length_error": 0.450,  ← duração de sequências chuvosas
  "dry_spell_length_error": 1.230,  ← duração de sequências secas
  "lag1_autocorr_error": 0.080,     ← autocorrelação temporal
  "coverage_80": 0.813,             ← cobertura do intervalo 80% (ideal: 0.80)
  "coverage_90": 0.898,             ← cobertura do intervalo 90% (ideal: 0.90)
  "sampling_time_ms": 45.3,         ← tempo para gerar 1000 amostras
  "n_parameters": 8432,             ← parâmetros da rede
  "evaluation_protocol": {
    "type": "temporal_holdout",
    "holdout_ratio": 0.2,
    "train_size": 2629,
    "eval_size": 658
  }
}
```

**Pontos de atenção:**
- `wet_day_freq_error > 0.15` → o modelo está errando muito na frequência de chuva
- `corr_rmse > 0.3` → correlações espaciais mal capturadas
- `coverage_80` muito abaixo de 0.80 → modelo sub-disperso (intervalos muito estreitos)
- `lag1_autocorr_error > 0.1` → persistência temporal mal capturada

---

## 11. Resultados e Conclusões

### Tabela Completa de Resultados

| Modelo            | Composite↓ | Wasserstein↓ | Wet Freq Err↓ | Parâmetros |
|-------------------|------------|--------------|---------------|------------|
| **hurdle_simple** | **0.140**  | 1.603        | **0.080**     | ~8.000     |
| vae               | 0.443      | 2.569        | 0.246         | ~270.000   |
| hurdle_vae        | 0.504      | 2.829        | 0.270         | ~150.000   |
| real_nvp          | 0.717      | 3.399        | 0.618         | ~1.500.000 |
| flow_match        | 0.724      | 3.600        | 0.496         | ~480.000   |
| copula            | 0.736      | 5.891        | 0.319         | 0          |
| best_v2_ldm*      | —          | **1.287**    | —             | ~500.000   |

*Medido separadamente em VAE_Tests, não diretamente comparável

### Principais Aprendizados

**1. Simplicidade vence com dados escassos:**
O `hurdle_simple` com ~8.000 parâmetros supera todos os modelos mais sofisticados com milhares de parâmetros. Com apenas ~3.000 dias de dados, modelos grandes sofrem de overfitting.

**2. Indução correta de estrutura é crucial:**
O hurdle model separa o problema de forma fisicamente motivada (ocorrência vs quantidade). Isso é mais eficiente do que deixar o modelo aprender essa estrutura do zero.

**3. Log-Normal é uma boa escolha para precipitação:**
Usar a distribuição correta (Log-Normal) em vez de MSE gaussiano faz diferença significativa na qualidade dos extremos gerados.

**4. Cópula gaussiana captura correlação espacial bem:**
A combinação rede neural (para distribuições marginais) + cópula gaussiana (para correlação espacial) é um híbrido poderoso.

**5. Modelos temporais não ajudam muito aqui:**
`hurdle_temporal` e `latent_flow` adicionam complexidade temporal mas não melhoram significativamente sobre `hurdle_simple`. Isso sugere que, para este dataset, a memória temporal não é a limitação principal.

### Onde ir Daqui

- **Mais dados:** Com mais dados históricos, modelos mais complexos (LDM, latent_flow) poderiam superar o hurdle_simple
- **Condicionamento sazonal:** Adicionar mês do ano como feature (os modelos `_mc` fazem isso)
- **Covariáveis climáticas:** ENSO, temperatura do oceano, etc.
- **Modelos de área:** Interpolação espacial entre estações

---

## Apêndice: Glossário

| Termo | Definição |
|-------|-----------|
| **ActNorm** | Normalização de ativação aprendida por canal — inicializada como BatchNorm mas determinística na inferência (usada no GLOW) |
| **Batch** | Subconjunto dos dados processado em cada passo de gradiente |
| **BCE** | Binary Cross-Entropy — loss para classificação binária |
| **Conditioning** | Injetar informação extra (ex: mês do ano) nos modelos para gerar amostras específicas para aquela condição |
| **CVAE** | Conditional VAE — VAE que recebe uma variável de condição no encoder e decoder, aprendendo `p(x|c)` |
| **DDIM** | Denoising Diffusion Implicit Models — amostragem determinística para DDPM |
| **DDPM** | Denoising Diffusion Probabilistic Models — modelo de difusão |
| **ELBO** | Evidence Lower Bound — objetivo variacional do VAE |
| **EMA** | Exponential Moving Average — média ponderada exponencial dos parâmetros |
| **FiLM** | Feature-wise Linear Modulation — modulação de ativações via escala e translação aprendidas de uma condição externa (`h * (1 + scale) + shift`) |
| **GLOW** | Generative Flow with Invertible 1×1 Convolutions — fluxo normalizante com ActNorm + mistura completa de dimensões (InvertibleLinear) + acoplamento afim |
| **GRU** | Gated Recurrent Unit — rede neural recorrente com gates |
| **Hurdle Model** | Modelo com dois estágios: ocorrência (binária) + quantidade (contínua) |
| **KL Divergence** | Kullback-Leibler — medida de diferença entre distribuições |
| **Latente** | Representação comprimida dos dados em dimensão menor |
| **NLL** | Negative Log-Likelihood — negativo da log-probabilidade dos dados |
| **nn.Embedding** | Tabela de lookup do PyTorch: mapeia inteiros (categorias) para vetores reais aprendidos |
| **Normal-Score** | Transformação de ranque para scores gaussianos |
| **ODE** | Equação Diferencial Ordinária — equação de evolução temporal |
| **Softplus** | `log(1+exp(x))` — alternativa suave ao ReLU, sempre positivo |
| **Wasserstein** | Distância de transporte ótimo entre distribuições |

---

*Documento gerado em 03/03/2026 — Framework PrecipModels v1.1 (inclui modelos _mc e variantes adicionais)*
