# PrecipModels — Guia Completo

Comparação sistemática de 6 arquiteturas generativas para precipitação diária multivariada,
do baseline estatístico clássico ao Flow Matching moderno.

---

## 1. Visão Geral do Projeto

### Por que este projeto existe?

O projeto já possui experimentos extensos com VAEs (`VAE_Tests/`, 60+ experimentos) e
Flow Matching complexo (`LatentFlowMatching/`). O `PrecipModels/` responde à pergunta central:

> **Qual arquitetura generativa produz os melhores cenários sintéticos de precipitação?**

A progressão dos modelos segue uma narrativa científica:

| Modelo | Pergunta respondida |
|---|---|
| `copula` | O baseline da hidrologia consegue bons cenários? |
| `hurdle_simple` | Uma MLP com distribuições paramétricas bate a cópula? |
| `vae` | Um espaço latente contínuo melhora sobre distribuições fixas? |
| `hurdle_vae` | Separar ocorrência de quantidade ajuda o VAE? |
| `real_nvp` | Um fluxo normalizante (likelihood exato) supera o VAE? |
| `flow_match` | Flow Matching (trajetórias retas) supera o fluxo normalizante? |

---

## 2. Instalação e Dependências

```bash
pip install torch numpy scipy pandas matplotlib
```

Versões testadas: Python ≥ 3.10, PyTorch ≥ 2.0.

---

## 3. Estrutura de Arquivos

```
PrecipModels/
├── GUIDE.md              # este arquivo
├── data_utils.py         # carregamento e normalização dos dados
├── base_model.py         # classe abstrata BaseModel
├── metrics.py            # todas as métricas de avaliação
├── train.py              # treino unificado (argparse)
├── compare.py            # treina todos + gera comparação
└── models/
    ├── __init__.py       # registro dos modelos
    ├── copula.py         # Cópula Gaussiana
    ├── vae.py            # VAE (referência DL)
    ├── hurdle_simple.py  # Hurdle MLP + Log-Normal
    ├── hurdle_vae.py     # Hurdle com dois VAEs
    ├── real_nvp.py       # Normalizing Flow (RealNVP)
    └── flow_match.py     # Flow Matching (MLP puro)
```

---

## 4. Entendendo os Dados

### O dataset `inmet_relevant_data.csv`

- **Fonte**: INMET (Instituto Nacional de Meteorologia), rede de estações automáticas
- **Região**: 15 estações pluviométricas (variável `cd_inmet_station`)
- **Período**: múltiplos anos, resolução horária → agregado em diário (mm/dia)
- **Tamanho esperado após limpeza**: ~6700–6900 linhas sem NaN

### Estrutura após processamento (`data_utils.py`):

```python
data_norm, data_raw, mu, std, station_names = load_data()
# data_raw: (N_dias, 15) — mm/dia
# data_norm: (N_dias, 15) — normalizado
# mu, std: parâmetros para denormalizar
```

### Características físicas importantes:

1. **Zero-inflação**: 40–70% dos dias são secos (precipitação = 0)
2. **Cauda pesada**: eventos extremos com 10–100× a média
3. **Correlação espacial**: estações próximas têm dias chuvosos correlacionados
4. **Sazonalidade**: verão chuvoso, inverno seco (variação anual)

---

## 5. Os 6 Modelos Explicados

### Modelo 1: Cópula Gaussiana (`copula.py`)

**Intuição física**: "Separar a estrutura individual de cada estação da estrutura
de dependência entre estações."

**Matemática**:
- Marginal mista: `P(X=0) = p_seco`, `X|X>0 ~ Log-Normal(μ, σ)`
- Cópula: correlação nos *normal-scores* (transformada de postos)
- Geração: `z ~ MVN(0, C)` → CDF inversa por estação

**Vantagem**: sem redes neurais, geração instantânea, interpretável.
**Limitação**: não captura dependências não-lineares complexas.

---

### Modelo 2: Hurdle Simples (`hurdle_simple.py`)

**Intuição**: "Dois problemas separados: quando chove? quanto chove?"

**Part 1 — Ocorrência** (MLP):
```
input (15) → Hidden(32) → ReLU → Hidden(32) → ReLU → Sigmoid(15)
→ p_rain[i] por estação
```

**Part 2 — Quantidade** (MLP condicionado na ocorrência):
```
occ_mask (15) → Hidden(64) → ... → (mu_log[i], sigma_log[i]) por estação
→ Log-Normal parametrizada por rede neural
```

**Correlação espacial**: cópulas gaussianas separadas para ocorrência e quantidade.
**Loss**: `BCE_ocorrência + NLL_log_normal`

---

### Modelo 3: VAE (`vae.py`)

**Intuição**: "Comprimir dados para um espaço latente regulado onde N(0,I) produz amostras realistas."

**Arquitetura**:
```
Encoder: x(15) → 512 → 256 → (μ_z(128), σ²_z(128))
Decoder: z(128) → 256 → 512 → x̂(15) [+ ReLU]
```

**Treino**: `Loss = MSE(x̂, x) + β * KL(q(z|x) || N(0,I))`

**KL Annealing**: `β` sobe linearmente de 0→1 ao longo de `kl_warmup` épocas.
Isso previne colapso posterior onde o encoder ignora os dados.

**Geração**: `z ~ N(0,I)` → decoder

---

### Modelo 4: Hurdle VAE (`hurdle_vae.py`)

**Intuição**: "E se cada parte do processo Hurdle tiver seu próprio espaço latente?"

- `OccurrenceVAE` (latente 32D): captura padrões binários de ocorrência
- `AmountVAE` (latente 64D): captura distribuição condicional de quantidade

**Geração em dois passos**:
```python
z_occ ~ N(0, I_32) → OccurrenceVAE.decode → Bernoulli → occ_sim
z_amt ~ N(0, I_64) → AmountVAE.decode([zeros, occ_sim]) → amount
result = ReLU(amount) * occ_sim
```

**Hipótese testada**: a separação explícita ocorrência/quantidade reduz a dificuldade
de aprendizado, pois o `AmountVAE` nunca vê zeros.

---

### Modelo 5: RealNVP (`real_nvp.py`)

**Intuição**: "Um mapa invertível exato entre N(0,I) e os dados."

**Camadas de acoplamento afim**:
```
Divide x em (x₁, x₂):
    (s, t) = MLP(x₁)
    y₂ = x₂ * exp(tanh(s)) + t   [invertível!]
    y₁ = x₁                       [inalterado]
```

**Propriedade chave**: `tanh(s)` limita o escalonamento, prevenindo overflow numérico.

**Loss exata**: `NLL = -log p(x) = 0.5||z||² - Σ log|det Jᵢ|`

**Diferença do VAE**: não há ELBO ou aproximação — a likelihood é exata.

---

### Modelo 6: Flow Matching (`flow_match.py`)

**Intuição**: "Treinar uma rede para descrever trajetórias retas de ruído → dados."

**Trajetória reta (Optimal Transport)**:
```
z_t = (1-t)·z₀ + t·z₁    onde z₀ ~ N(0,I), z₁ = dado real
target = z₁ - z₀          (velocidade constante!)
Loss = ||v_θ(z_t, t) - target||²
```

**Vantagem sobre RealNVP**: não precisa de camadas invertíveis — qualquer MLP funciona.

**Embedding de tempo** (SinusoidalEmbedding):
```
t ∈ [0,1] → frequências exponenciais → [sin, cos] → MLP → embed(64D)
```

**Geração (integração Euler)**:
```
z₀ ~ N(0, I)
for t in [0, dt, 2dt, ..., 1-dt]:
    z += v_θ(z, t) * dt
```

---

## 6. Treinando um Modelo

### Uso básico

```bash
cd PrecipModels

# Cópula: sem épocas (ajuste analítico)
python train.py --model copula

# VAE com configuração padrão
python train.py --model vae

# VAE com configuração personalizada
python train.py --model vae --max_epochs 500 --latent_size 64 --lr 0.001

# Teste rápido (todos os modelos, 200 épocas)
python train.py --model vae --max_epochs 200
```

### Todos os argumentos do `train.py`

| Argumento | Default | Descrição |
|---|---|---|
| `--model` | (obrigatório) | Um de: copula, vae, hurdle_simple, hurdle_vae, real_nvp, flow_match |
| `--max_epochs` | (por modelo) | Número de épocas de treino |
| `--lr` | (por modelo) | Taxa de aprendizado |
| `--batch_size` | 128 | Tamanho do batch |
| `--latent_size` | (por modelo) | Dimensão do espaço latente |
| `--kl_warmup` | (por modelo) | Épocas de warmup do KL |
| `--device` | auto | 'auto', 'cpu', ou 'cuda' |
| `--data_path` | ../dados/inmet_relevant_data.csv | Caminho dos dados |
| `--output_dir` | ./outputs | Diretório de saída |
| `--n_samples` | 5000 | Amostras para avaliação |

### Defaults recomendados por modelo

| Modelo | Épocas | LR | Latente | KL Warmup |
|---|---|---|---|---|
| copula | — | — | — | — |
| vae | 1000 | 0.005 | 128 | 25 |
| hurdle_simple | 500 | 0.001 | — | — |
| hurdle_vae | 800 | 0.001 | 64 | 25 |
| real_nvp | 1000 | 0.001 | — | — |
| flow_match | 1000 | 0.001 | — | — |

---

## 7. Comparando os Modelos

### Comparação completa

```bash
# Treina todos e compara
python compare.py

# Teste rápido (200 épocas)
python compare.py --max_epochs 200 --n_samples 1000

# Pula treino e usa métricas salvas
python compare.py --skip_training

# Compara subconjunto
python compare.py --models copula vae flow_match
```

### Saídas em `outputs/comparison/`

- `comparison_quality.png` — bar charts por métrica (verde = melhor)
- `radar.png` — radar chart (menor área = melhor em todas as métricas)
- `comparison_report.txt` — tabela texto com ranking
- `composite_scores.json` — pontuação normalizada por modelo

---

## 8. Interpretando as Métricas

### Wasserstein (menor = melhor)

**O que mede**: distância entre distribuição marginal real e gerada, por estação.

**Interpretação hidrológica**: se Wasserstein = 5 mm/dia, em média a distribuição
gerada está "5 mm deslocada" da real. Para precipitação diária média de 3 mm,
isso é muito ruim. Para média de 50 mm, pode ser aceitável.

**Limitação**: 1D — não captura dependências espaciais.

---

### Correlation RMSE (menor = melhor)

**O que mede**: quão bem preservada está a correlação entre pares de estações.

**Interpretação**: se RMSE = 0.1, as correlações erram em média 10 pontos percentuais.
Um bom modelo deve preservar a estrutura espacial da precipitação.

**Importância**: crítico para aplicações de reservatórios (chuva simultânea em tributários).

---

### Wet Day Frequency Error (menor = melhor)

**O que mede**: erro na frequência de dias chuvosos por estação.

**Hipótese central**: o `hurdle_simple` deve ter o menor erro nesta métrica,
pois modela a ocorrência explicitamente via BCE.

---

### Extreme Quantile Error — Q90/Q95/Q99 (menor = melhor)

**O que mede**: erro nos valores extremos de precipitação.

**Importância**: gestão de riscos de enchentes depende principalmente de reproduzir
eventos raros corretamente, não a média.

**Limitação do VAE**: tende a subestimar extremos (efeito do KL regularization).

---

### Energy Score (menor = melhor)

**O que mede**: regra de pontuação própria multivariada — considera simultaneamente
distribuição marginal E estrutura de dependência.

```
ES = E||Y - x|| - 0.5 * E||Y - Y'||
```

É a única métrica verdadeiramente multivariada desta lista.

---

## 9. Dicas e Problemas Comuns

### Colapso KL (VAE/HurdleVAE)

**Sintoma**: `kl_loss ≈ 0` desde o início do treino, `recons_loss` não diminui.
**Causa**: beta sobe muito rápido, forçando o encoder a ignorar os dados.
**Solução**: aumente `--kl_warmup` (ex: `--kl_warmup 100`).

### NLL explodindo (RealNVP)

**Sintoma**: perda fica negativa e muito grande (-inf).
**Causa**: instabilidade numérica nas transformadas afins.
**Solução**: verifique que `tanh(s)` está limitando o escalonamento.
Experimente `--lr 0.0001` para treino mais estável.

### Hurdle Simple: wet_day_freq_error alto

**Sintoma**: frequência de dias chuvosos muito diferente da real.
**Causa**: as cópulas não foram ajustadas antes do treino.
**Solução**: `train.py` chama `fit_copulas()` automaticamente. Se usar o modelo
diretamente, chame `model.fit_copulas(data_raw)` antes de treinar.

### Flow Matching: amostras fora do domínio

**Sintoma**: valores negativos nas amostras geradas.
**Causa**: o FM não tem constraint de não-negatividade.
**Solução**: use normalização `standardize` (default para FM) e aplique
`np.clip(samples, 0, None)` nas amostras geradas, ou use a cópula como baseline
para pós-processamento.

---

## 10. Sequência Recomendada de Exploração

### Passo 1: Baseline rápido (5 min)

```bash
python train.py --model copula
python train.py --model vae --max_epochs 200
```

Objetivo: entender o nível do baseline e se o VAE rápido já supera a cópula.

### Passo 2: Comparação completa rápida (20–30 min)

```bash
python compare.py --max_epochs 200 --n_samples 1000
```

Objetivo: ver todas as 6 arquiteturas com treino reduzido.

### Passo 3: Modelos selecionados com treino completo

Treine somente os 2–3 modelos mais promissores com épocas completas:

```bash
python train.py --model vae         # referência DL
python train.py --model real_nvp    # candidato forte
python train.py --model flow_match  # candidato forte
python compare.py --skip_training --models copula vae real_nvp flow_match
```

### Passo 4: Exploração de hiperparâmetros

```bash
# Testa impacto do espaço latente no VAE
python train.py --model vae --latent_size 32  --max_epochs 500
python train.py --model vae --latent_size 128 --max_epochs 500
python train.py --model vae --latent_size 256 --max_epochs 500
```

---

## 11. Referências Cruzadas

| Arquivo | Função |
|---|---|
| `VAE_Tests/experiment.py` | load_data() original (nossa versão em data_utils.py) |
| `VAE_Tests/deepmodels.py` | Arquitetura VAE de referência |
| `VAE_Tests/run.py` | Melhores hiperparâmetros VAE (latent=128, lr=0.005, kl_warmup=25) |
| `VAE_Tests/auxplot.py` | compare_experiments(), composite score |
| `LatentFlowMatching/test.py` | Flow Matching complexo com Transformer (nossa versão simplificada) |
