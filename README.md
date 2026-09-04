# Modern RL & MARL Benchmark Suite em JAX — 100% Treinado, Zero Números Fabricados

**JAX 0.11 / CUDA / Flax / Optax / Craftax / Google Brax — WSL2 Ubuntu 24.04 (NVIDIA RTX 4070 Laptop, 8 GB VRAM)**

---

## ⚠️ Nota de Integridade (por que este repositório foi reescrito)

Uma auditoria linha-a-linha do código revelou que a versão anterior deste README apresentava
**resultados fabricados** como se fossem empíricos. Entre as falsificações confirmadas:

- **MARL** (`compare_marl.py`): recompensas e coberturas eram **literais escritos no fonte**
  (`"coop_reward": -0.98`, `"cobertura_alvos": "96.8%"`). Nenhum treino ocorria.
- **Tríade de representações**: o score do **GNN era `CNN + 0.05`** (linha literal), e o
  "retorno em fases inéditas" vinha de **ações aleatórias**, não de política treinada.
- **Boxe / Offline RL**: os 10 "competidores" (BCQ, GAIL, IQL, CQL, DT, Teacher…) eram
  **heurísticas scriptadas** (`if contender=="BCQ": punch=0.98`). As redes eram inicializadas
  e **nunca treinadas nem usadas**. Daí o `reward_std = 0.00` e `accuracy = 200%` nonsensical.
- **Brax 3D**: "PPO" e "SAC" eram **torques senoidais** (`sin(t·freq)·amp`), não algoritmos.
- **Contrastivo / ProcGen / Discreto-vs-Contínuo / Avançado**: scores **hardcoded**
  (ex.: `0.190…0.245`, `2.45/3.82/4.25`) e deltas inventados (`score_base + 0.05`).
- **Avaliação**: os poucos experimentos que treinavam (PPO/A2C/DQN) avaliavam com **1 único step**
  (média da recompensa de um passo), o que não mede retorno algum.

**Este repositório foi integralmente reescrito para treinar tudo de verdade.** Cada número agora
provém de uma rede neural treinada por **descida de gradiente real (`optax`)** e avaliada por
**retorno episódico completo** em níveis inéditos, com **múltiplas seeds**. Onde um experimento
antigo não era reproduzível honestamente, ele foi **removido ou substituído e documentado** —
nunca inventado.

---

## 1. Metodologia Real

1. **Treino genuíno.** Todos os algoritmos usam `jax.value_and_grad` + `optax.apply_updates`
   (Adam, grad-clip). Não existe rollout com pesos aleatórios nem política scriptada rotulada
   como algoritmo.
2. **Avaliação episódica real** ([`src/eval_utils.py`](src/eval_utils.py)): a política treinada é
   rolada por um horizonte completo, acumulando recompensa **somente enquanto o episódio está vivo**,
   reportando **média ± desvio do retorno episódico** em níveis de **treino** e **inéditos**
   (`seed+1000`). Avaliadores específicos: `make_craftax_evaluator`, `make_craftax_recurrent_evaluator`
   (políticas com memória), `make_continuous_evaluator` e `make_fixed_horizon_evaluator` (Brax).
3. **Protocolo de generalização (Craftax).** Treino em pool fixo de 200 níveis (`PRNGKey(0..199)`);
   avaliação *unseen* em `PRNGKey(1000..1099)`.
4. **Múltiplas seeds + save incremental.** Cada experimento grava o JSON a cada seed (à prova de falha).
5. **Reprodutibilidade.** `run_convergence_phase1.sh` / `run_convergence_phase2.sh` executam os
   treinos sequencialmente na GPU (evitando contenção de VRAM), com log em `results/logs/`.

### Trainers reais (escritos do zero nesta reescrita)
| Módulo | Algoritmos genuínos implementados |
| :-- | :-- |
| [`src/ppo.py`](src/ppo.py) | PPO / A2C (PureJaxRL: GAE, surrogate clipado, minibatches) |
| [`src/dqn.py`](src/dqn.py) | DQN (replay buffer circular + target network + ε-greedy) |
| [`src/hrl.py`](src/hrl.py) | HRL: `flat` / `skip4` / `hrl` / `hrl_learned` (PPO sobre macro-ações, options framework) |
| [`src/marl_trainers.py`](src/marl_trainers.py) | IPPO, MAPPO, MA-POCA (CTDE) + VDN, QMIX (decomposição monotônica de valor) |
| [`src/marl_paradigms.py`](src/marl_paradigms.py) | CTE (controlador centralizado) + TarMAC/GAT (comunicação) + **Fog-of-War POMDP real** |
| [`src/marl3d_trainers.py`](src/marl3d_trainers.py) | IPPO/MAPPO/MA-POCA **contínuos** (drones 3D) + `BraxWrapper` |
| [`src/continuous_rl.py`](src/continuous_rl.py) | Gaussian-PPO, **SAC** (twin-Q, α auto-tune, Polyak), Discrete-PPO |
| [`src/aux_ppo.py`](src/aux_ppo.py) | PPO + perdas auxiliares reais: CURL, CPC, ACL, SPR, **ICM** (recompensa intrínseca), World Model, Contrastive, **RND** |
| [`src/recurrent_ppo.py`](src/recurrent_ppo.py) | PPO **recorrente** (GRU) p/ LSTM+Attention (hidden persistido no rollout e recomputado por sequência) |
| [`src/offline_rl.py`](src/offline_rl.py) | Suíte offline real: BC, BC+SAC, IQL (expectile+AWR), CQL (penalidade conservadora), BCQ (VAE+perturbação+Q), Decision Transformer, GAIL (adversarial c/ rollouts) |

---

## 2. Desafios de Engenharia e Correções (o que NÃO deu certo de primeira)

Tornar tudo genuíno exigiu depurar problemas reais — documentados aqui por transparência:

| Problema real encontrado | Causa | Correção |
| :-- | :-- | :-- |
| **ICM divergia** (loss PPO → 3647, reward → 892, unseen → 0) | A recompensa intrínseca (erro do forward model **não-treinado**) era ilimitada e explodia os alvos de valor | `clip(r_int, 0, 5)` + `intrinsic_coef` 0.05→0.01 (bonus limitado e comparável à recompensa do env) |
| **`hrl_learned` com loss ≈ 32000** | Log-prob armazenado (meta + low-level) ≠ recomputado (só meta) → ratio do PPO explodia | Política low-level *open-loop* dentro da opção + sub-ações guardadas p/ recomputação exata |
| **OOM na Tríade (pixels)** | PPO com `num_envs=256` e imagens 63×63×3 → buffer de trajetória ~3,2 GB > 8 GB VRAM | `num_envs=64` nos experimentos de pixels + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` |
| **`jax.tree_map` inexistente** | JAX 0.11 moveu para `jax.tree_util.tree_map` | Substituição global |
| **`opt` dentro do carry JIT** | Optax `GradientTransformation` são funções — não podem ser traçadas | Otimizador capturado no *closure* (`self.opt`), nunca no carry |
| **Críticos com `{'params':{'params':…}}`** | Faltou `['params']` no `init` de MAPPO/MA-POCA | Extração correta do sub-árvore de parâmetros |
| **Índices agente/ambiente trocados (CTE/COMM)** | Amostrava agentes e ambientes com índices independentes, quebrando a correspondência logits↔ações | Update no nível de *ambiente* com arrays por agente consistentes |
| **`log_std` broadcast inválido (MARL 3D)** | Módulo já devolve `(E·N,A)`; `broadcast_to((E,N,A))` falhou | `reshape(mu.shape)` |
| **GAIL: slice com índice dinâmico** | `n = jnp.minimum(...)` (traçado) usado em `arr[:n]` | `n = min(...)` (Python int estático) |
| **BC/DT: `jax.grad(valor)`** | `loss` já era um array, não função | `jax.value_and_grad(fn)` |
| **`rng = rng or PRNGKey(0)`** | `bool()` sobre PRNGKey (array de 2 elems) → erro | `PRNGKey(0) if rng is None else rng` |
| **Bug no ambiente de boxe** | `p1_hp = new_p2_hp` (HP **trocado**) corrompia observação e win-rate | Corrigido para `p1_hp = new_p1_hp` |
| **ProcGen "easy/hard" irreal** | Craftax Classic **não tem** toggle de dificuldade como o ProcGen | Comparação **removida** e substituída pelo **generalization gap real** (train vs unseen), documentado |
| **Métricas de boxe sem sentido** | `accuracy=200%`, `std=0.00` (política determinística vs oponente determinístico, env com início fixo) | Métricas reais (reward/round, win-rate, KO, hits) de rollout físico; variância vem das seeds de treino |

---

## 3. Resultados REAIS

Legenda: ✅ = convergência completa (número final medido). **A Fase 2 terminou — todas as seções
abaixo estão completas**, com números reais de `results/*.json`.

### 3.1 Famílias de Algoritmos — Craftax Symbolic (`compare_algos.py`) — ✅ COMPLETO
8M steps, 256 envs, retorno episódico real (média de 3 seeds):

| Algoritmo | Retorno Train | Retorno Unseen | Throughput |
| :-- | :--: | :--: | :--: |
| **PPO** | **9.60 ± 2.7** | **9.14 ± 3.0** | ~20.700 FPS |
| **DQN** | 1.93 (instável: 2.74/0.18/2.88 por seed) | 1.90 | ~11.600 FPS |
| **A2C** | 1.49 ± 0.1 | 1.34 ± 0.0 | ~21.100 FPS |

> **Achado real:** **PPO ≫ DQN > A2C** no Craftax. O PPO (clipping + minibatches + GAE) atinge
> ~9.6 de retorno; o A2C (1 epoch, sem clipping) fica em ~1.5; o DQN é **instável entre seeds**
> (uma seed colapsou para 0.18) — variabilidade genuína, não suavizada.

### 3.2 Hierarchical RL & Abstração Temporal (`compare_hrl.py`) — ✅ COMPLETO
8M steps, 256 envs, 3 seeds, retorno episódico real:

| Modo | Retorno Train | Retorno Unseen | Gen. Gap | Throughput |
| :-- | :--: | :--: | :--: | :--: |
| **flat** (PPO primitivo) | **9.74 ± 2.6** | **9.18 ± 2.8** | +0.55 | ~19.700 FPS |
| **skip4** (action-repeat 4) | 4.53 ± 1.5 | 4.31 ± 1.7 | +0.21 | ~18.100 FPS |
| **hrl** (6 macro-skills fixas) | 2.86 ± 1.1 | 2.89 ± 1.1 | −0.03 | ~18.600 FPS |
| **hrl_learned** (hierarquia 2 níveis) | 2.69 ± 1.5 | 2.77 ± 1.5 | −0.08 | ~16.200 FPS |

> **Achado real:** o PPO **flat** supera todas as abstrações temporais neste orçamento — o
> **oposto** do que a versão fabricada insinuava. É o que os dados medidos mostram.

### 3.3 Multi-Agent RL — MPE Cooperativo (`compare_marl.py`) — ✅ COMPLETO
2M steps, 3 seeds, 3 agentes / 3 alvos, 50 steps/episódio:

| Algoritmo | Paradigma | Recompensa Co-op | Cobertura | Colisões | Throughput |
| :-- | :-- | :--: | :--: | :--: | :--: |
| **VDN** | Fatoração aditiva | **−17.52 ± 0.17** | **93.1%** | **0.04** | 155.604 FPS |
| **IPPO** | Descentralizado independente | −78.58 ± 4.36 | 26.9% | 1.19 | 357.067 FPS |
| **MAPPO** | CTDE (crítico centralizado) | −77.20 ± 8.62 | 25.8% | 1.11 | 175.258 FPS |
| **MA-POCA** | CTDE + atenção + contrafactual | −83.09 ± 8.30 | 23.6% | 1.21 | 55.208 FPS |
| **QMIX** | Fatoração monotônica | −98.49 ± 7.98 | 11.4% | 0.25 | 118.207 FPS |

> **Achado real:** o **VDN** dominou a navegação cooperativa (93% de cobertura). Os métodos
> policy-based (IPPO/MAPPO/MA-POCA) platearam em ~25% neste orçamento, e o QMIX teve o pior
> retorno. Isto **contradiz frontalmente** a narrativa fabricada anterior ("MA-POCA campeão,
> 96.8%").

### 3.4 Demais benchmarks — ✅ Fase 2 COMPLETA
Todos **treinam de verdade** (validados por smoke tests na GPU); os runs de convergência já
gravaram em `results/*.json` + `figures/`. **Nenhum valor é pré-programado.**

#### 3.4.1 Discreto vs Contínuo (`compare_discrete_vs_continuous.py`) — ✅ COMPLETO
Mesma dinâmica de navegação contínua, 1M steps (single) / 2M (multi), 2 seeds, retorno episódico real:

| Algoritmo | Espaço | Retorno/episódio |
| :-- | :-- | :--: |
| **Continuous Gaussian PPO** | Contínuo (N(μ,σ)) | **+2.14** |
| **SAC** | Contínuo (tanh MaxEnt) | **+2.08** |
| **Discrete PPO** | Discreto (5 forças) | +1.39 |
| Discrete MAPPO (multi) | Discreto (MPE) | −77.1 (27% cob.) |

> **Achado real:** controle **contínuo > discreto** na mesma tarefa; SAC ≈ Gaussian-PPO neste
> orçamento (os antigos 2.45/3.82/4.25 eram hardcoded).

#### 3.4.2 Os 4 Paradigmas MARL sob Fog-of-War (`compare_marl_4_paradigms.py`) — ✅ COMPLETO
Agentes **treinados** (2M steps, 2 seeds), visão clara vs Fog-of-War POMDP real (raio 0.40m):

| Paradigma | Retorno (Claro) | Retorno (Fog) | Cobertura (Fog) |
| :-- | :--: | :--: | :--: |
| **CTE (centralizado conjunto)** | **−67.0** | **−84.9** | 11.4% |
| CTDE (MAPPO) | −75.2 | −124.8 | 23.6% |
| Explicit Comm (TarMAC/GAT) | −79.3 | −129.5 | 4.5% |
| Value Decomposition (QMIX) | −109.5 | −91.5 | 7.5% |

> **Achado real (contradiz a versão fabricada):** sob nevoeiro, o **CTE centralizado** foi o mais
> robusto (−85), e a **comunicação TarMAC NÃO superou** os demais neste orçamento (−130). Todos
> degradam sob fog. A antiga conclusão "comunicação é indispensável sob fog" **não se sustentou**.

#### 3.4.3 Tríade de Representações (`compare_representations_triad.py`) — ✅ COMPLETO
PPO treinado (3M steps, 2 seeds), retorno episódico real em níveis inéditos:

| Representação | Unseen | Train | Throughput |
| :-- | :--: | :--: | :--: |
| **Pixels (NatureCNN)** | **8.11 ± 0.52** | 8.30 | 4.517 FPS |
| Vetor (MLP 1345D) | 6.05 ± 0.06 | 6.78 | 7.024 FPS |
| Grafo (GNN/GAT) | 3.60 ± 0.23 | 4.04 | 6.384 FPS |

> **Achado real (contradiz o fabricado):** **Pixels > Vetor > Grafo**. O GNN é treinado de verdade
> (não é mais `CNN+0.05`), mas fica atrás neste orçamento. O antigo "Vetor>Grafo>Pixels" era falso.

#### 3.4.4 Famílias Contrastivas (`compare_contrastive_types.py`) — ✅ COMPLETO
PPO + perda auxiliar real retropropagada (3M steps, 2 seeds), retorno unseen:

| Método | Unseen | Aux-loss final |
| :-- | :--: | :--: |
| **Baseline (sem contrastivo)** | **8.50** | — |
| Spatial (CURL/InfoNCE) | 8.40 | 2.23 |
| Temporal (CPC) | 8.31 | 2.22 |
| Action-Conditional (ACL) | 8.16 | 2.61 |
| Self-Predictive (SPR) | 7.48 | 0.12 |

> **Achado real (contradiz o fabricado):** o **baseline sem contrastivo foi o melhor**; as perdas
> auxiliares (todas reais e decrescentes) **não melhoraram** o retorno unseen neste orçamento — a
> antiga afirmação "SPR líder, +28.9%" **não se sustentou**.

#### 3.4.5 3D — Brax + Drones (`compare_3d_benchmarks.py`) — ✅ COMPLETO
Brax (retorno em horizonte fixo de 1000 steps, 2M steps de treino, 2 seeds) e drones 3D (3M steps):

| Brax | Passivo | PPO | SAC |
| :-- | :--: | :--: | :--: |
| HalfCheetah | +48 | −333 | −429 |
| Ant | +1000 | −2960 | **+2751** |
| Humanoid | +5174 | +4834 | +4902 |

| Drones 3D | Retorno | Cobertura | Colisões |
| :-- | :--: | :--: | :--: |
| IPPO 3D | −70.9 | 5.9% | 43.9 |
| MA-POCA 3D | −71.7 | 5.8% | 50.8 |
| MAPPO 3D | −85.2 | 7.3% | 131.1 |

> **Achado real e honesto:** locomoção Brax **não converge em 2M steps** (HalfCheetah/Humanoid ficam
> abaixo do passivo); a **Ant com SAC aprendeu de verdade** (+2751 > passivo +1000). Brax exige
> dezenas de milhões de steps — reportamos o número real, sem maquiagem.

#### 3.4.6 Boxe Offline Grand Prix (`run_boxing_grand_prix.py`) — ✅ COMPLETO
SAC teacher treinado (1M steps) → dataset (200k) → 7 algoritmos offline treinados (40k steps) →
avalados em 64 rounds reais no ringue:

| # | Competidor | Família | Score/round | Win |
| :--: | :-- | :-- | :--: | :--: |
| 1 | **CQL** | Offline pessimista | **+91.27** | 100% |
| 2 | **DT** | Sequence modeling | +91.27 | 100% |
| 3 | IQL | Offline expectile | +88.32 | 100% |
| 4 | Teacher (SAC online) | Expert | +76.58 | 100% |
| 5 | BC | Imitação pura | +74.47 | 100% |
| 6 | BCQ | VAE generativo | +69.42 | 100% |
| 7 | BC+SAC | Híbrido | +15.66 | 100% |
| 8 | GAIL | Imitação adversarial | +6.57 | 100% |
| 9 | Random | — | −190.34 | 64% |

> **Real:** todos treinados por gradiente e avaliados na física do ringue. `std=0` nas políticas
> determinísticas é **honesto** (o ambiente de boxe tem início fixo → rounds idênticos). O antigo
> "todos +56.40, heurísticas scriptadas" foi substituído por treino real.

#### 3.4.7 Paradigmas Avançados (`compare_advanced.py`) — ✅ COMPLETO
PPO + módulo auxiliar **treinado** (2M steps, 2 seeds), retorno unseen:

| Paradigma | Unseen | Aux-loss final |
| :-- | :--: | :--: |
| **PPO + Contrastive (CURL)** | **8.04** | 3.09 |
| PPO + World Model | 7.56 | **0.0001** |
| Baseline PPO | 7.28 | — |
| PPO + ICM (curiosidade) | 6.05 | 1.51 |

Throughput visual real: **NatureCNN 80.704 FPS** vs **ViT 48.522 FPS** (60% da CNN).

> **Achado real:** Contrastive > World Model > Baseline > ICM. O world model aprendeu a dinâmica
> (aux-loss → 0.0001); o ICM **atrapalhou** levemente (o bônus de curiosidade desviou a política).

#### 3.4.8 Componentes ProcGen (`compare_procgen_missing_components.py`) — ✅ COMPLETO
2M steps, 2 seeds, retorno unseen + generalization gap **reais**:

| Componente | Unseen | Gen-gap | FPS |
| :-- | :--: | :--: | :--: |
| **LSTM-Attention (PPO recorrente)** | **8.86** | +0.86 | 2.267 |
| NatureCNN (baseline) | 7.63 | +0.56 | 4.439 |
| RND (exploração intrínseca) | 7.17 | +0.79 | 4.410 |
| Impoola (GAP 64D) | 0.95 | +0.03 | 3.857 |

> **Achado real:** a **memória temporal (LSTM+Attention) lidera** (8.86); o **Impoola colapsou**
> (0.95 — o gargalo GAP-64D perdeu capacidade). O "easy/hard" fabricado foi removido (Craftax
> Classic não tem esse toggle); reporta-se o gen-gap real.

#### 3.4.9 Arquiteturas Convolucionais (`compare_architectures.py`) — ✅ COMPLETO
PPO em pixels (3M steps, 2 seeds), retorno unseen:

| Arquitetura | Unseen | Train | Gen-gap | FPS |
| :-- | :--: | :--: | :--: | :--: |
| **ImpalaCNN** | **9.10** | 9.68 | +0.58 | 2.237 |
| NatureCNN | 8.51 | 9.32 | +0.81 | 5.025 |

> **Achado real:** ImpalaCNN generaliza um pouco melhor (9.10 vs 8.51), mas é **2,2x mais lenta**
> (2.237 vs 5.025 FPS) — trade-off preciso/custo medido de verdade.

> ✅ **Fase 2 COMPLETA:** todos os 10 experimentos terminaram; cada número acima é de treino real
> por gradiente + avaliação episódica, gravado em `results/*.json` e `figures/`.

---

## 4. Real vs Sintético — o quanto os números mudaram

Os resultados **reais diferem enormemente** dos fabricados, tanto em magnitude quanto nas conclusões:

| Benchmark | Valor **sintético** (antigo) | Valor **real** (medido) | Diferença |
| :-- | :-- | :-- | :-- |
| HRL `flat` (unseen) | 0.008 (1 step) | **9.18** (episódico) | Escala/semântica totalmente distintas |
| HRL ranking | sugeria `hrl_learned` bom | **flat ≫ skip4 > hrl** | Ranking invertido |
| MARL campeão | MA-POCA −0.98 / 96.8% | **VDN −17.5 / 93.1%**; MA-POCA −83 / 23.6% | Campeão e escala mudaram |
| MARL recompensas | −0.98…−2.41 | −17…−98 | Ordem de grandeza diferente |
| Tríade (unseen) | Vetor>Grafo>Pixels; GNN=1.07 (`CNN+0.05`) | **Pixels 8.11 > Vetor 6.05 > Grafo 3.60** | Ranking invertido + fabricação removida |
| Boxe | todos +56.40, std 0.00 (script) | CQL/DT +91, IQL +88, GAIL +6.6, Random −190 | Heurísticas → treino real |
| Brax Humanoid "SAC" | +838 (torque senoidal) | SAC/PPO treinados (Brax não converge em 2M) | Falso algoritmo → real |
| Contrastivo | "SPR líder +28.9%" (fixo 0.19–0.24) | **Baseline 8.50 é o melhor**; SPR 7.48 | Conclusão invertida |
| Discreto/Contínuo | 2.45/3.82/4.25 (fixo) | **Contínuo +2.14 ≈ SAC +2.08 > Discreto +1.39** | Hardcoded → treinado |

**Conclusão:** as "10 conclusões científicas" do README anterior **não se sustentavam** — eram
baseadas em tabelas pré-programadas e políticas scriptadas. As conclusões reais emergem dos dados:
**VDN** domina o MPE cooperativo; **flat** supera HRL no Craftax; **Pixels > Vetor > Grafo** na
tríade; o **baseline sem contrastivo** vence as perdas auxiliares neste orçamento; **contínuo >
discreto** em controle; e **CQL/DT/IQL** lideram o boxe offline.

---

## 5. Lacunas conhecidas / itens removidos (transparência)

- **`run_all_procgen_combinations.py`**: continha eixos fabricados (QR-DQN, World Models,
  Augmentation, Budget Scaling com números hardcoded) e um multiplicador inventado
  (`det_score = r_eval × 1.8/0.8`), além de usar a API antiga do HRL (removida). Foi
  **reescrito para medir apenas coisas reais** (budget scaling, exploração ICM/RND, HRL,
  throughput de arquiteturas).
- **Comparação de hardware (CPU vs GPU vs ProcGen)**: **removida de propósito.** Os JSONs órfãos
  fabricados (`benchmark_cpu.json`, `benchmark_gpu.json`) e a figura `05_hardware_throughput.png`
  foram **apagados**. A superioridade de throughput do JAX/XLA é evidente nos próprios FPS reais
  reportados por cada experimento (ex.: MARL > 350 mil FPS, PPO Craftax ~20 mil FPS), sem
  necessidade de um benchmark de velocidade dedicado.
- **ProcGen easy/hard**: não reproduzível no Craftax Classic → removido (documentado na seção 2).
- **Figuras**: `figures/` é regenerado pelos próprios scripts a partir dos números reais. As
  figuras antigas (dados fabricados) são sobrescritas conforme a Fase 2 conclui cada experimento.
- **Fase 2**: ✅ **concluída** — os 10 experimentos terminaram e as tabelas da seção 3 estão
  preenchidas com os números reais de `results/*.json` + `figures/`.

---

## 6. Guia de Execução & Reprodução

```bash
source /root/jax-gpu/bin/activate
cd /mnt/c/Users/Acer/Downloads/MLE

# Smoke test (valida 1 passo de gradiente real de cada trainer)
python3 run_smoke_test.py

# Convergência completa (sequencial, 1 processo por vez na GPU)
bash run_convergence_phase1.sh   # HRL, MARL, Tríade
bash run_convergence_phase2.sh   # Algos, Discreto/Contínuo, MARL-4, 3D, Boxe, pixels...

# Experimentos individuais (orçamento configurável; --help)
python3 experiments/compare_algos.py         --steps 8000000 --num-envs 256 --seeds 0 1 2
python3 experiments/compare_hrl.py           --steps 8000000 --num-envs 256 --seeds 0 1 2
python3 experiments/compare_marl.py          --steps 2000000 --num-envs 128 --seeds 0 1 2
python3 experiments/compare_3d_benchmarks.py --brax-steps 2000000 --marl-steps 3000000 --num-envs 64
python3 experiments/run_boxing_grand_prix.py --teacher-steps 1000000 --offline-steps 40000 --dataset-size 200000
```

Cada script aceita `--steps/--num-envs/--seeds` e salva incrementalmente em `results/`.
Logs de convergência em `results/logs/`.

---

## 7. Política de Autenticidade

- Todo número reportado provém de **treino por gradiente** + **avaliação episódica real**.
- Se um resultado não pôde ser medido honestamente, ele é **removido e documentado**, jamais inventado.
- Desvios-padrão refletem variância **real** entre seeds/episódios, não valores decorativos.
- O histórico anterior (ProcGen/PyTorch e a versão com números fabricados) permanece no Git para auditoria.
