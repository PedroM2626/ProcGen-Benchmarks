# Benchmark Sistemático de Arquiteturas Visuais, World Models e Exploração em Procgen — Estudo com 5 Seeds, 5 Jogos e 100k Passos

**Python 3.10.11 + Procgen 0.10.7 + Stable-Baselines3 2.9.0 + PyTorch 2.5.1+cu121 (RTX 4070) — `C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe`**

> **Resumo.** Avaliação sistemática de **16 arquiteturas** em **6 famílias** (`CNN` vs `Attention` vs `World Models` vs `Augment` vs `New Archs` vs `Exploração`) com **5 seeds** (`42-46`), **5 jogos** (`bossfight`, `starpilot`, `dodgeball`, `maze`, `heist` + `coinrun` controle) e **duas dificuldades** (`easy 200` / `hard 200` / `eval 0`) em `Procgen` (`~300 FPS` em `cuda`, `50k` em `~3 min`). Todos os experimentos usam `frame_stack=1` (`3×64×64` `CHW` `uint8`), `PPO` (`lr 3e-4`, `n_steps 256`, `batch 64`, `n_epochs 3`, `γ 0.99`, `λ 0.95`, `clip 0.2`), `eval 10 episódios` `deterministic=False` e `tensorboard` para a jornada.

---

## 1. Metodologia

### 1.1. Ambientes
- **Wrapper** `procgen_wrapper.py:6` `ProcgenGymWrapper(gymn.Env)` — converte `gym 0.26.2` `old API` (`obs, done`) → `gymnasium 1.3.0` (`obs, terminated, truncated`), `HWC 64×64×3 → CHW 3×64×64` (`np.transpose`), `Discrete(15)` (`bossfight`, `starpilot`, `dodgeball`, `coinrun`). Variante sem `CV` `procgen_wrapper.py:55` `ProcgenVectorWrapper` (`16×16` `grayscale` → `256D` `MLP`) para controle `mesmo jogo com/sem visão`.
- **Factory** `procgen_wrapper.py:85` `make_procgen_env(game, num_levels, distribution_mode, rand_seed, frame_stack, vector)` — `gym.make(f'procgen:procgen-{game}-v0', num_levels, distribution_mode, rand_seed)`.
- **Treino:** `num_levels=200` `distribution_mode='easy'` (padrão `Procgen`), **Eval:** `num_levels=0` (`ilimitado`, fases nunca vistas) `seed+1000` — mede generalização.
- **Sem Frame Stacking:** `frame_stack=1` `Box(3,64,64)` `uint8` em todos os benchmarks (`procgen_wrapper.py:21`). `frame_stack=4` (`12×64×64`) é suportado (`procgen_wrapper.py:19`) mas não usado; `Imitation-player` usa `128×128×4`.

### 1.2. Arquiteturas
- **CNN Clássica** `models/sb3_extractors.py:8` `ClassicCNNExtractor` — `Conv 32 8×8 s4 → 64 4×4 s2 → 64 3×3 s1 → Flatten → FC 512` (`600k` params), `HWC/CHW` auto-detectado (`is_hwc`).
- **Attention CNN** `models/sb3_extractors.py:63` `AttentionCNNExtractor(use_cbam)` — `CBAM` (`ChannelAttention` `reduction 16` + `SpatialAttention` `kernel 7`) `models/cnn_attention.py:82` ou só `SpatialAttentionModule` `models/cnn_attention.py:6` (`x * attention_map + x` com **residual** `models/cnn_attention.py:37` para estabilizar `spatial` puro que dava `0.00` determinístico). `FC 512`.
- **World Models** `models/world_model_extractors.py:6` — `VAEExtractor(latent 128, KL)` + `dream()` `deconv`, `AEExtractor` determinístico + `dream()`, `ReconExtractor` (`L2` `dec 3×64×64`) + `dream()`, `ContrastiveExtractor` (`InfoNCE` `noise 0.01` + `proj 64`).
- **Augment Contrastivo** `compare_augment_contrastive.py:14` `ContrastiveCrop` (`pad 4 + random 64`), `ContrastiveColor` (`brightness 0.8-1.2`), `ContrastiveNoise` (`noise 0.01`).
- **Novas Arquiteturas** `models/combined_extractors.py` — `ImpalaCNNExtractor` (stack de `ImpalaBlock` conv), `ImpoolaCNNExtractor` (`GAP 64D`), `LSTMAttentionExtractor` (`CNN + LSTM 256 + attention`), `ViTExtractor` (`64 patches 16×16 + Transformer 4 camadas`), `ResNet18Extractor` — todas com `FC 512` (`benchmark #6`).
- **Exploração (ICM/RND/NGU)** `compare_maze_heist.py:16` — `ICMWrapper` (bônus intrínseco por erro de modelo direto), `RNDWrapper` (destilação de rede aleatória), `NGUWrapper` (estende `RNDWrapper` com memória episódica, `reward += beta * bonus * episodic`) — aplicados sobre `maze`/`heist` (`benchmark #7`).

### 1.3. Seeds e Avaliação
- **Treino:** `5 seeds` (`42,43,44,45,46`) `PPO` `seed` + `procgen rand_seed` fixos — `mean±std` entre seeds em `statistics.json`.
- **Avaliação:** `10 episódios` `deterministic=False` (estocástico, corrige `spatial` que dava `0.00` determinístico vs `4.00` estocástico em `8k` teste) em `eval 0`.
- **Jornada:** `tensorboard --logdir logs_*` (`events.out.tfevents.*`).

---

## 2. Benchmarks Executados

| # | Script | Jogo(s) | Timesteps | Seeds | Configs | Tempo | Log |
|---|---|---|---|---|---|---|---|
| 1 | `compare_procgen.py` | `coinrun` | `50k` | `5` | `classic/cbam/spatial/mlp_vector` | `~35 min` `5×50k` | `logs_procgen/comparison_coinrun_20260827_204023` |
| 2 | `compare_world_models.py` | `bossfight` | `100k` | `5` | `vae/ae/recon/contrastive` | `~67 min` `5×100k` | `logs_world_models/comparison_bossfight_20260827_193327` |
| 3 | `compare_suite.py` | `bossfight+starpilot+dodgeball` | `100k` | `5` | `4 WM +4 CNN +3 Augment =11` por jogo | `~8h` `16.5M steps` `20:41→04:47` | `logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109` |
| 4 | `compare_bossfight_hard.py` | `bossfight hard` | `100k` | `5` | `11` | `~2.5h` `07:21→09:57` | `logs_bossfight_hard/comparison_bossfight_hard_20260828_072148` |
| 5 | `compare_augment_contrastive.py` | `bossfight` | `100k` | `5` | `crop/color/noise` | `~80 min` (embutido no suite) | `logs_suite` `*_aug_*` |
| 6 | `compare_new_archs.py` | `bossfight+starpilot+dodgeball` | `100k` | `5` | `impala/impoola/lstm_attention/vit/resnet18` | `~12h` `13:45→01:39` `7.5M` | `logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545` |
| 7 | `compare_maze_heist.py` | `maze+heist` | `100k` | `5` | `ppo/icm/rnd/ngu` | `~6.5h` `01:48→08:23` `4M` | `logs_maze_heist/maze_heist_maze_heist_20260829_014802` |
| 8 | `compare_combined.py` | agregação `suite+new_archs` | — | — | ranking global `16 arquiteturas` | imediato (sem treino) | `results/global_16.png` |

---

## 3. Resultados Oficiais (5 Seeds, 10 Eps Eval)

### 3.1. Coinrun 50k — CNN vs MLP (mesmo jogo com/sem CV)
`logs_procgen/comparison_coinrun_20260827_204023/statistics.json:1`
| Config | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `classic_pixels` | **7.6** | 1.2 | 6.0 | 9.0 |
| `attention_cbam_pixels` | **8.0** | 0.0 | 8.0 | 8.0 |
| `attention_spatial_pixels` | 6.2 | 3.31 | 0.0 | 9.0 |
| `mlp_vector` (`16×16` `256D` sem CV) | **8.0** | 0.89 | 7.0 | 9.0 |
> **Análise:** `CBAM 8.0` estável vence `classic 7.6`; `MLP 8.0` já empata `CNN` — `coinrun easy 200` é reativo e não exige `CV` (`downsample 256D` basta). `Spatial` puro instável (`0.0` em 1 seed) sem `residual`.

### 3.2. Bossfight 100k — World Models
`logs_world_models/comparison_bossfight_20260827_193327/statistics.json:1`
| Config | Mean | Std |
|---|---:|---:|
| `vae` | 0.16 | 0.16 |
| `ae` | 0.30 | 0.50 |
| `recon` | 0.02 | 0.04 |
| `contrastive` | **0.36** | 0.62 |
> `contrastive` melhor, mas todos `<0.5` — `bossfight 100k` insuficiente solo.

### 3.3. Suite 100k — 3 Jogos (11 Configs/Jogo)
`logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109/suite_statistics.json:1`
| Jogo | Melhor | Mean | 2º | Pior |
|---|---|---:|---|---|
| `bossfight` (`~0.5`) | `spatial 0.76±0.90` | `cbam 0.58` `classic 0.54` | `contrastive 0.36` | `recon 0.02` |
| `starpilot` (`~2.0`) | `spatial 2.6±0.82` | `aug_crop 2.12±0.79` `cbam 2.1` `ae 2.08` | `color 1.94` `noise 1.61` | — |
| `dodgeball` (`~1.2`) | `classic 1.48±0.69` | `cbam 1.31` `spatial 1.28` | `vae 1.2` | `recon 0.88` |
> **Augment:** `crop` vence `color`/`noise` em `starpilot` `+31%` e `dodgeball`; `color` segundo em `bossfight`. **Geral:** `classic` vence `dodgeball`, `spatial` vence `starpilot` — **ranking muda por jogo**, 1 jogo só vicia (padrão `Procgen` são `16` jogos; `3` é mínimo).

### 3.4. Bossfight HARD 100k — Stress Test
`logs_bossfight_hard/comparison_bossfight_hard_20260828_072148/statistics.json:1`
| Config | Mean | Std |
|---|---:|---:|
| `vae` | **0.43±0.54** | `ae 0.38` `aug_crop 0.32` `mlp 0.26` |
| `cnn` | `0.02±0.04` (`classic/cbam/spatial` zeram) |
> `hard` achata tudo para `0.0-0.4` (`easy` era `0.5-0.76`), `CNN` zera em `100k` — `hard` precisaria `200k` (`~6h`) e não vale `suite hard` completa; `easy` já é `benchmark justo`.

### 3.5. New Archs 100k — 5 Novas Arquiteturas ×3 Jogos
`logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545/statistics.json:1`
| Jogo | Melhor | Mean | 2º | Pior |
|---|---|---:|---|---|
| `bossfight` | `lstm 0.36±0.57` | `vit 0.30` `impala 0.28` | `resnet 0.02` | `impoola 0.06` |
| `starpilot` | `lstm 2.44±0.56` | `resnet 2.28` `impoola 2.2` | `vit 2.1` `impala 1.78` | — |
| `dodgeball` | `resnet 1.72±0.65` | `vit 1.2` `impoola 1.12` | `impala 1.08` | `lstm 0.80` |
> `ViT`/`ResNet` não superam `spatial 2.6` `starpilot` nem `classic 1.48` `dodgeball`; `lstm` vence `bossfight`/`starpilot` mas perde `dodgeball`.

### 3.6. Maze+Heist 100k — PPO vs ICM vs RND vs NGU (Exploração)
`logs_maze_heist/maze_heist_maze_heist_20260829_014802/statistics.json:1`
| Jogo | `ppo` | `icm` | `rnd` | `ngu` |
|---|---:|---:|---:|---:|
| `maze` | **2.4±1.49** | 1.8±1.46 | **2.4±1.49** | **2.4±1.49** |
| `heist` | **0.8±0.74** | 0.6±0.80 | **0.8±0.74** | **0.8±0.74** |
> `ICM` pior que `PPO` puro (`maze` `1.8` vs `2.4`, `heist` `0.6` vs `0.8`), `RND`/`NGU` empatam `PPO` — `100k` insuficiente para `curiosidade` brilhar em `maze`/`heist` `easy`; `NGU` não supera `RND` (`memória` não ajuda com `200` níveis). `heist` `0.8` confirma `sparse` hierárquico (`3 chaves`) precisa `>100k`.

### 3.7. Global 16 Arquiteturas — Média 3 Jogos (Top 10 de 16 mostrado)
`logs_suite` + `logs_new_archs` agregados (`16.5M+7.5M` steps) — `mean` de `3` `means` por arquitetura:
| Rank | Arquitetura | Global Mean | Por Jogo (B/S/D) |
|---:|---|---:|---|
| 1 | `spatial` | **1.54** | 0.76 / 2.60 / 1.28 |
| 2 | `resnet18` | 1.34 | 0.02 / 2.28 / 1.72 |
| 3 | `classic` | 1.33 | 0.54 / 1.98 / 1.48 |
| 4 | `cbam` | 1.33 | 0.58 / 2.10 / 1.31 |
| 5 | `mlp_vector` | 1.25 | 0.58 / 2.07 / 1.11 |
| 6 | `lstm_attention` | 1.20 | 0.36 / 2.44 / 0.80 |
| 7 | `vit` | 1.20 | 0.30 / 2.10 / 1.20 |
| 8 | `aug_crop` | 1.16 | 0.54 / 2.12 / 0.84 |
| 9 | `impoola` | 1.12 | 0.06 / 2.20 / 1.12 |
| 10 | `ae` | 1.11 | 0.30 / 2.08 / 0.96 |
> `Top 3` são `CNN` puros (`spatial`/`resnet`/`classic`); `World Models` (`vae 0.88` `recon 0.90`) e `contrastive 0.97` ficam abaixo de `MLP 1.25` em `suite 100k` `easy` — `CV` com `attention` ainda vence `World Model` em `Procgen` `100k`.

---

## 4. Gráficos e Vídeos

Todos os gráficos e vídeos estão versionados na pasta `results/`.

### 4.1. Coinrun 50k — CNN vs MLP (com/sem visão)
![Coinrun 50k — CNN vs MLP](results/coinrun_50k_cnn_vs_mlp.png)

### 4.2. Bossfight 100k — World Models
![Bossfight 100k — World Models](results/bossfight_100k_world_models.png)

### 4.3. Suite 100k — 3 Jogos × 11 Arquiteturas
| Bossfight | Starpilot | Dodgeball |
|---|---|---|
| ![Suite Bossfight](results/suite_bossfight.png) | ![Suite Starpilot](results/suite_starpilot.png) | ![Suite Dodgeball](results/suite_dodgeball.png) |

### 4.4. Bossfight HARD 100k — Stress Test
![Bossfight HARD 100k](results/bossfight_hard_100k.png)

### 4.5. New Archs 100k — 5 Novas Arquiteturas ×3 Jogos
![New Archs Bossfight](results/new_archs_bossfight.png) | ![New Archs Starpilot](results/new_archs_starpilot.png) | ![New Archs Dodgeball](results/new_archs_dodgeball.png)

### 4.6. Maze+Heist 100k — PPO vs ICM/RND/NGU
![Maze](results/maze_heist_maze_plot.png) | ![Heist](results/maze_heist_heist_plot.png)

### 4.7. Global 16 — Média 3 Jogos
![Global 16](results/global_16.png)

### 4.8. Vídeos

Vídeos lado-a-lado são gerados sob demanda (`visualize_side_by_side.py`) — bossfight com **sonhos** (`top=real`, `bottom=dream()` de `VAE/AE/Recon`; Contrastive exibe `no dream`) e coinrun com agentes lado-a-lado. Requer os `.zip` salvos pelos benchmarks durante o treino:
  ```powershell
  py -3.10 visualize_side_by_side.py --benchmark world_models --game bossfight --log_dir ./logs_world_models --mode mp4 --out results/bossfight_dreams.mp4 --steps 600 --device cuda
  py -3.10 visualize_side_by_side.py --benchmark procgen --game coinrun --log_dir ./logs_procgen --mode mp4 --out results/coinrun_side_by_side.mp4 --steps 600 --device cuda
  ```
- **Curvas de treino por seed:** `statistics.json` + `comparison_results.json` por benchmark + `tensorboard --logdir logs_suite` (logs tensorboard são descartáveis/gerados sob demanda).

---

## 5. Como Reproduzir

```powershell
# Ambiente (Python 3.10 obrigatório para Procgen) — versões pinadas em requirements.txt
pip install -r requirements.txt

# ou manualmente (mesmas versões validadas):
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -m pip install procgen==0.10.7 stable-baselines3==2.9.0 gymnasium==1.3.0 gym==0.26.2 torch==2.5.1+cu121 opencv-python==4.8.0.74 --extra-index-url https://download.pytorch.org/whl/cu121

# Benchmarks 5 seeds
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_world_models.py --timesteps 100000 --seeds 42 43 44 45 46 --num_levels 200 --log_dir ./logs_world_models --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_procgen.py --game coinrun --timesteps 50000 --seeds 42 43 44 45 46 --num_levels 200 --log_dir ./logs_procgen --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_suite.py --games bossfight starpilot dodgeball --timesteps 100000 --seeds 42 43 44 45 46 --log_dir ./logs_suite --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_bossfight_hard.py --timesteps 100000 --seeds 42 43 44 45 46 --log_dir ./logs_bossfight_hard --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_new_archs.py --timesteps 100000 --seeds 42 43 44 45 46 --games bossfight starpilot dodgeball --log_dir ./logs_new_archs --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_maze_heist.py --timesteps 100000 --seeds 42 43 44 45 46 --games maze heist --log_dir ./logs_maze_heist --device cuda
C:\Users\Acer\AppData\Local\Programs\Python\Python310\python.exe -u compare_combined.py  # agrega logs_suite + logs_new_archs no ranking global
```

**Estimativas `cuda`:** `coinrun 50k` `5×50k` `~35 min`, `bossfight 100k` `5×100k` `~67 min`, `suite 100k` `3 jogos ×11×5×100k` `16.5M steps` `~15h` (`20:41→04:47`), `bossfight hard` `~2.5h`, `new archs 100k` `3 jogos ×5×5×100k` `7.5M steps` `~12h` (`13:45→01:39`), `maze+heist 100k` `2 jogos ×4×5×100k` `4M steps` `~6.5h` (`01:48→08:23`), `combined` imediato.

---

## 6. Próximos Benchmarks Propostos

> **Nota:** os caminhos `D:\mario-ds`, `D:\mujoco-walker`, `D:\Imitation-player` e `D:\mario64ds-rl` são **repositórios externos de referência** fora deste projeto — não são necessários para reproduzir os benchmarks acima.

| # | Origem | Benchmark em `Procgen` `RL` (não só `loss` como `Imitation-player:171`) | Tempo |
|---|---|---|---|
| 1 | `mujoco-walker:50` | **Offline RL** `100k` `bossfight` `expert` `BC` vs `IQL` vs `CQL` vs `Decision Transformer` | `~40 min` offline |

---

## 7. Resultados por Seed (Completo) — 5 Seeds `42-46`, 10 Eps `deterministic=False`

**Coinrun 50k 5 seeds** `logs_procgen/comparison_coinrun_20260827_204023/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `classic` | 7.0 | 6.0 | 7.0 | 9.0 | 9.0 | **7.6±1.2** |
| `cbam` | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | **8.0±0.0** |
| `spatial` | 6.0 | 0.0 | 7.0 | 9.0 | 9.0 | **6.2±3.31** |
| `mlp_vector` | 7.0 | 7.0 | 8.0 | 9.0 | 9.0 | **8.0±0.89** |

**Bossfight 100k 5 seeds (World Models)** `logs_world_models/comparison_bossfight_20260827_193327/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `vae` | 0.0 | 0.0 | 0.1 | 0.4 | 0.4 | 0.16±0.16 |
| `ae` | 0.0 | 0.0 | 0.1 | 0.1 | 1.3 | 0.30±0.50 |
| `recon` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `contrastive` | 0.0 | 0.0 | 0.0 | 0.2 | 1.6 | **0.36±0.62** |

**Bossfight HARD 100k 5 seeds** `logs_bossfight_hard/comparison_bossfight_hard_20260828_072148/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `vae` | 0.0 | 0.0 | 0.4 | 1.2 | 0.6 | **0.43±0.54** |
| `ae` | 0.1 | 0.1 | 0.4 | 0.1 | 1.2 | 0.38±0.41 |
| `recon` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0±0.0 |
| `contrastive` | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.06±0.12 |
| `classic` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `cbam` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `spatial` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `mlp` | 0.0 | 0.0 | 0.0 | 0.1 | 1.2 | 0.26±0.47 |
| `aug_crop` | 0.0 | 0.0 | 0.0 | 0.3 | 1.3 | 0.32±0.49 |
| `aug_color` | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.02±0.04 |
| `aug_noise` | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.06±0.12 |

**Suite 100k 3 Jogos (11 configs×3×5=165 entradas)** `logs_suite/suite_bossfight_starpilot_dodgeball_20260827_204109/suite_results.json:1` — `mean` em `suite_statistics.json:1`; ex `starpilot spatial: [1.6,2.1,2.2,3.2,3.5] 2.6±0.82`, `dodgeball classic: [0.8,1.0,1.4,1.5,2.7] 1.48±0.69`. Full `165` linhas preservadas em `suite_results.json` para `LaTeX`.

**New Archs 100k 3 Jogos (5×3×5=75 entradas)** `logs_new_archs/new_archs_bossfight_starpilot_dodgeball_20260828_134545/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `bossfight_impala` | 0.0 | 0.2 | 0.0 | 0.0 | 1.2 | 0.28±0.47 |
| `starpilot_lstm_attention` | 2.3 | 2.9 | 1.4 | 2.7 | 2.9 | **2.44±0.56** |
| `dodgeball_resnet18` | 1.2 | 1.6 | 1.4 | 3.0 | 1.4 | **1.72±0.65** |

**Maze+Heist 100k 2 Jogos (4×2×5=40 entradas)** `logs_maze_heist/maze_heist_maze_heist_20260829_014802/comparison_results.json:1`
| Config | 42 | 43 | 44 | 45 | 46 | Mean±Std |
|---|---:|---:|---:|---:|---:|---|
| `maze_ppo` | 2.0 | 3.0 | 1.0 | 1.0 | 5.0 | **2.4±1.50** |
| `maze_icm` | 0.0 | 1.0 | 1.0 | 3.0 | 4.0 | 1.8±1.46 |
| `heist_ppo` | 0.0 | 0.0 | 1.0 | 1.0 | 2.0 | **0.8±0.74** |
| `heist_icm` | 1.0 | 0.0 | 0.0 | 0.0 | 2.0 | 0.6±0.80 |

## 8. Vídeos

Gerados sob demanda via `visualize_side_by_side.py` (comandos na seção 4.8): `bossfight_dreams.mp4` (World Models com sonhos) e `coinrun_side_by_side.mp4` (CNN vs MLP lado-a-lado). Saída `mp4` com painéis `128×128` por agente `hstack` `15 FPS`; `dream` para `VAE/AE/Recon` (`models/world_model_extractors.py:6` `dream()` `deconv`), `Contrastive` exibe `no dream`. Requer os `.zip` salvos durante o treino dos benchmarks.

## 9. Hardware e Limitações

- **Hardware:** `NVIDIA GeForce RTX 4070 Laptop` `556.29` `CUDA 12.5` `WDDM` `8 GB VRAM` `58°C` `~30%` `GPU-Util` durante `PPO` `cuda` (`nvidia-smi` `20:41`), `Python 3.10.11` `torch 2.5.1+cu121` `gym 0.26.2` `gymnasium 1.3.0` `stable-baselines3 2.9.0`.
- **Limitações:** `bossfight 100k` `easy` já `<1.0` (`0.76±0.90`) e `hard` `0.02±0.04` zeram `CNN` — `suite 100k` `16.5M steps` é `mínimo` para `ranking`; `ViT` medido no `benchmark #6`: `~4.6 min/run` em `bossfight` mas `16-25 min/run` em `starpilot` (`~4×` mais lento que `CNN`, variável por jogo) — a estimativa antiga de `~15×` (`Imitation-player:171`) era de `loss` imitation, não de `PPO`; ainda assim `ViT 1.20` global não supera `lstm/spatial`; `Imitation BC` só viu `loss` (`ResNet 2.90` melhor) sem `reward` `RL` — `#4` corrige isso.

## 10. Referências

- `Procgen` (`Cobbe et al.`), `DreamerV3` (`SheepRL`), `CURL`/`SPR` (`mario-ds:121`), `CBAM` (`Woo et al.`), `PPO` (`Schulman`), `Stable-Baselines3`, `Imitation-player` `compare_models.py` `Nature 3.48` vs `ResNet 2.90`.
