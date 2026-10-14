# Bumper Beam × GeoTransolver 架構筆記（已對照原始碼修正）

對應實驗：`python train.py --config-name=bumper_geotransolver_oneshot`
核對用檔案：

- [conf/bumper_geotransolver_oneshot.yaml](../crash/conf/bumper_geotransolver_oneshot.yaml)
- [conf/model/geotransolver_one_shot.yaml](../crash/conf/model/geotransolver_one_shot.yaml)
- [rollout.py](../crash/rollout.py)（`GeoTransolverOneShot.forward`）
- [datapipe.py](../crash/datapipe.py)
- [physicsnemo/models/geotransolver/geotransolver.py](../../../../physicsnemo/models/geotransolver/geotransolver.py)
- [physicsnemo/models/geotransolver/context_projector.py](../../../../physicsnemo/models/geotransolver/context_projector.py)
- [physicsnemo/nn/module/gale.py](../../../../physicsnemo/nn/module/gale.py)
- [physicsnemo/nn/module/physics_attention.py](../../../../physicsnemo/nn/module/physics_attention.py)
- [bumper_beam/generate_dataset.py](bumper_beam/generate_dataset.py)

---

## 🔹 Input / Output 是什麼？

以 bumper beam 撞擊資料集為例（對應論文 Eq. 15a/15b 的 one-shot 公式）：

- **Input x⁽⁰⁾**：初始（未變形）網格節點座標 `x['coords']`，`[N, 3]`
- **Input f**：全域設計參數 —— `velocity_x`、`thickness_scale`、`rwall_origin_y`，共 3 個，對應 `global_dim: 3`
- **Output ŷ**：`[N, 50, 5]`，one-shot 一次 forward pass 直接吐出整個撞擊過程的節點時序場

`num_time_steps: 51`（含 t0），rollout steps `T = 51 - 1 = 50`；`out_dim: 250 = 50 × 5`。5 個頻道不是純位移，而是 **3(位置) + effective_plastic_strain(1) + stress_vm(1)**，因為這個實驗的 `dynamic_targets` 設了 `[effective_plastic_strain, stress_vm]`。模型輸出的前 3 個頻道是「殘差 + t0 座標」還原出的絕對位置，後 2 個頻道是逐點塑性應變與 von Mises 應力的直接預測值（不是殘差）。

也就是 ŷ = f(x⁽⁰⁾, f) → 位置軌跡 + 應力/應變場，本質上是在取代一次 OpenRadioss/LS-DYNA 撞擊模擬，且順便把後處理常看的應力/塑性應變場也一起代理掉了。

**這個實驗沒有 per-node 的靜態特徵**：`datapipe.static_features: []`。所以 `x['features']` 是空的 `[N, 0]`，逐點厚度（如果保桿本身有變厚度設計）目前沒有被餵進這個模型——只有一個全域共用的 `thickness_scale` 純量。若要支援逐點厚度分布，需要另外在 `static_features` 補上（例如 `[thickness]`），這是目前這份設定檔的限制，不是架構限制。

---

## 🔹 三個容易搞混的「厚度/尺寸」變數

- **geometric scaling**（DOE 變數，`generate_dataset.py` 的 `geometry_scales`）：`(sx, sy, sz)` **各軸獨立縮放**，5 組 DOE 值以 `(1,1,1)` 為基準，只動 y 軸或 z 軸（寬/高方向），例如 `(1,0.5,1)`、`(1,1,2)`。它不是「保桿整體變長變短」，也**沒有**進 `global_features`——因為每次模擬的 `x['coords']` 座標本身就是縮放後的結果，Geometry Tokenizer 直接讀座標就能感知形狀差異，不需要額外標量。
- **thickness_scale**（global，`f` 三個成員之一）：整個保桿共用一個縮放係數（DOE 值 `[1.0, 0.7, 1.3]`），影響殼元素剛度/質量，不改變座標形狀。
- **per-node thickness**：架構上支援（`static_features` 可以填 `[thickness]`），但 bumper beam 這個實驗**目前沒有用**，`static_features: []`。

模型怎麼在沒有額外輸入「長度/尺寸」的情況下知道哪個 case 比較大——尺寸差異本身就寫在座標數值裡：Geometry Tokenizer 和 Multiscale feature extraction 都是直接讀原始座標去編碼，尺寸資訊隱含在座標分佈裡。

DOE 總數：5(geometry_scales) × 3(velocities: -5/-3/-7 沿 x) × 3(thickness_scales) × 1(rwall_diameter) × 3(rwall_origins) = **135 次模擬**。訓練用 121 筆、驗證 5 筆（config `num_training_samples`/`num_validation_samples`），其餘進 test split。

---

## 🔹 架構為何吃得下 CAE 的 input？

CAE 輸出是不規則網格/點雲，節點間的物理耦合（撞擊力怎麼從正面傳到末端）不一定沿網格拓樸走。MeshGraphNets 這類 GNN 只在相鄰節點間傳訊息，撞擊力要傳到保桿另一端得堆很多層（逐跳擴散）。GeoTransolver 靠 GALE（Geometry-Aware Latent Embeddings）attention，讓任意兩節點能透過一組**固定大小的 context token** 一步互通關係。

bumper beam 實際生效的超參數（**`conf/model/geotransolver_one_shot.yaml` 裡明確覆寫**，不是 class 預設值）：

| 參數 | bumper beam 實際值 | GeoTransolver class 預設值 |
|---|---|---|
| `n_hidden` | 256（沒覆寫，用預設） | 256 |
| `n_head` | 8（沒覆寫，用預設） | 8 |
| `n_layers` | **6**（config 明確覆寫） | 4 |
| `slice_num`（context/self-attn slice 數） | **128**（config 明確覆寫） | 32 |
| `include_local_features` | `true`（config 明確覆寫） | `false` |
| `radii` / `neighbors_in_radius` | `[0.05, 0.25]` / `[8, 32]`（沒覆寫） | 同左 |
| `n_hidden_local` | 32（沒覆寫） | 32 |
| `attention_type` | `"GALE"`（沒覆寫，非 FLARE） | `"GALE"` |

### ① 原始輸入（實際丟進模型的張量）

- 節點座標 `x['coords']`：`[N, 3]`——這份座標在 forward 裡**被用三次**：
  1. concat 進 `fx = torch.cat([coords, features])` → 作為 `local_embedding`（bumper beam 因 `features` 為空，`fx` 實際上就等於 `coords`）
  2. 作為 `geometry` 參數 → 餵給 Geometry Tokenizer
  3. 作為 `local_positions` 參數 → 餵給 Multiscale feature extraction 的 ball query
- 節點靜態特徵 `x['features']`：`[N, 0]`（本實驗為空）
- 全域參數 `f`：`[1, 3]`

**座標正規化**：這個 crash pipeline 用的是 **z-score**（每軸 `pos_mean`/`pos_std`，見 `datapipe.py`），不是 min-max 線性映射到 `[-1,1]`。所以 ball query 半徑 `0.05`/`0.25` 的物理意義是「幾個標準差」，不是「座標範圍的 5%/25%」。半徑仍然是跑分割(split)相對值、不受保桿絕對尺寸影響（因為統計量是用訓練集算出來重複套用），但如果新 case 尺寸大幅超出訓練集統計範圍，正規化後座標可能落到分布外——這正是 README 額外提供 `GuardedGeoTransolver` 這個 OOD guardrail 的原因。

### ② Context 建構：三個 Tokenizer（都在 GlobalContextBuilder 裡）

Geometry / Global / Multiscale 三個 tokenizer 全部是 `ContextProjector`，作用不是把每個點的 embedding 串起來，而是各自用「per-token 線性投影 → softmax 加權聚合」把 N 個點壓縮成**固定數量**的 slice token，形狀 `(B, H, S, D_c)`，跟 N 完全無關：

- **Geometry Tokenizer**：吃 `geometry=coords` `[N,3]` → `[B, 8, 128, 32]`
- **Global Tokenizer**：吃 `f` `[1,3]` → `[B, 8, 128, 32]`
- **Multiscale feature extraction**（`MultiScaleFeatureExtractor`，底層 `BQWarp` ball query，半徑 `[0.05,0.25]`、鄰居數 `[8,32]`）：以每個點的位置為中心，聚合鄰居的幾何特徵成局部多尺度描述，再各自 tokenize 成 `[B, 8, 128, 32]` × 2 個 scale

三路在最後一維 concat：

```
context_dim = dim_head × len(radii) × 1(functional stream) + dim_head(geometry) + dim_head(global)
            = 32 × 2 + 32 + 32 = 128
```

**Cₓ 形狀是 `[1, 8, 128, 128]`**（batch, heads, slice_num, context_dim），S 和 D_c 都是固定超參數，不會隨保桿網格節點數 N 增加而膨脹——這正是模型能吃下大型不規則網格、複雜度不隨 N 平方成長的關鍵之一。

`include_local_features=True` 額外把 multiscale 的局部特徵 concat 到主 token 上：

```
effective_hidden = n_hidden + n_hidden_local × len(radii) = 256 + 32×2 = 320
```

**GALE blocks（自注意力、交叉注意力、MLP）內部實際運作的 hidden dim 是 320，不是 256**，只有最後一層 `ln_mlp_out` 把 320 → 250（`out_dim`）。

### ③ GALE Block：Cross-attention + Self-attention

每層 GALE block（`hidden_dim=320`，`n_head=8` → `dim_head=40`）：

- **Cross-attention**：Query 來自 token 自己的投影，Key/Value 來自上面固定大小的 Context `[1,8,128,128]`，做標準 scaled-dot-product attention，輸出形狀 `[1, N_token, 320]` 對齊回原本的 token。
- **Self-attention（GALE，非 FLARE）**：這是標準 Transolver 式的「physics attention」，**不是**用一個可學習參數矩陣 `Gₘ` 做 cross-attention。實際流程：
  1. 每個 token 的投影特徵過線性層 `in_project_slice`，得到 per-token 的 slice logits `[N, H, S]`（`S=slice_num=128`）
  2. 對 `S` 維做 softmax，得到 per-token 的 slice 權重
  3. 用這組權重對特徵做加權平均聚合成 `S` 個 slice token：`slice_token[H,S,D] = normed_weights.T @ fx`
  4. 在 `S` 個 slice token 之間做標準 self-attention
  5. **用同一組 slice 權重**把 attended 的 `S` 個 slice token 投影回 `N` 個節點

  真正用可學習全域 query（`q_global = nn.Parameter(...)`）做 encode-decode 的是 **FLARE 變體（`GALE_FA`）**，對應另一個實驗 `bumper_geoflare_oneshot.yaml`，不是這份 bumper_geotransolver_oneshot 用的標準 GALE。兩者輸出 shape 相同，但聚合機制不同（前者是資料依賴的 softmax pooling，後者是固定可學習 query 的 attention）。

- **混合**：`Y = σ(α)·Y_self + (1-σ(α))·Y_cross`，`α`（`state_mixing`）是每層獨立、可學習的純量參數，初始化為 0（sigmoid(0)=0.5，初始平均混合）。

### ④ Decoder

`ln_mlp_out`：`LayerNorm(320) → Linear(320, 250)`，`[N, 320] → [N, 250]`，再 reshape 成 `[N, 50, 5]`；前 3 個頻道加回 t0 座標得到絕對位置，後 2 個頻道是塑性應變/von Mises 應力的直接預測。

**「加回 t0 座標」是 forward 內部寫死的一步，train/inference 用同一套邏輯，不是兩階段設計**（[rollout.py:97-117](../crash/rollout.py#L97-L117)）：

1. `GeoTransolverOneShot.forward()` 內部，模型 raw 輸出先 reshape 成 `[N, 50, 5]`（`_oneshot_output`）
2. 緊接著 `_oneshot_add_coords()` 把 t0 的（正規化後）座標加回前 3 個頻道：`pred[:, :, :3] += coords.unsqueeze(1)`，後 2 個頻道（應變/應力）不加
3. `forward()` 直接回傳這個「已加回 t0 座標」的 `pred`——這一步不是可選的後處理，寫在 forward 裡面

訓練時（[train.py:249-253](../crash/train.py#L249-L253)）：

```python
pred = self.model(sample=sample, data_stats=self.data_stats)  # 已經加回 t0 座標
target = sample.node_target                                    # 絕對位置（+ 應變/應力）
loss = self.criterion(pred, target)                            # MSE 直接比對絕對位置
```

`sample.node_target`（[datapipe.py:361-394](../crash/datapipe.py#L361-L394)）裡的位置頻道 `pos_rollout = pos_seq[1:]` 本身就是每個未來時刻的**絕對（正規化）位置**，不是相對 t0 的差值。

所以更準確的說法是：loss 算的是絕對位置的 MSE，但因為 forward 內部把網路的原始輸出加上一個固定的 t0 座標 offset 才輸出，反向傳播時這個 offset 是常數（不參與梯度），等價於強迫網路只需要學「殘差」這部分。train/eval/推論共用同一個 `forward()`，沒有額外的「inference 再補回」的分離邏輯。

---

## 🔹 每個 case 的節點數量都一樣嗎？

對 bumper beam 這個資料集來說，**N 是固定的**。實際核對 run1、run10、run100 三個已跑完的 case，VTK 輸出的 `POINTS` 都是 **13882**，完全一致。

原因：135 組 DOE 只是對同一份底模（`Bumper_Beam_AP_meshed_0000.rad`）做參數變異——`/NODE` 座標整體縮放（`geo_scales`）、殼元素厚度縮放（`thick_scale`）、初速度、剛性牆位置——都是在原本的節點/單元拓樸上做仿射變換或數值修改，**不是重新網格化**，所以每個 case 的節點數、連接關係完全相同，只有座標數值、厚度、邊界條件不同。

要注意兩點：

1. **這是 bumper beam 這個資料集的特性，不是 GeoTransolver 架構的限制**。模型（以及 `datapipe.py`/`rollout.py`）寫法上完全支援每個 sample 的 N 不同——這也是為什麼 README 的 TODO 提到 `batch_size` 目前固定為 1（`Support batch_size > 1: The pipeline currently uses batch_size=1 due to variable node counts per sample`）。這句話是講給整個 crash 系列聽的（包含 body-in-white 整車撞擊、crushcan），那些資料集不同 run 之間 mesh 解析度/拓樸可能真的不同；bumper beam 剛好是特例，N 剛好都一樣。
2. 就算 N 剛好固定，pipeline 依然是逐一 sample 處理（`batch_size=1`），不會因為 N 固定就自動幫你 batch 起來——要吃到 batch>1 的訓練效率，還是得照 README TODO 補 padding/batching 邏輯。

### N 固定會比較好學習嗎？

不完全是——要拆成兩層看。

**對 GeoTransolver/GALE 架構本身，N 固不固定其實無所謂**：整條 pipeline 是逐 token（逐節點）獨立做線性投影，聚合時是對 N 做加權平均求和（`slice_token = normed_weights.T @ fx`，[physics_attention.py:200-203](../../../../physicsnemo/nn/module/physics_attention.py#L200-L203)），這個聚合對節點順序、節點數量都不敏感——是 permutation-invariant 的 set/point-cloud 式操作，不依賴「第 i 個節點永遠對應同一個位置」這種 index correspondence 來學習。就算 N 每個 case 都不同，這個架構一樣吃得下（這也是為什麼 README 說 pipeline 原生支援 variable N，只是懶得做 batch>1 的 padding）。

**真正讓「這個資料集」好學的，是 N 固定背後的原因，不是 N 固定這件事本身**：bumper beam 的 135 組 case 是同一份底模仿射變換出來的——座標整體縮放、厚度縮放、速度/牆位改變，拓樸完全沒變。這代表：

- 訓練分布的幾何多樣性其實很窄：模型只需要學會「同一個形狀在不同縮放/厚度/衝擊條件下怎麼變形」，不需要學「不同拓樸/不同網格密度下怎麼變形」——後者難得多。
- 因為節點對應的物理位置在 135 組 case 裡幾乎一致（同一顆網格只是被拉伸/壓縮），ball query 抓到的鄰居結構、局部幾何 pattern 在不同 case 間高度相似，Multiscale feature extractor 學到的局部特徵比較容易 transfer。

這些才是讓訓練變簡單的原因，根源是「資料生成方式選擇不重新網格化」，不是「N 這個數字剛好相等」。

**風險**：因為模型從沒在訓練時看過不同節點數/不同網格密度的 bumper beam，沒辦法從這個資料集知道它對「換一顆網格密度不同、拓樸不同的真實保桿」是否還能 generalize——`GuardedGeoTransolver` 監控的是全域參數範圍跟幾何 latent 的 kNN 距離，不是專門針對「網格離散化方式改變」設計的防呆機制。這個資料集能驗證的是「同拓樸、不同縮放/工況」的泛化能力，不是「不同網格」的泛化能力。

---

## 🔹 為什麼這個模型適合 crash dynamics 這個問題？

1. 論文比較了四種時間推進策略（AR rollout / teacher forcing / one-shot / time-conditional）：AR rollout 在較長時間步（T=50）時明顯發散不穩定，one-shot 沒有誤差累積問題，推論成本是 O(1) 而非 O(T)，在 bumper beam 案例上準確度與訓練成本取得最佳平衡（README：one-shot 1 sec/epoch vs time-conditional 29 sec/epoch，驗證 MSE 5.42e-3 vs 4.12e-3，差距不大但訓練成本差近 30 倍）。
2. 論文額外提出的 GeoFLARE（GALE_FA）版本把 self-attention 換成透過少量可學習 global query 的 encode-decode 路由，記憶體再省，準確度在 one-shot 比較中也是最高（README：GeoFlare > GeoTransolver > Transolver > MeshGraphNet）。
3 . 對應到實務痛點：crashworthiness 設計需要跑大量不同板厚/幾何組合的迭代優化，傳統 FEA 一次模擬要跑上小時；這種能吃複雜幾何、one-shot 推論、context 大小不隨網格密度膨脹的架構，正好切中「快速大量設計探索」的需求。

一句話心得：這篇論文把「幾何怎麼餵進 Transformer」和「時間軸怎麼推進最穩」這兩個工程問題想清楚，但實作細節（context 是固定大小的壓縮表示、self-attention 是 softmax pooling 而非可學習 query attention、bumper beam 目前沒有逐點厚度輸入）跟直覺上「Transformer 全連接 attention」的簡化敘事有不少落差，讀原始碼比只讀架構圖更準。

---

## 參考

- [Automotive Crash Dynamics Modeling Accelerated with Machine Learning](https://arxiv.org/pdf/2510.15201)
- [GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer](https://arxiv.org/pdf/2512.20399)
- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)
