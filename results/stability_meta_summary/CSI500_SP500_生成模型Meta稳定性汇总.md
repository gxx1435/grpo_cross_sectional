# CSI500 / SP500 生成模型 Meta 稳定性汇总

## 设定（各组相同）

- 冻结最新 Alpha + 对应生成模型 checkpoint（**不训练**）
- **Meta**：G 条生成样本 ∪ Teacher（MVO / MaxSharpe / RiskParity）→ 再聚合
- **G** ∈ {8, 32, 64, 128}
- **seed** ∈ {1…20}；每日噪声 `hash(date, seed)`
- 同 (日, G, seed) 共用一个 Meta 池，比较四种聚合：
  - Pred-Utility（argmax）
  - Mean utility（等权平均）
  - Top-k（k=3，按 pred_utility）
  - Softmax（τ=1.0）
- 指标：全年 OOS **单利 Σ** 的 across-seed **mean±std**
- 配权池：Top30

覆盖实验：

| 市场 | 生成模型 | 目录 |
| --- | --- | --- |
| CSI500 | Diffusion | `results/CSI500/strict_fixed_oos_diff_stability_meta_seeds/` |
| CSI500 | SS-FM | `results/CSI500/strict_fixed_oos_ssfm_stability_meta_seeds/` |
| CSI500 | FM | `results/CSI500/strict_fixed_oos_fm_stability_meta_seeds/` |
| SP500 | SS-FM | `results/S&P500/strict_fixed_oos_top30_ssfm_stability_meta_seeds_rerun/` |
| SP500 | Diffusion | `results/S&P500/strict_fixed_oos_top30_diff_stability_meta_seeds/` |
| SP500 | FM | `results/S&P500/strict_fixed_oos_top30_fm_stability_meta_seeds/` |

## 1. 分市场 × 生成模型明细

### CSI500 · SS-FM

产物：`results/CSI500/strict_fixed_oos_ssfm_stability_meta_seeds/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | 0.3668±0.1751 | 0.5005±0.0498 | 0.3430±0.0902 | 0.5003±0.0498 |
| G=32 | 0.4121±0.1831 | 0.5051±0.0268 | 0.4258±0.0802 | 0.5051±0.0268 |
| G=64 | 0.4663±0.1988 | 0.5108±0.0213 | 0.4897±0.1444 | 0.5107±0.0212 |
| G=128 | 0.4882±0.1455 | 0.5110±0.0144 | 0.5136±0.1105 | 0.5110±0.0144 |

### CSI500 · FM

产物：`results/CSI500/strict_fixed_oos_fm_stability_meta_seeds/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | 0.1772±0.0718 | 0.4987±0.0216 | 0.3211±0.0434 | 0.4985±0.0216 |
| G=32 | 0.2570±0.1027 | 0.5184±0.0110 | 0.3597±0.0456 | 0.5183±0.0110 |
| G=64 | 0.3194±0.1188 | 0.5259±0.0087 | 0.4098±0.0491 | 0.5259±0.0087 |
| G=128 | 0.3735±0.1131 | 0.5257±0.0065 | 0.4478±0.0818 | 0.5257±0.0065 |

### CSI500 · Diffusion

产物：`results/CSI500/strict_fixed_oos_diff_stability_meta_seeds/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | 0.4039±0.2325 | 0.5085±0.0602 | 0.3826±0.1110 | 0.5086±0.0602 |
| G=32 | 0.7319±0.2903 | 0.5300±0.0345 | 0.6202±0.1842 | 0.5302±0.0345 |
| G=64 | 0.7499±0.2798 | 0.5409±0.0278 | 0.7534±0.1205 | 0.5411±0.0279 |
| G=128 | 0.8828±0.2281 | 0.5510±0.0238 | 0.7984±0.1722 | 0.5513±0.0239 |

### SP500 · SS-FM

产物：`results/S&P500/strict_fixed_oos_top30_ssfm_stability_meta_seeds_rerun/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | 0.0847±0.0630 | 0.0687±0.0252 | 0.0602±0.0499 | 0.0687±0.0252 |
| G=32 | 0.0747±0.0700 | 0.0687±0.0157 | 0.0478±0.0399 | 0.0687±0.0157 |
| G=64 | 0.0217±0.0636 | 0.0698±0.0096 | 0.0291±0.0458 | 0.0698±0.0096 |
| G=128 | 0.0146±0.0736 | 0.0691±0.0064 | 0.0187±0.0465 | 0.0691±0.0064 |

### SP500 · FM

产物：`results/S&P500/strict_fixed_oos_top30_fm_stability_meta_seeds/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | 0.0797±0.0390 | 0.0268±0.0137 | 0.0564±0.0328 | 0.0269±0.0137 |
| G=32 | 0.0844±0.0659 | 0.0062±0.0068 | 0.0367±0.0328 | 0.0062±0.0068 |
| G=64 | 0.0782±0.0486 | 0.0009±0.0038 | 0.0329±0.0323 | 0.0009±0.0038 |
| G=128 | 0.0652±0.0639 | -0.0019±0.0040 | 0.0336±0.0272 | -0.0019±0.0040 |

### SP500 · Diffusion

产物：`results/S&P500/strict_fixed_oos_top30_diff_stability_meta_seeds/`

| G | Pred-Utility | Mean | Top-k | Softmax |
| --- | --- | --- | --- | --- |
| G=8 | -0.0237±0.1134 | -0.0001±0.0282 | 0.0039±0.0566 | -0.0001±0.0282 |
| G=32 | -0.0878±0.1146 | -0.0185±0.0262 | -0.0429±0.0508 | -0.0185±0.0262 |
| G=64 | -0.1074±0.0789 | -0.0141±0.0165 | -0.0622±0.0672 | -0.0141±0.0165 |
| G=128 | -0.1722±0.0749 | -0.0203±0.0116 | -0.0869±0.0502 | -0.0203±0.0116 |

## 2. Pred-Utility 横比（同市场）

### CSI500 · Pred-Utility across generators

| G | Diffusion | SS-FM | FM |
| --- | --- | --- | --- |
| G=8 | 0.4039±0.2325 | 0.3668±0.1751 | 0.1772±0.0718 |
| G=32 | 0.7319±0.2903 | 0.4121±0.1831 | 0.2570±0.1027 |
| G=64 | 0.7499±0.2798 | 0.4663±0.1988 | 0.3194±0.1188 |
| G=128 | 0.8828±0.2281 | 0.4882±0.1455 | 0.3735±0.1131 |

### SP500 · Pred-Utility across generators

| G | SS-FM | Diffusion | FM |
| --- | --- | --- | --- |
| G=8 | 0.0847±0.0630 | -0.0237±0.1134 | 0.0797±0.0390 |
| G=32 | 0.0747±0.0700 | -0.0878±0.1146 | 0.0844±0.0659 |
| G=64 | 0.0217±0.0636 | -0.1074±0.0789 | 0.0782±0.0486 |
| G=128 | 0.0146±0.0736 | -0.1722±0.0749 | 0.0652±0.0639 |

## 3. 简要结论

### CSI500

- **Diffusion**：Pred-Utility 最优约在 `G=128`，`0.8828±0.2281`
- **SS-FM**：Pred-Utility 最优约在 `G=128`，`0.4882±0.1455`
- **FM**：Pred-Utility 最优约在 `G=128`，`0.3735±0.1131`
- Mean / Softmax 通常 **std 更小**；Pred-Utility 均值可更高但跨 seed 波动更大（视市场/模型而定）。

### SP500

- **SS-FM**：Pred-Utility 最优约在 `G=8`，`0.0847±0.0630`
- **FM**：Pred-Utility 最优约在 `G=32`，`0.0844±0.0659`
- **Diffusion**：Pred-Utility 最优约在 `G=8`，`-0.0237±0.1134`
- Mean / Softmax 通常 **std 更小**；Pred-Utility 均值可更高但跨 seed 波动更大（视市场/模型而定）。

