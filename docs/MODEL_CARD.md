# 模型卡（Model Card）— EGFR 活性预测双引擎

> 维护：改进总控代理 · 2026-10-24（P0-07）
> 配套文档：数据卡 `docs/DATA_CARD.md`（数据口径）、`EXPERIMENTS.md`（实验证据）。
> 原则：**每个指标标注产生它的脚本与产物文件**；无产物出处的数字不得写入本卡。

---

## 模型 1 — 随机森林（RF）「部署模型」

| 属性 | 值 | 来源 |
|---|---|---|
| 文件 | `rf_egfr_model_final.pkl`（25 MB） | 仓库既有 |
| 类型 | `sklearn.RandomForestClassifier` | EXP-005 checkpoint 逆向 |
| 超参 | `n_estimators=200, max_depth=15, max_features='sqrt', min_samples_leaf=2, criterion='gini', bootstrap=True, random_state=42` | 同上（从 pkl 内部读出） |
| 输入特征 | 12 个 RDKit 描述符（`feature_names.json`）：MolWt, NumHAcceptors, NumHDonors, NumRotatableBonds, RingCount, FractionCSP3, HeavyAtomCount, NumHeteroatoms, NumSaturatedRings, NumAromaticRings, TPSA, MolLogP | EXP-005 |
| 特征重要性 Top5 | MolWt .146 / TPSA .129 / MolLogP .115 / NumRotatableBonds .099 / FractionCSP3 .099 | EXP-005 |
| 训练数据 | 数据集 A：13,286 条，median split（见数据卡） | EXP-010b 复现证实 |
| 元数据缺失 | 无 sklearn 版本标记；`oob_score=False`；无训练时间/数据指纹 | EXP-005（已知缺陷） |

**指标（README 口径，全部可复现）**

| 指标 | 值 | 产生方式 |
|---|---|---|
| 5 折 CV ROC-AUC | **0.8680 ± 0.0050** | `python scripts/train_rf_legacy_protocol.py` → `artifacts/rf_legacy_protocol_metrics.json` |
| 5 折 CV Accuracy | **0.7851 ± 0.0073** | 同上。注：median split 下类别 1.03:1 平衡，Accuracy 与 BAcc（0.7845）几乎相等，**无不平衡虚高** |
| 5 折 CV MCC | 0.5712 ± 0.0149 | 同上 |
| 5 折 CV BAcc / F1 | 0.7845 / 0.7955 | 同上 |
| 独立测试集（20%） | AUC 0.8701 / ACC 0.7893 / MCC 0.5792 | 同上（与 CV 样本互斥） |

**复现模型对照**：`artifacts/rf_legacy_protocol.pkl`（27 MB，复现产物，不替代部署模型）与部署模型在 teach_lab 协议测试集上判定一致率 69%、概率相关 0.65——证实部署 pkl 的训练数据版本与当前快照**不同源**（EXP-022）。

---

## 模型 2 — 图神经网络（GCN）

| 属性 | 值 | 来源 |
|---|---|---|
| 文件 | `gcn_egfr_best_model.pth`（180 KB） | 仓库既有 |
| 结构 | GCNConv×3（13→128→128→128）+ BatchNorm×2 + Linear(128→64) + Linear(64→1)，**43,649 参数** | EXP-006 checkpoint 逆向 |
| checkpoint 形态 | 裸 `OrderedDict` state_dict，**无任何元数据**（无 epoch / AUC / 数据版本） | EXP-006（已知缺陷） |
| 输入 | 13 维原子特征（与 `gnn_predictor.py` 严格同序）：atomic_num, degree, formal_charge, hybridization, is_aromatic, total_hs, implicit_valence, radical_electrons, isotope, mass/100, total_valence, has_implicit_h, is_N_or_O | `utils/teach_lab.smiles_to_graph` |
| 训练数据 | 数据集 A（同 RF） | 原始 `gcn_cv_results.csv` 与 median split 数据吻合 |

**原始 CV 结果（`data/raw/gcn_cv_results.csv`，逐折存档）**

| 指标 | 值 | 口径说明 |
|---|---|---|
| 5 折 CV ROC-AUC | **0.8445 ± 0.0085**（ddof=1）/ 0.0076（ddof=0） | README 写的 ±0.008 对应**总体标准差 ddof=0** |
| 5 折 CV Accuracy | **0.7671 ± 0.0122**（ddof=1）/ 0.0109（ddof=0） | README ±0.011 同上 |
| 逐折 best_epoch | 22 / 18 / 24 / 15 / 23 | 说明原训练的收敛窗口覆盖至 25 epochs |

**可执行复现（P0-02，EXP-019，已完成）**：`python scripts/train_gnn.py --dataset legacy --tag legacy --epochs 25`
→ `artifacts/gnn_metrics_legacy.json` + `artifacts/gcn_repro_legacy.pth`（178.7 KB，sha256 a509aa53…，未覆盖原 pth）。

| 指标 | 复现实测 | 原始 CSV | README 声称 |
|---|---|---|---|
| 5 折 CV AUC | **0.8517 ± 0.0061** | 0.8445 ± 0.0085 | 0.845 ± 0.008 ✅ |
| 5 折 CV Accuracy | **0.7719 ± 0.0061** | 0.7671 ± 0.0122 | 0.767 ± 0.011 ✅ |
| 5 折 CV BAcc / MCC | 0.7714 / 0.5463 | （原始未记录） | — |
| 独立测试集（n=2,658） | AUC 0.8541 / BAcc 0.7745 / MCC 0.5506 | — | — |

逐折对照（复现 vs 原始 CSV）：0.8557/0.8455/0.8484/0.8484/0.8603 vs 0.8501/0.8361/0.8376/0.8424/0.8561，**全部 Δ<0.011**。
与现有 `gcn_egfr_best_model.pth` 同测试集对照：现有 pth AUC **0.8766**（更优），判定一致率 88.75%、概率相关 0.939 → **同数据源确认，无泄漏信号（<0.99）**。

---

## ⚠️ F-12：两套特征集不可互换（本卡最重要的警告）

部署 RF 与教学实验室小模型使用**不同的 12 维描述符**，仅 5 个同名：

| # | 部署（`feature_names.json`） | 教学（`teach_lab.DESCRIPTOR_NAMES`） |
|---|---|---|
| 1 | MolWt | MolWt ✅ |
| 2 | NumHAcceptors | LogP ❌ |
| 3 | NumHDonors | HBD ❌ |
| 4 | NumRotatableBonds | HBA ❌ |
| 5 | RingCount | TPSA |
| 6 | FractionCSP3 | RotatableBonds ❌ |
| 7 | HeavyAtomCount | AromaticRings ❌ |
| 8 | NumHeteroatoms | HeavyAtoms ❌ |
| 9 | NumSaturatedRings | FractionCSP3 ❌ |
| 10 | NumAromaticRings | RingCount |
| 11 | TPSA | NumHeteroatoms |
| 12 | MolLogP | MolMR ❌ |

交叉使用会**静默产出错误预测且不报错**（12 个特征 7 个错位）。当前代码无防线；P1 阶段加特征名校验断言。

**F-13 关联缺陷**：部署路径 `real_predictor.smiles_to_features` 任何描述符计算异常→静默填 0（无日志）。复现脚本已显式计数（legacy 数据零填充，见 `rf_legacy_protocol_metrics.json`）；生产路径修复归入 P0-03 之后的工程卡。

---

## 已知限制（诚实清单）

1. **泛化边界（P0-05，EXP-021，10 seeds）**：零共享骨架划分下 RF AUC **0.8215 ± 0.0292**，比随机划分（0.8680）低 **4.7 个点**——模型对全新骨架的泛化属领域正常水平但明显弱于随机划分口径；单 seed 波动 8.7 个点，任何单次划分结论不可靠。**材料引用规则：跨骨架泛化必须用 0.82 口径，不得用 0.87 冒充。**
2. **无校准分析**：预测概率未验证可靠性（P1-01）。吉非替尼仅得 0.60 提示分布偏保守（EXP-004）。
3. **无适用域定义**：任意 SMILES 都会给概率（P1-02）。
4. **GNN 低于 RF 约 1.6 个 AUC 点**（复现 0.852 vs 0.868，同数据同 split 下 RF 全 5 折领先）：答辩口径——小数据（13k）+ 2D 描述符任务上，GCN 相对描述符模型常无优势，此为文献常见现象，不是实现错误；现有 pth（0.8766）显示调优后可接近 RF【P1-07 消融进一步证实】。
5. **median split 的语义**：标签含义是「高于该靶点数据中位水平」，**不是**药理学活性阈值（1 µM）。写材料时不得表述为「活性/无活性化合物」的二分。
6. **仅 EGFR**：其他靶点须经教学实验室现场下载训练（小模型，教学协议）。
7. **降级链**：已全部确定性化并显式标注层级（P0-03）；`fallback`/`minimal` 层为规则基线，结果不具科研效力，UI 有醒目警告。

## 引用规范

- 论文/申报书引用指标 → 本卡表格 + 对应 `artifacts/*.json` 文件名。
- 禁止引用无产物出处的数字（包括历史 PPT、旧版 README 中的任何未列于此卡的值）。
