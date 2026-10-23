# 数据卡（Data Card）— EGFR 活性数据集

> 维护：改进总控代理 · 2026-10-24（P0-06）
> 原则：**每一个数字都可由脚本重跑得出**，来源标注到产物文件。
> 本项目存在**两套数据集与两套标注协议**（F-14），引用指标时必须注明用的是哪一套。

---

## 数据集 A — legacy 原始训练集（内置模型 / README 口径）

| 属性 | 值 |
|---|---|
| 文件 | `data/raw/egfr_all_ic50_full.csv` |
| sha256 | `8525319917e534b72a5cb3c440f23dee61379460082cb88a333948854a0e7dc4` |
| 行数 | **13,286**（列：`SMILES`, `pIC50`） |
| 重复 SMILES | 0（导出前已完成去重） |
| 缺失值 | 0 |
| pIC50 分布 | min 1.60 / Q1 5.61 / **median 7.00** / Q3 7.98 / max 17.30 / mean 6.80 / std 1.51 |
| 标注协议 | **median split**：pIC50 ≥ 7.0000（= 中位数）→ 活性；无丢弃 |
| 标签分布 | 活性 6,750（50.81%）/ 非活性 6,536（49.19%），不平衡比 1.03:1 |
| 来源 | ChEMBL CHEMBL203（EGFR）IC50 活性数据导出。**具体 API 查询参数不可考【待核验】**——导出发生在仓库之外，本卡以 sha256 冻结当前版本 |
| 使用者 | `rf_egfr_model_final.pkl`（README RF 指标）、原始 GNN 训练（`data/raw/gcn_cv_results.csv`） |
| 复现脚本 | `scripts/train_rf_legacy_protocol.py`（RF）、`scripts/train_gnn.py --dataset legacy`（GNN） |

**已知质量问题（必须披露）**：
1. **1 条 pIC50 = 17.30**（IC50 ≈ 5×10⁻⁹ nM）超出物理合理上限，对应阿法替尼类共价抑制剂（`C=CC(=O)N1CC[C@H](Oc2nc(-c3n[nH]c(=O)[nH]3)cc3ccccc23)C1`）。占比 1/13,286，对 CV 指标影响可忽略（复现 AUC 0.8680 与声称一致），但正式材料引用分布统计时需注明。

---

## 数据集 B — teach_lab 快照（教学实验室现场训练口径）

| 属性 | 值 |
|---|---|
| 文件 | `data/raw/chembl_egfr_activities.csv` |
| sha256 | `79068a1ed591d3ee650101057d70c288424df38456d5702ded7c9fc4e479c70c` |
| 拉取时间 | 2026-10-24（UTC），chembl_webresource_client 0.10.9，耗时 99.1 s |
| 查询参数 | target=CHEMBL203（EGFR）· type=IC50 · relation='=' · assay_type='B' · organism=Homo sapiens · limit 2000 → **1,998 条**（API 报告 raw_count 10,000，受 limit 截断） |
| 重复 / 缺失 | 0 / 0，单位全部 nM |
| pIC50 分布 | min 1.26 / Q1 5.68 / median 6.80 / Q3 7.78 / max 11.52 / mean 6.74 / std 1.40 |
| 标注协议 | **阈值过滤**：pIC50 ≥ 6.0 → 活性；≤ 5.0 → 非活性；**5.0 < pIC50 < 6.0 的中间带丢弃** |
| 清洗流水线 | 1,998 → 去重（-0）→ SMILES 校验（-0）→ 阈值过滤（**-392**，19.6%）→ 1,606 条 |
| 标签分布 | 活性 1,381（86.0%）/ 非活性 225（14.0%），不平衡比 ~6:1 |
| 使用者 | `scripts/train_rf.py` 复现管线、教学实验室「现场训练」功能 |
| 复现脚本 | `scripts/train_rf.py`（快照优先，无快照时自动拉取并落盘） |

---

## 两套协议对照（为什么有两套？）

| 维度 | A：legacy（README/内置模型） | B：teach_lab（教学现场） |
|---|---|---|
| 样本量 | 13,286 | 1,606（自 1,998 快照清洗） |
| 标注 | median split（无丢弃） | 阈值 6.0 + 丢弃中间带 |
| 正例率 | 50.81%（平衡） | 86.0%（6:1 不平衡） |
| 回答的问题 | 「pIC50 是否高于靶点中位水平」 | 「pIC50 是否达到 1 µM 药理学阈值」 |
| Accuracy 可用性 | ✅ 可用（BAcc 0.7845 ≈ ACC 0.7851） | ⚠️ 有虚高（多数类基线 0.8598，净增益仅 +0.035，EXP-012） |
| 适用场景 | **正式指标、论文、申报书** | **教学演示**（展示清洗决策对数据分布的影响） |

**引用规则**：正式材料中的 0.867±0.005 / 0.782±0.005 / 0.845 / 0.767 / 13,286 / 50.8% 全部出自数据集 A；
任何来自数据集 B 的数字（1,606 / 86.0% / AUC 0.8815）不得与 A 口径混排比较。

**特征不可互换警告（F-12）**：部署模型用 `feature_names.json` 的 12 个描述符，教学小模型用
`teach_lab.DESCRIPTOR_NAMES` 的 12 个，**仅 5 个同名、顺序不同**。两套模型与两套特征计算路径严格绑定，交叉使用会静默产出错误预测。

---

## 复现命令

```bash
# 数据集 A · RF（约 12 s，CPU）
python scripts/train_rf_legacy_protocol.py
# 数据集 A · GNN（约 41 min，CPU，EXP-019 已完成）
python scripts/train_gnn.py --dataset legacy --tag legacy --epochs 25
# 数据集 B · RF（约 2 min，首次需联网拉快照）
python scripts/train_rf.py
```

产物与指纹见 `artifacts/*_metrics.json`（含数据 sha256、超参、逐折结果、环境版本）。

## 维护纪律

- 训练只读快照，不实时拉 ChEMBL（数据版本漂移 = 不可复现，DECISIONS D-04）。
- 任何数据变更必须更新本卡 + 重新计算 sha256 + 在 EXPERIMENTS.md 记录 EXP 编号。
- 快照元数据：`data/raw/chembl_egfr_activities.meta.json`（数据集 B）、本文档 sha256（数据集 A）。
