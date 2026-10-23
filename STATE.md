# STATE — 《药尘光》科研化改进总控状态

> 本文件由改进总控代理维护。每轮任务卡执行后更新。
> 最后更新：2026-10-24（第 3 轮）

---

## ⚠️ 第 3 轮重要更正（覆盖第 1-2 轮的部分结论）

人类提供了 `data/raw/egfr_all_ic50_full.csv`（13,286 行）与 `data/raw/gcn_cv_results.csv` 后：

1. **README 的 RF 四项声称全部复现成功**（EXP-010b）：
   `scripts/train_rf_legacy_protocol.py` 用 **median split（pIC50 ≥ 中位数 7.0）** 在 13,286 条原始数据上得到
   CV AUC **0.8680 ± 0.0050**（称 0.867±0.005）、Accuracy **0.7851 ± 0.0073**（称 0.782±0.005）、
   正例率 **50.81%**（称 50.8%）、样本量 **13,286**（称 13,286）。
2. **第 1-2 轮「指标虚高 / 13,286 无法复现 / Accuracy 是伪指标」的判断撤销**。真实原因是
   **训练代码未入库 + 两套标注协议并存未文档化**，不是数据造假：
   - **legacy 协议**（内置模型，README 口径）：13,286 条，median split，类别 1.03:1 平衡 → Accuracy 0.785 ≈ BAcc 0.784，**无不平衡陷阱**。
   - **teach_lab 协议**（教学实验室现场训练）：1,606 条（阈值 6.0 + 丢弃中间带），86% 活性 → 在此协议下 Accuracy 确有虚高（EXP-012 结论仅适用于该协议）。
3. **GNN 声称值有真实来源文件，且已完整复现（EXP-019，第 3 轮）**：`gcn_cv_results.csv` 5 折 AUC 均值 0.8445（称 0.845）、ACC 0.7671（称 0.767）；`scripts/train_gnn.py --dataset legacy` 复现实测 **0.8517±0.0061 / 0.7719±0.0061**，逐折 Δ<0.011。现有 pth 在同测试集 AUC 0.8766，与复现模型一致率 88.75%、概率相关 0.939 → GNN pth 与数据集 A 同源。
4. 第 1-2 轮完成的泄漏排查（EXP-013 骨架划分、EXP-014 标签打乱）是按 teach_lab 协议做的，结论方向有效，但正式口径应改在 legacy 协议上重跑（P0-05）。

---

## 一、项目定位判断（代理自主判定）

| 维度 | 现状自评 | 目标 |
|---|---|---|
| 工程可靠性 | **8/10** — 32 单测全过；三层降级链全部确定性且诚实标注；假预测器与合成数据重建清零 | 8/10 ✅ |
| 科研可信度 | **9/10**（3→5.5→6.5→7→7.5→8→8.5→**9**）— P0+P1 全部收官：四项声称闭环复现、口径绑定、10-seed 骨架外推、校准（ECE 5.3%）、适用域悬崖（NN<0.4 崩至随机）、基线对比、失败案例归因、消融矩阵、GNN 排序、**时序外部验证诚实失败（AUC 0.53）并完成归因**。15 问闭环 13 个 | 8/10 ✅ |
| 教学可证性 | **4/10** — 教学设计极好，但零学习者数据（P2，需真人参与） | 7/10 |
| 材料齐全度 | **8.5/10** — 数据卡+模型卡+失败案例库+31 条实验记录+校准图+**申报材料四件套已对齐证据链（P3-06）**；剩 P2/P3 新增内容（按人类指示暂停） | 8/10 ✅ |

**结论：P0 + P1 全部收官；P3 既有申报材料已完成事实性修订与证据注入（第 9 轮）。**
审稿 15 问闭环 13 个（剩 Q3 不可考已冻结、Q14 需真人、Q15 低优先）。
**最重要的诚实结论**：模型在同批爬取数据内部表现稳定（0.82–0.87 区间，视划分方式），
但**不具备时间外推能力**（EXP-029，新记录 AUC 0.53，且非结构距离可解释）——所有对外材料已按此口径约束。
待人类指示：P2 教学评估（需真人）、P3 新增内容（报告/预印本骨架）、现场训练 0.9763 出处核验。

---

## 二、本轮实测环境（可复现基线）

```
解释器  C:\Users\dadamingli\AppData\Roaming\mamba\envs\egfr-md\python.exe
版本    Python 3.11.15
启动    python -m streamlit run app.py  →  /_stcore/health = 200 ✅（第 4 轮复检通过）
单测    pytest -q  →  32 passed（20 原有 + 4 registry 对账 + 8 预测器确定性）✅
导入    39 个模块全部 OK ✅
推理    吉非替尼 → RF 0.6005(活性) / GCN 0.6124(活性) ✅
依赖    streamlit 1.60.0 · rdkit 2023.09.6 · sklearn 1.9.0 · numpy 1.26.4
        pandas 1.5.3 · torch 2.5.1+cpu · torch_geometric 2.4.0 · shap 0.46.0
        openmm 8.2 · mdtraj · plip · openbabel 3.1.0 · smina.exe 在 env\Library\bin ✅
网络    ChEMBL API 可达（HTTP 200）✅
```

---

## 三、已完成的文件级审计（事实，非推测）

### 模型溯源（实测 pkl / pth 内部）

**RF 模型 `rf_egfr_model_final.pkl`（25 MB）**

| 属性 | 实测值 |
|---|---|
| 类型 | `RandomForestClassifier` |
| `n_estimators` | 200 |
| `n_features_in_` | **12** |
| `max_depth` | 15（所有树一致） |
| `max_features` | `'sqrt'` |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |
| `criterion` / `bootstrap` | gini / True |
| `class_weight` | None |
| `oob_score` | **False**（无 OOB 可用） |
| `_sklearn_version` | **缺失** —— 训练时未写入版本标记 |
| 特征名 | `feature_names.json` 的 12 个 RDKit 描述符 |
| Top5 重要性 | MolWt .146 / TPSA .129 / MolLogP .115 / NumRotatableBonds .099 / FractionCSP3 .099 |

**GNN 模型 `gcn_egfr_best_model.pth`（180 KB）**

| 属性 | 实测值 |
|---|---|
| checkpoint 类型 | 裸 `OrderedDict`，**无任何元数据** |
| 键数 | 20 个张量 |
| 键名 | `conv1/2/3.{bias,lin.weight}` + `bn1/2.{weight,bias,running_mean,running_var,num_batches_tracked}` + `lin1/2.*` |
| 结构 | GCNConv×3(13→128→128→128) + BN×2 + Linear(128,64) + Linear(64,1) |
| 参数量 | 43,649 |
| 附带指标 | **无**（auc/epoch/seed/数据版本一概没有） |

### 已确认的问题（P0 级别，影响科研可信度）

| ID | 问题 | 证据位置 | 严重度 |
|---|---|---|---|
| F-01 | ~~「5 折 CV AUC 0.867±0.005」无代码来源~~ **已解决（第 3 轮）**：`scripts/train_rf_legacy_protocol.py` 复现 0.8680±0.0050 | artifacts/rf_legacy_protocol_metrics.json | ✅ 已闭环 |
| F-02 | ~~std=±0.005 无从产生~~ **已解决**：median split 协议 5 折 std 实测 0.0050 | 同上 | ✅ 已闭环 |
| F-03 | ~~GNN 指标硬编码在返回字典~~ **已修复（P0-08）**：`gnn_predictor.py` 运行时值改引 `model_registry.GNN_PERF`（0.8517/0.7719） | ✅ 已闭环 |
| F-04 | ~~界面显示的 GNN AUC 三处矛盾~~ **已修复**：Config 与运行时同引 `model_registry.py` 单一事实源 | ✅ 已闭环 |
| F-05 | ~~界面显示的 RF AUC 与 README 矛盾~~ **已修复**：0.8695 → 0.868（registry） | ✅ 已闭环 |
| F-06 | ~~「200+ 描述符」~~ **已全部修复**：README 3 处 + app.py 5 处 → 「12 个描述符（200 棵树）」 | ✅ 已闭环 |
| F-07 | ~~无训练脚本~~ **RF+GNN 全部闭环**：`train_rf.py` / `train_rf_legacy_protocol.py` / `train_gnn.py --dataset legacy` 均可重跑 | scripts/ | ✅ 已闭环 |
| F-08 | ~~假预测器 `MinimalEGFRPredictor` 用 `hash()` 造随机数~~ **已修复（P0-03）**：移入 `minimal_predictor.py`，确定性启发式 + tier 标签 | ✅ 已闭环 |
| F-09 | ~~`fallback_predictor` 用 `hash()`+高斯噪声~~ **已修复**：噪声移除，两层规则基线确定性输出 + tier 标签 + 空串拦截 | ✅ 已闭环 |
| **F-15** | **`real_predictor._rebuild_model` 用合成正态分布数据训假 RF 冒充真模型** | ✅ 已修复（P0-03，第 5 轮发现并移除；模型缺失时显式 None → 降级链接管） |
| F-10 | ~~无数据泄漏防护证明~~ **已闭环**：teach_lab 协议（EXP-013/014）+ legacy 协议 10-seed scaffold split（EXP-021，零骨架重叠 AUC 0.8215±0.0292，诚实口径：随机划分对新骨架泛化偏乐观 4.7 点） | ✅ 已闭环 |
| F-11 | 训练时阈值/去重规则未落盘，模型无法对应到具体数据版本 | legacy 协议已落盘（rf_legacy_protocol_metrics.json）；teach_lab 现场训练未落盘 | 🟡 半闭环 |
| **F-12** | **部署模型与教学实验室用两套不同特征名** | `feature_names.json` 12 名 vs `teach_lab.DESCRIPTOR_NAMES` 12 名，**仅 5 个同名**（详见下） | 🔴 致命（易导致特征错位静默出错） |
| **F-13** | **部署模型特征计算静默填 0** | `real_predictor.py:203-208` 任何异常→`0.0`，无日志、无计数 | 🟠 高 |
| **F-14** | **两套标注协议并存且未文档化**（legacy median split vs teach_lab 阈值+丢弃） | EXP-010b/010c；README 未注明口径来源 | 🟠 高（P0-06/07/08 核心议题） |

#### F-12 细节（本轮新发现）

两套特征集名字不同、顺序不同、语义部分重叠：

| 序号 | `feature_names.json`（部署 pkl 用） | `teach_lab.DESCRIPTOR_NAMES`（教学小模型用） |
|---|---|---|
| 1 | MolWt | MolWt ✅同名 |
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

仅在 `feature_names.json`：`HeavyAtomCount, MolLogP, NumAromaticRings, NumHAcceptors, NumHDonors, NumRotatableBonds, NumSaturatedRings`
仅在 `DESCRIPTOR_NAMES`：`AromaticRings, HBA, HBD, HeavyAtoms, LogP, MolMR, RotatableBonds`

**影响**：如果有人把 `teach_lab` 训练出的模型套到 `real_predictor` 的特征计算上（或反之），**12 个特征里有 7 个对不上，模型会静默产出错误预测且不报错**。教学实验室的「小模型」与「内置模型」不可互换，但代码里没有任何一道防线。

**代理处置**：P0-01 脚本已改为**严格复刻 `real_predictor.smiles_to_features`**（含静默填 0 行为），并显式统计填充次数写入 metrics.json。两套特征的关系需在 P0-07 模型卡中明确写出。

### 已确认的工程问题（P1/P2）

| ID | 问题 | 证据 |
|---|---|---|
| E-01 | ~~`environment_md.yml` 声明 python=3.10 / torch=2.1.2，实际 3.11.15 / 2.5.1~~ | ✅ 已修复（C-03，第 3 轮穿插） |
| E-02 | ~~`requirements.txt` 要求 `pandas>=2.0.0`，实际 1.5.3~~ | ✅ 已修复（C-04，改为 >=1.5.3，兼容 2.x） |
| E-03 | ~~README 两处仓库名 `ai-egfr-platform`，实际 remote 是 `aidd-platform`~~ | ✅ 已修复（C-01，第 3 轮穿插） |
| E-04 | 25 MB pkl 直接进 git，未走 LFS | 未修（C-06，需人类决策） |
| E-05 | `__pycache__` 残留 cpython-314 的 `_ucw_*.pyc`，源码已不存在 | 待清理（安全，可随时做） |
| E-06 | ~~`git worktree` 挂着 prunable 的 Linux 死路径~~ | ✅ 已修复（C-05，worktree prune 完成） |
| E-07 | 分支 `openclaw/teach-lab-chembl-status-nav` 悬空未处理 | 需人类确认是否删除 |

### 已确认的教学/材料缺口

| ID | 缺口 |
|---|---|
| M-01 | 零学习者数据（无前后测、无问卷、无评分记录） |
| M-02 | ~~无模型卡 / 数据卡~~ → **数据卡 + 模型卡均已完成**（`docs/DATA_CARD.md` P0-06、`docs/MODEL_CARD.md` P0-07，第 3 轮穿插；GNN 复现值待 P0-02 回填） |
| M-03 | 无失败案例记录（哪些分子预测错、为什么） |
| M-04 | 无校准分析（概率是否可信） |
| M-05 | 无适用域定义（AD） |
| M-06 | ~~无实验记录文件~~ → EXPERIMENTS.md 已建立并持续维护（EXP-000~023） |

---

## 四、已完成

- [x] 环境实测（Python/依赖/网络/端口/单测/推理）
- [x] 39 个模块导入验证
- [x] RF/GNN checkpoint 内部结构逆向
- [x] 全仓库 CV 代码搜索（结论：不存在）
- [x] 降级链假预测器定位
- [x] 发现 F-12：两套特征名冲突（7/12 对不上）
- [x] 建立 `scripts/ data/ docs/ artifacts/` 目录骨架
- [x] 建立 6 个状态文件
- [x] **P0-01 完成**：`scripts/train_rf.py` 真实数据跑通，产出 1,998 条快照 + 指标 JSON
- [x] **EXP-010** 5 折 CV 复现（teach_lab 协议）：AUC **0.8815 ± 0.0339**
- [x] **EXP-012** 揭示类别不平衡陷阱（teach_lab 协议下）：Accuracy 0.8944 但多数类基线 0.8598
- [x] **EXP-013** 骨架划分排除泄漏：零重叠 AUC 0.8767 > 随机 0.8593
- [x] **EXP-014** 打乱标签对照：0.49 ≈ 随机，证明无标签泄漏
- [x] **P0-01b 完成（第 3 轮）**：`scripts/train_rf_legacy_protocol.py` 在 13,286 条 median split 数据上**完全复现 README 四项声称**（AUC 0.8680±0.0050 / ACC 0.7851±0.0073 / 50.81% / 13,286），产物 `artifacts/rf_legacy_protocol.pkl` + metrics.json
- [x] **F-01/F-02/F-07(RF) 闭环**，新增 F-14（两协议并存未文档化）
- [x] `train_gnn.py` 扩展 `--dataset legacy --tag` 支持（复现 README GNN 口径，产物隔离不覆盖）

## 五、两套协议对照（第 3 轮定性，取代旧的「四处偏差」表）

| 维度 | legacy 协议（README/内置模型口径） | teach_lab 协议（教学实验室现场训练） |
|---|---|---|
| 数据 | `data/raw/egfr_all_ic50_full.csv`，13,286 条 | `data/raw/chembl_egfr_activities.csv`，1,998 条快照 |
| 标注 | **median split**：pIC50 ≥ 7.0（=中位数）→ 活性，无丢弃 | 阈值 6.0 + 丢弃中间带（5.0<pIC50<6.0 的样本丢弃） |
| 正例率 | 50.81%（1.03:1 平衡） | 86.0%（6:1 不平衡） |
| 复现脚本 | `scripts/train_rf_legacy_protocol.py` ✅ | `scripts/train_rf.py` ✅ |
| RF 结果 | AUC **0.8680±0.0050** / ACC 0.7851 / BAcc 0.7845 / MCC 0.5712 | AUC 0.8815±0.0339 / ACC 0.8935（虚高，净增益仅 +0.035） |
| GNN 结果 | 复现（EXP-019）：AUC **0.8517±0.0061** / ACC 0.7719 / BAcc 0.7714 / MCC 0.5463；原始 CSV 0.8445/0.7671；现有 pth 同测试 0.8766（更优） | 未训练 |
| 产物 | `artifacts/rf_legacy_protocol.pkl` + metrics.json | `artifacts/rf_repro.pkl` + rf_metrics.json |
| 适用场景 | **正式指标、论文、申报书引用** | **教学演示**（展示清洗决策对分布的影响） |

**两套协议都真实、都可重跑，但回答的问题不同**：
- legacy：这个模型对「pIC50 是否高于中位」的判断准不准（平衡任务，Accuracy 可用）。
- teach_lab：这个模型对「pIC50 ≥ 6（1 µM 药理学阈值）」的判断准不准（不平衡任务，必须看 BAcc/MCC）。

**风险（F-14）**：README/UI 未注明任何一处数字来自哪套协议。读者会拿 teach_lab 协议的 86% 正例率去质疑 legacy 的 50.8%，或反之。P0-06/07/08 统一解决。

## 六、进行中

- [x] **P0 + P1 全部收官**（第 8 轮，2026-10-24）。剩余实验级待补：EXP-031（新记录标签噪声量化）、DUD-E 跨源验证（可选）。
- [ ] **下一阶段**：P2 教学评估方案（我起草方案/题库，真实数据需人类采集）→ P3 材料写作（技术报告/预印本/申报书/答辩 PPT，证据链已齐）。

## 七、阻塞（需人类）

当前 **0 项**。训练类命令已改由代理直接执行（DECISIONS D-09）。
剩余人类决策仅 1 项低优先：C-06 pkl 转 LFS（涉及 git 历史重写，需人类拍板时机）。

---

## 八、代理自主决策摘要

见 `DECISIONS.md`。关键三条：
1. **不删除任何现有模型文件**，新模型另存并对比，旧模型保留作对照。
2. **指标口径以「代码能重跑出来」为唯一标准**，凡不能重跑的声称一律降级为【待补】。
3. **先补可复现性，不动 UI**。P0 期间禁止改任何 `pages/` 文件，避免引入回归。
