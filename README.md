# 🧬 药尘光 · EGFR 抑制剂智能发现与设计平台

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-3D9970?logo=python&logoColor=white)](https://www.rdkit.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

🔗 **在线体验**：[https://ai-egfr-platform.streamlit.app/](https://ai-egfr-platform.streamlit.app/)

---

## 📖 项目简介

**药尘光**是一款面向 **AIDD（AI 辅助药物设计）教学** 的交互式 Web 平台，以 EGFR 激酶抑制剂为切入点，集成 **随机森林（RF）** 与 **图神经网络（GNN）** 双引擎，覆盖从公共数据库挖掘、分子活性预测、成药性评估、化学空间探索、蛋白-配体结构分析、分子对接、分子动力学模拟到 MM-GBSA 结合自由能计算的全流程。

> *"双核驱动，理形相生"* —— 随机森林捕捉「经验之理」，图神经网络感知「结构之形」，双引擎相互验证，让 AI 决策透明可解释。

**教学定位**：适合药学、化学、生物信息学等专业的本科生/研究生，零基础上手，浏览器即开即用。通过渐进式标签页引导，直观对比传统特征工程与深度学习方法在药物发现中的表现。

---

## 🧭 AIDD 学习路径（13 个顶层标签页，部分含子标签）

| 阶段 | # | 页面 | 子标签 | 教学目标 |
|:---:|---|------|------|----------|
| 🔰 入门 | 1 | 🏠 **首页** | — | 建立整体认知 + 快速上手引导 |
| 📥 数据 | 2 | 📦 **数据获取** | — | 学会从 ChEMBL / PubChem 获取化合物数据 |
| 🤖 预测 | 3 | 🧪 **分子预测** | — | RF + GNN 双引擎 + SHAP 解释 + 不确定性 |
| 🧪 评估 | 4 | 🧪 **分子评估** | 🛡️药物筛选 · 🔍化学依据 | Lipinski / PAINS / Brenk + 描述符 + 相似性 |
| 🎯 设计 | 5 | 🎯 **药效团设计** | — | 3D 药效团特征提取与模型生成 |
| 🗺️ 探索 | 6 | 🗺️ **化学空间** | 🧩分子聚类 · 🧩公共子结构 | Butina 聚类 + UMAP + MCS 骨架发现 |
| 🔬 结构 | 7 | 🔬 **结构分析** | 🔗3D可视化 · 💊相互作用 | 3D 渲染 + PLIP 非共价作用检测 |
| 🔗 对接 | 8 | 🔗 **分子对接** | 🔗单分子 · 🧩批量 | Smina 精确对接 + 虚拟筛选排序 |
| ⚛️ 模拟 | 9 | ⚛️ **分子动力学** | ⚛️MD模拟 · ⚛️MM-GBSA | OpenMM 全原子 MD + 结合自由能 (ΔG) |
| 🧬 拓展 | 10 | 🧬 **激酶相似性** | — | KLIFS-IFP 激酶组选择性分析 |
| | 11 | 🧬 **分子生成** | — | LSTM 自回归生成新颖 EGFR 抑制剂 |
| ⚡ 整合 | 12 | ⚙️ **自动化流程** | — | 预测→筛选→药效团→相似性一键串联 |
| 📊 总结 | 13 | 📊 **模型与系统** | 📈性能 · 🏗️架构 · 📚关于 | 模型评估 + 双引擎架构 + 技术栈 + 项目背景 |

> **设计理念**：遵循「数据获取 → 单点分析 → 多维探索 → 结构可视化 → 计算模拟 → 流程整合 → 总结反思」的 AIDD 认知规律。  
> 每个标签页和子标签均有 **🎓 教学弹窗**，点击即可学习相关理论——从 Lipinski 五规则到 AMBER 力场，从 SHAP 瀑布图到 LSTM 温度采样。

### 🗂️ 功能速览

| 模块 | 页面 | 一句话说明 |
|------|------|------|
| 📥 数据层 | 数据获取 | ChEMBL / PubChem 检索 + CSV 上传 |
| 🤖 预测层 | 分子预测 | RF + GNN 双引擎 + SHAP 解释 + 不确定性估计 |
| 🧪 评估层 | 分子评估 | 成药性 (Lipinski) + 毒性 (PAINS/Brenk) + 理化性质 |
| 🎯 设计层 | 药效团设计 | 3D 药效团特征提取 |
| 🗺️ 探索层 | 化学空间 | Butina 聚类 + UMAP + MCS 骨架发现 |
| 🔬 结构层 | 结构分析 | 3D 渲染 + PLIP 相互作用检测 |
| 🔗 对接层 | 分子对接 | 单分子精确对接 + 批量虚拟筛选 |
| ⚛️ 模拟层 | 分子动力学 | OpenMM MD + MM-GBSA 结合自由能 |
| 🧬 拓展层 | 激酶相似性 · 分子生成 | KLIFS 选择性 + LSTM 从头设计 |
| ⚡ 整合层 | 自动化流程 | 预测→筛选→药效团→相似性一键串联 |
| 📊 总结层 | 模型与系统 | 性能 + 架构 + 技术栈 + 背景（四合一） |

---

## ✨ 核心功能详解

### 📦 数据获取 —— AIDD 的起点
- **ChEMBL 检索**：按靶点名称（如 EGFR）获取化合物活性数据，支持 pIC50 / IC50 过滤
- **PubChem 相似性搜索**：输入 SMILES，按 Tanimoto 相似度查找类似化合物
- **文件上传**：支持 CSV / Excel，自动检测 SMILES 列
- **一键流转**：获取结果可直接送入分子聚类或自动化流程

### 🧪 分子活性预测 —— 双引擎核心
- **随机森林（RF）**：200+ RDKit 分子描述符，5 折交叉验证 AUC ≈ **0.867**
- **图神经网络（GNN）**：3 层 GCN，13 维原子特征，端到端学习分子图拓扑
- **三种模式**：标准（RF）、高级（GNN）、双模型对比
- **可解释性**：
  - **SHAP 瀑布图**：分解每个描述符对预测的贡献
  - **不确定性估计**：通过随机森林 100 棵树的投票方差评估预测可信度
- 支持批量 SMILES 与 CSV 导出

### 🛡️ 药物筛选 —— 成药性关卡
- **Lipinski 五规则（Ro5）**：MW ≤ 500、LogP ≤ 5、HBA ≤ 10、HBD ≤ 5
- **毒性警报**：PAINS（泛测定干扰化合物）+ Brenk（不良子结构）
- 支持单分子评估与批量筛选，输出统计报告与可视化图表

### 🔍 化学依据 —— 分子性质剖析
- 分子描述符计算（LogP、TPSA、HBA/HBD、可旋转键等）
- 基于 Morgan 指纹的参考分子相似性搜索
- RF 与 GNN 分子表示对比

### 🎯 药效团设计 —— 活性特征提取
- 从多个活性分子中提取共同药效团特征（氢键供/受体、疏水区、芳香环）
- 生成 3D 药效团模型，指导分子优化方向

### 🧩 分子聚类 —— 化学空间探索
- **Butina 算法**：基于 Tanimoto 距离的层次聚类
- 支持 Morgan / RDKit 指纹，可调距离阈值
- **UMAP 降维可视化**：化学空间 2D 投影，按簇着色
- 簇浏览器展示代表分子结构，支持一键送入自动化流程

### 🧩 最大公共子结构（MCS）—— 骨架发现
- 识别多个活性分子的最大公共子结构
- 可视化公共骨架，辅助骨架跃迁与先导化合物优化
- 预置 EGFR 抑制剂系列模板

### 🔗 3D 结构可视化 —— 蛋白-配体观察
- PDB ID 或本地文件加载蛋白-配体复合物
- 多种显示样式：cartoon、stick、sphere、surface
- 实时旋转、缩放、平移，支持自定义配色
- 推荐 EGFR 结构：3POZ（Takeda-285）、1M17（埃罗替尼）、2ITY（吉非替尼）

### 💊 蛋白-配体相互作用 —— 分子间作用力
- 自动检测氢键、疏水接触、π-π 堆积、盐桥等非共价相互作用
- 2D 相互作用图谱 + 3D 可视化联动
- 支持从 MD 模拟页面直接衔接数据

### 🔗 分子对接 —— 计算结合模式
- 基于 Smina（AutoDock Vina 分支）的配体-蛋白对接模拟
- 自定义对接盒子（grid box），可视化结合姿态与打分

### 🧩 批量对接 —— 虚拟筛选方法论
- 多配体 SMILES 并行对接至同一靶点蛋白
- 按结合能排序，对比分析不同配体的亲和力
- 参考 TeachOpenCADD T015 + T018

### ⚛️ 分子动力学模拟 —— 蛋白-配体动态行为
- 基于 **OpenMM** 对蛋白-配体复合物进行全原子 MD 模拟
- 三步标签页：输入设置 → 执行监控（异步模拟 + 实时进度） → 轨迹分析
- 支持轨迹下载与下游分析衔接
- 参考 TeachOpenCADD T019

### ⚛️ MM-GBSA —— 结合自由能估算
- 从 MD 轨迹计算蛋白-配体结合自由能 (ΔG = G_complex - G_receptor - G_ligand)
- 使用 GB-Neck2 隐式溶剂模型，基于 OpenMM + MDTraj，单轨迹协议
- 支持多帧采样、结果可视化与下载

### 🧬 激酶相似性 —— 选择性分析
- 基于 **KLIFS-IFP** 相互作用指纹比较不同激酶的结合模式
- 预置 12 种激酶：EGFR, ErbB2, ErbB4, CDK2, CDK4, MET, KDR, LCK, SRC, ABL1, BRAF, p38α
- 评估化合物对 EGFR 家族成员及脱靶激酶的选择性

### 🧬 分子生成 —— AI 驱动的从头设计
- 字符级 LSTM 自回归生成新 SMILES 分子
- 借鉴 **REINVENT** 范式，支持温度采样和迁移学习（在 EGFR 抑制剂上微调）
- 参考 TeachOpenCADD T034

### ⚙️ 自动化流程 —— 一键全流程
按需组合 6 大模块，串联为完整筛选管线：
1. 🌲 随机森林预测
2. 🧠 GNN 预测
3. 💊 ADME / Ro5 筛选
4. ⚠️ 不良子结构筛查
5. 🎯 药效团匹配
6. 🔍 相似性搜索

支持单分子、CSV 批量、从其他页面导入数据。

### 📊 模型与系统 —— 总结与反思
将「模型分析」「技术详情」「关于项目」三合一，包含：
- **模型性能**：RF / GNN 的 AUC、准确率、特征重要性排序、混淆矩阵、训练曲线
- **双引擎架构图**：SMILES → RF 分支 / GNN 分支 → 集成决策
- **技术栈一览**：Streamlit + PyTorch Geometric + RDKit + scikit-learn + OpenMM + SHAP
- **项目背景**：数据来源（ChEMBL 13,286 化合物）、开源协议、致谢

### 📤 KNIME 导出 —— 对接外部工作流
- 支持将平台分析结果导出为 KNIME 兼容格式（CSV + 元数据 + 工作流描述符）
- 与 TeachOpenCADD-KNIME W1-W8 工作流对齐
- 一键下载 ZIP 包，可直接导入 KNIME Analytics Platform

---

## 🚀 快速开始

### 本地运行（推荐 conda）

```bash
# 1. 克隆仓库
git clone https://github.com/d7ftjy8n4j-cell/ai-egfr-platform.git
cd ai-egfr-platform

# 2. 创建虚拟环境（需要 conda 以支持 OpenMM/OpenBabel 等编译依赖）
conda env create -f environment_md.yml
conda activate egfr

# 3. 安装 PyPI 依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

### 最小化安装（仅核心预测功能）

```bash
pip install streamlit rdkit-pypi scikit-learn pandas numpy
streamlit run app.py
```

> 无 conda 环境时，分子对接（需 OpenBabel + Smina）、分子动力学（需 OpenMM）、MM-GBSA（需 OpenMM + MDTraj）和蛋白-配体作用分析（需 PLIP）将自动降级但其他功能正常。

### Streamlit Cloud 一键部署

1. 将代码推送到 GitHub 仓库
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 **New app** → 选择仓库 → 主文件设为 `app.py`
4. 点击 **Deploy**（已内置 `packages.txt` 和 `Dockerfile.dockerfile`）

---

## 📁 项目结构

```
.
├── app.py                      # 主入口（st.navigation 13 页，5 组合并标签）
├── requirements.txt            # PyPI 依赖（Streamlit Cloud 兼容）
├── packages.txt                # apt 系统依赖
├── environment_md.yml          # conda 全栈环境（含 OpenMM/OpenBabel）
├── Dockerfile.dockerfile       # Docker 镜像
│
├── 🧠 模型                     # rf_egfr_model_final.pkl / gcn_egfr_best_model.pth
├── 🔧 引擎                     # 12 个核心模块（predictor / filter / docking / md / …）
├── 📂 pages/                   # 10 个独立页面（clustering / pipeline / docking / md / …）
├── 🛠️ utils/                   # 5 个工具模块（fetcher / cluster / pipeline / shap / …）
└── 🧩 components/              # 可复用 UI 组件（knime_export）
```

> 完整文件清单见仓库。核心逻辑集中在根目录 `.py` 模块中，页面 UI 在 `pages/`，纯工具函数在 `utils/`。

---

## 🎯 使用指南

### 推荐学习路线（AIDD 教学 · 12 步）

| 步骤 | 操作 | 学习目标 |
|:--:|------|----------|
| 1 | 📦 **数据获取** → 从 ChEMBL 检索 EGFR 抑制剂数据 | 了解公共化合物数据库 |
| 2 | 🧪 **分子预测** → 输入吉非替尼 SMILES，对比两种模型 | 体验 AI 预测，理解 RF vs GNN + SHAP |
| 3 | 🧪 **分子评估** → 评估吉非替尼的成药性与毒性风险 | 掌握 Lipinski 规则 + 理化性质 |
| 4 | 🎯 **药效团设计** → 从已知活性分子提取药效团特征 | 理解活性关键基团 |
| 5 | 🗺️ **化学空间** → 聚类降维 + MCS 骨架发现 | 探索化学空间与结构-活性关系 |
| 6 | 🔬 **结构分析** → 加载 2ITY 观察 3D 结合 + 相互作用 | 理解蛋白-配体三维结合 |
| 7 | 🔗 **分子对接** → 单分子精确对接 + 批量虚拟筛选 | 掌握计算对接与筛选方法论 |
| 8 | ⚛️ **分子动力学** → MD 模拟 + MM-GBSA 结合自由能 | 从静态到动态，定量评估亲和力 |
| 9 | 🧬 **激酶相似性** → 比较 EGFR 与 ErbB2 等家族成员 | 理解激酶选择性 |
| 10 | 🧬 **分子生成** → AI 生成新颖 EGFR 抑制剂候选分子 | 体验生成式 AI 药物设计 |
| 11 | ⚙️ **自动化流程** → 运行一键全流程管线 | 串联各模块，形成完整认知 |
| 12 | 📊 **模型与系统** → 查看性能指标与系统架构 | 总结反思，理解全局 |

### 示例 SMILES

| 药物 | SMILES |
|------|--------|
| 吉非替尼 (Gefitinib) | `COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4` |
| 埃罗替尼 (Erlotinib) | `CCOCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OC` |
| 奥希替尼 (Osimertinib) | `CN1CCN(CCOC2=C(C=C3C(=C2)N=CN=C3NC4=CC(=C(C=C4)F)Cl)OC)C1` |

---

## 📊 模型性能

| 模型 | AUC (5-fold CV) | 准确率 | 特征 | 可解释性 |
|------|:---:|:---:|------|:---:|
| 随机森林 | **0.867 ± 0.005** | **0.782 ± 0.005** | 200+ RDKit 描述符 | ⭐⭐⭐ 高 (SHAP + 不确定性) |
| GNN (GCN) | **0.845 ± 0.008** | **0.767 ± 0.011** | 13 维原子特征 | ⭐⭐ 中 |

> **训练数据**：ChEMBL EGFR 靶点（CHEMBL203），IC50 (nM) 筛选去重后 **13,286** 个唯一化合物（50.8% 活性）。

### 模型选择建议

| 场景 | 推荐 | 理由 |
|------|:--:|------|
| 需要可解释的特征重要性 | RF | 特征工程透明 + SHAP 瀑布图 |
| 探索新型骨架分子 | GNN | 端到端学习，不依赖预设描述符 |
| 高可靠性要求 | 双模型 | 结论一致时可信度高 |
| 快速批量筛选 | RF | 推理速度快 |
| 需要量化预测可信度 | RF | 内置不确定性估计 |

---

## 🛠️ 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| Web 框架 | Streamlit ≥ 1.56 | 交互式界面 |
| 传统 ML | scikit-learn | 随机森林模型 |
| 深度学习 | PyTorch + PyTorch Geometric | GNN 图神经网络 + LSTM 分子生成 |
| 模型解释 | SHAP | 特征贡献分析 |
| 化学信息学 | RDKit | 分子解析、描述符、指纹 |
| 3D 可视化 | py3Dmol / nglview | 蛋白-配体结构渲染 |
| 分子对接 | Smina (AutoDock Vina) | 计算结合姿态与亲和力 |
| 分子动力学 | OpenMM | 全原子 MD 模拟 |
| 自由能计算 | MDTraj + OpenMM | MM-GBSA 结合自由能 |
| 蛋白-配体作用 | PLIP | 非共价相互作用检测 |
| 降维可视化 | UMAP-learn | 化学空间 2D 投影 |
| 数据处理 | pandas / numpy | 数据清洗与统计 |
| 数据获取 | chembl_webresource_client | ChEMBL API 访问 |
| 激酶分析 | KLIFS REST API | 激酶结合模式相似性 |
| 工作流导出 | KNIME 兼容格式 | 外部工作流对接 |

---

## 📝 依赖说明

### 核心依赖（`requirements.txt`，Streamlit Cloud 可用）

| 包 | 最低版本 | 说明 |
|----|:---:|------|
| `streamlit` | 1.56.0 | Web 框架 |
| `rdkit-pypi` | 2022.9.5 | 化学信息学核心 |
| `scikit-learn` | 1.3.0 | 随机森林 |
| `torch` | 2.1.2 | PyTorch 后端 |
| `torch-geometric` | 2.4.0 | 图神经网络 |
| `pandas` / `numpy` | 1.5.0 / 1.19.3 | 数据处理 |
| `shap` | 0.41.0 | 模型可解释性 |
| `umap-learn` | 0.5.5 | 化学空间降维 |
| `plotly` | 5.18.0 | 交互式图表 |

### 可选依赖（需 conda 本地安装，Streamlit Cloud 不可用）

| 包 | 对应页面 | 说明 |
|----|----------|------|
| `openmm` / `pdbfixer` / `openff-toolkit` | 分子动力学 | MD 模拟引擎 |
| `mdtraj` | MM-GBSA | 轨迹分析与自由能计算 |
| `smina` / `openbabel` | 分子对接 / 批量对接 | 对接引擎与分子格式转换 |
| `plip` | 蛋白-配体作用 | 相互作用自动检测 |

以上包因需要 SWIG / C++ 编译环境，建议通过 conda 安装：
```bash
conda install -c conda-forge openmm openmmforcefields openff-toolkit pdbfixer mdtraj openbabel plip smina
```

---

## 🙏 致谢与资源

- **数据来源**：[ChEMBL](https://www.ebi.ac.uk/chembl/)（EMBL-EBI）、[PubChem](https://pubchem.ncbi.nlm.nih.gov/)（NCBI）、[KLIFS](https://klifs.net/)（激酶结构数据库）
- **教程参考**：[TeachOpenCADD](https://github.com/volkamerlab/TeachOpenCADD)（T001, T007, T015, T018, T019, T034）
- **开源工具**：RDKit、PyTorch Geometric、Streamlit、scikit-learn、OpenMM、SHAP
- **项目仓库**：[GitHub](https://github.com/d7ftjy8n4j-cell/ai-egfr-platform)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！本项目遵循开源精神，所有代码可供教学与科研复用。

## 📄 许可证

[MIT License](LICENSE)

---

## ❓ 常见问题

<details>
<summary><b>Streamlit Cloud 部署后某些页面报错？</b></summary>

部分页面依赖需 C++ 编译的包（OpenMM、OpenBabel、PLIP），无法在 Streamlit Cloud 构建。这些页面在云端会自动降级显示安装指引。**完整功能需本地 conda 环境**：
```bash
conda env create -f environment_md.yml
conda activate egfr
streamlit run app.py
```
</details>

<details>
<summary><b>分子动力学模拟需要多久？</b></summary>

教学演示（5,000 步 ≈ 10 ps）：CPU 约 2-5 分钟，GPU 约 30 秒。研究级模拟（500,000 步 ≈ 1 ns）：GPU 约 30-60 分钟。**推荐在有 NVIDIA GPU 的本地机器上运行**。
</details>

<details>
<summary><b>如何导入自己的分子数据？</b></summary>

在「📦 数据获取」页面支持 CSV/Excel 上传，只需包含 SMILES 列即可。也可通过 ChEMBL 靶点名或 PubChem 相似性搜索在线获取。获取后可直接送入分子聚类或自动化流程。
</details>

<details>
<summary><b>预测结果可靠吗？如何解读？</b></summary>

- **双模型一致** → 高可信度
- **双模型不一致** → 该分子可能具有特殊结构（RF 依赖预设描述符，GNN 学习图拓扑），建议参考 SHAP 解释和不确定性估计
- **不确定性低 + 概率远离 0.5** → 预测可信
- **不确定性高或概率接近 0.5** → 建议结合分子对接、MD 模拟进一步验证
</details>

<details>
<summary><b>页面间如何传递数据？</b></summary>

```
📦 数据获取 ──→ 🧩 分子聚类 ──→ ⚙️ 自动化流程
     │                                    ▲
     └────────────────────────────────────┘
🧪 分子预测 ──────────────────────────────┘
⚛️ 分子动力学 ──→ 💊 蛋白-配体作用（自动衔接）
```
所有中间结果通过 `st.session_state` 跨页面共享，无需手动导出/导入。
</details>

---

<div align="center">

**双核驱动，理形相生**  
*从微观尘埃中寻找治愈之光*

</div>
