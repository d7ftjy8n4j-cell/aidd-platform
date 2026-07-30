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

**药尘光**是一款面向 **AIDD（AI 辅助药物设计）教学** 的交互式 Web 平台，以 EGFR 激酶抑制剂为切入点，集成 **随机森林（RF）** 与 **图神经网络（GNN）** 双引擎，覆盖从公共数据库挖掘、分子活性预测、成药性评估、化学空间探索到蛋白-配体相互作用分析的全流程。

> *"双核驱动，理形相生"* —— 随机森林捕捉「经验之理」，图神经网络感知「结构之形」，双引擎相互验证，让 AI 决策透明可解释。

**教学定位**：适合药学、化学、生物信息学等专业的本科生/研究生，零基础上手，浏览器即开即用。通过渐进式标签页引导，直观对比传统特征工程与深度学习方法在药物发现中的表现。

---

## 🧭 AIDD 学习路径（16 个标签页，按认知逻辑编排）

| 阶段 | # | 页面 | 核心内容 | 教学目标 |
|:---:|---|------|----------|----------|
| 🔰 入门 | 1 | 🏠 **首页** | 系统概览、模型状态、使用统计 | 建立整体认知 |
| 📥 数据 | 2 | 📦 **数据获取** | ChEMBL / PubChem 检索、CSV 上传 | 学会获取公开化合物数据 |
| 🤖 预测 | 3 | 🧪 **分子预测** | RF + GNN 双引擎活性预测 | 体验 AI 预测分子活性 |
| 🧪 评估 | 4 | 🛡️ **药物筛选** | Lipinski 五规则、PAINS / Brenk 毒性 | 理解成药性评估 |
| 🔬 分析 | 5 | 🔍 **化学依据** | 描述符计算、相似性搜索 | 掌握分子理化性质分析 |
| 🎯 设计 | 6 | 🎯 **药效团设计** | 3D 药效团特征提取 | 理解「活性关键基团」 |
| 🗺️ 探索 | 7 | 🧩 **分子聚类** | Butina 聚类 + UMAP 可视化 | 探索化学空间多样性 |
| | 8 | 🧩 **公共子结构** | 最大公共子结构（MCS）分析 | 发现活性分子的共同骨架 |
| 🔗 结构 | 9 | 🔗 **3D 结构** | 蛋白-配体复合物交互式 3D 渲染 | 观察三维结合模式 |
| | 10 | 💊 **蛋白-配体作用** | 氢键、疏水、π-π 堆积等相互作用 | 理解分子间作用力 |
| | 11 | 🔗 **分子对接** | AutoDock Vina 对接模拟 | 体验计算对接流程 |
| 🧬 拓展 | 12 | 🧬 **激酶相似性** | 激酶组序列/结构相似性分析 | 理解激酶选择性 |
| ⚡ 整合 | 13 | ⚙️ **自动化流程** | 预测→筛选→药效团→相似性一键串联 | 体验完整 CADD 管线 |
| 📊 总结 | 14 | 📊 **模型与系统** | 特征重要性、混淆矩阵、架构图、技术栈 | 理解模型性能与系统设计 |

> **设计理念**：遵循「数据获取 → 单点分析 → 多维探索 → 结构可视化 → 流程整合 → 总结反思」的 AIDD 认知规律，每步均有 🎓 教学弹窗引导。

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
- **可解释性**：RF 输出特征重要性，GNN 展示图结构学习原理
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

### 🔗 3D 结构可视化 —— 蛋白-配体观察
- PDB ID 或本地文件加载蛋白-配体复合物
- 多种显示样式：cartoon、stick、sphere、surface
- 实时旋转、缩放、平移，支持自定义配色
- 推荐 EGFR 结构：3POZ（Takeda-285）、1M17（埃罗替尼）、2ITY（吉非替尼）

### 💊 蛋白-配体相互作用 —— 分子间作用力
- 自动检测氢键、疏水接触、π-π 堆积、盐桥等非共价相互作用
- 2D 相互作用图谱 + 3D 可视化联动
- 理解关键氨基酸残基对配体结合的贡献

### 🔗 分子对接 —— 计算结合模式
- 基于 AutoDock Vina 的配体-蛋白对接模拟
- 自定义对接盒子（grid box），可视化结合姿态与打分

### 🧬 激酶相似性 —— 选择性分析
- 基于序列/结构的激酶组相似性比较
- 评估化合物对 EGFR 家族成员（及脱靶激酶）的选择性

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
将原「模型分析」「技术详情」「关于项目」三合一，包含：
- **模型性能**：RF / GNN 的 AUC、准确率、特征重要性排序、混淆矩阵、训练曲线
- **双引擎架构图**：SMILES → RF 分支 / GNN 分支 → 集成决策
- **技术栈一览**：Streamlit + PyTorch Geometric + RDKit + scikit-learn
- **项目背景**：数据来源（ChEMBL 13,286 化合物）、开源协议、致谢

---

## 🚀 快速开始

### 本地运行（推荐 conda）

```bash
# 1. 克隆仓库
git clone https://github.com/d7ftjy8n4j-cell/ai-egfr-platform.git
cd ai-egfr-platform

# 2. 创建虚拟环境
conda create -n egfr python=3.10
conda activate egfr

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

### Streamlit Cloud 一键部署

1. 将代码推送到 GitHub 仓库
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 **New app** → 选择仓库 → 主文件设为 `app.py`
4. 点击 **Deploy**（已内置 `packages.txt` 和 `Dockerfile`）

---

## 📁 项目结构

```
.
├── app.py                              # 主应用入口（st.navigation 16 页架构）
├── requirements.txt                    # Python 依赖
├── packages.txt                        # 系统级依赖（Streamlit Cloud 用）
├── Dockerfile.dockerfile               # Docker 镜像
├── LICENSE                             # MIT 许可证
├── README.md                           # 本文件
│
├── 🧠 模型文件
│   ├── rf_egfr_model_final.pkl         # 随机森林模型（5 折 CV）
│   ├── gcn_egfr_best_model.pth         # GNN 模型（5 折 CV）
│   └── feature_names.json              # 特征名称清单
│
├── 🔧 核心模块
│   ├── real_predictor.py               # RF 预测器
│   ├── gnn_predictor.py                # GNN 预测器
│   ├── fallback_predictor.py           # 降级预测器（兜底方案）
│   ├── chem_filter.py                  # ADME / Ro5 / PAINS / Brenk 筛选
│   ├── chem_insight_safe.py            # 化学洞察与相似性搜索
│   ├── pharmacophore_streamlit.py      # 药效团分析引擎
│   ├── structure_viz.py                # 3D 可视化引擎
│   ├── molecule_utils.py               # 分子处理工具
│   ├── mcs_utils.py                    # MCS 最大公共子结构
│   ├── docking_utils.py                # 分子对接工具
│   ├── interaction_utils.py            # 蛋白-配体相互作用
│   └── protein_ligand_streamlit.py     # 蛋白-配体页面核心
│
├── 📂 页面模块（pages/）
│   ├── page_data_acquisition.py        # 数据获取
│   ├── page_clustering.py              # 分子聚类
│   ├── page_automated_pipeline.py      # 自动化流程
│   ├── protein_ligand_interaction.py   # 蛋白-配体相互作用
│   ├── kinase_similarity.py            # 激酶相似性
│   ├── molecular_docking.py            # 分子对接
│   └── mcs_analysis.py                 # MCS 公共子结构
│
└── 🛠️ 工具模块（utils/）
    ├── data_fetcher.py                 # ChEMBL / PubChem 数据获取
    ├── cluster_engine.py               # Butina 聚类引擎
    └── pipeline.py                     # 自动化流程编排器
```

---

## 🎯 使用指南

### 推荐学习路线（AIDD 教学）

| 步骤 | 操作 | 学习目标 |
|:--:|------|----------|
| 1 | 📦 **数据获取** → 从 ChEMBL 检索 EGFR 抑制剂数据 | 了解公共化合物数据库 |
| 2 | 🧪 **分子预测** → 输入吉非替尼 SMILES，对比两种模型结果 | 体验 AI 预测，理解 RF vs GNN |
| 3 | 🛡️ **药物筛选** → 评估吉非替尼的成药性与毒性风险 | 掌握 Lipinski 规则 |
| 4 | 🔍 **化学依据** → 查看分子的 LogP、TPSA 等描述符 | 理解理化性质计算 |
| 5 | 🎯 **药效团设计** → 从已知活性分子中提取药效团特征 | 理解活性关键基团 |
| 6 | 🧩 **分子聚类** + **公共子结构** → 探索化学空间与共同骨架 | 发现结构-活性关系 |
| 7 | 🔗 **3D 结构** → 加载 2ITY 观察吉非替尼与 EGFR 的结合 | 理解蛋白-配体三维结合 |
| 8 | 💊 **蛋白-配体作用** → 分析关键氢键与疏水接触 | 理解分子间作用力 |
| 9 | 🔗 **分子对接** → 对接一个虚拟分子到 EGFR | 体验计算对接流程 |
| 10 | 🧬 **激酶相似性** → 比较 EGFR 与 ErbB2 等家族成员 | 理解激酶选择性 |
| 11 | ⚙️ **自动化流程** → 运行一键全流程管线 | 串联各模块，形成完整认知 |
| 12 | 📊 **模型与系统** → 查看模型性能与系统架构 | 总结反思，理解全局 |

### 示例 SMILES

| 药物 | SMILES |
|------|--------|
| 吉非替尼 (Gefitinib) | `COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4` |
| 埃罗替尼 (Erlotinib) | `CCOCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OC` |
| 奥希替尼 (Osimertinib) | `CN1CCN(CCOC2=C(C=C3C(=C2)N=CN=C3NC4=CC(=C(C=C4)F)Cl)OC)C1` |

### 跨页面数据流

```
                    ┌──────────────────────────────┐
                    │      ⚙️ 自动化流程（终点）      │
                    └──────────────────────────────┘
                          ▲           ▲
                          │           │
        ┌─────────────────┘           └─────────────────┐
        │                                                 │
        ▼                                                 ▼
┌───────────────┐                                   ┌──────────────┐
│  📦 数据获取   │──────▶ 🧩 分子聚类 ──────▶        │ 🧪 分子预测   │
└───────────────┘                                   └──────────────┘
```

---

## 📊 模型性能

| 模型 | AUC (5-fold CV) | 准确率 | 特征 | 可解释性 |
|------|:---:|:---:|------|:---:|
| 随机森林 | **0.867 ± 0.005** | **0.782 ± 0.005** | 200+ RDKit 描述符 | ⭐⭐⭐ 高 |
| GNN (GCN) | **0.845 ± 0.008** | **0.767 ± 0.011** | 13 维原子特征 | ⭐⭐ 中 |

> **训练数据**：ChEMBL EGFR 靶点（CHEMBL203），IC50 (nM) 筛选去重后 **13,286** 个唯一化合物（50.8% 活性）。

### 模型选择建议

| 场景 | 推荐 | 理由 |
|------|:--:|------|
| 需要可解释的特征重要性 | RF | 特征工程透明 |
| 探索新型骨架分子 | GNN | 端到端学习，不依赖预设描述符 |
| 高可靠性要求 | 双模型 | 结论一致时可信度高 |
| 快速批量筛选 | RF | 推理速度快 |

---

## 🛠️ 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| Web 框架 | Streamlit ≥ 1.28 | 交互式界面 |
| 传统 ML | scikit-learn | 随机森林模型 |
| 深度学习 | PyTorch + PyTorch Geometric | GNN 图神经网络 |
| 化学信息学 | RDKit | 分子解析、描述符、指纹 |
| 3D 可视化 | py3Dmol / nglview | 蛋白-配体结构渲染 |
| 分子对接 | AutoDock Vina | 计算结合姿态与亲和力 |
| 降维可视化 | UMAP-learn | 化学空间 2D 投影 |
| 数据处理 | pandas / numpy | 数据清洗与统计 |
| 数据获取 | chembl_webresource_client | ChEMBL API 访问 |

---

## 📝 依赖说明

### 核心依赖（`requirements.txt`）

| 包 | 最低版本 | 说明 |
|----|:---:|------|
| `streamlit` | 1.28.0 | Web 框架 |
| `rdkit` | 2022.9.5 | 化学信息学核心 |
| `scikit-learn` | 1.3.2 | 随机森林 |
| `torch` | 2.1.2 | GNN 后端 |
| `pandas` / `numpy` | 1.5.3 | 数据处理 |

### 可选依赖（按需安装）

| 包 | 对应页面 | 说明 |
|----|----------|------|
| `chembl_webresource_client` | 数据获取 | ChEMBL 数据库检索 |
| `umap-learn` | 分子聚类 | UMAP 降维可视化 |
| `py3dmol` | 3D 结构 | 蛋白-配体 3D 渲染 |
| `plip` | 蛋白-配体作用 | 相互作用自动检测 |
| `nglview` | 分子对接 | 对接结果可视化 |
| `openbabel` | 分子对接 | 分子格式转换 |

### 部署注意
Streamlit Cloud 需通过 `packages.txt` 安装 `autodock-vina` 等系统级依赖。

---

## 🙏 致谢与资源

- **数据来源**：[ChEMBL](https://www.ebi.ac.uk/chembl/)（EMBL-EBI）、[PubChem](https://pubchem.ncbi.nlm.nih.gov/)（NCBI）
- **教程参考**：[TeachOpenCADD](https://github.com/volkamerlab/TeachOpenCADD)（T001, T007, T033, T035）
- **开源工具**：RDKit、PyTorch Geometric、Streamlit、scikit-learn
- **项目仓库**：[GitHub](https://github.com/d7ftjy8n4j-cell/ai-egfr-platform)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！本项目遵循开源精神，所有代码可供教学与科研复用。

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

**双核驱动，理形相生**  
*从微观尘埃中寻找治愈之光*

</div>
