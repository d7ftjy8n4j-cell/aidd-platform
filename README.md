# 🧬 药尘光 · EGFR抑制剂智能发现与设计平台

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-3D9970?logo=python&logoColor=white)](https://www.rdkit.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**药尘光** 是一款面向 EGFR 抑制剂的 **教学友好型 Web 平台**，集成随机森林（RF）与图神经网络（GNN）双引擎，提供从数据获取、分子预测、成药性筛选到化学空间探索的全流程功能。

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

🔗 **在线体验**：[https://ai-egfr-platform.streamlit.app/](https://ai-egfr-platform.streamlit.app/)

---

## 📋 功能总览（12 大标签页）

| # | 页面 | 功能 |
|---|------|------|
| 🏠 | **首页** | 系统概览、模型状态、使用统计 |
| 🧪 | **分子预测** | RF + GNN 双引擎活性预测，支持单分子 / 批量 / 双模型对比 |
| 🛡️ | **药物筛选** | Lipinski 五规则 (ADME)、PAINS / Brenk 毒性警报 |
| 🔍 | **化学依据** | 分子描述符计算、已知活性分子相似性搜索 |
| 🎯 | **药效团设计** | 3D 药效团特征提取与模型生成 |
| 🔗 | **3D 结构** | 蛋白-配体复合物交互式 3D 可视化 |
| 📊 | **模型分析** | 特征重要性、混淆矩阵、训练曲线 |
| 🧩 | **分子聚类** | Butina 聚类 + UMAP 降维可视化，化学空间探索 |
| 📦 | **数据获取** | 从 ChEMBL / PubChem / 文件上传获取化合物数据 |
| ⚙️ | **自动化流程** | 一键全流程：预测 → 筛选 → 子结构 → 药效团 → 相似性 |
| 🔬 | **技术详情** | 系统架构、技术栈、特征工程对比 |
| 📚 | **关于项目** | 背景、核心理念、致谢 |

---

## ✨ 核心功能详情

### 🔬 双引擎智能预测 (`🧪 分子预测`)
- **随机森林 (RF)**：基于 200+ 个 RDKit 分子描述符，5 折交叉验证 AUC ≈ **0.870**
- **图神经网络 (GNN)**：3 层 GCN，13 维原子特征，端到端学习分子拓扑结构
- **三种预测模式**：标准（RF）、高级（GNN）、双模型对比
- 支持批量 SMILES 输入与结果导出

### 🛡️ 药物筛选 (`🛡️ 药物筛选`)
- **Lipinski 五规则 (Ro5)**：分子量 ≤ 500、LogP ≤ 5、HBA ≤ 10、HBD ≤ 5
- **毒性警报**：PAINS 泛测定干扰化合物筛查 + Brenk 不良子结构检测
- 支持单分子分析与批量数据集筛选

### 🧩 分子聚类 (`🧩 分子聚类`)
- **Butina 算法**：基于 Tanimoto 距离的层次聚类
- 支持 Morgan / RDKit 指纹，可调距离阈值和指纹参数
- **UMAP 降维可视化**：化学空间 2D 投影，按簇着色
- 簇浏览器：查看每个簇的代表分子结构图与簇内相似度
- 支持将代表性分子一键送入自动化流程

### 📦 数据获取 (`📦 数据获取`)
- **ChEMBL 检索**：按靶点名称获取化合物活性数据（支持 pIC50 过滤）
- **PubChem 相似性搜索**：输入 SMILES，按 Tanimoto 相似度查找类似化合物
- **文件上传**：支持 CSV / Excel 格式，自动检测 SMILES 列
- 获取结果可一键送入自动化流程或分子聚类

### ⚙️ 自动化流程 (`⚙️ 自动化流程`)
将各模块组合为可配置的完整筛选管线：
1. 🌲 随机森林预测
2. 🧠 GNN 图神经网络预测
3. 💊 ADME / Ro5 成药性筛选
4. ⚠️ 不良子结构筛查
5. 🎯 药效团匹配
6. 🔍 相似性搜索

支持单个 SMILES、批量 CSV 上传、从数据获取 / 聚类页面导入。

### 🎨 3D 分子可视化 (`🔗 3D 结构`)
- 支持蛋白-配体复合物（PDB ID / 本地文件）交互式 3D 渲染
- 多种显示样式：cartoon、stick、sphere，可自定义配色
- 实时旋转、缩放、平移

### 🔍 化学依据 (`🔍 化学依据`)
- 分子描述符计算（LogP、TPSA、HBA / HBD 等）
- 参考分子相似性搜索（Morgan 指纹）
- RF / GNN 表示对比

### 📊 模型分析 (`📊 模型分析`)
- 随机森林特征重要性排序图
- GNN 混淆矩阵与训练曲线
- 双模型性能对比

---

## 🚀 快速开始

### 本地运行（推荐使用 conda）

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

已配置 `packages.txt` 和 `Dockerfile`，可直接部署到 [Streamlit Cloud](https://share.streamlit.io)：

1. 将代码推送到 GitHub 仓库
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app" → 选择仓库 → 设置主文件为 `app.py`
4. 点击 "Deploy"

---

## 📁 项目结构

```
.
├── app.py                          # 主应用（st.navigation 12 页架构）
├── requirements.txt                # Python 依赖
├── packages.txt                    # 系统依赖（供 Streamlit Cloud）
├── Dockerfile.dockerfile           # Docker 镜像配置
├── LICENSE                         # MIT 许可证
├── README.md                       # 本文件
│
├── 🧠 模型文件
│   ├── rf_egfr_model_final.pkl     # 随机森林模型（交叉验证训练）
│   ├── gcn_egfr_best_model.pth     # GNN 模型（交叉验证训练）
│   └── feature_names.json          # 特征名称列表
│
├── 📊 可视化资源
│   ├── feature_importance.png      # RF 特征重要性图
│   ├── gcn_confusion_matrix.png    # GNN 混淆矩阵
│   └── gcn_training_history.png    # GNN 训练曲线
│
├── 🔧 核心模块
│   ├── real_predictor.py           # 随机森林预测器
│   ├── gnn_predictor.py            # GNN 预测器
│   ├── chem_filter.py              # ADME / Ro5 筛选器
│   ├── chem_insight_safe.py        # 化学洞察与相似性搜索
│   ├── pharmacophore_streamlit.py  # 药效团分析
│   └── structure_viz.py            # 3D 可视化引擎
│
├── 📂 独立页面模块（pages/）
│   ├── page_automated_pipeline.py  # ⚙️ 自动化流程页面
│   ├── page_clustering.py          # 🧩 分子聚类页面
│   └── page_data_acquisition.py    # 📦 数据获取页面
│
└── 🛠️ 工具模块（utils/）
    ├── pipeline.py                 # 自动化流程编排器
    ├── data_fetcher.py             # ChEMBL / PubChem 数据获取
    └── cluster_engine.py           # Butina 分子聚类引擎
```

---

## 🎯 使用指南

### 推荐学习路径

1. **🧪 分子预测** → 输入 SMILES，体验双引擎对比预测
2. **🛡️ 药物筛选** → 评估分子的成药性与安全性
3. **🔍 化学依据** → 查看分子描述符与已知活性分子相似性
4. **🎯 药效团设计** → 提取活性分子的 3D 药效特征
5. **🔗 3D 结构** → 观察蛋白-配体相互作用
6. **📊 模型分析** → 理解模型性能与特征重要性
7. **📦 数据获取** → 从公开数据库获取化合物数据
8. **🧩 分子聚类** → 探索化学空间多样性
9. **⚙️ 自动化流程** → 一键全流程综合评估

### 示例 SMILES（吉非替尼）
```
COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4
```

### 跨页面数据流
```
📦 数据获取 ──┬──▶ 🧩 分子聚类 ──┬──▶ ⚙️ 自动化流程
              │                   │
              └───────────────────┘
🧪 分子预测 ─────────────────────────▶ ⚙️ 自动化流程
```

---

## 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| Web 框架 | Streamlit | 1.28.0+ |
| 机器学习 | scikit-learn | 1.3.2 |
| 深度学习 | PyTorch / PyTorch Geometric | 2.1.2+ |
| 化学信息学 | RDKit | 2022.9.5+ |
| 3D 可视化 | py3Dmol / stmol | 0.3.0+ |
| 降维可视化 | UMAP-learn | 可选 |
| 数据处理 | pandas / numpy | 1.5.3+ |
| 数据获取 | chembl_webresource_client | 可选 |

---

## 📊 模型性能（5 折交叉验证）

| 模型 | AUC | 准确率 | 特征 |
|------|-----|--------|------|
| 随机森林 | **0.867 ± 0.005** | **0.782 ± 0.005** | 200+ RDKit 分子描述符 |
| GNN | **0.845 ± 0.008** | **0.767 ± 0.011** | 13 维原子特征（GCN） |

> 数据来源：ChEMBL EGFR 靶点（CHEMBL203），IC50 (nM) 筛选去重后共 **13,286** 个唯一化合物。

---

## 📝 依赖说明

### Python 依赖
核心依赖见 `requirements.txt`。关键包：
- `streamlit` ≥ 1.28.0 — Web 框架
- `rdkit` — 化学信息学核心
- `scikit-learn` — 随机森林模型
- `torch` + `torch-geometric` — 图神经网络
- `pandas` / `numpy` — 数据处理

### 可选依赖
- `chembl_webresource_client` — ChEMBL 数据库访问（数据获取页面需要）
- `umap-learn` — 分子聚类页面的 UMAP 降维可视化
- `py3dmol` + `stmol` — 3D 结构可视化

### 部署注意
在 Streamlit Cloud 部署时，需通过 `packages.txt` 安装 `autodock-vina` 等系统级依赖。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。本项目遵循开源精神，所有代码可供教学与科研复用。

---

## 📄 许可证

[MIT License](LICENSE)

---

## 🙏 致谢

- **数据**：ChEMBL 数据库（EMBL-EBI）、PubChem（NCBI）
- **教程**：TeachOpenCADD (T001, T007, T033, T035)
- **工具**：RDKit, PyTorch Geometric, Streamlit, scikit-learn

---

<div align="center">

**双核驱动，理形相生**  
*从微观尘埃中寻找治愈之光*

</div>"
