"""pytest 单测 —— utils/teach_lab.py（教学实验室核心逻辑）。

设计原则
--------
* **完全离线**：所有输入都是本地合成的 DataFrame，不访问 ChEMBL / 网络。
* 合成数据里刻意包含：完全相同的 SMILES 字符串、同一分子的不同写法（会归一成同一个
  规范 SMILES）、非法 SMILES，以及覆盖活性 / 非活性 / 中间地带的 pIC50。
* 与项目硬约束对齐的断言：GNN 原子特征必须 **13 维**、GCN 层名必须是
  ``conv1/conv2/conv3 + bn1/bn2 + lin1/lin2``（匹配 gcn_egfr_best_model.pth）。

运行方式（Windows / conda env ``egfr-md``，在仓库根目录执行）::

    set KMP_DUPLICATE_LIB_OK=TRUE
    python.exe -m pytest tests/test_teach_lab.py -q
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_teach_lab():
    """导入被测模块。

    优先走正常的包导入（``utils.teach_lab``）；若 ``utils/__init__.py`` 因环境缺少
    某个子模块的依赖而导入失败，则回退为「按文件路径加载」。teach_lab.py 本身只依赖
    标准库 + numpy/pandas/rdkit/torch，没有相对导入，因此回退加载同样有效。
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import utils.teach_lab as module  # type: ignore[import-not-found]

        return module
    except Exception:  # pragma: no cover - 取决于 utils 其他子模块的环境依赖
        spec = importlib.util.spec_from_file_location(
            "teach_lab_under_test", REPO_ROOT / "utils" / "teach_lab.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


teach_lab = _load_teach_lab()


def _require_rdkit():
    pytest.importorskip("rdkit")
    if not getattr(teach_lab, "RDKIT_AVAILABLE", False):
        pytest.skip("RDKit 不可用，跳过依赖分子解析的用例")


def _require_torch():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if not getattr(teach_lab, "TORCH_AVAILABLE", False):
        pytest.skip("PyTorch / PyG 不可用，跳过 GNN 用例")


# ============================================================================
# 合成数据
# ============================================================================
#: 原始活性数据（模拟 ChEMBL 下载结果）。每行 (smiles, pIC50, 备注)。
#: 12 行，设计意图见行内注释；期望的逐阶段流失在 test_clean_activity_data_stage_counts 中。
SYNTHETIC_RAW_ROWS = [
    # ---- 完全相同的 SMILES 字符串：只有这一对被「去重」阶段移除 ----
    ("CCO", 8.0, "乙醇，活性（pIC50 >= 6）"),
    ("CCO", 5.0, "与上一行完全相同的字符串，去重时移除（保留 pIC50 更高的那条）"),
    # ---- 同一分子的不同写法：必须记在「SMILES 有效性检查」阶段 ----
    ("OCC", 7.5, "乙醇的另一种写法，规范化后与 CCO 相同"),
    ("CC(=O)Oc1ccccc1C(=O)O", 6.5, "阿司匹林，活性"),
    ("O=C(O)c1ccccc1OC(C)=O", 6.2, "阿司匹林的另一种写法"),
    ("c1ccccc1O", 4.0, "苯酚，非活性（pIC50 <= 5）"),
    ("Oc1ccccc1", 4.5, "苯酚的另一种写法"),
    # ---- 非法 SMILES：必须记在「SMILES 有效性检查」阶段 ----
    ("not_a_smiles", 7.0, "RDKit 无法解析"),
    ("C1CC", 6.8, "环未闭合，RDKit 无法解析"),
    # ---- 中间地带：必须记在「活性阈值过滤」阶段 ----
    ("c1ccccc1", 5.5, "苯，5 < pIC50 < 6，落在阈值中间地带"),
    # ---- 正常保留的非活性样本 ----
    ("Cc1ccccc1", 3.0, "甲苯，非活性"),
    ("CCCCC", 5.0, "戊烷，非活性（恰好等于 阈值-1）"),
]


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """合成原始数据框（列：smiles / pIC50）。"""
    return pd.DataFrame(
        [(s, v) for s, v, _ in SYNTHETIC_RAW_ROWS], columns=["smiles", "pIC50"]
    )


def _synthetic_descriptor_frame(n_per_class: int = 20, seed: int = 20260916) -> pd.DataFrame:
    """合成「12 个描述符 + label」数据框，供 RF 用例使用（不需要 RDKit）。"""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label in (1, 0):
        center = 0.0 if label == 1 else 3.0
        for _ in range(n_per_class):
            rows.append(rng.normal(center, 1.0, size=len(teach_lab.DESCRIPTOR_NAMES)))
            labels.append(label)
    df = pd.DataFrame(rows, columns=list(teach_lab.DESCRIPTOR_NAMES))
    df["label"] = labels
    return df


#: 20 个合法小分子，供 GNN 用例使用（活性/非活性各 10 个，便于 8:2 分层划分）
GNN_SMILES = [
    "CCO", "CCCO", "CCCCO", "CCN", "CCCN",
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "CC(=O)O", "CC(=O)N",
    "CCCC", "CCCCC", "CCOC", "COC", "CCS",
    "CCCl", "C1CCCCC1", "c1ccncc1", "NCCN", "CC(C)O",
]


@pytest.fixture()
def gnn_df() -> pd.DataFrame:
    """合成清洗后数据框（列：smiles / label），供 GNN 用例使用。"""
    labels = [1 if i % 2 == 0 else 0 for i in range(len(GNN_SMILES))]
    return pd.DataFrame({"smiles": list(GNN_SMILES), "label": labels})


# ============================================================================
# 1. 数据清洗：5 个 DataStage，且每一步 remaining/removed 正确
# ============================================================================
def test_clean_activity_data_returns_five_stages(raw_df):
    _require_rdkit()
    clean_df, stages = teach_lab.clean_activity_data(raw_df)

    assert len(stages) == 5
    assert all(isinstance(s, teach_lab.DataStage) for s in stages)
    assert [s.name for s in stages[:3]] == ["原始数据", "去重", "SMILES 有效性检查"]
    assert stages[3].name.startswith("活性阈值过滤")
    assert stages[4].name == "分子描述符计算"

    # 清洗结果结构：smiles / pic50 / label + 12 个描述符（顺序一致）
    assert list(clean_df.columns) == ["smiles", "pic50", "label"] + list(
        teach_lab.DESCRIPTOR_NAMES
    )
    assert not clean_df.isna().any().any()


def test_clean_activity_data_stage_counts(raw_df):
    """逐阶段 remaining / removed 的精确断言（手算见 SYNTHETIC_RAW_ROWS 注释）。"""
    _require_rdkit()
    clean_df, stages = teach_lab.clean_activity_data(raw_df)

    expected = {
        # name 前缀 -> (remaining, removed)
        "原始数据": (12, 0),
        "去重": (11, 1),
        "SMILES 有效性检查": (6, 5),
        "活性阈值过滤": (5, 1),
        "分子描述符计算": (5, 0),
    }

    for stage in stages:
        key = next(k for k in expected if stage.name.startswith(k))
        remaining, removed = expected[key]
        assert (stage.remaining, stage.removed) == (remaining, removed), stage.name

    # 每一阶段的 remaining/removed 之和守恒；整体流失 = 12 - 5
    assert stages[0].remaining + stages[0].removed == len(raw_df)
    for prev, cur in zip(stages, stages[1:]):
        assert cur.remaining + cur.removed == prev.remaining, cur.name
    assert len(clean_df) == stages[-1].remaining == 5


def test_dedup_counts_only_identical_smiles_strings(raw_df):
    """「去重」只统计完全相同的 SMILES 字符串（本用例中只有 CCO 这一对）。"""
    _require_rdkit()
    _, stages = teach_lab.clean_activity_data(raw_df)
    dedup = stages[1]

    assert dedup.name == "去重"
    assert dedup.removed == 1  # 只有 ("CCO", 8.0) 与 ("CCO", 5.0) 是完全相同的字符串


def test_canonicalization_and_invalid_smiles_counted_in_stage_3(raw_df):
    """同分子异写法 + 非法 SMILES 都必须记在「SMILES 有效性检查」阶段。"""
    _require_rdkit()
    clean_df, stages = teach_lab.clean_activity_data(raw_df)
    validity = stages[2]

    assert validity.name == "SMILES 有效性检查"
    # 5 条 = 2 条非法（not_a_smiles / C1CC）+ 3 条同分子异写法（OCC / 阿司匹林异构写法 / 苯酚异构写法）
    assert validity.removed == 5
    assert "2 条" in validity.note and "3 条" in validity.note

    # 归一后同一分子只留一条：结果里不再出现不同写法重复的分子
    assert len(clean_df) == 5
    assert sorted(clean_df["smiles"]) == sorted(
        ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "Oc1ccccc1", "Cc1ccccc1", "CCCCC"]
    )


def test_threshold_stage_drops_middle_zone(raw_df):
    """中间地带（pIC50=5.5）被阈值阶段移除，标签只保留 0/1。"""
    _require_rdkit()
    clean_df, stages = teach_lab.clean_activity_data(raw_df)
    threshold_stage = stages[3]

    assert threshold_stage.removed == 1
    assert set(clean_df["label"].unique()) == {0, 1}
    assert teach_lab.summarize_labels(clean_df) == {"total": 5, "active": 2, "inactive": 3}
    # 活性样本 pIC50 >= 阈值，非活性样本 pIC50 <= 阈值-1
    threshold = teach_lab.DEFAULT_PIC50_THRESHOLD
    assert (clean_df.loc[clean_df["label"] == 1, "pic50"] >= threshold).all()
    assert (clean_df.loc[clean_df["label"] == 0, "pic50"] <= threshold - 1.0).all()


def test_clean_activity_data_rejects_bad_input():
    _require_rdkit()
    with pytest.raises(ValueError):
        teach_lab.clean_activity_data(pd.DataFrame())
    with pytest.raises(ValueError):
        teach_lab.clean_activity_data(pd.DataFrame({"smiles": ["CCO"]}))  # 缺 pIC50 列


def test_data_stage_retention():
    stage = teach_lab.DataStage(name="x", description="y", remaining=3, removed=1)
    assert stage.retention == pytest.approx(0.75)
    assert teach_lab.DataStage(name="x", description="y", remaining=0, removed=0).retention == 0.0


# ============================================================================
# 2. 描述符：12 个键、顺序等于 DESCRIPTOR_NAMES；非法 SMILES 返回 None
# ============================================================================
def test_compute_descriptors_keys_and_order(raw_df):
    _require_rdkit()
    feats = teach_lab.compute_descriptors("CCO")  # 乙醇

    assert feats is not None
    assert len(feats) == 12
    assert list(feats.keys()) == list(teach_lab.DESCRIPTOR_NAMES)
    assert list(feats.keys()) == teach_lab.DESCRIPTOR_NAMES  # 顺序即特征向量顺序
    assert all(isinstance(v, float) and math.isfinite(v) for v in feats.values())

    # 乙醇的已知值（MolWt 46.069 / HBD 1 / 重原子 3）
    assert feats["MolWt"] == pytest.approx(46.069, abs=1e-3)
    assert feats["HBD"] == 1.0
    assert feats["HeavyAtoms"] == 3.0


def test_compute_descriptors_invalid_smiles_returns_none():
    _require_rdkit()
    assert teach_lab.compute_descriptors("not_a_smiles") is None
    assert teach_lab.compute_descriptors("C1CC") is None
    assert teach_lab.compute_descriptors(None) is None


# ============================================================================
# 3. pIC50 换算
# ============================================================================
def test_pic50_from_nm():
    assert teach_lab.pic50_from_nm(100) == pytest.approx(7.0, abs=1e-12)
    assert teach_lab.pic50_from_nm(1) == pytest.approx(9.0, abs=1e-12)
    assert teach_lab.pic50_from_nm(1000) == pytest.approx(6.0, abs=1e-12)


def test_pic50_from_nm_non_positive_is_nan():
    for bad in (0, 0.0, -1, -100.5, None, "abc"):
        assert math.isnan(teach_lab.pic50_from_nm(bad)), bad


# ============================================================================
# 4. 随机森林：两类可训练；单类别抛 ValueError
# ============================================================================
def test_train_rf_model_two_classes():
    df = _synthetic_descriptor_frame(n_per_class=20)
    result = teach_lab.train_rf_model(df, n_estimators=20, random_state=42)

    assert result.model_name == "随机森林 (RF)"
    assert result.n_features == len(teach_lab.DESCRIPTOR_NAMES) == 12
    assert result.n_train + result.n_test == len(df) == 40
    assert result.n_test == 8  # test_size=0.2
    assert 0.0 <= result.accuracy <= 1.0
    assert math.isfinite(result.auc)
    assert 0.0 <= result.auc <= 1.0

    assert len(result.extra["feature_importances"]) == 12
    assert {name for name, _ in result.extra["feature_importances"]} == set(
        teach_lab.DESCRIPTOR_NAMES
    )
    importances = [v for _, v in result.extra["feature_importances"]]
    assert importances == sorted(importances, reverse=True)
    assert result.extra["train_seconds"] >= 0.0
    assert 0.0 <= result.extra["positive_rate"] <= 1.0


def test_train_rf_model_single_class_raises():
    df = _synthetic_descriptor_frame(n_per_class=10)
    single = df[df["label"] == 1].reset_index(drop=True)
    assert single["label"].nunique() == 1

    with pytest.raises(ValueError):
        teach_lab.train_rf_model(single)

    with pytest.raises(ValueError):
        teach_lab.train_rf_model(pd.DataFrame())


# ============================================================================
# 5. 建图与模型结构（硬约束：13 维原子特征 + 固定层名）
# ============================================================================
def test_smiles_to_graph_atom_feature_dim_is_13():
    _require_torch()
    assert teach_lab.ATOM_FEATURE_DIM == 13

    data = teach_lab.smiles_to_graph("CCO", label=1)  # 乙醇：3 原子 2 键
    assert data.x.shape == (3, 13)
    assert data.x.shape[1] == 13
    assert data.edge_index.shape == (2, 4)  # 无向图两条边各建双向
    assert data.y.item() == 1.0

    torch = pytest.importorskip("torch")
    larger = teach_lab.smiles_to_graph(teach_lab.EXAMPLE_SMILES)
    assert larger.x.shape[1] == 13
    assert larger.x.dtype == torch.float32


def test_smiles_to_graph_invalid_smiles_raises():
    _require_torch()
    with pytest.raises(ValueError):
        teach_lab.smiles_to_graph("not_a_smiles")


def test_teach_gcn_module_names_and_shapes():
    _require_torch()
    model = teach_lab.TeachGCN()

    assert [name for name, _ in model.named_children()] == [
        "conv1", "conv2", "conv3", "bn1", "bn2", "lin1", "lin2",
    ]
    state = model.state_dict()
    for key in ("conv1", "conv2", "conv3", "bn1", "bn2", "lin1", "lin2"):
        assert any(k.startswith(key + ".") for k in state), key
    # 与 gcn_egfr_best_model.pth 的硬约束：输入特征维度 13、隐藏维度 128
    assert tuple(state["conv1.lin.weight"].shape) == (128, 13)
    assert tuple(state["bn1.weight"].shape) == (128,)
    assert tuple(state["lin1.weight"].shape) == (64, 128)
    assert tuple(state["lin2.weight"].shape) == (1, 64)


def test_teach_gcn_forward_output_shape():
    _require_torch()
    model = teach_lab.TeachGCN()
    model.eval()
    data = teach_lab.smiles_to_graph("CCO")
    with teach_lab.torch.no_grad():
        out = model(data)
    # 单个图 -> [1, 1] 的 logit
    assert tuple(out.shape) == (1, 1)


# ============================================================================
# 6. GNN 训练：3 个 epoch 跑通，回调 3 次，history['loss'] 长度 3
# ============================================================================
def test_train_gnn_model_three_epochs(gnn_df):
    _require_torch()
    calls = []

    def epoch_callback(epoch, train_loss, val_auc):
        calls.append((epoch, train_loss, val_auc))

    result = teach_lab.train_gnn_model(gnn_df, epochs=3, epoch_callback=epoch_callback)

    assert result.model_name == "图神经网络 (GCN)"
    assert len(calls) == 3  # epoch_callback 恰好被调用 3 次
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(math.isfinite(loss) for _, loss, _ in calls)

    history = result.extra["history"]
    assert len(history["loss"]) == 3
    assert len(history["val_auc"]) == 3
    assert all(math.isfinite(v) for v in history["loss"])

    assert result.n_features == teach_lab.ATOM_FEATURE_DIM == 13
    assert result.n_train + result.n_test == len(gnn_df) == 20
    assert result.extra["epochs"] == 3
    assert result.extra["from_scratch"] is True
    assert result.extra["train_seconds"] >= 0.0


def test_train_gnn_model_single_class_raises():
    _require_torch()
    single = pd.DataFrame({"smiles": GNN_SMILES[:10], "label": [1] * 10})
    with pytest.raises(ValueError):
        teach_lab.train_gnn_model(single, epochs=1)

    with pytest.raises(ValueError):
        teach_lab.train_gnn_model(pd.DataFrame(), epochs=1)
