# pages/page_teach_lab.py
"""
🎓 教学实验室（Teach Lab）页面
==============================

四步闭环，每一步都可见、可解释：

    步骤 1  从 ChEMBL 下载真实活性数据
    步骤 2  清洗数据（用 DataStage 逐个展开，看清每一步流失了多少样本）
    步骤 3  现场训练两个小模型：随机森林（秒级）+ GNN（分钟级，含实时训练曲线）
    步骤 4  用刚训练好的模型预测新分子

全部核心逻辑在 ``utils/teach_lab.py``；本文件只负责交互与可视化。

页面状态（st.session_state）：
    tl_raw_df     步骤 1 的原始数据
    tl_clean_df   步骤 2 的清洗后数据
    tl_pipeline   步骤 2 的清洗阶段列表（List[DataStage]）
    tl_rf_result  步骤 3.1 的随机森林训练结果
    tl_gnn_result 步骤 3.2 的 GNN 训练结果
    tl_target     当前靶点名

约束：下载或清洗后必须清空下游结果，避免用户看到过期的模型。

作者：dadamingli
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# 绘图：matplotlib（图表标签统一用 ASCII，避免中文字体缺失导致方块）
import matplotlib.pyplot as plt

# 分子 2D 结构图（RDKit 缺失时降级为文字提示，不影响其它流程）
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    _RDKIT_DRAW_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于环境
    _RDKIT_DRAW_AVAILABLE = False

from utils import teach_lab as tl


# ============================================================================
# 常量与状态管理
# ============================================================================
#: 本页面用到的 session_state 键及默认值
_STATE_DEFAULTS: Dict[str, Any] = {
    "tl_raw_df": None,
    "tl_clean_df": None,
    "tl_pipeline": None,
    "tl_rf_result": None,
    "tl_gnn_result": None,
    "tl_target": "EGFR",
}

#: 每个步骤需要清空的「下游」键（下载/清洗后必须清空，避免展示过期模型）
_DOWNSTREAM_KEYS: Dict[str, List[str]] = {
    "download": ["tl_clean_df", "tl_pipeline", "tl_rf_result", "tl_gnn_result"],
    "clean": ["tl_rf_result", "tl_gnn_result"],
    "rf": [],
    "gnn": [],
}


def _init_state() -> None:
    """初始化本页面用到的 session_state 键（幂等）。"""
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _clear_downstream(level: str) -> None:
    """清空某个步骤的下游结果。

    Args:
        level: "download"（清空清洗+两模型）/ "clean"（清空两模型）。
    """
    for key in _DOWNSTREAM_KEYS.get(level, []):
        st.session_state[key] = _STATE_DEFAULTS[key]


def _render_molecule_image(smiles: str, size: tuple = (320, 240)) -> bool:
    """在页面上画一个分子的 2D 结构图。

    Args:
        smiles: 分子 SMILES。
        size: 图片尺寸。

    Returns:
        是否成功绘制。
    """
    if not _RDKIT_DRAW_AVAILABLE:
        st.caption("RDKit 未安装，无法绘制 2D 结构。")
        return False
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        st.caption("SMILES 无法解析，无法绘制结构。")
        return False
    try:
        st.image(Draw.MolToImage(mol, size=size), caption="分子 2D 结构（RDKit 绘制）")
        return True
    except Exception as exc:  # pragma: no cover - 绘图失败不影响主流程
        st.caption(f"结构绘制失败: {exc}")
        return False


# ============================================================================
# 图表
# ============================================================================
def _plot_roc(result: tl.TrainingResult) -> Optional[plt.Figure]:
    """画测试集 ROC 曲线（含对角线参考）。"""
    fpr, tpr = result.extra.get("roc", ([], []))
    if not fpr:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"{result.model_name} (AUC={result.auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_feature_importance(result: tl.TrainingResult) -> Optional[plt.Figure]:
    """画特征重要性横向条形图（按重要性降序，最重要在最上方）。"""
    importances = result.extra.get("feature_importances")
    if not importances:
        return None
    names = [name for name, _ in importances]
    values = [value for _, value in importances]

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(names) + 1.2))
    # barh 从下往上绘制：先反转，让重要性最高的显示在最上方
    ax.barh(names[::-1], values[::-1], color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_gnn_history(loss_history: List[float], auc_history: List[float]) -> plt.Figure:
    """画 GNN 训练曲线（左：训练 Loss；右：测试集 Val AUC）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    epochs = list(range(1, len(loss_history) + 1))

    axes[0].plot(epochs, loss_history, marker="o", ms=3, color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, auc_history, marker="o", ms=3, color="#1f77b4")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation AUC")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================================
# 步骤 1：下载数据
# ============================================================================
def _render_step1() -> None:
    """步骤 1：从 ChEMBL 下载真实活性数据并展示原始数据。"""
    st.subheader("步骤 1 · 从 ChEMBL 下载真实活性数据")
    st.caption(
        "输入靶点名（EGFR / HER2 / BTK …），选择活性类型与下载量。"
        "数据来自 ChEMBL 真实记录，不是模拟数据——所以会出现噪声、重复和缺失，这正是清洗环节存在的理由。"
    )

    with st.form("tl_download_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            target_input = st.text_input(
                "靶点名称",
                value=str(st.session_state.get("tl_target") or "EGFR"),
                help="ChEMBL 靶点检索关键词，例如 EGFR、HER2、BTK。建议用基因/蛋白通用名。",
            )
        with col2:
            activity_type = st.selectbox(
                "活性类型",
                options=["IC50", "Ki", "Kd", "EC50"],
                index=0,
                help="IC50=半数抑制浓度（最常用）；Ki=抑制常数；Kd=解离常数；EC50=半数效应浓度。",
            )
        with col3:
            max_records = st.number_input(
                "最大下载量",
                min_value=50,
                max_value=2000,
                value=300,
                step=50,
                help=(
                    "最多保留多少个分子。下载耗时取决于 ChEMBL 服务器的响应速度"
                    "（实测单次请求可能需要几十秒），300 条约几秒到一分钟，2000 条约需数分钟。"
                    "ChEMBL 响应会被客户端缓存 24 小时，重复下载同一个靶点会很快。"
                ),
            )
        submitted = st.form_submit_button("⬇️ 从 ChEMBL 下载", type="primary", width="stretch")

    if submitted:
        with st.status("正在从 ChEMBL 下载数据…", expanded=True) as status_box:
            log_placeholder = st.empty()

            def _status_callback(msg: str) -> None:
                log_placeholder.info(msg)

            outcome = tl.fetch_target_activities(
                target_name=target_input,
                activity_type=activity_type,
                max_records=int(max_records),
                status_callback=_status_callback,
            )

            if outcome["success"]:
                # 新数据到来：保存并清空下游（清洗结果 + 两个模型）
                st.session_state.tl_raw_df = outcome["df"]
                st.session_state.tl_target = outcome["target_name"]
                _clear_downstream("download")
                status_box.update(label="下载完成 ✅", state="complete", expanded=False)
            else:
                status_box.update(label="下载失败 ❌", state="error", expanded=True)
                st.error(outcome["message"])

    raw_df: Optional[pd.DataFrame] = st.session_state.get("tl_raw_df")
    if raw_df is None or raw_df.empty:
        st.info("还没有数据。点击上面的按钮，先去 ChEMBL 把真实数据拉下来。")
        return

    # ---- 下载概要 ----
    st.success(
        f"当前数据：靶点 **{st.session_state.get('tl_target')}** · "
        f"{raw_df['activity_type'].iloc[0]} · {len(raw_df)} 个分子"
        f"（ChEMBL 原始活性记录数已在上方日志中）"
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("下载分子数", f"{len(raw_df)}")
    col_b.metric("pIC50 中位数", f"{float(raw_df['pIC50'].median()):.2f}")
    col_c.metric("pIC50 范围", f"{float(raw_df['pIC50'].min()):.2f} ~ {float(raw_df['pIC50'].max()):.2f}")

    st.markdown("**原始数据前 20 行**（pIC50 由 IC50[nM] 换算：pIC50 = 9 − log10(IC50)）")
    st.dataframe(raw_df.head(20), width="stretch", height=320)

    with st.expander("🔬 示例分子：一条真实记录长什么样", expanded=True):
        sample_row = raw_df.iloc[0]
        col_img, col_info = st.columns([1, 2])
        with col_img:
            _render_molecule_image(str(sample_row["smiles"]))
        with col_info:
            st.markdown(
                f"- **ChEMBL ID**：`{sample_row['molecule_chembl_id']}`\n"
                f"- **标准活性值**：{sample_row['standard_value']:.1f} {sample_row['standard_units']}（{sample_row['activity_type']}）\n"
                f"- **换算后 pIC50**：{sample_row['pIC50']:.2f}"
                f"（pIC50 每 +1，活性强 10 倍）\n"
                f"- **SMILES**：`{sample_row['smiles']}`"
            )
            st.caption(
                "教学点：同一条活性数据在 ChEMBL 里有多种字段（standard_value 是标准化后的值）。"
                "我们统一换算成 pIC50 才能跨实验比较——这就是数据预处理的第一步。"
            )


# ============================================================================
# 步骤 2：清洗数据
# ============================================================================
def _render_step2() -> None:
    """步骤 2：清洗数据，并用 expander 逐个展示每个阶段的数据流失。"""
    st.subheader("步骤 2 · 清洗数据（看清每一步流失了多少）")
    st.caption(
        "清洗的顺序本身就是知识点：先去掉重复测量，再剔除解析失败的分子，"
        "最后用活性阈值把连续值变成二分类标签，再计算分子描述符。"
    )

    raw_df: Optional[pd.DataFrame] = st.session_state.get("tl_raw_df")
    if raw_df is None or raw_df.empty:
        st.info("请先完成步骤 1（下载数据）。")
        return

    threshold = st.slider(
        "活性阈值 (pIC50)",
        min_value=4.0,
        max_value=9.0,
        value=6.0,
        step=0.5,
        help=(
            "pIC50 ≥ 阈值的分子记为「活性」，pIC50 ≤ 阈值−1 的记为「非活性」，"
            "中间地带的样本语义模糊、直接丢弃。阈值 6.0 约等于 1 uM（1 微摩尔/升）。"
        ),
    )
    st.caption(
        f"当前阈值 **{threshold:.1f}**：≥ {threshold:.1f} → 活性；≤ {threshold - 1.0:.1f} → 非活性；"
        f"中间 {threshold - 1.0:.1f} ~ {threshold:.1f} 的样本会被丢弃。"
    )

    if st.button("🧹 开始清洗", type="primary", key="tl_clean_button"):
        progress_bar = st.progress(0.0, text="准备清洗…")

        def _progress_callback(stage_name: str, fraction: float) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=f"清洗中：{stage_name}")

        try:
            clean_df, stages = tl.clean_activity_data(
                raw_df,
                pic50_threshold=float(threshold),
                progress_callback=_progress_callback,
            )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"清洗失败：{exc}")
            return

        # 保存清洗结果，并清空下游模型（数据变了，旧模型就过期了）
        st.session_state.tl_clean_df = clean_df
        st.session_state.tl_pipeline = stages
        _clear_downstream("clean")
        progress_bar.progress(1.0, text="清洗完成 ✅")

    clean_df: Optional[pd.DataFrame] = st.session_state.get("tl_clean_df")
    stages: Optional[List[tl.DataStage]] = st.session_state.get("tl_pipeline")
    if clean_df is None or not stages:
        st.info("点击「开始清洗」，看看 300 条原始数据最后能剩下多少可用样本。")
        return

    # ---- 三个核心指标 ----
    summary = tl.summarize_labels(clean_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("总样本数", f"{summary['total']}")
    col2.metric("活性样本数", f"{summary['active']}")
    col3.metric("非活性样本数", f"{summary['inactive']}")
    if summary["active"] == 0 or summary["inactive"] == 0:
        st.warning(
            "某一类样本为 0，模型无法训练（AUC 无定义）。"
            "请把活性阈值调到两类样本都有的位置，或增大下载量。"
        )

    st.caption(
        f"从 {len(raw_df)} 条原始记录到 {summary['total']} 条可用于训练的数据，"
        f"整体保留率 {summary['total'] / max(len(raw_df), 1) * 100:.1f}%。"
    )

    # ---- 逐个阶段展开：每步的名称、描述、剩余、移除、备注 ----
    st.markdown("**清洗流水线明细**（点开每一步，看它为什么必要）")
    for index, stage in enumerate(stages, start=1):
        title = f"阶段 {index} · {stage.name} — 剩余 {stage.remaining} 条（本步移除 {stage.removed}）"
        with st.expander(title, expanded=(index == 1)):
            st.markdown(f"**这一步在做什么**：{stage.description}")
            metric_cols = st.columns(3)
            metric_cols[0].metric("保留", f"{stage.remaining}")
            metric_cols[1].metric("移除", f"{stage.removed}")
            metric_cols[2].metric("保留率", f"{stage.retention * 100:.1f}%")
            st.progress(min(max(stage.retention, 0.0), 1.0))
            if stage.note:
                st.caption(f"💡 {stage.note}")

    with st.expander("📋 清洗后数据预览（前 20 行）", expanded=False):
        st.dataframe(clean_df.head(20), width="stretch", height=320)
        st.caption(
            "label=1 表示活性，label=0 表示非活性；"
            "其后 12 列是 RDKit 分子描述符，它们就是随机森林的输入特征。"
        )


# ============================================================================
# 步骤 3：现场训练两个模型
# ============================================================================
def _render_step3() -> None:
    """步骤 3：现场训练随机森林与 GNN。"""
    st.subheader("步骤 3 · 现场训练两个小模型")
    st.caption(
        "同一个数据集，两种完全不同的分子表示：随机森林吃「12 个描述符」（手工特征），"
        "GNN 直接吃「分子图」（学习表示）。对比它们的表现，是理解表示学习最好的方式。"
    )

    clean_df: Optional[pd.DataFrame] = st.session_state.get("tl_clean_df")
    if clean_df is None or clean_df.empty:
        st.info("请先完成步骤 2（清洗数据）。")
        return

    # ---------------- 3.1 随机森林 ----------------
    st.markdown("#### 3.1 随机森林（秒级）")
    st.caption(
        "特征 = 12 个 RDKit 描述符；8:2 分层划分；n_estimators=100，max_depth=10。"
        "树模型不需要归一化，所以描述符可以「原汁原味」地喂进去。"
    )

    rf_col1, rf_col2 = st.columns([1, 3])
    with rf_col1:
        rf_clicked = st.button("🌲 训练随机森林", type="primary", key="tl_train_rf", width="stretch")
    with rf_col2:
        st.caption("训练耗时通常 < 1 秒；每次点击都会重新划分数据并重新训练。")

    if rf_clicked:
        with st.spinner("正在训练随机森林…"):
            try:
                rf_result = tl.train_rf_model(clean_df)
            except Exception as exc:
                st.error(f"训练失败：{exc}")
                rf_result = None
        if rf_result is not None:
            st.session_state.tl_rf_result = rf_result

    rf_result: Optional[tl.TrainingResult] = st.session_state.get("tl_rf_result")
    if rf_result is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("测试集 AUC", rf_result.auc_text)
        col_b.metric("测试集准确率", f"{rf_result.accuracy:.4f}")
        col_c.metric("训练 / 测试样本数", f"{rf_result.n_train} / {rf_result.n_test}")
        st.caption(
            f"特征数 {rf_result.n_features} · 训练用时 "
            f"{rf_result.extra.get('train_seconds', 0.0):.2f} 秒 · "
            f"训练集活性样本占比 {rf_result.extra.get('positive_rate', 0.0) * 100:.1f}%"
        )

        chart_col1, chart_col2 = st.columns([1, 1])
        with chart_col1:
            fig_roc = _plot_roc(rf_result)
            if fig_roc is not None:
                st.pyplot(fig_roc, width="stretch")
                plt.close(fig_roc)
        with chart_col2:
            fig_imp = _plot_feature_importance(rf_result)
            if fig_imp is not None:
                st.pyplot(fig_imp, width="stretch")
                plt.close(fig_imp)

        with st.expander("🔍 特征重要性明细（模型在「看」哪些描述符）", expanded=False):
            importance_df = pd.DataFrame(
                rf_result.extra.get("feature_importances", []),
                columns=["描述符", "重要性"],
            )
            importance_df["中文说明"] = importance_df["描述符"].map(tl.DESCRIPTOR_LABELS)
            st.dataframe(
                importance_df[["描述符", "中文说明", "重要性"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "教学点：特征重要性高 ≠ 因果相关，只说明这棵树模型「用得多」。"
                "换个数据集或换种子，排序可能就变了。"
            )

    # ---------------- 3.2 GNN ----------------
    st.markdown("#### 3.2 图神经网络 GNN（分钟级，可选）")
    st.caption(
        "13 维原子特征 + 3 层 GCN（conv1→bn1→ReLU→Dropout(0.5) → conv2→bn2→ReLU→Dropout(0.5) "
        "→ conv3 → global_mean_pool → lin1→ReLU→Dropout(0.5) → lin2）。"
        "GNN 不依赖手工描述符，直接从分子图的连接关系里学特征。"
    )

    status = tl.get_teach_lab_status()
    if not status["torch"]:
        st.warning(
            "PyTorch / PyTorch Geometric 不可用，GNN 训练已自动禁用（随机森林流程不受影响）。\n\n"
            f"原始错误：{status['torch_error']}"
        )
        return

    epochs = st.slider(
        "训练轮数 (epochs)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="tl_gnn_epochs",
        help="轮数越多模型见得越多，但 CPU 上耗时线性增长，且轮数过多反而可能过拟合。建议先跑 20 轮。",
    )
    st.caption(
        f"当前设置：{epochs} 轮 · Adam(lr=1e-3) · batch_size=32 · 8:2 分层划分。"
        "训练曲线会**逐轮实时刷新**，你可以看到 Loss 下降、Val AUC 爬升（或震荡）的过程。"
    )

    live_chart_placeholder = st.empty()
    if st.button("🕸️ 训练 GNN", type="primary", key="tl_train_gnn", width="stretch"):
        loss_history: List[float] = []
        auc_history: List[float] = []
        progress_bar = st.progress(0.0, text="正在把分子转成图…")

        def _epoch_callback(epoch: int, train_loss: float, val_auc: float) -> None:
            """每个 epoch 回调：刷新进度条与实时训练曲线。"""
            loss_history.append(float(train_loss))
            auc_history.append(float(val_auc))
            progress_bar.progress(
                min(epoch / max(epochs, 1), 1.0),
                text=f"Epoch {epoch}/{epochs} · Loss {train_loss:.4f} · Val AUC {val_auc:.4f}",
            )
            figure = _plot_gnn_history(loss_history, auc_history)
            live_chart_placeholder.pyplot(figure, width="stretch")
            plt.close(figure)

        with st.spinner("正在现场训练 GNN（CPU 上通常需要 1~3 分钟）…"):
            try:
                gnn_result = tl.train_gnn_model(
                    clean_df,
                    epochs=int(epochs),
                    epoch_callback=_epoch_callback,
                )
            except Exception as exc:
                progress_bar.empty()
                st.error(f"GNN 训练失败：{exc}")
                gnn_result = None

        if gnn_result is not None:
            st.session_state.tl_gnn_result = gnn_result
            progress_bar.progress(1.0, text="训练完成 ✅")

    gnn_result: Optional[tl.TrainingResult] = st.session_state.get("tl_gnn_result")
    if gnn_result is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("测试集 AUC", gnn_result.auc_text)
        col_b.metric("测试集准确率", f"{gnn_result.accuracy:.4f}")
        col_c.metric("训练 / 测试样本数", f"{gnn_result.n_train} / {gnn_result.n_test}")
        st.caption(
            f"原子特征 {gnn_result.n_features} 维 · {gnn_result.extra.get('epochs')} 轮 · "
            f"训练用时 {gnn_result.extra.get('train_seconds', 0.0):.1f} 秒 · "
            "本次是**从零现场训练**（没有加载项目里预训练的 gcn_egfr_best_model.pth）"
        )

        history = gnn_result.extra.get("history", {})
        loss_hist = history.get("loss", [])
        auc_hist = history.get("val_auc", [])
        if loss_hist and auc_hist:
            fig_history = _plot_gnn_history(loss_hist, auc_hist)
            st.pyplot(fig_history, width="stretch")
            plt.close(fig_history)

        fig_roc_gnn = _plot_roc(gnn_result)
        if fig_roc_gnn is not None:
            st.pyplot(fig_roc_gnn, width="stretch")
            plt.close(fig_roc_gnn)

        st.caption(
            "教学点：GNN 用了**同一批样本、同一种划分**，但表示方式不同。"
            "如果 GNN 没跑赢随机森林，往往是样本量太小——这正是小数据场景下"
            "「手工特征 + 树模型」依然常见的原因。"
        )


# ============================================================================
# 步骤 4：预测新分子
# ============================================================================
def _render_step4() -> None:
    """步骤 4：用刚才训练的两个模型预测新分子。"""
    st.subheader("步骤 4 · 用刚训练的模型预测新分子")
    st.caption(
        "输入一个 SMILES，看看两个模型分别怎么判断。"
        "注意：这两个模型是本页用几百个样本现场训练的小模型，"
        "与平台「分子预测」页的正式模型（AUC ~0.85）不是同一个——正好可以体会数据量对模型的影响。"
    )

    rf_result: Optional[tl.TrainingResult] = st.session_state.get("tl_rf_result")
    gnn_result: Optional[tl.TrainingResult] = st.session_state.get("tl_gnn_result")
    if rf_result is None and gnn_result is None:
        st.info("请先在步骤 3 训练至少一个模型（随机森林即可）。")
        return

    smiles = st.text_input(
        "输入 SMILES",
        value=tl.EXAMPLE_SMILES,
        key="tl_predict_smiles",
        help="默认值是吉非替尼（Gefitinib）——一个真实的 EGFR 抑制剂，可以先用它试试。",
    )
    if not smiles or not smiles.strip():
        st.info("请输入一个 SMILES 字符串。")
        return

    smiles = smiles.strip()
    col_structure, col_result = st.columns([1, 2])

    with col_structure:
        with st.expander("🧪 输入分子结构", expanded=True):
            _render_molecule_image(smiles, size=(300, 230))

    with col_result:
        rf_prediction: Optional[Dict[str, Any]] = None
        if rf_result is not None:
            rf_prediction = tl.predict_smiles_rf(rf_result, smiles)
            st.markdown("**随机森林 (RF)**")
            if rf_prediction["success"]:
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("活性概率", f"{rf_prediction['probability_active'] * 100:.1f}%")
                col_p2.metric("判断", rf_prediction["label"])
                st.caption("RF 只看 12 个描述符——它「看不到」原子的连接方式。")
            else:
                st.warning(f"RF 预测不可用：{rf_prediction['error']}")

        if gnn_result is not None:
            gnn_prediction = tl.predict_smiles_gnn(gnn_result, smiles)
            st.markdown("**图神经网络 (GCN)**")
            if gnn_prediction["success"]:
                col_g1, col_g2 = st.columns(2)
                col_g1.metric("活性概率", f"{gnn_prediction['probability_active'] * 100:.1f}%")
                col_g2.metric("判断", gnn_prediction["label"])
                st.caption(
                    f"GCN 把分子看成图：{gnn_prediction['num_atoms']} 个原子节点、"
                    f"{gnn_prediction['num_bonds']} 条键边。"
                )
            else:
                st.warning(f"GNN 预测不可用：{gnn_prediction['error']}")

    if rf_prediction is not None and rf_prediction["success"]:
        with st.expander("🔍 这个分子的 12 个描述符（RF 的全部输入）", expanded=False):
            descriptor_df = pd.DataFrame(
                [
                    {
                        "描述符": name,
                        "中文说明": tl.DESCRIPTOR_LABELS.get(name, name),
                        "数值": value,
                    }
                    for name, value in rf_prediction["descriptors"].items()
                ]
            )
            st.dataframe(descriptor_df, width="stretch", hide_index=True)

    st.caption(
        "教学点：两个模型给出不同概率是正常的——它们「看到」的信息本就不同。"
        "真实项目里我们会用多个模型互相校验，而不是盲信单一模型的分值。"
    )


# ============================================================================
# 页面入口
# ============================================================================
def page_teach_lab() -> None:
    """🎓 教学实验室页面入口（在 app.py 的 st.navigation 中注册）。"""
    _init_state()

    st.header("🎓 教学实验室：亲手走一遍药物发现的四步闭环")
    st.caption(
        "不给你答案，给你流程。四步：**下载真实数据 → 清洗（看清流失）→ 现场训练两个模型 → 预测新分子**。"
        "每一步都在这个页面上发生，你可以随时改参数、重跑、对比。"
    )

    status = tl.get_teach_lab_status()
    status_cols = st.columns(3)
    status_cols[0].metric("RDKit（分子解析）", "✅ 可用" if status["rdkit"] else "❌ 不可用")
    status_cols[1].metric("ChEMBL（数据下载）", "✅ 可用" if status["chembl"] else "❌ 不可用")
    status_cols[2].metric("PyTorch + PyG（GNN）", "✅ 可用" if status["torch"] else "❌ 不可用")
    if not (status["rdkit"] and status["chembl"]):
        st.error(
            "核心依赖缺失，教学实验室无法完整运行：\n"
            f"- RDKit: {status['rdkit_error']}\n"
            f"- ChEMBL: {status['chembl_error']}"
        )

    with st.popover("🎓 教学点：这个页面在教什么"):
        st.markdown(
            """
            **核心问题：AI 模型的结果是怎么来的？**

            1. **数据决定上限**：活性数据来自真实实验，有噪声、有重复、有模糊地带。
               清洗时你会看到 300 条数据最后可能只剩下三分之一。
            2. **标签是人为定义的**：pIC50 ≥ 6 算活性、≤ 5 算非活性，中间扔掉——
               换一个阈值，模型和结论都会变。
            3. **特征是人的假设**：随机森林用 12 个描述符，每一个都是化学家定义的量；
               GNN 不用定义，直接从分子图里学。
            4. **评估要看 AUC 而不是准确率**：类别不平衡时，全部猜「非活性」也能有高准确率。
            5. **训练曲线会说话**：Loss 不降 = 学不动；Val AUC 下降 = 过拟合苗头。

            这个页面的目的不是得到最好的模型，而是让你**看清每一步做了什么选择**。
            """
        )

    _render_step1()
    st.divider()
    _render_step2()
    st.divider()
    _render_step3()
    st.divider()
    _render_step4()
    st.divider()

    with st.expander("♻️ 重置教学实验室（清空本页所有结果）", expanded=False):
        st.caption("只清空本页的下载/清洗/训练结果，不影响平台其它模块和正式模型。")
        if st.button("确认清空", key="tl_reset_button"):
            for key, default in _STATE_DEFAULTS.items():
                st.session_state[key] = default
            st.success("已重置教学实验室状态。")
