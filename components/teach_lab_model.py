# components/teach_lab_model.py
"""
教学实验室「小模型」复用组件
============================

🎓 教学实验室里现场训练出来的 RF / GNN 是**会话级模型**（保存在 `st.session_state`），
本组件把它们包装成可复用的「打分器」，供其他页面直接调用：

    * 🧩 分子聚类：给簇代表分子活性打分，看"结构上聚成一类的分子，活性是否也相似"
    * 🧬 分子生成：给生成出来的候选分子打分 / 筛选（这正是 REINVENT 里 scorer 的角色）

设计约定（与教学实验室一致）：
    * 模型只从 `st.session_state["tl_rf_result"] / ["tl_gnn_result"]` 取，不落盘、不跨会话
    * RF 用 12 个 RDKit 描述符打分；GNN 用 13 维原子特征建图打分
    * 打分结果的活性含义 = **该模型训练时那个靶点**（模型是在教学实验室里用哪个靶点数据
      训出来的，就代表哪个靶点），页面上必须把这句话写清楚，避免误用

作者：dadamingli
"""

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from utils import teach_lab as tl

#: 教学实验室在 session_state 中存放模型结果的两个键
_RF_STATE_KEY = "tl_rf_result"
_GNN_STATE_KEY = "tl_gnn_result"


def teach_lab_model_status() -> Dict[str, Any]:
    """读取当前会话里教学实验室训练出的模型。

    Returns:
        ``{"rf": TrainingResult|None, "gnn": TrainingResult|None,
           "has_rf": bool, "has_gnn": bool, "target": str}``
    """
    rf_result: Optional[tl.TrainingResult] = st.session_state.get(_RF_STATE_KEY)
    gnn_result: Optional[tl.TrainingResult] = st.session_state.get(_GNN_STATE_KEY)
    return {
        "rf": rf_result,
        "gnn": gnn_result,
        "has_rf": rf_result is not None,
        "has_gnn": gnn_result is not None,
        "target": str(st.session_state.get("tl_target") or "未知靶点"),
    }


def score_smiles(
    smiles_list: Sequence[str],
    use_rf: bool = True,
    use_gnn: bool = True,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """用教学实验室训练的小模型给一批 SMILES 打分。

    Args:
        smiles_list: 待打分的 SMILES 列表。
        use_rf: 是否使用随机森林模型（会话里没有该模型时自动跳过）。
        use_gnn: 是否使用 GNN 模型（会话里没有该模型时自动跳过）。
        progress_callback: ``(已完成, 总数)`` 回调。

    Returns:
        DataFrame，列随可用模型而定：
            ``smiles`` / ``rf_probability`` / ``rf_label`` / ``gnn_probability`` /
            ``gnn_label`` / ``consensus_score``（可用模型的平均概率）
        不可解析或模型不可用时对应单元格为 ``NaN``。
    """
    status = teach_lab_model_status()
    frame = pd.DataFrame({"smiles": [str(s) for s in smiles_list]})

    if use_rf and status["has_rf"]:
        rf_df = tl.predict_dataframe_rf(status["rf"], frame["smiles"].tolist(), progress_callback)
        frame["rf_probability"] = rf_df["rf_probability"]
        frame["rf_label"] = rf_df["rf_label"]

    if use_gnn and status["has_gnn"]:
        gnn_df = tl.predict_dataframe_gnn(status["gnn"], frame["smiles"].tolist(), progress_callback)
        frame["gnn_probability"] = gnn_df["gnn_probability"]
        frame["gnn_label"] = gnn_df["gnn_label"]

    probability_columns = [c for c in ("rf_probability", "gnn_probability") if c in frame.columns]
    if probability_columns:
        frame["consensus_score"] = frame[probability_columns].mean(axis=1, skipna=True)
    return frame


def render_teach_lab_scoring(
    smiles_list: Sequence[str],
    *,
    title: str,
    key_prefix: str,
    threshold_default: float = 0.5,
    caption: Optional[str] = None,
    download_name: str = "teach_lab_scores.csv",
) -> Optional[pd.DataFrame]:
    """渲染一个「用小模型打分」区块，返回打分后的 DataFrame（未打分时返回 None）。

    Args:
        smiles_list: 待打分分子。
        title: 区块标题。
        key_prefix: 本区块所有 widget/缓存键的前缀（同页多处使用时必须不同）。
        threshold_default: 「活性概率阈值」滑块的默认值。
        caption: 标题下的说明文字。
        download_name: 导出 CSV 的文件名。
    """
    st.subheader(title)
    if caption:
        st.caption(caption)

    status = teach_lab_model_status()
    if not status["has_rf"] and not status["has_gnn"]:
        st.info(
            "这里还没有可用的小模型。请先到 **🎓 教学实验室** 走完 "
            "步骤 2（清洗）与步骤 3（现场训练），训好的 RF / GNN 会自动出现在这里。"
        )
        return None

    model_badges = []
    if status["has_rf"]:
        model_badges.append(f"随机森林（AUC {status['rf'].auc_text}）")
    if status["has_gnn"]:
        model_badges.append(f"GNN（AUC {status['gnn'].auc_text}）")
    st.caption(
        f"可用小模型：{' · '.join(model_badges)} ——都是在教学实验室用 **{status['target']}** "
        "数据现场训练的，因此这里的「活性」指的是**该靶点**的活性。"
    )

    col_rf, col_gnn, col_thresh = st.columns([1, 1, 2])
    with col_rf:
        use_rf = st.checkbox(
            "用随机森林打分",
            value=status["has_rf"],
            disabled=not status["has_rf"],
            key=f"{key_prefix}_use_rf",
            help="RF 只看 12 个 RDKit 描述符（分子量/LogP/氢键/环数…），速度快。",
        )
    with col_gnn:
        use_gnn = st.checkbox(
            "用 GNN 打分",
            value=status["has_gnn"],
            disabled=not status["has_gnn"],
            key=f"{key_prefix}_use_gnn",
            help="GNN 把分子当图看（13 维原子特征 + 键连接），与 RF 的视角完全不同。",
        )
    with col_thresh:
        threshold = st.slider(
            "活性概率阈值（用于筛选）",
            min_value=0.0,
            max_value=1.0,
            value=float(threshold_default),
            step=0.05,
            key=f"{key_prefix}_threshold",
            help="平均活性概率 ≥ 该阈值才视为「候选活性分子」，用来筛选表格和导出结果。",
        )

    if not use_rf and not use_gnn:
        st.warning("请至少选择一个模型再打分。")
        return None

    if st.button("🔎 用小模型打分", type="primary", key=f"{key_prefix}_run"):
        progress_bar = st.progress(0.0, text="正在打分…")

        def _progress(done: int, total: int) -> None:
            progress_bar.progress(min(done / max(total, 1), 1.0), text=f"正在打分… {done}/{total}")

        try:
            scores = score_smiles(smiles_list, use_rf=use_rf, use_gnn=use_gnn, progress_callback=_progress)
        except Exception as exc:
            progress_bar.empty()
            st.error(f"打分失败：{exc}")
            return None

        progress_bar.progress(1.0, text="打分完成 ✅")
        st.session_state[f"{key_prefix}_scores"] = scores

    scores: Optional[pd.DataFrame] = st.session_state.get(f"{key_prefix}_scores")
    if scores is None or scores.empty:
        st.caption("点上面的按钮开始打分。")
        return None

    # 只统计"确实被打分"的行：注意不能用 scores[["rf_probability", "gnn_probability"]] 直接索引，
    # 只启用其中一个模型时另一列根本不存在，会抛 KeyError（教训：先取交集再索引）
    probability_columns = [c for c in ("rf_probability", "gnn_probability") if c in scores.columns]
    total_scored = (
        int(scores[probability_columns].notna().any(axis=1).sum()) if probability_columns else 0
    )
    hits = scores[scores["consensus_score"] >= threshold] if "consensus_score" in scores.columns else scores.iloc[0:0]

    metric_cols = st.columns(3)
    metric_cols[0].metric("待打分分子", f"{len(scores)}")
    metric_cols[1].metric("成功打分", f"{int(total_scored)}")
    metric_cols[2].metric(f"平均概率 ≥ {threshold:.2f}", f"{len(hits)}")

    display_columns = [c for c in (
        "smiles", "rf_probability", "rf_label", "gnn_probability", "gnn_label", "consensus_score"
    ) if c in scores.columns]
    sorted_scores = scores.sort_values("consensus_score", ascending=False) if "consensus_score" in scores.columns else scores
    st.dataframe(
        sorted_scores[display_columns].round(3),
        width="stretch",
        height=280,
        hide_index=True,
    )

    if len(hits) == 0:
        st.warning(
            f"没有分子的平均活性概率达到 {threshold:.2f}。"
            "教学点：生成/聚类出来的分子大多不含该靶点所需的药效特征，"
            "这正是「生成很多、能用的很少」的真实写照；可以把阈值调低一点再看排序。"
        )

    csv_data = sorted_scores[display_columns].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 下载打分结果 (CSV)",
        data=csv_data,
        file_name=download_name,
        mime="text/csv",
        key=f"{key_prefix}_download",
    )
    return sorted_scores


def render_teach_lab_model_hint(context: str = "") -> None:
    """在页面醒目处提示「小模型可以在哪些地方复用」。

    Args:
        context: 当前页面的名字，用于组织文案。
    """
    status = teach_lab_model_status()
    if not status["has_rf"] and not status["has_gnn"]:
        return
    st.caption(
        f"💡 检测到本会话已训练过小模型（{status['target']}）：{context}"
    )


def build_smiles_list_from_summary(summary: Any, molecules: List[Any]) -> List[str]:
    """从聚类结果里取出「簇代表分子」的 SMILES（供打分使用）。

    Args:
        summary: ``utils.cluster_engine.ClusteringSummary``。
        molecules: 与聚类输入顺序一致的 RDKit Mol 列表。

    Returns:
        代表分子 SMILES 列表（保持簇顺序，索引非法时跳过）。
    """
    from rdkit import Chem

    smiles: List[str] = []
    for result in getattr(summary, "results", []):
        index = getattr(result, "centroid_index", None)
        if index is None or not (0 <= index < len(molecules)):
            continue
        try:
            smiles.append(Chem.MolToSmiles(molecules[index]))
        except Exception:
            continue
    return smiles
