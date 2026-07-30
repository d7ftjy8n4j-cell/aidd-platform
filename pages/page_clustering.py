"""
分子聚类页面 - 基于 Butina 算法

对化合物库进行结构聚类，探索化学空间多样性，
支持降维可视化、簇浏览和代表性分子筛选。
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import rdBase

import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from utils.cluster_engine import ClusterEngine, ClusteringSummary
from components.knime_export import knime_export_section


def show_clustering_page():
    """分子聚类主页面"""
    st.title("🧩 分子聚类分析")
    st.caption(
        "基于 Butina 算法对分子进行结构聚类，探索化学空间多样性"
    )

    # ---- 初始化引擎 ----
    if "cluster_engine" not in st.session_state:
        st.session_state.cluster_engine = ClusterEngine()

    # ===================== 侧边栏：数据输入 =====================
    st.sidebar.header("📥 输入数据")

    # 自动检测是否有来自数据获取页面的数据
    has_batch_data = (
        "batch_smiles_list" in st.session_state
        and st.session_state.batch_smiles_list
    )
    batch_source = st.session_state.get("batch_data_source", "数据获取")

    # 有数据时始终切换为"从数据获取模块导入"（覆盖用户之前可能选错的状态）
    if has_batch_data:
        st.session_state["clustering_input_option"] = "📂 从数据获取模块导入"

    input_option = st.sidebar.radio(
        "选择输入方式",
        [
            "📂 从数据获取模块导入",
            "📄 上传 CSV 文件",
            "✏️ 手动输入 SMILES 列表",
        ],
        key="clustering_input_option"
    )

    if has_batch_data and input_option != "📂 从数据获取模块导入":
        st.sidebar.info(
            "检测到当前会话中已有获取的化合物数据，请切换到“📂 从数据获取模块导入”开始聚类。",
            icon="ℹ️"
        )
        if st.sidebar.button(
            "使用已获取数据",
            key="clustering_switch_to_imported_data"
        ):
            st.session_state.clustering_input_option = "📂 从数据获取模块导入"
            st.experimental_rerun()

    molecules: list = []
    mol_ids: list = []

    if input_option == "📂 从数据获取模块导入":
        if has_batch_data:
            smiles_list = st.session_state.batch_smiles_list
            st.sidebar.success(
                f"✅ 已从「{batch_source}」导入 {len(smiles_list)} 个分子"
            )
            if st.sidebar.button("🔄 刷新数据", key="clustering_refresh"):
                st.rerun()
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    molecules.append(mol)
                    mol_ids.append(f"mol_{i} ({smi[:20]}...)")
        else:
            st.sidebar.warning(
                "⚠️ 请先在「📦 数据获取」页面获取化合物数据\n\n"
                "支持的操作：\n"
                "- 按靶点名称从 ChEMBL 检索\n"
                "- 按分子结构相似性搜索\n"
                "- 手动输入 SMILES 列表"
            )

    elif input_option == "📄 上传 CSV 文件":
        uploaded = st.sidebar.file_uploader(
            "选择 CSV 文件", type=["csv"]
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                # 自动识别 SMILES 列
                possible_cols = [
                    "smiles", "SMILES", "Smiles", "canonical_smiles"
                ]
                smiles_col = next(
                    (c for c in possible_cols if c in df.columns), None
                )
                if smiles_col:
                    for _, row in df.iterrows():
                        smi = str(row[smiles_col])
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            molecules.append(mol)
                            mol_ids.append(smi[:25])
                    st.sidebar.success(f"成功加载 {len(molecules)} 个分子")
                else:
                    st.sidebar.error(
                        "CSV 文件缺少 SMILES 列，"
                        f"当前列: {', '.join(df.columns.tolist())}"
                    )
            except Exception as e:
                st.sidebar.error(f"文件解析失败: {e}")

    else:  # 手动输入
        smi_text = st.sidebar.text_area(
            "每行一个 SMILES", height=200
        )
        if smi_text.strip():
            for i, line in enumerate(smi_text.splitlines()):
                smi = line.strip()
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        molecules.append(mol)
                        mol_ids.append(f"mol_{i}: {smi[:25]}")
                    else:
                        st.sidebar.warning(f"无效 SMILES (行 {i+1}): {smi[:30]}")

    if not molecules:
        st.info("👈 请在左侧边栏输入分子数据以开始聚类分析")
        return

    st.sidebar.markdown(f"**已加载分子数**: `{len(molecules)}`")

    # ===================== 侧边栏：聚类参数 =====================
    st.sidebar.header("⚙️ 聚类参数")
    cutoff = st.sidebar.slider(
        "距离阈值 (cutoff)",
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="簇内距离需 < cutoff，越小簇越紧密（但簇数越多）"
    )
    fingerprint_type = st.sidebar.selectbox(
        "指纹类型",
        ["morgan", "rdkit"],
        help="Morgan: 圆形指纹（推荐）; RDKit: 拓扑路径指纹"
    )
    radius = (
        st.sidebar.slider(
            "Morgan 半径", min_value=1, max_value=6, value=2
        )
        if fingerprint_type == "morgan"
        else 2
    )
    n_bits = st.sidebar.selectbox(
        "指纹长度", [1024, 2048, 4096], index=1
    )

    if st.sidebar.button("🚀 运行聚类", type="primary"):
        with st.spinner("正在计算指纹和距离矩阵..."):
            engine = ClusterEngine(
                fingerprint_type=fingerprint_type,
                radius=radius,
                n_bits=n_bits,
            )
            summary = engine.cluster(
                molecules, cutoff=cutoff, ids=mol_ids
            )
            st.session_state["cluster_summary"] = summary
            st.session_state["cluster_molecules"] = molecules
            st.session_state["cluster_mol_ids"] = mol_ids
            st.session_state["cluster_engine"] = engine
            st.toast(
                f"聚类完成: {summary.n_clusters} 个簇, "
                f"{summary.n_singletons} 个单例"
            )
            st.rerun()

    # ===================== 结果展示 =====================
    if "cluster_summary" not in st.session_state:
        return

    summary: ClusteringSummary = st.session_state["cluster_summary"]
    molecules = st.session_state["cluster_molecules"]
    mol_ids = st.session_state["cluster_mol_ids"]
    engine = st.session_state["cluster_engine"]

    st.header("📊 聚类结果")

    # ---- 概要指标 ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总分子数", len(molecules))
    col2.metric("簇数量", summary.n_clusters)
    col3.metric("单例数", summary.n_singletons)
    col4.metric("最大簇", summary.largest_cluster_size)

    # ---- 簇大小分布 ----
    st.subheader("📈 簇大小分布")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(summary.cluster_sizes))
    )
    ax.bar(
        range(1, len(summary.cluster_sizes) + 1),
        summary.cluster_sizes,
        color=colors,
    )
    ax.set_xlabel("簇编号（按大小排序）")
    ax.set_ylabel("分子数量")
    ax.set_title("簇大小分布")
    # 标注前几个簇
    for i, size in enumerate(summary.cluster_sizes[:5]):
        ax.text(i + 1, size + 0.3, str(size), ha="center", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    # ---- 降维可视化 (UMAP) ----
    st.subheader("🗺️ 化学空间可视化")
    try:
        import umap

        fps = [
            engine._fingerprint_generator.GetFingerprint(m)
            for m in molecules
        ]
        fp_array = np.array([list(fp) for fp in fps])

        n_neighbors = min(15, len(fp_array) - 1)
        reducer = umap.UMAP(
            n_neighbors=max(2, n_neighbors),
            min_dist=0.1,
            random_state=42,
        )
        embedding = reducer.fit_transform(fp_array)

        # 着色：按簇 ID
        cluster_labels = np.full(len(molecules), -1, dtype=int)
        for r in summary.results:
            for idx in r.member_indices:
                cluster_labels[idx] = r.cluster_id

        fig, ax = plt.subplots(figsize=(8, 6))
        n_clusters = summary.n_clusters
        cmap = plt.cm.tab20 if n_clusters <= 20 else plt.cm.gist_rainbow
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=cluster_labels,
            cmap=cmap,
            s=12,
            alpha=0.7,
        )
        ax.set_title("UMAP 降维投影 (按簇着色)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        st.pyplot(fig)
        plt.close(fig)

    except ImportError:
        st.info("💡 安装 `umap-learn` 可启用化学空间降维可视化")
    except Exception as e:
        st.warning(f"降维可视化生成失败: {e}")

    # ---- 查看单个簇 ----
    st.subheader("🔍 浏览簇")
    cluster_options = [
        f"Cluster {r.cluster_id} (大小: {r.size})"
        for r in summary.results
    ]
    selected_label = st.selectbox(
        "选择一个簇查看详情", cluster_options
    )
    if selected_label:
        cluster_id = int(selected_label.split()[1])
        selected = summary.results[cluster_id]
        st.write(
            f"**簇 {cluster_id}**: {selected.size} 个分子, "
            f"中心分子索引 {selected.centroid_index}"
        )

        # 绘制代表分子
        if selected.representative_mols:
            n_show = min(10, len(selected.representative_mols))

            # 图例：前几个分子的 ID
            legend_ids = [
                mol_ids[i] if i < len(mol_ids) else f"mol_{i}"
                for i in selected.member_indices[:n_show]
            ]
            img = Draw.MolsToGridImage(
                selected.representative_mols[:n_show],
                molsPerRow=min(5, n_show),
                subImgSize=(220, 180),
                legends=legend_ids,
            )
            st.image(img)

            if selected.size > 1:
                avg_sim = (
                    np.mean(selected.intra_similarities)
                    if selected.intra_similarities
                    else 1.0
                )
                st.caption(
                    f"簇内平均相似度 (vs 中心): {avg_sim:.3f}"
                )

    # ---- 导出 ----
    st.subheader("💾 导出与联动")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 导出簇中心 SMILES"):
            center_indices = [r.centroid_index for r in summary.results]
            center_smiles = [
                Chem.MolToSmiles(molecules[i]) for i in center_indices
            ]
            df_export = pd.DataFrame({
                "cluster_id": [r.cluster_id for r in summary.results],
                "cluster_size": [r.size for r in summary.results],
                "smiles": center_smiles,
            })
            csv_data = df_export.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "下载 CSV",
                data=csv_data,
                file_name="cluster_centers.csv",
                mime="text/csv",
            )
    with col2:
        already_sent_to_pipeline = (
            st.session_state.get("batch_data_source") == "clustering"
            and st.session_state.get("batch_smiles_list")
        )
        if already_sent_to_pipeline:
            st.success(
                f"✅ 已发送 {len(st.session_state.batch_smiles_list)} 个代表分子 — "
                "请切换到「⚙️ 自动化流程」页面"
            )
        else:
            if st.button("🚀 代表分子送入分析", type="primary"):
                idxs, _ = engine.get_representative_subset(
                    summary, molecules, max_compounds=1000
                )
                smiles_subset = [
                    Chem.MolToSmiles(molecules[i]) for i in idxs
                ]
                st.session_state.batch_smiles_list = smiles_subset
                st.session_state.batch_data_source = "clustering"
                st.session_state.pipeline_input_mode = "📦 已导入数据"
                st.session_state.pop("pipeline_results", None)
                st.session_state.pop("pipeline_smiles_list", None)
                st.rerun()

    # KNIME 导出
    try:
        export_rows = []
        for r in summary.results:
            row = {"cluster_id": r.cluster_id, "size": r.size}
            if hasattr(r, "centroid_smiles"):
                row["smiles"] = r.centroid_smiles
            export_rows.append(row)
        if export_rows:
            knime_export_section(
                pd.DataFrame(export_rows),
                title="聚类分析结果",
                key_prefix="cluster_knime",
            )
    except Exception:
        pass


if __name__ == "__main__":
    show_clustering_page()
