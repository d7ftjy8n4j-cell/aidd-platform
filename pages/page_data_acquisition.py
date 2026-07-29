"""
数据获取页面 - 从 ChEMBL / PubChem 获取化合物数据
提供三种数据获取方式：按靶点检索、相似性搜索、文件上传
获取的数据可一键送入自动化分析流程。
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from utils.data_fetcher import DataFetcher, CompoundRecord, FetchResult


def show_data_acquisition():
    """数据获取主页面"""
    st.title("📦 数据获取中心")
    st.caption(
        "从公开数据库中获取化合物数据，一键送入自动化分析流程"
    )

    # ---- 初始化 fetcher ----
    if "fetcher" not in st.session_state:
        st.session_state.fetcher = DataFetcher()

    # ---- 模式选择 ----
    mode = st.radio(
        "选择数据获取方式",
        [
            "🔬 按靶点/疾病检索 (ChEMBL)",
            "🧪 按结构相似性检索 (PubChem)",
            "📄 上传已有文件",
        ],
        horizontal=True,
    )

    fetch_result: FetchResult = None

    # ===================== 模式 1: ChEMBL =====================
    if mode == "🔬 按靶点/疾病检索 (ChEMBL)":
        _render_chembl_mode()
        # ChEMBL 模式将结果存入 session_state 后 rerun，
        # 从 session 中恢复结果以供展示
        fetch_result = st.session_state.pop("fetch_result", None)

    # ===================== 模式 2: PubChem =====================
    elif mode == "🧪 按结构相似性检索 (PubChem)":
        fetch_result = _render_pubchem_mode()

    # ===================== 模式 3: 文件上传 =====================
    else:
        _render_upload_mode()

    # ===================== 结果展示与联动 =====================
    if fetch_result is not None:
        _render_fetch_result(fetch_result)

    # ---- 页脚 ----
    st.divider()
    st.caption("""
    **📌 数据来源说明**
    - ChEMBL: 欧洲生物信息学研究所 (EMBL-EBI) 公开药物发现数据库
    - PubChem: 美国国家医学图书馆 (NCBI) 化合物数据库
    - 数据仅供学术研究使用，请遵守各数据库的使用条款
    """)


# ===================== 渲染函数 =====================

def _render_chembl_mode():
    """渲染 ChEMBL 检索表单"""
    st.markdown("""
    **从 ChEMBL 获取指定靶点的化合物活性数据**
    - 输入靶点名称（如 "EGFR", "BRCA1", "HER2"）
    - 可选活性类型和最小 pIC50 阈值
    """)

    col1, col2 = st.columns(2)
    with col1:
        target_name = st.text_input("靶点名称", value="EGFR")
        activity_type = st.selectbox(
            "活性类型", ["IC50", "EC50", "Ki", "Kd"], index=0
        )
    with col2:
        min_pic50 = st.number_input(
            "最小 pIC50 (可选, 0=不过滤)",
            min_value=0.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
        )
        max_compounds = st.slider(
            "最大获取数量", min_value=5, max_value=200, value=50, step=5
        )

    if st.button("🔍 开始检索", type="primary", key="chembl_search"):
        with st.spinner(f"正在从 ChEMBL 检索 '{target_name}'..."):
            res = st.session_state.fetcher.fetch_by_target(
                target_name=target_name,
                activity_type=activity_type,
                max_compounds=max_compounds,
                min_pic50=min_pic50 if min_pic50 > 0 else None,
            )
            st.session_state.fetch_result = res
            st.rerun()


def _render_pubchem_mode() -> FetchResult:
    """渲染 PubChem 相似性搜索表单，返回 fetch_result 或 None"""
    st.markdown("""
    **在 PubChem 中查找与输入分子结构相似的化合物**
    - 输入查询分子的 SMILES
    - 设置相似度阈值和最大返回数量
    """)

    query_smiles = st.text_area(
        "输入查询分子的 SMILES",
        value=st.session_state.get("last_smiles", ""),
        height=80,
    )
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "相似度阈值 (%)", min_value=70, max_value=100, value=85, step=5
        )
    with col2:
        max_records = st.slider(
            "最大返回数量", min_value=5, max_value=50, value=20, step=5
        )

    if st.button("🔍 开始检索", type="primary", key="pubchem_search"):
        if query_smiles.strip():
            with st.spinner("正在从 PubChem 检索相似化合物..."):
                res = st.session_state.fetcher.fetch_similar_by_smiles(
                    smiles=query_smiles.strip(),
                    threshold=threshold,
                    max_records=max_records,
                )
                st.session_state.fetch_result = res
                st.rerun()
        else:
            st.warning("请输入有效的 SMILES")

    # 如果 session 中有上次的结果，渲染它
    return st.session_state.pop("fetch_result", None)


def _render_upload_mode():
    """渲染文件上传模式"""
    st.markdown("""
    **上传本地化合物数据文件**
    - 支持 CSV、Excel 格式（需包含 SMILES 列）
    """)

    uploaded = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            st.error("不支持的文件格式")
            return

        st.write(f"✅ 成功读取 {len(df)} 条记录")

        # 检测 SMILES 列
        possible_cols = [
            "smiles", "SMILES", "Smiles",
            "canonical_smiles", "structure",
        ]
        smiles_col = next(
            (c for c in possible_cols if c in df.columns), None
        )

        if smiles_col:
            st.success(f"检测到 SMILES 列: '{smiles_col}'")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 导入分析", type="primary"):
                smiles_list = df[smiles_col].dropna().tolist()
                st.session_state.batch_smiles_list = smiles_list
                st.session_state.batch_data_source = "upload"
                st.session_state.pipeline_input_mode = "📦 已导入数据"
                st.session_state.clustering_input_option = "📂 从数据获取模块导入"
                st.toast(f"✅ 已导入 {len(smiles_list)} 个分子")
                st.info("💡 请前往「⚙️ 自动化流程」页面进行分析")
        else:
            st.error(
                f"未找到 SMILES 列，请确保包含以下列之一: "
                f"{', '.join(possible_cols)}"
            )
    except Exception as e:
        st.error(f"文件解析失败: {e}")


def _render_fetch_result(fetch_result: FetchResult):
    """渲染数据获取结果"""
    st.divider()
    st.subheader(f"📊 检索结果 ({fetch_result.source})")

    if not fetch_result.success:
        st.error(f"❌ 检索失败: {fetch_result.error}")
        return

    # 概要指标
    col1, col2, col3 = st.columns(3)
    with col1:
        query_display = (
            fetch_result.query[:30] + "..."
            if len(fetch_result.query) > 30
            else fetch_result.query
        )
        st.metric("查询", query_display)
    with col2:
        st.metric("获取化合物数", fetch_result.total_count)
    with col3:
        st.metric("耗时", f"{fetch_result.fetch_time:.1f}s")

    if not fetch_result.compounds:
        st.warning("没有返回化合物")
        return

    # 构建 DataFrame 展示
    data = []
    for comp in fetch_result.compounds:
        data.append({
            "SMILES": (
                comp.smiles[:80] + "..."
                if len(comp.smiles) > 80
                else comp.smiles
            ),
            "ChEMBL ID": comp.chembl_id or "-",
            "PubChem CID": comp.pubchem_cid or "-",
            "活性值": (
                f"{comp.activity_value:.2f}"
                if comp.activity_value is not None
                else "-"
            ),
            "来源": comp.source,
        })
    df_results = pd.DataFrame(data)
    st.dataframe(df_results, use_container_width=True)

    # 下载按钮
    csv_data = df_results.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 下载结果 CSV",
        data=csv_data,
        file_name=(
            f"data_{fetch_result.source}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        mime="text/csv",
    )

    # 一键送入 Pipeline
    if st.button("🚀 全部送入自动化分析", type="primary", key="send_to_pipeline"):
        smiles_list = [
            comp.smiles
            for comp in fetch_result.compounds
            if comp.smiles
        ]
        st.session_state.batch_smiles_list = smiles_list
        st.session_state.batch_data_source = fetch_result.source
        # 清空旧的流程结果缓存，避免与新数据混淆
        st.session_state.pop("pipeline_results", None)
        st.session_state.pop("pipeline_smiles_list", None)
        st.toast(f"✅ 已发送 {len(smiles_list)} 个分子到自动化流程")
        st.success("💡 请点击左侧导航栏「⚙️ 自动化流程」查看")
        st.rerun()

    # 分子结构预览 (前 5 个)
    with st.expander("🔬 分子结构预览 (前 5 个)"):
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mols = []
            for comp in fetch_result.compounds[:5]:
                mol = Chem.MolFromSmiles(comp.smiles)
                if mol:
                    mols.append(mol)
            if mols:
                img = Draw.MolsToGridImage(
                    mols, molsPerRow=min(len(mols), 5),
                    subImgSize=(200, 150),
                )
                st.image(img)
            else:
                st.caption("无法绘制分子结构")
        except ImportError:
            st.caption("RDKit 未安装，结构预览不可用")
        except Exception as e:
            st.caption(f"结构预览不可用: {e}")


# ---- 页面入口 ----
if __name__ == "__main__":
    show_data_acquisition()
