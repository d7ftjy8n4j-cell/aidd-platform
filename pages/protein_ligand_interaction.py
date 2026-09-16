"""
蛋白-配体相互作用分析页面 (PLIP)
"""
import streamlit as st
from interaction_utils import analyze_plip


def page_protein_ligand_interaction():
    """蛋白-配体相互作用分析主函数"""
    st.title("🔬 蛋白-配体相互作用分析 (PLIP)")

    # ---- MD 模拟导出衔接 ----
    md_pdb_content = st.session_state.get("md_ready_pdb_content")
    md_source = st.session_state.get("md_ready_source", "")
    if md_pdb_content:
        st.success(f"🔗 **已从「{md_source}」自动加载结构** — 展示 MD 模拟代表性构象的相互作用")
        if st.button("❌ 清除 MD 数据，切换手动输入"):
            st.session_state.pop("md_ready_pdb_content", None)
            st.session_state.pop("md_ready_source", None)
            st.rerun()

    st.markdown("""
    **功能说明**：输入 PDB ID 或上传 PDB 文件，自动分析蛋白与配体之间的：
    氢键、疏水作用、盐桥、π-π堆积、卤键 等非共价相互作用，并通过 3D 结构可视化高亮显示。
    """)

    # ---------- 输入区域 ----------
    col1, col2 = st.columns(2)

    with col1:
        input_type = st.radio("选择输入方式", ["PDB ID", "上传PDB文件"], index=0, horizontal=True)

    pdb_id = None
    pdb_content = None
    analyze_btn = False

    # 如果 MD 导出了结构，自动使用它
    if md_pdb_content:
        pdb_content = md_pdb_content if isinstance(md_pdb_content, bytes) else md_pdb_content.encode("utf-8")
        st.info(f"📐 当前使用 MD 模拟代表性构象 (来源: {md_source})")
        analyze_btn = st.button("🚀 分析 MD 构象", type="primary")
    else:
        with col2:
            if input_type == "PDB ID":
                pdb_id = st.text_input("输入 PDB ID（如 3POZ, 3UG5）", value="3POZ",
                                       help="PDB 数据库中的结构 ID，需包含配体共晶结构")
                if pdb_id.strip():
                    analyze_btn = st.button("🚀 开始分析", type="primary")
            else:
                uploaded_file = st.file_uploader("上传 PDB 文件", type=['pdb', 'ent'])
                if uploaded_file:
                    pdb_content = uploaded_file.read()
                    analyze_btn = st.button("🚀 开始分析", type="primary")

    # ---------- 执行分析 ----------
    if analyze_btn:
        if not pdb_id and not pdb_content:
            st.warning("请提供有效的 PDB ID 或上传文件")
            st.stop()

        with st.spinner("⏳ 正在下载结构并调用 PLIP 分析（可能需要 10-20 秒）..."):
            try:
                df, html_str, pdb_path = analyze_plip(
                    pdb_id=pdb_id.strip() if pdb_id else None,
                    pdb_content=pdb_content
                )

                # ---- 显示结果 ----
                st.success(f"✅ 分析完成！共发现 {len(df)} 条相互作用")

                # 统计卡片
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("总相互作用数", len(df))
                if not df.empty:
                    col_b.metric("最多类型", df['类型'].value_counts().index[0])
                    col_c.metric("涉及残基数", df['蛋白残基'].nunique())
                else:
                    col_b.metric("最多类型", "-")
                    col_c.metric("涉及残基数", 0)

                # 左右布局：3D 可视化 + 表格
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.subheader("🧊 3D 相互作用可视化")
                    st.caption("蛋白质以卡通形式显示，配体以球棍模型高亮")
                    try:
                        st.components.v1.html(html_str, height=550, scrolling=False)
                    except Exception as e:
                        st.warning(f"3D 可视化渲染失败: {e}")
                        st.info(f"PDB 文件已下载至: {pdb_path}")

                with col_right:
                    if not df.empty:
                        st.subheader("📋 相互作用列表")
                        st.dataframe(df, width="stretch", height=400)

                        # 类型分布图
                        st.subheader("📊 相互作用类型分布")
                        type_counts = df['类型'].value_counts()
                        st.bar_chart(type_counts)
                    else:
                        st.info("未检测到蛋白-配体相互作用，可能该 PDB 不含配体或配体识别失败。")

            except Exception as e:
                st.error(f"分析失败: {e}")
                st.info("""
                **常见原因**：
                - PDB 结构中不含配体（尝试 3POZ, 3UG5, 1M17 等经典 EGFR 共晶结构）
                - 文件格式不标准
                - 网络无法访问 PDB 数据库
                """)
