"""
激酶结合模式相似性分析页面 (KLIFS-IFP)
"""
import streamlit as st
import pandas as pd
import numpy as np
from interaction_utils import fetch_klifs_ifps, compute_ifp_distance_matrix, plot_ifp_heatmap


# ---------- 预置激酶列表（覆盖 EGFR 相关主要激酶）----------
KINASE_OPTIONS = {
    "EGFR":   "Epidermal growth factor receptor",
    "ErbB2":  "Receptor tyrosine-protein kinase erbB-2",
    "ErbB4":  "Receptor tyrosine-protein kinase erbB-4",
    "CDK2":   "Cyclin-dependent kinase 2",
    "CDK4":   "Cyclin-dependent kinase 4",
    "MET":    "Hepatocyte growth factor receptor",
    "KDR":    "Vascular endothelial growth factor receptor 2",
    "LCK":    "Tyrosine-protein kinase Lck",
    "SRC":    "Proto-oncogene tyrosine-protein kinase Src",
    "ABL1":   "Tyrosine-protein kinase ABL1",
    "BRAF":   "Serine/threonine-protein kinase B-raf",
    "p38a":   "Mitogen-activated protein kinase 14",
}


def page_kinase_similarity():
    """激酶结合模式相似性分析主函数"""
    st.title("🧬 激酶结合模式相似性分析 (KLIFS-IFP)")

    with st.popover("🎓 教学点"):
        st.markdown("""
        **激酶选择性——药物设计的关键挑战**：

        - **激酶组** (Kinome)：人类基因组编码约 518 种激酶，ATP 结合口袋高度保守
        - 许多激酶抑制剂因选择性差而产生脱靶毒性
        - **KLIFS** 数据库收录了所有公开的激酶-配体共晶结构

        **IFP（相互作用指纹）方法**：
        - 将蛋白-配体相互作用编码为二进制指纹（某残基-某作用类型-有/无）
        - 基于 IFP 的相似度计算能揭示**结合模式的异同**
        - 可用于：选择性评估、脱靶预测、激酶谱分析

        **预置激酶说明**：
        - **EGFR/ErbB2/ErbB4**：ErbB 家族，抗癌核心靶点
        - **CDK2/CDK4**：细胞周期调控激酶
        - **MET/KDR**：RTK 类激酶
        - **LCK/SRC/ABL1**：非受体酪氨酸激酶
        - **BRAF/p38α**：丝/苏氨酸激酶

        > 数据来源：[KLIFS](https://klifs.net/) REST API，自动过滤高质量人源晶体结构。
        """)

    st.markdown("""
    **功能说明**：基于 KLIFS 数据库的**相互作用指纹 (IFP)**，比较不同激酶的结合模式相似性。
    可用于评估化合物的潜在**脱靶风险**或寻找**替代靶点**。

    所有数据来源于 KLIFS（激酶-配体相互作用指纹与结构数据库），自动过滤高质量人源晶体结构。
    """)

    # ---------- 激酶选择 ----------
    selected = st.multiselect(
        "选择要比较的激酶（至少选 2 个）",
        options=list(KINASE_OPTIONS.keys()),
        default=["EGFR", "ErbB2", "CDK2", "MET"],
        format_func=lambda x: f"{x} ({KINASE_OPTIONS[x]})",
        help="选择多个激酶以比较它们的配体结合模式相似性"
    )

    if st.button("📊 分析结合模式相似性", type="primary"):
        if len(selected) < 2:
            st.warning("请至少选择 2 个激酶进行比较")
            st.stop()

        with st.spinner(f"⏳ 正在从 KLIFS 数据库获取 {len(selected)} 个激酶的结构数据..."):
            try:
                # 1. 获取 IFP 数据
                ifp_df = fetch_klifs_ifps(selected)

                if ifp_df.empty:
                    st.error("未获取到任何结构数据。可能原因：激酶名称不匹配、无高质量结构、或网络无法访问 KLIFS。")
                    st.info("""
                    **排查建议**：
                    - 检查激酶名称是否与 KLIFS 数据库一致
                    - 尝试减少激酶数量、换用经典激酶（如 EGFR, ABL1）
                    - 确认网络可以访问 https://klifs.net
                    """)
                    st.stop()

                # 2. 数据覆盖情况
                st.subheader("📊 各激酶可用高质量结构数量")
                if "kinase.klifs_name" in ifp_df.columns:
                    coverage = ifp_df.groupby("kinase.klifs_name").size().sort_values(ascending=True)
                else:
                    coverage = pd.Series([len(ifp_df)], index=[", ".join(selected)])
                st.bar_chart(coverage)

                # 显示数据表格
                with st.expander("📋 原始 IFP 数据预览", expanded=False):
                    st.dataframe(ifp_df.head(20), use_container_width=True)

                # 3. 计算距离矩阵
                dist_matrix, labels = compute_ifp_distance_matrix(ifp_df)

                # 4. 热图
                st.subheader("🔥 结合模式相似性热图")
                st.caption("Jaccard 距离：0 = 完全一致，1 = 完全不同。距离越小，结合模式越相似。")
                fig = plot_ifp_heatmap(dist_matrix, labels)
                st.pyplot(fig)

                # 5. 脱靶风险提示
                st.subheader("⚠️ 潜在脱靶风险分析")

                df_dist = pd.DataFrame(dist_matrix, index=labels, columns=labels)
                risk_found = False
                n = len(labels)

                for i in range(n):
                    for j in range(i + 1, n):
                        d = df_dist.iloc[i, j]
                        if d < 0.3:
                            st.warning(
                                f"🔥 **{labels[i]}** ↔ **{labels[j]}** "
                                f"结合模式高度相似 (Jaccard 距离 = {d:.3f})，存在交叉反应风险。"
                            )
                            risk_found = True
                        elif d < 0.5:
                            st.info(
                                f"⚠️ **{labels[i]}** ↔ **{labels[j]}** "
                                f"结合模式中等相似 (Jaccard 距离 = {d:.3f})，需关注选择性。"
                            )
                            risk_found = True

                if not risk_found:
                    st.success("✅ 所选激酶间的结合模式差异较大（Jaccard 距离 ≥ 0.5），靶点选择性良好。")

                st.success("分析完成！")

            except Exception as e:
                st.error(f"分析失败: {e}")
                st.info("""
                **可能原因**：
                - KLIFS 远程数据库暂时不可用
                - 选中的激酶在 KLIFS 中没有满足过滤条件的结构
                - 网络连接问题

                建议：稍后重试，或减少激酶数量、换用经典激酶组合。
                """)
