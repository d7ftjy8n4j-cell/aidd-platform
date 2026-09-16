# components/knime_export.py
"""
可复用 KNIME 导出组件 - 嵌入任意 Streamlit 页面底部
"""

import streamlit as st
import pandas as pd
from typing import Optional
from utils.knime_export_utils import KNIMEExporter, WORKFLOW_SUGGESTIONS


def knime_export_section(
    data: Optional[pd.DataFrame],
    title: str = "分子数据",
    key_prefix: str = "knime",
    metadata: Optional[dict] = None,
) -> None:
    """
    在页面底部渲染 KNIME 导出区域。

    参数
    ----------
    data : Optional[pd.DataFrame]
        要导出的分子数据 (必须含 SMILES 列；允许 None/空)
    title : str
        导出文件名称前缀
    key_prefix : str
        Streamlit widget key 前缀 (避免多页面 key 冲突)
    metadata : dict, optional
        附加元数据 (pdb_id, 参数等)
    """
    if data is None or data.empty:
        return

    with st.expander("📤 导出到 KNIME", expanded=False):
        st.markdown(f"""
        **TeachOpenCADD-KNIME** 提供 8 个互联工作流 (W1-W8)，
        覆盖从数据获取到蛋白-配体分析的完整 CADD 管道。
        将当前 **{title}** 数据导出，在 KNIME 中继续深度分析。
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            wf_name = st.text_input(
                "工作流名称",
                value=f"药尘光_{title}",
                key=f"{key_prefix}_wf_name",
            )
        with col_b:
            fmt_choice = st.radio(
                "导出格式",
                ["CSV 文件", "ZIP 完整包 (CSV + 元数据)"],
                key=f"{key_prefix}_fmt",
            )

        btn_col, tip_col = st.columns([1, 3])
        with btn_col:
            do_export = st.button(
                "📥 导出", type="primary",
                key=f"{key_prefix}_export_btn",
                width="stretch",
            )

        with tip_col:
            try:
                exporter = KNIMEExporter(data, metadata)
                suggested = WORKFLOW_SUGGESTIONS.get(exporter.module_type, "W1-W8")
                st.caption(f"💡 推荐后续 KNIME 工作流: **{suggested}**")
            except (ValueError, AttributeError) as _e:
                # 数据缺 SMILES 列等真实错误：向用户明示，而不是静默吞掉
                st.caption(f"⚠️ 无法识别导出数据: {_e}")
                st.caption("💡 [TeachOpenCADD-KNIME Hub](https://hub.knime.com/volkamerlab/space/TeachOpenCADD)")

        if do_export:
            try:
                exporter = KNIMEExporter(data, metadata)

                if "ZIP" in fmt_choice:
                    zip_bytes = exporter.to_workflow_zip(wf_name)
                    st.download_button(
                        label=f"📦 下载 {wf_name}.zip",
                        data=zip_bytes,
                        file_name=f"{wf_name}.zip",
                        mime="application/zip",
                        key=f"{key_prefix}_dl_zip",
                    )
                else:
                    csv_bytes = exporter.to_csv_bytes()
                    st.download_button(
                        label=f"📄 下载 {wf_name}_data.csv",
                        data=csv_bytes,
                        file_name=f"{wf_name}_data.csv",
                        mime="text/csv",
                        key=f"{key_prefix}_dl_csv",
                    )

                st.success(f"✅ 导出成功！共 {len(data)} 个分子，{data.shape[1]} 个属性。")
                st.info(f"""
                **在 KNIME 中继续分析**:
                1. 新建工作流 → 拖入 **File Reader** 节点
                2. 加载导出的 CSV 文件
                3. 参考 **{WORKFLOW_SUGGESTIONS.get(exporter.module_type, 'W1-W8')}** 工作流
                4. 更多教学: [TeachOpenCADD-KNIME Hub](https://hub.knime.com/volkamerlab/space/TeachOpenCADD)
                """)

            except Exception as e:
                st.error(f"导出失败: {e}")
