"""
预测系统状态栏组件 —— 供分子预测和自动化流程页面复用
"""
import streamlit as st
import pandas as pd
from datetime import datetime


def render_prediction_status_bar(get_predictors_available):
    """
    以 popover 弹出框形式显示模型状态、使用统计和快捷操作。
    仅在预测相关页面调用。

    Args:
        get_predictors_available: callable() -> dict[str, bool]
            返回 {'rf': bool, 'gnn': bool} 表示两个预测器是否在线。
    """
    with st.popover("⚙️ 预测系统状态", icon="⚙️"):
        available = get_predictors_available()

        st.markdown("### 模型状态")
        rf_status = "✅ 在线" if available.get('rf') else "❌ 离线"
        gnn_status = "✅ 在线" if available.get('gnn') else "❌ 离线"
        st.write(f"- 随机森林: {rf_status}")
        st.write(f"- GNN模型: {gnn_status}")

        st.divider()

        st.markdown("### 📈 使用统计")
        st.metric("总预测次数", st.session_state.get("prediction_count", 0))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重置统计", width="stretch", key="reset_stat_bar"):
                st.session_state.prediction_count = 0
                st.rerun()

        with col2:
            has_data = bool(st.session_state.get("last_smiles"))
            if st.button("📥 导出结果", width="stretch", key="export_stat_bar",
                         disabled=not has_data):
                _show_export_dialog()


def _show_export_dialog():
    """导出预测结果弹窗"""
    @st.dialog("📥 导出预测结果", icon="📊")
    def export_dialog():
        export_data = {}
        if st.session_state.get("last_rf_result"):
            rf_result = st.session_state.last_rf_result
            if isinstance(rf_result, dict) and "error" not in rf_result:
                export_data['rf'] = rf_result
        if st.session_state.get("last_gnn_result"):
            gnn_result = st.session_state.last_gnn_result
            if isinstance(gnn_result, dict) and gnn_result.get("success", True):
                export_data['gnn'] = gnn_result

        if export_data:
            df = _build_export_dataframe(export_data)
            st.dataframe(df, width="stretch", hide_index=True)
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            filename = f"egfr_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="📥 下载CSV文件", data=csv,
                file_name=filename, mime="text/csv"
            )
        else:
            st.warning("没有可用的模型结果")
        if st.button("关闭", type="secondary", key="close_export_bar"):
            st.rerun()
    export_dialog()


def _build_export_dataframe(results_dict):
    """构建导出用的 DataFrame，不依赖 app.py 中的 get_model_performance"""
    data = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for model_type, result in results_dict.items():
        if isinstance(result, dict):
            row = {
                '时间戳': timestamp,
                'SMILES': st.session_state.get('last_smiles', ''),
                '模型': model_type.upper(),
            }
            if 'error' not in result and result.get('success', True):
                row['预测结果'] = '活性' if result.get('prediction') == 1 else '非活性'
                row['活性概率'] = f"{result.get('probability_active', 0):.4f}"
                row['置信度'] = result.get('confidence', '中')
            else:
                row['预测结果'] = '失败'
                row['错误信息'] = result.get('error', '未知错误')
            data.append(row)
    return pd.DataFrame(data)
