# pages/page_automated_pipeline.py
"""
自动化药物发现流程页面
一键运行从预测到筛选的全流程，快速评估分子的成药潜力。
"""

import streamlit as st
import pandas as pd
import io
import time
from datetime import datetime

from utils.pipeline import Pipeline, SingleMoleculeResult
from components.knime_export import knime_export_section


@st.cache_resource
def _get_pipeline() -> Pipeline:
    """缓存 Pipeline 实例，避免重复初始化各组件"""
    return Pipeline()


def page_automated_pipeline():
    """⚙️ 自动化流程页面"""
    
    st.header("⚙️ 自动化药物发现流程")
    st.caption("一键运行从双模型预测到成药性筛选的全流程，快速评估分子成药潜力。")
    
    with st.popover("🎓 教学点"):
        st.markdown("""学习药物发现的典型筛选流程：
        1. **活性预测**：AI模型预测分子是否对EGFR有抑制活性
        2. **成药性筛选**：ADME/Ro5评估口服生物利用度
        3. **毒性警报**：PAINS/Brenk子结构标注潜在风险
        4. **药效团匹配**：检查分子是否含有关键药效特征
        5. **相似性搜索**：查找已知活性化合类似物
        
        通过配置化流程，理解每一步在药物发现中的作用。""")
    
    pipeline = _get_pipeline()
    
    # ---- 是否已有缓存的流程结果（影响输入区折叠状态） ----
    has_cached_results = bool(st.session_state.get("pipeline_results"))

    # ---- 输入区（结果存在时自动折叠，让结果更突出） ----
    with st.expander("📝 输入分子与流程配置", expanded=not has_cached_results):
        # 检查是否有来自数据获取/聚类页面的批量数据 (不使用 pop，保留数据)
        has_batch_data = (
            "batch_smiles_list" in st.session_state
            and st.session_state.batch_smiles_list
        )
        batch_source = st.session_state.get("batch_data_source", "数据获取")

        # 有批量数据时始终切换为"已导入数据"模式
        if has_batch_data:
            st.session_state["pipeline_input_mode"] = "📦 已导入数据"

        input_mode = st.radio(
            "选择输入方式",
            ["单个SMILES", "批量上传CSV", "📦 已导入数据"],
            horizontal=True,
            key="pipeline_input_mode"
        )

        if has_batch_data and input_mode != "📦 已导入数据":
            st.info(
                "检测到当前会话中已有导入的批量分子数据。\n"
                "请切换到“📦 已导入数据”模式，或继续使用其他输入方式。",
                icon="ℹ️"
            )
            if st.button(
                "切换到已导入数据",
                key="pipeline_switch_to_imported_data"
            ):
                st.session_state.pipeline_input_mode = "📦 已导入数据"
                st.experimental_rerun()

        smiles_list: list = []

        # Mode 1: 单个 SMILES
        if input_mode == "单个SMILES":
            default_smiles = st.session_state.get('last_smiles', '') or "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1"
            smiles = st.text_area(
                "输入SMILES",
                value=default_smiles,
                height=100,
                help="输入一个分子的SMILES表示",
                key="pipeline_smiles_input"
            )
            if smiles.strip():
                smiles_list = [smiles.strip()]

        # Mode 2: 上传 CSV
        elif input_mode == "批量上传CSV":
            uploaded = st.file_uploader(
                "上传CSV文件 (需包含 'smiles' 列)",
                type=["csv"],
                key="pipeline_csv_upload"
            )
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    if 'smiles' in df.columns:
                        smiles_list = df['smiles'].dropna().astype(str).tolist()
                        st.success(f"已加载 {len(smiles_list)} 个分子")

                        # 预览
                        with st.expander("📋 分子列表预览", expanded=False):
                            st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.error("CSV必须包含 'smiles' 列，当前列名: " + ", ".join(df.columns.tolist()))
                except Exception as e:
                    st.error(f"CSV读取失败: {e}")

        # Mode 3: 已从数据获取/聚类页面导入
        else:
            if has_batch_data:
                smiles_list = list(st.session_state.batch_smiles_list)
                st.success(f"✅ 已加载 {len(smiles_list)} 个分子 (来源: {batch_source})")

                # 可编辑的文本区域（自动填充 batch_smiles_list）
                edit_text = "\n".join(smiles_list)
                edited = st.text_area(
                    "可编辑的SMILES列表 (每行一个)",
                    value=edit_text,
                    height=200,
                    help="从数据获取或聚类结果自动填入，可手动编辑",
                    key="pipeline_edited_smiles"
                )
                if edited.strip():
                    smiles_list = [s.strip() for s in edited.splitlines() if s.strip()]
            else:
                st.info("💡 请先前往「📦 数据获取」页面获取化合物数据，或在「🧩 分子聚类」页面筛选代表性分子。")
                st.caption("也可使用「批量上传CSV」模式直接上传本地文件。")

        # ---- 流程步骤配置 (在 expander 内) ----
        st.divider()
        st.markdown("**⚙️ 流程步骤配置**")
        st.caption("勾选需要运行的步骤")
        col1, col2 = st.columns(2)
        with col1:
            enable_rf = st.checkbox("🌲 随机森林预测", value=True, 
                                     help="基于200+ RDKit分子描述符的RF模型")
            enable_gnn = st.checkbox("🧠 GNN图神经网络预测", value=True,
                                      help="基于分子图结构的图卷积网络模型")
            enable_adme = st.checkbox("💊 ADME/Ro5成药性筛选", value=True,
                                       help="计算Lipinski五规则指标")
        with col2:
            enable_substructure = st.checkbox("⚠️ 不良子结构筛查", value=True,
                                               help="检测PAINS和Brenk毒性警报")
            enable_pharmacophore = st.checkbox("🎯 药效团匹配", value=False,
                                                help="提取并匹配分子药效团特征")
            enable_similarity = st.checkbox("🔍 相似性搜索", value=False,
                                             help="搜索ChEMBL中结构相似的已知活性化合物")
    
    # ---- 启动按钮 ----
    st.divider()
    can_run = len(smiles_list) > 0
    if not can_run:
        st.info("👆 请先输入或上传SMILES分子")
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_clicked = st.button(
            "🚀 启动流程",
            type="primary",
            use_container_width=True,
            disabled=not can_run
        )
    
    # ---- 执行流程 ----
    if run_clicked and smiles_list:
        with st.status("正在执行自动化流程...", expanded=True) as status:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total = len(smiles_list)
            for i, smi in enumerate(smiles_list):
                short_smi = (smi[:40] + "...") if len(smi) > 40 else smi
                status_text.text(f"处理中 [{i+1}/{total}]: {short_smi}")
                
                # 每隔10个分子让出控制权，避免前端卡顿
                if i > 0 and i % 10 == 0:
                    time.sleep(0.01)
                
                res = pipeline.run_single_molecule(
                    smi,
                    enable_rf=enable_rf,
                    enable_gnn=enable_gnn,
                    enable_adme=enable_adme,
                    enable_substructure=enable_substructure,
                    enable_pharmacophore=enable_pharmacophore,
                    enable_similarity=enable_similarity,
                )
                results.append(res)
                progress_bar.progress((i + 1) / total)
            
            status.update(label="✅ 流程完成！", state="complete")
            status_text.text("")
        
        # ---- 存储结果（转为字典便于 session_state 序列化） ----
        serializable_results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        st.session_state['pipeline_results'] = serializable_results
        st.session_state['pipeline_smiles_list'] = smiles_list
        # 触发重绘以显示完整结果（run_clicked 会在重绘后变 False，但结果已持久化）
        st.rerun()
    
    # ---- 始终展示已缓存的结果（无论是否刚运行完） ----
    if 'pipeline_results' in st.session_state and st.session_state['pipeline_results']:
        _render_pipeline_results(pipeline)


def _render_pipeline_results(pipeline: Pipeline):
    """统一的结果渲染函数 — 无论首次运行还是缓存恢复都展示完整结果"""
    results = st.session_state['pipeline_results']
    
    st.divider()
    st.subheader("📊 流程结果汇总")
    
    # 汇总表格
    df_summary = pipeline.results_to_dataframe(results)
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RF活性概率": st.column_config.NumberColumn(format="%.4f"),
            "GNN活性概率": st.column_config.NumberColumn(format="%.4f"),
        }
    )
    
    # 统计信息
    verdicts = df_summary["最终判定"].tolist()
    recommended = sum(1 for v in verdicts if v.startswith("✅"))
    active_but_poor = sum(1 for v in verdicts if v.startswith("⚠️ 活性"))
    inactive = sum(1 for v in verdicts if v.startswith("❌"))
    divergent = sum(1 for v in verdicts if "分歧" in v or "无法" in v)
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("✅ 推荐", recommended)
    with col_s2:
        st.metric("⚠️ 活性/成药性差", active_but_poor)
    with col_s3:
        st.metric("❌ 非活性", inactive)
    with col_s4:
        st.metric("⚡ 需人工判断", divergent)
    
    st.divider()
    
    # ---- 详细报告 ----
    st.subheader("📄 详细报告")
    
    # 如果只有一个分子，直接展示；多个则用选择器
    if len(results) == 1:
        _show_detailed_report(pipeline, results[0])
    else:
        # 多分子选择
        idx_options = [
            f"[{i+1}] {r['smiles'][:50]}... → {r['summary'].get('final_verdict', 'N/A')}" 
            for i, r in enumerate(results)
        ]
        selected_idx = st.selectbox(
            "选择分子查看详细报告",
            range(len(results)),
            format_func=lambda i: idx_options[i],
            key="pipeline_detail_selector"
        )
        _show_detailed_report(pipeline, results[selected_idx], idx=selected_idx + 1)
    
    # 清空结果按钮
    st.divider()
    if st.button("🔄 清空结果并重新开始", key="clear_pipeline_results"):
        st.session_state.pop('pipeline_results', None)
        st.session_state.pop('pipeline_smiles_list', None)
        st.rerun()

    # KNIME 导出
    if results is not None and len(results) > 0:
        try:
            export_rows = []
            for r in results:
                row = {"smiles": r.smiles}
                for attr in dir(r):
                    if not attr.startswith("_") and attr != "smiles":
                        val = getattr(r, attr, None)
                        if isinstance(val, (str, int, float, bool)) and val is not None:
                            row[attr] = val
                export_rows.append(row)
            if export_rows:
                import pandas as _pd
                knime_export_section(
                    _pd.DataFrame(export_rows),
                    title="自动化流程结果",
                    key_prefix="pipeline_knime",
                )
        except Exception:
            pass


def _show_detailed_report(pipeline: Pipeline, result: SingleMoleculeResult, idx: int = 1):
    """展示单个分子的详细报告和下载选项"""
    
    # Tab：格式化报告 ｜ 原始JSON
    tab_md, tab_json = st.tabs(["📝 Markdown报告", "🔧 原始数据 (JSON)"])
    
    with tab_md:
        report = pipeline.generate_report(result, format="markdown")
        st.markdown(report)
        
        # 下载按钮
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=report.encode('utf-8'),
                file_name=f"pipeline_report_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            # CSV 汇总下载
            df = pipeline.results_to_dataframe([result])
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 下载汇总 (CSV)",
                data=csv_data.encode('utf-8'),
                file_name=f"pipeline_summary_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with tab_json:
        import json
        # 过滤掉不可序列化的对象
        def _json_safe(obj):
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_json_safe(i) for i in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        safe_result = _json_safe(result)
        st.json(safe_result)
        
        st.download_button(
            label="📥 下载原始数据 (JSON)",
            data=json.dumps(safe_result, ensure_ascii=False, indent=2),
            file_name=f"pipeline_result_{timestamp}.json",
            mime="application/json",
        )
