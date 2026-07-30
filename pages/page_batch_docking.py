# pages/page_batch_docking.py
"""
批量对接与结果对比页面 (Smina)
支持多个配体 SMILES 同时对接到一个靶点蛋白，排序结果并对比分析。

参考: TeachOpenCADD T015 (单分子对接) + T018 (批量虚拟筛选)
"""

import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import logging

from components.knime_export import knime_export_section


def page_batch_docking():
    """批量对接与结果对比主函数"""

    # ---------- 检查依赖 ----------
    smina_available = _check_smina()
    openbabel_available = _check_openbabel()

    if not smina_available or not openbabel_available:
        st.title("🧩 批量对接与结果对比")
        if not smina_available:
            st.error("❌ 未检测到 Smina 命令行工具")
            st.info(
                "安装方法：\n"
                "- **macOS**: `brew install smina`\n"
                "- **Linux**: 下载预编译包到 `/usr/local/bin/smina`\n"
                "- **conda**: `conda install -c conda-forge smina`\n"
                "- **其他**: 参见 https://sourceforge.net/projects/smina/"
            )
        if not openbabel_available:
            st.error("❌ 未检测到 openbabel Python 绑定")
            st.info("安装方法：`conda install -c conda-forge openbabel`")
        return

    # ---------- 页面头部 ----------
    st.title("🧩 批量对接与结果对比")

    st.markdown("""
    **功能说明**：基于 Smina (AutoDock Vina 分支)，将一个靶点蛋白与多个配体同时对接，
    按结合能排序对比，快速筛选最优候选化合物。

    - **输入**：靶点蛋白（PDB ID 或上传）+ 多个配体（SMILES 列表 / CSV 上传）
    - **输出**：排序对比表 + 结合能分布图 + 统计摘要 + 结果导出

    ---
    **典型应用**：EGFR 激酶 (PDB: 2ITO) × 候选配体库 → 按对接打分排名
    """)

    # ---------- 侧边栏：输入设置 ----------
    with st.sidebar:
        st.header("📥 输入设置")

        # === 蛋白输入 ===
        st.subheader("🧬 靶点蛋白")
        input_type = st.radio(
            "蛋白来源",
            ["PDB ID", "上传PDB文件"],
            index=0,
            key="batch_input_type",
        )
        pdb_id = None
        pdb_content = None
        if input_type == "PDB ID":
            pdb_id = st.text_input(
                "输入 PDB ID",
                "2ITO",
                help="4 位 PDB 代码，如 2ITO, 3POZ",
                key="batch_pdb_id",
            )
        else:
            uploaded_file = st.file_uploader(
                "上传 PDB 文件",
                type=["pdb", "ent"],
                key="batch_pdb_upload",
            )
            if uploaded_file:
                pdb_content = uploaded_file.read()
                st.success(f"已加载: {uploaded_file.name}")

        # === 配体输入 ===
        st.subheader("💊 配体列表")
        input_method = st.radio(
            "配体输入方式",
            ["SMILES 文本框", "上传 CSV 文件"],
            index=0,
            key="batch_ligand_method",
        )

        smiles_list = []
        if input_method == "SMILES 文本框":
            default_smiles = (
                "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4\n"
                "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC\n"
                "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2F)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC=CC=N5"
            )
            smiles_text = st.text_area(
                "输入 SMILES (每行一个)",
                default_smiles,
                height=150,
                help="每行输入一个配体的 SMILES 字符串",
                key="batch_smiles_text",
            )
            smiles_list = [
                s.strip() for s in smiles_text.split("\n") if s.strip()
            ]
        else:
            csv_file = st.file_uploader(
                "上传 CSV (需含 smiles 列)",
                type=["csv"],
                key="batch_csv_upload",
            )
            if csv_file:
                try:
                    df_csv = pd.read_csv(csv_file)

                    # 智能列选择器：让用户指定包含 SMILES 的列
                    candidate_cols = [
                        c for c in df_csv.columns
                        if "smile" in c.lower() or "canonical" in c.lower()
                        or "structure" in c.lower() or "smi" in c.lower()
                    ]
                    # 补齐到全部列供用户选择
                    all_cols = list(df_csv.columns)
                    default_idx = 0
                    for i, c in enumerate(all_cols):
                        if c in candidate_cols:
                            default_idx = i
                            break

                    smiles_col = st.selectbox(
                        "选择 SMILES 所在的列",
                        all_cols,
                        index=default_idx,
                        help="请确保所选列包含有效的 SMILES 字符串",
                        key="batch_smiles_col_select",
                    )

                    raw_smiles = df_csv[smiles_col].dropna().astype(str).tolist()
                    # 过滤掉明显无效的 SMILES（长度 < 3 或 > 500）
                    smiles_list = [
                        s.strip() for s in raw_smiles
                        if 2 < len(s.strip()) < 500
                    ]
                    skipped = len(raw_smiles) - len(smiles_list)
                    if skipped > 0:
                        st.warning(f"已自动跳过 {skipped} 个无效/过短的 SMILES 条目")
                    st.success(f"从列 `{smiles_col}` 解析到 {len(smiles_list)} 个有效配体")
                except Exception as e:
                    st.error(f"CSV 解析失败: {e}")

        # === 常用分子模板 ===
        with st.expander("📋 常用 EGFR 抑制剂模板"):
            templates = {
                "Gefitinib": (
                    "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
                ),
                "Erlotinib": (
                    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
                ),
                "Lapatinib": (
                    "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2F)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC=CC=N5"
                ),
                "Osimertinib": (
                    "CN1CCN(CC1)C2=CC(=C(C=C2)NC3=NC=C(C(=N3)NC4=C(C=C(C=C4)N(C)C(=O)C=C)N5C=CC(=O)N5)C)OC"
                ),
            }
            cols_t = st.columns(4)
            for idx, (name, smi) in enumerate(templates.items()):
                with cols_t[idx]:
                    if st.button(f"添加 {name}", key=f"batch_tpl_{idx}"):
                        current = st.session_state.get("batch_smiles_text", "")
                        if current:
                            st.session_state["batch_smiles_text"] = current.strip() + "\n" + smi
                        else:
                            st.session_state["batch_smiles_text"] = smi
                        st.rerun()

        # === 对接参数 ===
        st.subheader("⚙️ 对接参数")
        num_poses = st.slider(
            "每配体构象数", 3, 20, 9,
            help="每个配体保留的最优构象数量",
            key="batch_num_poses",
        )
        exhaustiveness = st.slider(
            "搜索精度", 4, 32, 8,
            help="4=快速, 8=标准, 16=精细, 32=极精细",
            key="batch_exhaustiveness",
        )
        n_jobs = st.slider(
            "并行线程数", 1, 4, 1,
            help="同时对接的配体数量（需 CPU 支持）",
            key="batch_n_jobs",
        )
        ligand_resname = st.text_input(
            "共晶配体残基名 (可选)",
            "",
            help="留空则自动检测非水非蛋白的第一个残基",
            key="batch_resname",
        )

        st.divider()
        docking_info = st.empty()  # 对接状态信息占位

    # ---------- 主区域：配体数量提示 ----------
    st.caption(f"当前加载 **{len(smiles_list)}** 个配体待对接")

    # ---------- 执行批量对接 ----------
    if st.button("🚀 开始批量对接", type="primary", key="batch_run_btn", use_container_width=True):
        if not smiles_list:
            st.error("请至少输入一个配体 SMILES")
            st.stop()
        if not pdb_id and not pdb_content:
            st.error("请提供 PDB ID 或上传 PDB 文件")
            st.stop()

        # 进度 UI
        progress_bar = st.progress(0, text="准备中...")
        status_text = st.empty()
        stage_placeholder = st.empty()

        # 阶段进度追踪
        stage_progress = {"protein": 0, "receptor_pdbqt": 0, "ligands": 0, "docking": 0}
        total_pipeline_steps = 4  # protein, receptor_pdbqt, ligands, docking

        def update_progress(stage, completed, total):
            """更新阶段进度 UI"""
            stage_progress[stage] = float(completed) / max(total, 1)

            stage_labels = {
                "protein": f"下载蛋白... ({completed}/{total})",
                "receptor_pdbqt": f"转换蛋白 PDBQT... ({completed}/{total})",
                "ligands": f"准备配体 PDBQT... ({completed}/{total})",
                "docking": f"执行对接... ({completed}/{total})",
            }

            # 计算管线整体进度
            weights = {"protein": 0.05, "receptor_pdbqt": 0.05, "ligands": 0.15, "docking": 0.75}
            overall = sum(
                stage_progress[s] * weights[s] for s in weights
            )
            progress_bar.progress(min(overall, 1.0), text=stage_labels.get(stage, ""))

        try:
            from utils.batch_docking_utils import run_batch_docking, cleanup_batch_docking

            with st.spinner(f"正在对接 {len(smiles_list)} 个配体..."
                           f"（精度={exhaustiveness}, 预计 {len(smiles_list) * 2}~{len(smiles_list) * 5} 分钟）"):
                result = run_batch_docking(
                    pdb_id=pdb_id,
                    pdb_content=pdb_content,
                    smiles_list=smiles_list,
                    ligand_resname=ligand_resname.strip() if ligand_resname.strip() else None,
                    num_poses=num_poses,
                    exhaustiveness=exhaustiveness,
                    n_jobs=n_jobs,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0, text="✅ 批量对接完成！")
            st.success(f"✅ 批量对接完成！成功 {result['stats']['n_success']}/{len(smiles_list)} 个配体")

            # ---- 显示结果 ----
            _render_batch_results(result)

            # 缓存 + 清理定时器
            st.session_state["batch_docking_last_result"] = result

            # 注册 Session State 清理
            if "batch_pending_cleanup" not in st.session_state:
                st.session_state["batch_pending_cleanup"] = []
            st.session_state["batch_pending_cleanup"].append(result)

        except subprocess.CalledProcessError as e:
            st.error(f"❌ Smina 执行失败 (退出码 {e.returncode})")
            if hasattr(e, "output") and e.output:
                with st.expander("查看错误详情"):
                    st.code(e.output, language="text")
        except ImportError as e:
            st.error(f"❌ 缺少依赖: {e}")
        except Exception as e:
            st.error(f"❌ 批量对接失败: {e}")
            import traceback
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc(), language="text")
        finally:
            progress_bar.empty()
            status_text.empty()
            stage_placeholder.empty()

    # ---- 上次结果回显 ----
    elif "batch_docking_last_result" in st.session_state:
        with st.expander("📌 上次批量对接结果（点击查看）", expanded=False):
            cached = st.session_state["batch_docking_last_result"]
            st.caption(f"共 {cached['ligand_count']} 个配体，"
                       f"成功 {cached['stats']['n_success']} 个")
            _render_batch_results(cached, compact=True)

    # ---------- 使用说明 ----------
    st.divider()
    with st.expander("📘 使用说明与常见问题", expanded=False):
        st.markdown("""
        ### 批量对接流程
        1. **选择靶点蛋白**: PDB ID 或上传含共晶配体的 PDB 文件
        2. **输入配体列表**: 在文本框中每行一个 SMILES，或上传含 `smiles` 列的 CSV
        3. **调整参数**: 设置构象数量、搜索精度和并行度
        4. **运行对接**: 每个配体独立对接，结果按结合能升序排列
        5. **分析结果**: 查看排序表、分布图、统计摘要
        6. **下载导出**: 批量下载所有配体结果

        ### 参数建议
        | 场景 | exhaustiveness | num_poses | 预计耗时/配体 |
        |------|---------------|-----------|-------------|
        | 快速预览 | 4 | 5 | ~30 秒 |
        | 标准筛选 | 8 | 9 | ~1-2 分钟 |
        | 精细比对 | 16 | 9 | ~3-5 分钟 |

        ### 与单分子对接的区别
        | | 单分子对接 | 批量对接 |
        |--|-----------|---------|
        | **场景** | 研究单个分子的结合模式 | 从分子库中筛选最优候选 |
        | **输出** | Top-N poses + 3D 可视化 | 排序对比表 + 统计摘要 |
        | **3D 可视化** | NGLView 交互式 | 结果表 + 可下载 poses |
        | **并行** | 不支持 | 支持多线程并行 |

        ### 数据流衔接
        ```
        分子预测 (QSAR) → 药物筛选 (ADME) → 批量对接 → Top 候选 → MD 模拟
        ```
        """)


# ---------- 结果渲染辅助函数 ----------

def _render_batch_results(result, compact=False):
    """渲染批量对接结果（表格、图表、统计）"""
    results_df = result["results_df"]
    stats = result["stats"]

    if compact:
        # 紧凑模式：仅显示简要表格
        display_cols = ["name", "smiles", "best_score", "pose_count"]
        display_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()
        display_df.columns = [
            "配体名称", "SMILES", "最佳结合能 (kcal/mol)", "构象数"
        ][:len(display_df.columns)]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        return

    # ---- 统计卡片 ----
    st.subheader("📊 对接统计")
    cols = st.columns(5)
    cols[0].metric("总配体数", stats["count"], border=True)
    cols[1].metric("成功", f"{stats['n_success']}", border=True)
    cols[2].metric("失败", f"{stats['n_failed']}",
                   delta=None if stats['n_failed'] == 0 else f"-{stats['n_failed']}",
                   delta_color="inverse",
                   border=True)

    if stats.get("best_score") is not None:
        cols[3].metric("最佳结合能", f"{stats['best_score']:.2f} kcal/mol", border=True)
        cols[4].metric("平均结合能", f"{stats['mean_score']:.2f} kcal/mol",
                       f"±{stats['std_score']:.2f}", border=True)
    else:
        cols[3].metric("最佳结合能", "N/A", border=True)
        cols[4].metric("平均结合能", "N/A", border=True)

    # ---- 对接口袋信息 ----
    with st.expander("🎯 对接参数详情", expanded=False):
        p = result["pocket"]
        st.json({
            "PDB": result["pdb_id"],
            "共晶配体残基名": result["ligand_resname"],
            "口袋中心": f"[{p['center'][0]:.2f}, {p['center'][1]:.2f}, {p['center'][2]:.2f}]",
            "口袋尺寸": f"[{p['size'][0]:.2f}, {p['size'][1]:.2f}, {p['size'][2]:.2f}]",
            "配体总数": result["ligand_count"],
            "成功对接": stats["n_success"],
            "对接失败": stats["n_failed"],
        })

    # ---- 排序对比表 ----
    st.subheader("🏆 对接结果排名 (按结合能升序)")
    display_cols = ["name", "smiles", "best_score", "worst_score", "pose_count"]
    display_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()

    col_names = {
        "name": "配体",
        "smiles": "SMILES",
        "best_score": "最佳结合能 (kcal/mol)",
        "worst_score": "最差结合能 (kcal/mol)",
        "pose_count": "构象数",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_names.items() if k in display_df.columns})

    # 为成功的行加颜色渐变
    if "最佳结合能 (kcal/mol)" in display_df.columns and len(display_df) > 0:
        def _color_score(val):
            try:
                v = float(val)
                if v < -9.5:
                    return "background-color: #c8e6c9"  # 深绿: 强结合
                elif v < -7.5:
                    return "background-color: #e8f5e9"  # 浅绿: 中等
                elif v < -6.0:
                    return "background-color: #fff9c4"  # 黄: 弱
                else:
                    return "background-color: #ffcdd2"  # 红: 很弱
            except Exception:
                return ""

        styled = display_df.style.map(_color_score, subset=["最佳结合能 (kcal/mol)"])
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.dataframe(display_df, use_container_width=True)

    # ---- 结合能分布图 ----
    st.subheader("📈 结合能分布")
    success_df = results_df[results_df["success"] == True]
    if len(success_df) > 0:
        chart_df = success_df.set_index("name")[["best_score"]].copy()
        chart_df.columns = ["结合能 (kcal/mol)"]
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.warning("无成功对接结果")

    # ---- 失败详情 ----
    failed_df = results_df[results_df["success"] != True]
    if len(failed_df) > 0:
        with st.expander(f"⚠️ {len(failed_df)} 个配体对接失败"):
            for _, row in failed_df.iterrows():
                st.text(f"{row['name']}: {row.get('error', '未知错误')}")

    # ---- 下载 ----
    st.subheader("💾 结果导出")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = results_df[[c for c in [
            "name", "smiles", "best_score", "worst_score", "pose_count", "success", "error"
        ] if c in results_df.columns]].to_csv(index=False)
        st.download_button(
            "📥 下载结果表 (CSV)",
            csv_data,
            f"batch_docking_results_{result['pdb_id']}.csv",
            "text/csv",
            key="batch_dl_csv",
        )

    with col_dl2:
        import io, zipfile

        # 计算估计总大小
        total_bytes = sum(
            len(row["_sdf_bytes"]) for _, row in results_df.iterrows()
            if row["success"] and row.get("_sdf_bytes")
        )
        total_mb = total_bytes / (1024 * 1024)

        # 仅最佳 pose 选项
        best_only = st.checkbox(
            "仅导出最佳 pose (每个配体保留第1个构象)",
            value=(total_mb > 20),
            help="勾选后每种配体只保留打分最优的一个构象，显著减小文件大小",
            key="batch_zip_best_only",
        )

        # 体积预警
        estimate_mb = total_mb if not best_only else total_mb * 0.2
        if estimate_mb > 20:
            st.warning(
                f"⚠️ 预计 ZIP 大小约 **{estimate_mb:.0f} MB**，"
                f"下载可能较慢。建议勾选「仅导出最佳 pose」减小体积。"
            )

        # 打包
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in results_df.iterrows():
                if row["success"] and row.get("_sdf_bytes"):
                    if best_only:
                        # 仅保留最佳 pose：解析 SDF，只取第一个分子
                        sdf_text = row["_sdf_bytes"].decode("utf-8", errors="replace")
                        parts = [p.strip() for p in sdf_text.split("$$$$\n") if p.strip()]
                        if parts:
                            zf.writestr(f"{row['name']}_best_pose.sdf", parts[0] + "\n$$$$\n")
                    else:
                        zf.writestr(f"{row['name']}_poses.sdf", row["_sdf_bytes"])
        buf.seek(0)

        zip_label = "📦 下载最佳构象 (ZIP)" if best_only else "📦 下载所有构象 (ZIP)"
        st.download_button(
            zip_label,
            buf.getvalue(),
            f"batch_docking_poses_{result['pdb_id']}.zip",
            "application/zip",
            key="batch_dl_zip",
        )

    # KNIME 导出
    if results_df is not None and not results_df.empty:
        knime_export_section(
            results_df,
            title="批量对接结果",
            key_prefix="batch_dock_knime",
            metadata={
                "pdb_id": result.get("pdb_id", ""),
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
            },
        )


# ---------- 依赖检测 ----------

@st.cache_resource
def _check_smina() -> bool:
    """检查 Smina 命令行是否可用（缓存结果）"""
    try:
        result = subprocess.run(
            ["smina", "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@st.cache_resource
def _check_openbabel() -> bool:
    """检查 openbabel Python 绑定是否可用（缓存结果）"""
    try:
        from openbabel import pybel  # noqa: F811
        return True
    except ImportError:
        return False
