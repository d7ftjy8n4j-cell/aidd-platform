# pages/page_mmgbsa.py
"""
MM-GBSA 结合自由能估算页面

从分子动力学 (MD) 轨迹计算蛋白-配体结合自由能 (ΔG)。
使用 GB-OBC1 (igb=2) 隐式溶剂模型，支持选择多帧采样、结果可视化与下载。
"""

import json
import os

import pandas as pd
import streamlit as st

from utils.mmgbsa_utils import run_mmgbsa

CACHE_TTL = 86400  # 24 小时


# ---------- 依赖检测 ----------
def _check_mmgbsa_deps():
    """检测 MM-GBSA 必需依赖"""
    errors = []
    warns = []
    try:
        import mdtraj  # noqa: F401
    except ImportError:
        errors.append("mdtraj")
    try:
        import openmm  # noqa: F401
    except ImportError:
        errors.append("openmm")

    return errors, warns


# ---------- 页面主入口 ----------
def page_mmgbsa():
    """MM-GBSA 绑定自由能估算"""

    st.title("⚛️ MM-GBSA 结合自由能估算")

    st.markdown("""
    本工具使用 **MM-GBSA** (Molecular Mechanics / Generalized Born Surface Area) 方法，
    从已完成的**分子动力学 (MD) 轨迹**中估算蛋白-配体体系的结合自由能 (ΔG)。

    **核心公式**： `ΔG_bind ≈ G_complex - G_receptor - G_ligand`

    相比对接打分，MM-GBSA 更准确地考虑了蛋白-配体在溶剂中的动态行为。
    """)

    # ---------- 依赖检测 ----------
    errs, warns = _check_mmgbsa_deps()
    if errs:
        st.error(
            "❌ 缺少必需依赖。本功能仅限**本地 conda 环境**使用，"
            "Streamlit Cloud 不支持安装 OpenMM。\n\n"
            "**本地安装命令**：\n"
            "```bash\n"
            "conda install -c conda-forge openmm mdtraj numpy pandas\n"
            "```"
        )
        with st.expander("📋 查看缺失包详情"):
            st.write(f"缺失: {', '.join(errs)}")
        return
    if warns:
        for w in warns:
            st.warning(w)

    # ---------- 输入区域 ----------
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📂 输入文件")

        # 如果 md_output 已有完整数据，默认切换到缓存模式
        _has_cached = bool(
            "md_output" in st.session_state
            and st.session_state.md_output
            and st.session_state.md_output.get("topology_pdb")
            and st.session_state.md_output.get("trajectory_xtc")
        )
        default_mode = 1 if _has_cached else 0

        input_mode = st.radio(
            "输入方式",
            ["手动路径", "会话缓存的 MD 结果"],
            index=default_mode,
            horizontal=True,
            key="mmgbsa_input_mode",
            help="手动输入文件路径，或从刚完成的 MD 模拟会话中读取",
        )
        # 缓存不完整时回退到手动路径（保证 radio 显示与实际输入一致）
        if input_mode == "会话缓存的 MD 结果" and not _has_cached:
            input_mode = "手动路径"

        top_file = ""
        traj_file = ""

        if input_mode == "会话缓存的 MD 结果":
            if (
                "md_output" in st.session_state
                and st.session_state.md_output
            ):
                md_out = st.session_state.md_output
                top_file = md_out.get("topology_pdb", "")
                traj_file = md_out.get("trajectory_xtc", "")

                if top_file and traj_file:
                    st.success("✅ 检测到会话中的 MD 结果")
                    st.code(f"拓扑: {top_file}\n轨迹: {traj_file}", language="text")
                else:
                    st.warning("⚠️ 会话数据不完整，请手动输入路径")
                    input_mode = "手动路径"
            else:
                st.info(
                    "暂无缓存的 MD 结果。请先运行 [分子动力学模拟](/page_molecular_dynamics)"
                    " 或在下方手动输入文件路径。"
                )
                input_mode = "手动路径"

        if input_mode == "手动路径":
            top_file = st.text_input(
                "拓扑文件路径 (*.pdb)",
                placeholder="/path/to/complex_solvated.pdb",
                help="MD 模拟输出的拓扑/结构文件",
            )
            traj_file = st.text_input(
                "轨迹文件路径 (*.xtc / *.dcd)",
                placeholder="/path/to/trajectory.xtc",
                help="MD 模拟输出的轨迹文件",
            )

    with col_right:
        st.subheader("⚙️ 计算参数")

        ligand_resname = st.text_input(
            "配体残基名",
            value="03P",
            help="PDB 中配体的残基名（如 03P, LIG, UNK）",
        )

        n_frames = st.slider(
            "采样帧数",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="从轨迹中均匀采样的帧数。越多越准确，但耗时更长。",
        )

        temperature = st.number_input(
            "模拟温度 (K)",
            min_value=100.0,
            max_value=500.0,
            value=300.0,
            step=10.0,
            help="MD 模拟时使用的温度",
        )

    # ---------- 执行计算 ----------
    st.divider()

    btn_disabled = not (top_file and traj_file)

    if st.button(
        "🚀 开始 MM-GBSA 计算",
        type="primary",
        use_container_width=True,
        disabled=btn_disabled,
    ):
        if not os.path.exists(top_file):
            st.error(f"❌ 拓扑文件不存在: {top_file}")
            st.stop()
        if not os.path.exists(traj_file):
            st.error(f"❌ 轨迹文件不存在: {traj_file}")
            st.stop()

        with st.spinner("⏳ MM-GBSA 计算中... 每帧约需 5-15 秒，请耐心等待。"):
            try:
                import numpy as np

                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total, traj_idx):
                    pct = current / total
                    progress_bar.progress(pct)
                    status_text.text(
                        f"  第 {current}/{total} 帧  (轨迹索引 {traj_idx})"
                    )
                    return True

                result = run_mmgbsa(
                    topology_file=top_file,
                    trajectory_file=traj_file,
                    ligand_resname=ligand_resname,
                    n_frames=n_frames,
                    temperature=temperature,
                    progress_callback=progress_callback,
                )

                progress_bar.progress(1.0)
                status_text.text("✅ 计算完成")

            except ValueError as e:
                st.error(f"❌ 数据错误: {e}")
                st.info(
                    "💡 请检查配体残基名是否正确。常见配体残基名: 03P, LIG, UNK, MOL。"
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ 计算失败: {e}")
                st.stop()

        # 结果存入 session_state：后续交互（下载/调参）重跑时不丢失
        st.session_state["mmgbsa_last_result"] = {
            **result,
            "_top_file": top_file,
            "_traj_file": traj_file,
        }
        st.rerun()

    # ---------- 结果展示（基于 session_state 缓存，不依赖本次点击） ----------
    cached = st.session_state.get("mmgbsa_last_result")
    if cached is not None:
        result = cached
        top_file = result.get("_top_file", top_file)
        traj_file = result.get("_traj_file", traj_file)

        st.success(f"✅ MM-GBSA 计算完成！（{result['n_frames']} 帧）")

        # 1. 核心指标
        col1, col2, col3, col4 = st.columns(4)
        dg = result["delta_g_mean"]
        dg_std = result["delta_g_std"]

        col1.metric(
            "平均 ΔG_bind",
            f"{dg:.2f} kcal/mol",
            help="负值越大 → 结合越强",
        )
        col2.metric(
            "标准差",
            f"± {dg_std:.2f} kcal/mol",
        )
        col3.metric(
            "采样帧数",
            result["n_frames"],
        )
        col4.metric(
            "配体残基",
            result["ligand_resname"],
        )

        # 结合强度评价
        if dg < -12:
            st.info("🔵 **结合强度：非常强** — 配体与靶点有很高的亲和力")
        elif dg < -8:
            st.success("🟢 **结合强度：强** — 结合比较牢固")
        elif dg < -5:
            st.warning("🟡 **结合强度：中等**")
        else:
            st.error("🔴 **结合强度：弱** — 可能不是理想配体")

        # 2. 每帧 ΔG 表格
        with st.expander("📊 逐帧 ΔG 数据", expanded=True):
            df = pd.DataFrame({
                "帧号": range(1, len(result["delta_g_all"]) + 1),
                "ΔG (kcal/mol)": [round(v, 2) for v in result["delta_g_all"]],
                "G_complex": [round(v, 2) for v in result["g_complex_all"]],
                "G_receptor": [round(v, 2) for v in result["g_receptor_all"]],
                "G_ligand": [round(v, 2) for v in result["g_ligand_all"]],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        # 3. 可视化
        st.subheader("📈 ΔG 分布")

        c1, c2 = st.columns(2)

        with c1:
            st.caption("逐帧 ΔG 变化")
            st.line_chart(
                df.set_index("帧号")["ΔG (kcal/mol)"],
                y_label="ΔG (kcal/mol)",
            )

        with c2:
            st.caption("各帧能量分量对比")
            st.bar_chart(
                df.set_index("帧号")[
                    ["G_complex", "G_receptor", "G_ligand"]
                ],
                stack=False,
            )

        # 统计信息
        with st.expander("📈 统计摘要"):
            arr = np.array(result["delta_g_all"])
            stats_df = pd.DataFrame({
                "指标": [
                    "均值", "标准差", "最小值", "25% 分位",
                    "中位数", "75% 分位", "最大值",
                ],
                "ΔG (kcal/mol)": [
                    round(np.mean(arr), 2),
                    round(np.std(arr), 2),
                    round(np.min(arr), 2),
                    round(np.percentile(arr, 25), 2),
                    round(np.median(arr), 2),
                    round(np.percentile(arr, 75), 2),
                    round(np.max(arr), 2),
                ],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 4. 下载
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 下载逐帧数据 (CSV)",
                data=csv_data,
                file_name="mmgbsa_per_frame.csv",
                mime="text/csv",
            )
        with col_dl2:
            result_for_export = {
                "delta_g_mean": result["delta_g_mean"],
                "delta_g_std": result["delta_g_std"],
                "n_frames": result["n_frames"],
                "ligand_resname": result["ligand_resname"],
                "topology_file": top_file,
                "trajectory_file": traj_file,
            }
            st.download_button(
                label="📥 下载结果摘要 (JSON)",
                data=json.dumps(result_for_export, indent=2, ensure_ascii=False),
                file_name="mmgbsa_summary.json",
                mime="application/json",
            )

    # ---------- 空状态提示（仅当没有缓存结果时） ----------
    else:
        st.info(
            "👈 请在左侧输入 MD 轨迹文件路径并设置参数后点击计算。"
            "\n\n"
            "**前置条件**：需要先完成 MD 模拟。你可以在 [分子动力学模拟](/page_molecular_dynamics) 页面生成轨迹。"
        )

    # ---------- 原理说明 ----------
    with st.expander("📚 MM-GBSA 原理"):
        st.markdown("""
        ### MM-GBSA 方法

        **MM-GBSA** (Molecular Mechanics / Generalized Born Surface Area) 是一种
        **末端点自由能**估算方法。

        #### 计算公式

        ```
        ΔG_bind = 〈G_complex - G_receptor - G_ligand〉
        ```

        对轨迹中的每一帧：
        1. 移除显式水和离子
        2. 用 GB (Generalized Born) 隐式溶剂模型替换
        3. 计算 3 个系统的势能：
           - **复合物** (蛋白 + 配体)
           - **受体** (仅蛋白)
           - **配体** (仅配体)
        4. 代入公式得到该帧的 ΔG

        #### 本实现的特点

        | 特性 | 说明 |
        |------|------|
        | 隐式溶剂 | GB-OBC1 (igb=2) |
        | 力场 | Amber14 + tip3p |
        | 采样 | 均匀采样 N 帧 |
        | 单轨迹 | 从同一轨迹中取复合物、受体、配体坐标 |
        | 非键截断 | 无周期性 (CutoffNonPeriodic) |

        #### 局限性

        - **忽略熵变**：未包含构象熵 (ΔS)，因此是"焓近似"
        - **忽略 SA 项**：未显式计算溶剂可及表面积 (SASA)
        - **力场精度**：依赖于 Amber 力场参数化质量
        - **采样充分性**：需要足够的轨迹时长和采样帧数

        #### 参考文献

        - Genheden & Ryde (2015) *Expert Opin. Drug Discov.*
        - Miller et al. (2012) *J. Chem. Theory Comput.*
        """)
