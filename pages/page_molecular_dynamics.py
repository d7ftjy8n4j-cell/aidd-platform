"""
pages/page_molecular_dynamics.py — 分子动力学模拟页面
基于 OpenMM 对蛋白-配体复合物进行 MD 模拟
参考: TeachOpenCADD T019

3 步标签页:
  Step 1: 输入设置（蛋白 + 配体 + 参数）
  Step 2: 执行与监控（异步模拟 + 实时进度）
  Step 3: 结果展示（轨迹分析 + 下载 + 下游衔接）
"""

import os
import sys
import tempfile
import subprocess
import json
import time
import logging
from pathlib import Path
import streamlit as st
import numpy as np

from md_utils import (
    _check_md_deps, _get_dep_errors,
    run_md_simulation, analyze_trajectory, save_merged_pdb,
    download_pdb, prepare_protein, prepare_ligand,
)

logger = logging.getLogger(__name__)

# ---- 常用配体 SMILES 模板 ----
LIGAND_TEMPLATES = {
    "03P (TAK-285, 3POZ)": {
        "resname": "03P",
        "smiles": "CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(c4)C(F)(F)F)c(Cl)c3)c12",
        "pdb": "3POZ",
    },
    "IRE (Gefitinib, 2ITY)": {
        "resname": "IRE",
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "pdb": "2ITY",
    },
    "AQ4 (Erlotinib, 1M17)": {
        "resname": "AQ4",
        "smiles": "CCOCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OC",
        "pdb": "1M17",
    },
}


def page_molecular_dynamics():
    """分子动力学模拟主页面 —— 3 步标签页"""

    st.title("⚛️ 分子动力学模拟 (MD)")
    st.caption("基于 OpenMM 对 EGFR 蛋白-配体复合物进行分子动力学模拟，观察原子运动与构象变化。")

    with st.popover("🎓 教学点"):
        st.markdown("""
        **分子动力学 (MD)** 通过牛顿力学模拟原子运动，帮助理解：
        - 蛋白-配体结合的**动态过程**（而非单一静态结构）
        - 构象变化与「**隐性结合口袋**」(cryptic binding sites) 的发现
        - **力场** (Force Field) 如何近似描述共价/非共价作用力
        - **周期性边界条件**如何消除有限盒子的人为边界效应

        > 📖 参考：TeachOpenCADD T019 · *J Med Chem* (2016), 59(9), 4035-4061
        """)

    # ==================== 依赖检查 ====================
    if not _check_md_deps():
        errors = _get_dep_errors()
        st.error("❌ MD 模拟所需依赖未安装")

        error_list = "\n".join([f"  - {name}" for name in errors])
        st.markdown(f"**缺失的包**：\n{error_list}")

        st.markdown("""
        **安装方法（本地 conda 环境）**：
        ```bash
        # 方式 1: 使用项目提供的专用环境文件
        conda env create -f environment_md.yml
        conda activate egfr-md

        # 方式 2: 手动安装
        conda install -c conda-forge openmm openmmforcefields openff-toolkit pdbfixer mdtraj
        ```

        > ⚠️ MD 模拟推荐使用 GPU 加速，且 **不适合 Streamlit Cloud 部署**。  
        > 云端用户仍可使用平台其他全部功能。
        """)
        return

    # ==================== 3 步标签页 ====================
    tab_input, tab_run, tab_results = st.tabs([
        "📝 Step 1: 输入设置",
        "⚡ Step 2: 执行与监控",
        "📊 Step 3: 结果展示",
    ])

    # ================================================================
    # Step 1: 输入设置
    # ================================================================
    with tab_input:
        st.subheader("🧬 蛋白-配体体系设置")

        # 快捷模板
        template_name = st.selectbox(
            "📋 快捷模板（自动填充下方参数）",
            ["自定义"] + list(LIGAND_TEMPLATES.keys()),
            help="选择经典 EGFR-抑制剂共晶结构模板",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🧬 蛋白结构")
            input_mode = st.radio(
                "输入方式", ["PDB ID", "上传 PDB 文件"],
                index=0, horizontal=True, key="md_input_mode",
            )
            if input_mode == "PDB ID":
                default_pdb = LIGAND_TEMPLATES[template_name]["pdb"] if template_name != "自定义" else "3POZ"
                pdb_id = st.text_input(
                    "PDB ID", value=default_pdb,
                    help="推荐 EGFR 结构: 3POZ (TAK-285), 2ITY (吉非替尼), 1M17 (埃罗替尼)",
                ).strip().upper()
                st.session_state["md_pdb_id"] = pdb_id
                st.session_state["md_pdb_content"] = None
            else:
                uploaded = st.file_uploader("上传 .pdb 文件", type=["pdb", "ent"], key="md_pdb_upload")
                if uploaded:
                    st.session_state["md_pdb_content"] = uploaded.read()
                    st.session_state["md_pdb_id"] = None
                    st.success(f"✅ 已加载: {uploaded.name}")
                else:
                    st.session_state["md_pdb_content"] = None
                    st.session_state["md_pdb_id"] = None

        with col2:
            st.markdown("#### 💊 配体设置")
            default_resname = LIGAND_TEMPLATES[template_name]["resname"] if template_name != "自定义" else "03P"
            default_smiles = LIGAND_TEMPLATES[template_name]["smiles"] if template_name != "自定义" else "CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(c4)C(F)(F)F)c(Cl)c3)c12"

            ligand_resname = st.text_input(
                "配体残基名（3字母）", value=default_resname,
                help="PDB 中配体的残基名。3POZ 的 TAK-285 残基名为 03P",
            ).strip()
            ligand_smiles = st.text_area(
                "配体 SMILES（键级修正）", value=default_smiles, height=70,
                help="从 PDB 网页获取的 Isomeric SMILES，用于修正配体键级和质子化状态",
            ).strip()
            st.session_state["md_ligand_resname"] = ligand_resname
            st.session_state["md_ligand_smiles"] = ligand_smiles

        # ---- 模拟参数 ----
        st.divider()
        st.subheader("⚙️ 模拟参数")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            total_steps = st.number_input(
                "模拟步数", min_value=100, max_value=500000,
                value=5000, step=1000, key="md_total_steps",
                help="每步 2 fs。5000 步 = 10 ps（教学演示），100k 步 = 200 ps",
            )
            sim_time_ps = total_steps * 0.002
            st.caption(f"≈ {sim_time_ps:.1f} ps 模拟时间")

        with col_b:
            write_interval = st.number_input(
                "轨迹保存间隔（步）", min_value=1, max_value=10000,
                value=max(1, total_steps // 50), step=100, key="md_write_interval",
            )
            n_frames = max(1, total_steps // write_interval)
            st.caption(f"≈ {n_frames} 帧轨迹")

        with col_c:
            temperature = st.slider(
                "温度 (K)", 100.0, 400.0, 300.0, 10.0,
                help="300 K ≈ 27°C（室温）",
                key="md_temperature",
            )

        with st.expander("🔧 高级参数"):
            cx, cy, cz = st.columns(3)
            with cx:
                padding = st.slider("水盒子 Padding (nm)", 0.5, 2.0, 1.0, 0.1, key="md_padding")
            with cy:
                ionic_strength = st.slider("离子强度 (M)", 0.0, 0.3, 0.15, 0.01, key="md_ionic")
            with cz:
                ph = st.slider("pH", 5.0, 9.0, 7.0, 0.5, key="md_ph")

        # 保存参数到 session
        for k, v in [("md_total_steps", total_steps), ("md_write_interval", write_interval),
                      ("md_temperature", temperature), ("md_padding", padding),
                      ("md_ionic", ionic_strength), ("md_ph", ph)]:
            st.session_state[k] = v

        # ---- 切换到 Step 2 ----
        can_run = (st.session_state.get("md_pdb_id") or st.session_state.get("md_pdb_content"))
        if can_run:
            if st.button("➡️ 进入模拟设置", type="primary", use_container_width=True):
                st.session_state["md_go_to_run"] = True
                st.rerun()
        else:
            st.warning("👈 请输入 PDB ID 或上传 PDB 文件")

    # ================================================================
    # Step 2: 执行与监控
    # ================================================================
    with tab_run:
        if not st.session_state.get("md_go_to_run"):
            st.info("👈 请先在 **Step 1** 中完成输入设置，然后点击「进入模拟设置」")
        else:
            st.subheader("⚡ 执行分子动力学模拟")

            pdb_id = st.session_state.get("md_pdb_id")
            pdb_content = st.session_state.get("md_pdb_content")
            ligand_resname = st.session_state.get("md_ligand_resname", "03P")
            ligand_smiles = st.session_state.get("md_ligand_smiles", "")
            total_steps = st.session_state.get("md_total_steps", 5000)
            write_interval = st.session_state.get("md_write_interval", 100)
            temperature = st.session_state.get("md_temperature", 300.0)
            padding = st.session_state.get("md_padding", 1.0)
            ionic_strength = st.session_state.get("md_ionic", 0.15)
            ph = st.session_state.get("md_ph", 7.0)

            # 显示配置摘要
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | 蛋白 | `{pdb_id or '上传文件'}` |
            | 配体残基 | `{ligand_resname}` |
            | 模拟步数 | {total_steps} ({total_steps * 0.002:.0f} ps) |
            | 温度 | {temperature} K |
            | 帧数 | ≈{max(1, total_steps // write_interval)} |
            """)

            # 异步执行：subprocess + 进度文件轮询
            if "md_process" not in st.session_state:
                st.session_state["md_process"] = None
                st.session_state["md_result"] = None
                st.session_state["md_error"] = None
                st.session_state["md_progress"] = 0.0
                st.session_state["md_status"] = ""
                st.session_state["md_analysis"] = None

            if st.button("🚀 开始分子动力学模拟", type="primary",
                         disabled=st.session_state["md_process"] is not None,
                         use_container_width=True):
                # 准备参数
                output_dir = tempfile.mkdtemp(prefix="md_worker_")
                worker_params = {
                    "pdb_id": pdb_id,
                    "pdb_content": None,  # 二进制不能通过 JSON 传；上传文件暂走内存
                    "ligand_resname": ligand_resname,
                    "ligand_smiles": ligand_smiles,
                    "total_steps": total_steps,
                    "write_interval": write_interval,
                    "temperature": temperature,
                    "padding": padding,
                    "ionic_strength": ionic_strength,
                    "ph": ph,
                    "output_dir": output_dir,
                }

                # 如果有上传的 PDB 内容，先写入临时文件
                if pdb_content:
                    pdb_tmp_path = os.path.join(output_dir, "input.pdb")
                    with open(pdb_tmp_path, "wb") as f:
                        f.write(pdb_content)
                    # 通知 worker 使用该文件（通过修改参数方式）
                    worker_params["pdb_content_path"] = pdb_tmp_path

                # 启动独立进程（绝对路径 + cwd 设为项目根目录）
                project_root = Path(__file__).resolve().parent.parent
                worker_script = project_root / "run_md_worker.py"
                cmd = [sys.executable, str(worker_script), json.dumps(worker_params)]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(project_root),
                    env=os.environ.copy(),  # 关键：继承 conda 环境变量（OpenMM .so 路径等）
                )
                # 写 PID 文件以便检测孤儿进程
                pid_file = os.path.join(output_dir, "worker.pid")
                with open(pid_file, "w") as pf:
                    pf.write(str(process.pid))
                st.session_state["md_process"] = process
                st.session_state["md_output_dir"] = output_dir
                st.session_state["md_result"] = None
                st.session_state["md_error"] = None
                st.session_state["md_progress"] = 0.0
                st.session_state["md_status"] = "⏳ 启动模拟进程..."
                st.session_state["md_analysis"] = None
                st.rerun()

            # 轮询进度
            process = st.session_state.get("md_process")
            if process is not None:
                output_dir = st.session_state.get("md_output_dir", "")
                progress_file = os.path.join(output_dir, "progress.txt")
                energy_csv = os.path.join(output_dir, "energy_log.csv")
                result_file = os.path.join(output_dir, "result.json")

                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()
                chart_placeholder = st.empty()

                # 取消按钮 + 进度条同行
                col_prog, col_cancel = st.columns([5, 1])
                cancel_clicked = col_cancel.button("⏹ 取消模拟", type="secondary",
                                                    use_container_width=True,
                                                    key=f"md_cancel_{id(process)}")

                if cancel_clicked:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                    st.session_state["md_process"] = None
                    st.session_state["md_status"] = "⏹ 模拟已取消"
                    st.session_state["md_error"] = "用户取消"
                    # 写取消标记供 worker 残留检测
                    stop_file = os.path.join(output_dir, "stop_signal.txt")
                    with open(stop_file, "w") as sf:
                        sf.write("cancelled")
                    st.warning("⏹ 模拟已取消。可调整参数后重新运行。")
                    time.sleep(1)
                    st.rerun()

                if process.poll() is None:
                    # 进程仍在运行，读取进度
                    pct = 0.0
                    if os.path.exists(progress_file):
                        with open(progress_file, "r") as f:
                            lines = f.read().strip().split("\n", 1)
                            try:
                                pct = float(lines[0])
                                st.session_state["md_progress"] = max(0.0, pct)
                            except ValueError:
                                pass
                            st.session_state["md_status"] = lines[1] if len(lines) > 1 else "运行中..."

                    progress_bar.progress(st.session_state["md_progress"])
                    status_placeholder.markdown(f"**{st.session_state['md_status']}**")

                    # 动态能量/温度折线图（防空文件读取）
                    if os.path.exists(energy_csv) and os.path.getsize(energy_csv) > 20:
                        try:
                            import pandas as pd
                            df = pd.read_csv(energy_csv)
                            if not df.empty and len(df) > 1:
                                chart_data = pd.DataFrame({
                                    "势能 (kJ/mol)": df["potential_energy_kjmol"].values,
                                    "温度 (K)": df["temperature_k"].values,
                                })
                                chart_placeholder.line_chart(chart_data, height=200)
                        except (pd.errors.EmptyDataError, Exception):
                            pass  # 首次空读取或列名不一致，静默跳过

                    time.sleep(2)
                    st.rerun()

                else:
                    # 进程已结束 → 先解析 result.json 判断状态
                    progress_bar.progress(1.0)
                    returncode = process.returncode

                    if os.path.exists(result_file):
                        with open(result_file, "r") as f:
                            result_data = json.load(f)

                        if result_data.get("status") == "error":
                            error_msg = result_data.get("error_message", "未知错误")
                            st.session_state["md_error"] = error_msg
                            st.session_state["md_status"] = f"❌ 模拟失败"
                            st.session_state["md_process"] = None
                            status_placeholder.error(f"❌ MD 模拟失败")
                            st.error(f"**错误详情**：{error_msg}")
                            st.markdown("""
                            **常见原因与建议**：
                            - 配体 SMILES 与 PDB 中残基不匹配 → 检查残基名和 SMILES
                            - PDBFixer 无法修复缺失残基 → 尝试更换 PDB ID
                            - 力场参数化失败 → 尝试减少模拟步数或调整 pH
                            """)
                            with st.expander("🔍 原始错误输出"):
                                stderr_output = process.stderr.read().decode("utf-8", errors="replace")
                                st.code(stderr_output[-2000:], language="text")
                            return  # 不再继续

                        if result_data.get("status") == "success":
                            st.session_state["md_result"] = result_data
                            st.session_state["md_analysis"] = result_data.get("analysis")
                            st.session_state["md_progress"] = 1.0
                            st.session_state["md_status"] = "✅ 模拟完成！"
                            st.session_state["md_process"] = None
                            status_placeholder.success("✅ 模拟完成！")
                            st.session_state["md_go_to_results"] = True
                            if st.button("➡️ 查看结果", type="primary"):
                                st.rerun()
                    else:
                        # result.json 不存在 → worker 崩溃
                        stderr_output = process.stderr.read().decode("utf-8", errors="replace")
                        st.session_state["md_error"] = stderr_output[-500:] or f"退出码: {returncode}"
                        st.session_state["md_status"] = "❌ 模拟进程异常退出"
                        st.session_state["md_process"] = None
                        status_placeholder.error("❌ 模拟进程异常退出（未生成结果文件）")
                        with st.expander("🔍 错误详情"):
                            st.code(stderr_output[-2000:] or "(无输出)", language="text")

    # ================================================================
    # Step 3: 结果展示
    # ================================================================
    with tab_results:
        result = st.session_state.get("md_result")
        analysis = st.session_state.get("md_analysis")

        if not result:
            st.info("👈 请先在 **Step 2** 中运行模拟")
        else:
            st.subheader("📊 模拟结果")

            # ---- 指标卡片 ----
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("总原子数", result.get("num_atoms", "N/A"))
            col_r2.metric("轨迹帧数", analysis.get("n_frames", "N/A") if analysis else "N/A")
            col_r3.metric("模拟步数", st.session_state.get("md_total_steps", "N/A"))
            col_r4.metric("温度", f"{st.session_state.get('md_temperature', 300)} K")

            # ---- 轨迹分析图表 ----
            if analysis:
                st.divider()
                st.subheader("📈 轨迹分析")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("**蛋白骨架 RMSD**")
                    if len(analysis["rmsd_protein"]) > 1:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(analysis["time_ps"], analysis["rmsd_protein"], color="#2196F3", linewidth=1.5)
                        ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="2.0 Å")
                        ax.set_xlabel("时间 (ps)")
                        ax.set_ylabel("RMSD (Å)")
                        ax.legend(fontsize=8)
                        ax.set_title("蛋白骨架 RMSD", fontsize=10)
                        fig.tight_layout()
                        st.pyplot(fig)
                        st.caption("RMSD 越大表示结构偏离初始构象越多。若曲线趋于平稳，说明体系已平衡。")
                    else:
                        st.info("帧数不足，无法绘制 RMSD 曲线")

                with chart_col2:
                    st.markdown("**配体重原子 RMSD**")
                    if len(analysis["rmsd_ligand"]) > 1:
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        ax2.plot(analysis["time_ps"], analysis["rmsd_ligand"], color="#FF9800", linewidth=1.5)
                        ax2.set_xlabel("时间 (ps)")
                        ax2.set_ylabel("配体 RMSD (Å)")
                        ax2.set_title("配体重原子 RMSD", fontsize=10)
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        st.caption("配体 RMSD 反映配体在结合口袋中的稳定性。")
                    else:
                        st.info("帧数不足")

                # RMSF 图
                if len(analysis["rmsf"]) > 0:
                    st.markdown("**残基柔性 (RMSF)**")
                    fig3, ax3 = plt.subplots(figsize=(8, 2.5))
                    ax3.bar(analysis["residue_ids"], analysis["rmsf"], color="#4CAF50", width=0.8)
                    ax3.set_xlabel("残基编号")
                    ax3.set_ylabel("RMSF (Å)")
                    ax3.set_title("蛋白残基 RMSF", fontsize=10)
                    fig3.tight_layout()
                    st.pyplot(fig3)
                    st.caption("RMSF 越高，残基越柔性。loop 区通常 RMSF 较高，α-螺旋/β-折叠较低。")

            # ---- 下载 ----
            st.divider()
            st.subheader("📥 下载结果")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                if os.path.exists(result.get("topology_pdb", "")):
                    with open(result["topology_pdb"], "r") as f:
                        st.download_button(
                            "📄 拓扑 PDB", data=f.read(),
                            file_name="md_topology.pdb", mime="chemical/x-pdb",
                            use_container_width=True,
                        )
            with col_dl2:
                if os.path.exists(result.get("trajectory_xtc", "")):
                    with open(result["trajectory_xtc"], "rb") as f:
                        st.download_button(
                            "🎬 轨迹 XTC", data=f.read(),
                            file_name="md_trajectory.xtc", mime="application/octet-stream",
                            use_container_width=True,
                        )
            with col_dl3:
                if os.path.exists(result.get("mean_pdb_path", "")):
                    with open(result["mean_pdb_path"], "r") as f:
                        st.download_button(
                            "📐 平均结构 PDB", data=f.read(),
                            file_name="md_mean_structure.pdb", mime="chemical/x-pdb",
                            use_container_width=True,
                        )

            # ---- 导出到蛋白-配体作用分析 ----
            st.divider()
            st.subheader("🔗 下游衔接")
            st.markdown("将 MD 平均结构发送到「**💊 蛋白-配体作用分析**」页面，查看动态过程中的关键相互作用。")

            if st.button("📤 导出平均结构到作用分析", use_container_width=True):
                mean_pdb = result.get("mean_pdb_path", "")
                if mean_pdb and os.path.exists(mean_pdb):
                    with open(mean_pdb, "r") as f:
                        st.session_state["md_ready_pdb_content"] = f.read()
                    st.session_state["md_ready_source"] = "⚛️ 分子动力学模拟"
                    st.success("✅ 已导出！请切换到「💊 蛋白-配体作用」标签页查看")
                else:
                    st.warning("平均结构不可用，请先完成模拟")

            # ---- 模拟日志 ----
            with st.expander("📋 模拟日志"):
                log = result.get("log", "")
                st.code(log[-3000:] if len(log) > 3000 else log, language="text")

            # ---- 教学延伸 ----
            with st.expander("💡 进阶学习：轨迹分析工具"):
                st.markdown("""
                | 工具 | 用途 | 环境 |
                |------|------|------|
                | **MDTraj** | RMSD/RMSF/氢键分析 (Python) | conda |
                | **MDAnalysis** | 更灵活的轨迹分析 (Python) | conda |
                | **VMD** | 轨迹可视化 + 渲染动画 | 桌面 |
                | **PyMOL** | 多帧 PDB 构象比对 | 桌面 |
                | **NGLview** | Jupyter 内交互式查看 | pip |

                **进阶分析指标**：
                - **RMSD**: 判断体系是否达到平衡（曲线趋于平坦）
                - **RMSF**: 找出蛋白灵活区域（潜在变构位点）
                - **氢键占有率**: 评估配体-蛋白关键相互作用的稳定性
                - **回转半径 (Rg)**: 评估蛋白整体构象紧凑程度
                - **PCA / 主成分分析**: 识别主要构象运动模式
                """)

