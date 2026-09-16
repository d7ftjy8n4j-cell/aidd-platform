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
import shutil
from pathlib import Path
import streamlit as st
import numpy as np

from md_utils import (
    _check_md_deps, _get_dep_errors,
    run_md_simulation, analyze_trajectory, save_merged_pdb,
    download_pdb, prepare_protein, prepare_ligand,
)

logger = logging.getLogger(__name__)

# ---- 持久化缓存 ----
def _save_result_cache(cache_file, result_data):
    """保存 MD 结果到持久化文件"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "result": {k: v for k, v in result_data.items()
                           if not k.startswith("_") and k != "log"},
                "analysis": result_data.get("analysis"),
            }, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass


def _clear_result_cache(cache_file):
    """清除持久化缓存"""
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except Exception:
        pass

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

    # ---- 持久化结果目录 ----
    _MD_CACHE_DIR = os.path.join(tempfile.gettempdir(), "egfr_md_cache")
    os.makedirs(_MD_CACHE_DIR, exist_ok=True)
    _RESULT_CACHE_FILE = os.path.join(_MD_CACHE_DIR, "last_result.json")
    _WORKER_LOCK_FILE = os.path.join(_MD_CACHE_DIR, "worker.lock")

    # ---- 初始化 session_state ----
    for key, default in [("md_go_to_run", False),
                          ("md_pdb_content", None),
                          ("md_output_dir", ""),
                          ("md_process", None),
                          ("md_result", None),
                          ("md_error", None),
                          ("md_progress", 0.0),
                          ("md_status", ""),
                          ("md_analysis", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ====== 刷新恢复：从 worker.lock 重连正在运行的模拟 ======
    if (st.session_state.get("md_process") is None
            and st.session_state.get("md_result") is None
            and os.path.exists(_WORKER_LOCK_FILE)):
        try:
            with open(_WORKER_LOCK_FILE, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            saved_dir = lock_data.get("output_dir", "")
            saved_pid = lock_data.get("pid", 0)
            # 检查进程是否还活着（仅 Windows 可用 OpenProcess；其他平台按 PID 存活判断）
            alive = False
            if saved_pid and saved_dir and os.path.isdir(saved_dir):
                if os.name == "nt":
                    try:
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        handle = kernel32.OpenProcess(0x0400, False, saved_pid)  # PROCESS_QUERY_INFORMATION
                        if handle:
                            kernel32.CloseHandle(handle)
                            alive = True
                    except Exception:
                        pass  # 非 Windows 或进程不存在
                else:
                    try:
                        os.kill(saved_pid, 0)  # 仅探测是否存在
                        alive = True
                    except (OSError, ProcessLookupError):
                        alive = False
            if alive:
                # 进程还在跑 → 恢复轮询状态
                st.session_state["md_output_dir"] = saved_dir
                st.session_state["md_process"] = "restored"  # 标记为非 None，触发轮询分支
                st.session_state["md_progress"] = 0.0
                st.session_state["md_status"] = "🔄 页面刷新，重连模拟进程..."
            else:
                # 进程已死 → 尝试从 result.json 恢复结果
                rf = os.path.join(saved_dir, "result.json")
                if os.path.exists(rf):
                    try:
                        with open(rf, "r", encoding="utf-8") as f:
                            worker_result = json.load(f)
                        if worker_result.get("status") == "success":
                            st.session_state["md_result"] = worker_result
                            st.session_state["md_analysis"] = worker_result.get("analysis")
                            st.session_state["md_progress"] = 1.0
                            st.session_state["md_status"] = "✅ 模拟完成"
                            _save_result_cache(_RESULT_CACHE_FILE, worker_result)
                    except Exception:
                        pass
                # 清理锁文件
                try:
                    os.remove(_WORKER_LOCK_FILE)
                except Exception:
                    pass
        except Exception:
            pass

    # ====== 从持久化结果缓存恢复（仅在无活动模拟时）======
    if (st.session_state.get("md_result") is None
            and st.session_state.get("md_process") is None
            and os.path.exists(_RESULT_CACHE_FILE)):
        try:
            with open(_RESULT_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            st.session_state["md_result"] = cached.get("result")
            st.session_state["md_analysis"] = cached.get("analysis")
            st.session_state["md_progress"] = 1.0
            st.session_state["md_status"] = "✅ 模拟完成（从缓存恢复）"
        except Exception:
            pass

    # ====== 从 worker 临时目录恢复结果（兜底）======
    if (st.session_state.get("md_result") is None
            and st.session_state.get("md_process") is None):
        import glob
        worker_dirs = sorted(glob.glob(os.path.join(tempfile.gettempdir(), "md_worker_*")),
                             key=os.path.getmtime, reverse=True)
        for wd in worker_dirs[:3]:
            rf = os.path.join(wd, "result.json")
            if os.path.exists(rf):
                try:
                    with open(rf, "r") as f:
                        worker_result = json.load(f)
                    if worker_result.get("status") == "success":
                        st.session_state["md_result"] = worker_result
                        st.session_state["md_analysis"] = worker_result.get("analysis")
                        st.session_state["md_progress"] = 1.0
                        st.session_state["md_status"] = "✅ 模拟完成（自动恢复）"
                        st.session_state["md_process"] = None
                        # 同时保存到缓存
                        _save_result_cache(_RESULT_CACHE_FILE, worker_result)
                        break
                except Exception:
                    pass

    st.title("⚛️ 分子动力学模拟 (MD)")
    st.caption("基于 OpenMM 对蛋白-配体复合物（示例：EGFR 体系）进行分子动力学模拟，观察原子运动与构象变化。")

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
        st.warning("⚠️ 当前环境里的 MD 依赖没有通过完整导入检查，页面将以降级模式运行；你可以继续试用基础输入和结果展示。")
        if errors:
            st.caption("检测到的问题：")
            st.code("\n".join([f"- {name}: {msg}" for name, msg in errors.items() if msg]), language="text")
        st.info("如需完整 MD 运行，请在本地 conda 环境中安装：conda install -c conda-forge openmm openmmforcefields openff-toolkit pdbfixer mdtraj")

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
            help="选择经典激酶共晶结构模板（示例：EGFR）",
            key="md_template_select",
        )

        # 模板默认值
        tpl = LIGAND_TEMPLATES.get(template_name, {})
        default_pdb = tpl.get("pdb", "3POZ")
        default_resname = tpl.get("resname", "03P")
        default_smiles = tpl.get("smiles", "CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(c4)C(F)(F)F)c(Cl)c3)c12")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🧬 蛋白结构")
            input_mode = st.radio(
                "输入方式", ["PDB ID", "上传 PDB 文件"],
                index=0, horizontal=True, key="md_input_mode",
            )

        with col2:
            st.markdown("#### 💊 配体设置")
            ligand_resname = st.text_input(
                "配体残基名（3字母）", value=default_resname,
                help="PDB 中配体的残基名。3POZ 的 TAK-285 残基名为 03P",
                key="md_ligand_resname",
            ).strip()
            ligand_smiles = st.text_area(
                "配体 SMILES（键级修正）", value=default_smiles, height=70,
                help="从 PDB 网页获取的 Isomeric SMILES，用于修正配体键级和质子化状态",
                key="md_ligand_smiles",
            ).strip()

        # 蛋白 PDB 输入（放在 radio 下方以正确渲染）
        pdb_id = None
        if input_mode == "PDB ID":
            pdb_id = st.text_input(
                "PDB ID", value=default_pdb,
                help="推荐示例结构（EGFR 体系）: 3POZ (TAK-285), 2ITY (吉非替尼), 1M17 (埃罗替尼)",
                key="md_pdb_id_input",
            ).strip().upper()
        else:
            uploaded = st.file_uploader("上传 .pdb 文件", type=["pdb", "ent"], key="md_pdb_uploader")
            if uploaded:
                st.session_state["md_pdb_content"] = uploaded.read()
                st.success(f"✅ 已加载: {uploaded.name}")
            elif st.session_state.get("md_pdb_content") is not None:
                # 用户用 X 移除了文件 → 同步清除缓存，避免用过期内容启动模拟
                st.session_state["md_pdb_content"] = None

        # ---- 模拟参数 ----
        st.divider()
        st.subheader("⚙️ 模拟参数")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            total_steps = st.number_input(
                "模拟步数", min_value=100, max_value=500000,
                value=5000, step=1000, key="md_total_steps",
                help="每步 2 fs（LangevinMiddleIntegrator）。5000 步 = 10 ps，25000 步 = 50 ps",
            )
            sim_time_ps = total_steps * 0.002  # 2 fs per step
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

        # ---- 切换到 Step 2 ----
        pdb_input = st.session_state.get("md_pdb_id_input", "")
        can_run = bool(pdb_input) or bool(st.session_state.get("md_pdb_content"))
        if can_run:
            if st.button("➡️ 进入模拟设置", type="primary", width="stretch"):
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

            pdb_id = st.session_state.get("md_pdb_id_input", "")
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
            st.info("💡 当前为纯蛋白 MD 模式（CHARMM36 力场），配体参数化将在后续版本支持。")
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | 蛋白 | `{pdb_id or '上传文件'}` |
            | 力场 | CHARMM36 + TIP3P |
            | 模拟步数 | {total_steps} ({total_steps * 0.002:.0f} ps) |
            | 温度 | {temperature} K |
            | 帧数 | ≈{max(1, total_steps // write_interval)} |
            """)

            if st.button("🚀 开始分子动力学模拟", type="primary",
                         disabled=st.session_state["md_process"] is not None,
                         width="stretch"):
                # 清除旧缓存
                _clear_result_cache(_RESULT_CACHE_FILE)
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
                with open(pid_file, "w", encoding="utf-8") as pf:
                    pf.write(str(process.pid))
                st.session_state["md_process"] = process
                st.session_state["md_output_dir"] = output_dir
                st.session_state["md_result"] = None
                st.session_state["md_error"] = None
                st.session_state["md_progress"] = 0.0
                st.session_state["md_status"] = "⏳ 启动模拟进程..."
                st.session_state["md_analysis"] = None
                # 写持久化锁文件（供刷新恢复）
                try:
                    with open(_WORKER_LOCK_FILE, "w") as lf:
                        json.dump({"output_dir": output_dir, "pid": process.pid}, lf)
                except Exception:
                    pass
                st.rerun()

            # 轮询进度
            process = st.session_state.get("md_process")
            if process is not None:
                output_dir = st.session_state.get("md_output_dir", "")
                progress_file = os.path.join(output_dir, "progress.txt")
                energy_csv = os.path.join(output_dir, "energy_log.csv")
                result_file = os.path.join(output_dir, "result.json")

                # 刷新恢复模式：无真实 Popen 对象，从文件轮询
                is_restored = process == "restored"

                progress_bar = st.progress(st.session_state.get("md_progress", 0.0))
                status_placeholder = st.empty()
                chart_placeholder = st.empty()

                # 取消按钮
                if not is_restored:
                    col_prog, col_cancel = st.columns([5, 1])
                    cancel_clicked = col_cancel.button("⏹ 取消模拟", type="secondary",
                                                        width="stretch",
                                                        key=f"md_cancel_{id(process)}")
                else:
                    cancel_clicked = st.button("⏹ 取消模拟", type="secondary",
                                                width="stretch",
                                                key="md_cancel_restored")

                if cancel_clicked:
                    if not is_restored:
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except Exception:
                            try:
                                process.kill()
                            except Exception:
                                pass
                    else:
                        # 恢复模式下杀 PID
                        try:
                            with open(_WORKER_LOCK_FILE, "r") as lf:
                                lock_data = json.load(lf)
                            pid = lock_data.get("pid", 0)
                            if pid:
                                try:
                                    import signal
                                    os.kill(pid, signal.SIGTERM)
                                except Exception:
                                    pass  # 可能已经死了
                        except Exception:
                            pass
                    st.session_state["md_process"] = None
                    st.session_state["md_status"] = "⏹ 模拟已取消"
                    st.session_state["md_error"] = "用户取消"
                    _clear_result_cache(_RESULT_CACHE_FILE)
                    try:
                        os.remove(_WORKER_LOCK_FILE)
                    except Exception:
                        pass
                    st.warning("⏹ 模拟已取消。可调整参数后重新运行。")
                    time.sleep(1)
                    st.rerun()

                # 读取进度和状态
                running = False
                if is_restored:
                    # 恢复模式：检查进程是否还活着
                    try:
                        with open(_WORKER_LOCK_FILE, "r") as lf:
                            lock_data = json.load(lf)
                        pid = lock_data.get("pid", 0)
                        if pid:
                            import ctypes
                            kernel32 = ctypes.windll.kernel32
                            handle = kernel32.OpenProcess(0x0400, False, pid)
                            if handle:
                                kernel32.CloseHandle(handle)
                                running = True
                    except Exception:
                        running = os.path.exists(progress_file)
                else:
                    running = process.poll() is None

                if running:
                    # 进程仍在运行，读取进度
                    pct = st.session_state.get("md_progress", 0.0)
                    if os.path.exists(progress_file):
                        try:
                            with open(progress_file, "r", encoding="utf-8") as f:
                                lines = f.read().strip().split("\n", 1)
                            pct = float(lines[0])
                            st.session_state["md_progress"] = max(0.0, pct)
                            st.session_state["md_status"] = lines[1] if len(lines) > 1 else "运行中..."
                        except (ValueError, IndexError):
                            pass

                    progress_bar.progress(st.session_state["md_progress"])
                    status_placeholder.markdown(f"**{st.session_state['md_status']}**")

                    # 能量/温度折线图
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
                        except Exception:
                            pass

                    time.sleep(2)
                    st.rerun()

                else:
                    # 进程已结束 → 清理锁文件，解析结果
                    try:
                        os.remove(_WORKER_LOCK_FILE)
                    except Exception:
                        pass
                    progress_bar.progress(1.0)

                    if os.path.exists(result_file):
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                result_data = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            # worker 可能被 kill 导致 result.json 写了一半（截断）
                            st.session_state["md_error"] = "结果文件损坏（worker 可能被强制终止）"
                            st.session_state["md_status"] = "❌ 模拟进程异常退出"
                            st.session_state["md_process"] = None
                            _clear_result_cache(_RESULT_CACHE_FILE)
                            status_placeholder.error("❌ 模拟进程异常退出（结果文件不完整）")
                            try:
                                os.remove(_WORKER_LOCK_FILE)
                            except Exception:
                                pass
                            return

                        if result_data.get("status") == "error":
                            error_msg = result_data.get("error_message", "未知错误")
                            st.session_state["md_error"] = error_msg
                            st.session_state["md_status"] = "❌ 模拟失败"
                            st.session_state["md_process"] = None
                            _clear_result_cache(_RESULT_CACHE_FILE)
                            status_placeholder.error("❌ MD 模拟失败")
                            st.error(f"**错误详情**：{error_msg}")
                            with st.expander("🔍 原始错误输出"):
                                try:
                                    stderr_output = process.stderr.read().decode("utf-8", errors="replace") if not is_restored else ""
                                    st.code(stderr_output[-2000:] or "(无输出)", language="text")
                                except Exception:
                                    st.code("(无法读取错误输出)", language="text")
                            return

                        if result_data.get("status") == "success":
                            st.session_state["md_result"] = result_data
                            st.session_state["md_analysis"] = result_data.get("analysis")
                            st.session_state["md_progress"] = 1.0
                            st.session_state["md_status"] = "✅ 模拟完成！"
                            st.session_state["md_process"] = None
                            _save_result_cache(_RESULT_CACHE_FILE, result_data)
                            status_placeholder.success("✅ 模拟完成！")
                            st.session_state["md_go_to_results"] = True
                            st.success("✅ 模拟已完成，请切换到 **Step 3** 标签页查看结果与分析。")
                    else:
                        st.session_state["md_error"] = "worker 崩溃（无结果文件）"
                        st.session_state["md_status"] = "❌ 模拟进程异常退出"
                        st.session_state["md_process"] = None
                        _clear_result_cache(_RESULT_CACHE_FILE)
                        status_placeholder.error("❌ 模拟进程异常退出（未生成结果文件）")

    # ================================================================
    # Step 3: 结果展示
    # ================================================================
    with tab_results:
        result = st.session_state.get("md_result")
        analysis = st.session_state.get("md_analysis")

        if not result:
            st.info("👈 请先在 **Step 2** 中运行模拟")
        else:
            # 🔗 自动同步到 MM-GBSA / 作用分析页面
            if st.session_state.get("md_output") is None:
                st.session_state["md_output"] = {
                    "topology_pdb": result.get("topology_pdb", ""),
                    "trajectory_xtc": result.get("trajectory_xtc", ""),
                    "mean_pdb_path": result.get("mean_pdb_path", ""),
                    "num_atoms": result.get("num_atoms", 0),
                    "platform": result.get("platform", ""),
                }
            st.subheader("📊 模拟结果")

            # ---- 指标卡片 ----
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("总原子数", result.get("num_atoms", "N/A"))
            col_r2.metric("轨迹帧数", analysis.get("n_frames", "N/A") if analysis else "N/A")
            col_r3.metric("模拟步数", st.session_state.get("md_total_steps", "N/A"))
            col_r4.metric("温度", f"{st.session_state.get('md_temperature', 300)} K")
            # 显示使用的计算平台（GPU/CPU）
            platform_info = result.get("platform", "未知")
            st.caption(f"🖥️ 计算平台: {platform_info}")

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
                    with open(result["topology_pdb"], "r", encoding="utf-8") as f:
                        st.download_button(
                            "📄 拓扑 PDB", data=f.read(),
                            file_name="md_topology.pdb", mime="chemical/x-pdb",
                            width="stretch",
                        )
            with col_dl2:
                if os.path.exists(result.get("trajectory_xtc", "")):
                    with open(result["trajectory_xtc"], "rb") as f:
                        st.download_button(
                            "🎬 轨迹 XTC", data=f.read(),
                            file_name="md_trajectory.xtc", mime="application/octet-stream",
                            width="stretch",
                        )
            with col_dl3:
                if os.path.exists(result.get("mean_pdb_path", "")):
                    with open(result["mean_pdb_path"], "r") as f:
                        st.download_button(
                            "📐 平均结构 PDB", data=f.read(),
                            file_name="md_mean_structure.pdb", mime="chemical/x-pdb",
                            width="stretch",
                        )

            # ---- 清除结果 ----
            st.divider()
            if st.button("🗑️ 清除模拟结果", type="secondary", width="stretch"):
                # 同时删除 worker 输出目录，否则刷新后兜底恢复会重新找回旧结果
                try:
                    import glob
                    out_dir = st.session_state.get("md_output_dir", "")
                    if out_dir and os.path.isdir(out_dir):
                        shutil.rmtree(out_dir, ignore_errors=True)
                    else:
                        for wd in glob.glob(os.path.join(tempfile.gettempdir(), "md_worker_*")):
                            shutil.rmtree(wd, ignore_errors=True)
                except Exception:
                    pass
                _clear_result_cache(_RESULT_CACHE_FILE)
                for key in ["md_result", "md_analysis", "md_process", "md_error",
                            "md_progress", "md_status", "md_go_to_run", "md_go_to_results",
                            "md_output_dir", "md_output"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

            # ---- 导出到蛋白-配体作用分析 ----
            st.divider()
            st.subheader("🔗 下游衔接")
            st.markdown("将 MD 平均结构发送到「**💊 蛋白-配体作用分析**」页面，查看动态过程中的关键相互作用。")

            if st.button("📤 导出平均结构到作用分析", width="stretch"):
                mean_pdb = result.get("mean_pdb_path", "")
                if mean_pdb and os.path.exists(mean_pdb):
                    with open(mean_pdb, "r", encoding="utf-8") as f:
                        st.session_state["md_ready_pdb_content"] = f.read()
                    st.session_state["md_ready_source"] = "⚛️ 分子动力学模拟"
                    # 同时写入 MM-GBSA 需要的 md_output
                    st.session_state["md_output"] = {
                        "topology_pdb": result.get("topology_pdb", ""),
                        "trajectory_xtc": result.get("trajectory_xtc", ""),
                        "mean_pdb_path": mean_pdb,
                        "num_atoms": result.get("num_atoms", 0),
                        "platform": result.get("platform", ""),
                    }
                    st.success("✅ 已导出到「💊 蛋白-配体作用」和「⚛️ MM-GBSA」页面")
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

