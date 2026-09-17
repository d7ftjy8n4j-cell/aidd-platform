# pages/molecular_docking.py
"""
分子对接与虚拟筛选页面 (Smina)
用户输入靶点蛋白 + 配体 SMILES，预测结合构象和亲和力。
"""

import subprocess
import sys
import pandas as pd
import streamlit as st

from components.knime_export import knime_export_section


def page_molecular_docking():
    """分子对接与虚拟筛选主函数"""

    # ---------- 检查 Smina 是否可用 ----------
    smina_available = _check_smina()

    # ---------- 页面头部 ----------
    st.title("🔗 分子对接与虚拟筛选 (Smina)")

    st.markdown("""
    **功能说明**：使用 Smina（AutoDock Vina 分支）预测小分子在靶点蛋白中的结合构象和亲和力。

    - **输入**：靶点蛋白（PDB ID 或 上传 PDB 文件）+ 配体分子（SMILES）
    - **输出**：Top-N 结合构象、结合能 (kcal/mol)、RMSD、3D 可视化

    ---
    **典型示例**：EGFR 激酶 (PDB: 2ITO) + Gefitinib (SMILES)
    """)

    if smina_available:
        try:
            from docking_utils import find_smina_executable

            _smina_path = find_smina_executable()
            if _smina_path:
                st.caption(f"✅ Smina 已就绪：`{_smina_path}`")
        except Exception:
            pass

    if not smina_available:
        st.error("❌ 未检测到 Smina 可执行文件（命令行工具）")
        st.markdown(
            """
            **真实原因（不是 Python 依赖冲突）**

            Smina 是用 C++ 编译的**原生可执行文件**（AutoDock Vina 的分支）。
            PyPI 上**并不存在 `smina` 这个包** —— `pip install smina` 会直接返回 404，
            所以它无法写进 `requirements.txt`，只能靠下面两条路装：

            | 途径 | 做法 | 适用场景 |
            |------|------|----------|
            | conda-forge | `conda install -c conda-forge smina` | 本地开发（本项目 `egfr-md` 环境走这条） |
            | 官方静态二进制 | 下载 [smina.static](https://sourceforge.net/projects/smina/) 放到 `/usr/local/bin/smina` 并 `chmod +x` | Linux / Docker 镜像 |

            **为什么 Streamlit Community Cloud 上必然没有**：云端构建只做两件事——
            `apt` 装 `packages.txt` 的系统包、`pip` 装 `requirements.txt` 的 Python 包；
            Debian 仓库里没有 smina，PyPI 里也没有，因此部署出来的容器天然缺这个二进制，
            页面只能提示未检测到。**这是环境限制，不是库版本/依赖冲突。**

            **三条可行路线**：
            1. **本地运行（推荐）**：双击 `启动药尘光.bat`（自动激活 `egfr-md`，smina 已装在
               `...\\envs\\egfr-md\\Library\\bin\\smina.exe`）；
            2. **Docker 部署**：用仓库 `Dockerfile`（已内置 smina 静态二进制下载步骤）构建镜像后部署；
            3. 只在云端体验其它功能：对接页会自动降级提示，其余页面不受影响。
            """
        )
        if sys.platform == "win32":
            st.info(
                "💡 若你是在 VS Code / PyCharm 里直接点运行（没经过 `conda activate egfr-md`），"
                "PATH 里不会包含该环境的 `Library\\bin`，即使 smina 已安装也会检测不到——"
                "请改用 `启动药尘光.bat` 启动。"
            )
        st.caption("详见 README → 常见问题 → 为什么分子对接页面提示未检测到 Smina。")
        return

    # ---------- 输入区域 ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 靶点蛋白")
        input_type = st.radio(
            "选择输入方式",
            ["PDB ID", "上传PDB文件"],
            index=0,
            key="docking_input_type",
        )

        pdb_id = None
        pdb_content = None

        if input_type == "PDB ID":
            pdb_id = st.text_input(
                "输入 PDB ID（如 2ITO, 3POZ）",
                "2ITO",
                help="4 位 PDB 代码，蛋白质需包含共晶配体以确定结合口袋",
                key="docking_pdb_id",
            )
        else:
            uploaded_file = st.file_uploader(
                "上传 PDB 文件",
                type=["pdb", "ent"],
                key="docking_pdb_upload",
            )
            if uploaded_file:
                pdb_content = uploaded_file.read()
                st.success(f"已加载: {uploaded_file.name}")

    with col2:
        st.subheader("💊 配体分子")
        default_smiles = (
            "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
        )
        ligand_smiles = st.text_area(
            "输入 SMILES",
            default_smiles,
            height=100,
            help="配体分子的 SMILES 字符串（默认示例为 Gefitinib）",
            key="docking_ligand_smiles",
        )

        # 常用分子快捷选择
        with st.expander("📋 常用分子模板"):
            templates = {
                "Gefitinib (易瑞沙)": (
                    "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
                ),
                "Erlotinib (特罗凯)": (
                    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
                ),
                "Lapatinib (泰克博)": (
                    "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2F)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC=CC=N5"
                ),
                "Osimertinib (泰瑞沙)": (
                    "CN1CCN(CC1)C2=CC(=C(C=C2)NC3=NC=C(C(=N3)NC4=C(C=C(C=C4)N(C)C(=O)C=C)N5C=CC(=O)N5)C)OC"
                ),
            }
            cols_t = st.columns(4)
            for idx, (name, smi) in enumerate(templates.items()):
                with cols_t[idx]:
                    if st.button(name.split("(")[0].strip(), key=f"tpl_{idx}"):
                        st.session_state["docking_ligand_smiles"] = smi
                        st.rerun()

        # 高级参数
        with st.expander("⚙️ 对接参数"):
            num_poses = st.slider(
                "生成构象数 (num_modes)",
                5, 50, 10,
                key="docking_num_poses",
            )
            exhaustiveness = st.slider(
                "搜索精度 (exhaustiveness)",
                4, 32, 8,
                help="值越高搜索越彻底，耗时也越长（4 快速, 8 默认, 16 精细, 32 极精细）",
                key="docking_exhaustiveness",
            )
            ligand_resname = st.text_input(
                "配体残基名（可选）",
                "",
                help="PDB 中配体的残基名（如 IRE, GEF）。留空则自动检测",
                key="docking_ligand_resname",
            )
            buffer_size = st.slider(
                "口袋缓冲距离 (Å)",
                2.0, 10.0, 5.0, 0.5,
                help="在共晶配体周围扩展的搜索空间",
                key="docking_buffer",
            )

    # ---------- 执行对接 ----------
    st.divider()

    if st.button("🚀 开始分子对接", type="primary", key="docking_run_btn"):
        # 输入校验
        if not pdb_id and not pdb_content:
            st.error("请提供 PDB ID 或上传 PDB 文件")
            st.stop()
        if not ligand_smiles or not ligand_smiles.strip():
            st.error("请输入配体 SMILES")
            st.stop()

        with st.spinner(
            f"⏳ 正在执行分子对接（精度={exhaustiveness}，预计 1~5 分钟）..."
        ):
            try:
                from docking_utils import run_docking

                result = run_docking(
                    pdb_id=pdb_id,
                    pdb_content=pdb_content,
                    ligand_smiles=ligand_smiles.strip(),
                    ligand_resname=ligand_resname.strip() if ligand_resname.strip() else None,
                    num_poses=num_poses,
                    exhaustiveness=exhaustiveness,
                    buffer=buffer_size,
                )

                # ---------- 显示结果 ----------
                st.success("✅ 对接完成！")

                # 1. 结果表格
                st.subheader("📊 对接结果")
                if result["results"]:
                    df = pd.DataFrame(result["results"])
                    df.columns = ["构象 #", "结合能 (kcal/mol)", "RMSD l.b.", "RMSD u.b."]
                    st.dataframe(df, width="stretch", hide_index=True)

                    # 结合能分布图
                    st.subheader("📈 结合能分布")
                    chart_data = pd.DataFrame({
                        "构象": [f"#{r['mode']}" for r in result["results"]],
                        "结合能 (kcal/mol)": [r["affinity"] for r in result["results"]],
                    })
                    st.bar_chart(
                        chart_data.set_index("构象"),
                        width="stretch",
                        color="#4CAF50",
                    )
                else:
                    st.warning("⚠️ 未提取到对接结果，请检查原始输出")

                # 2. 最佳构象卡片
                if result["results"]:
                    best = result["results"][0]
                    worst = result["results"][-1]
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric(
                        "最佳结合能",
                        f"{best['affinity']:.2f} kcal/mol",
                        border=True,
                    )
                    col_b.metric(
                        "最佳 RMSD (l.b.)",
                        f"{best['rmsd_lb']:.2f} Å",
                        border=True,
                    )
                    col_c.metric(
                        "生成构象数 / 能差",
                        f"{len(result['results'])}",
                        f"Δ = {worst['affinity'] - best['affinity']:.2f} kcal/mol",
                        border=True,
                    )

                    # 活性解释
                    affinity = best["affinity"]
                    if affinity < -9.5:
                        st.success(f"🟢 **强结合** — 结合能 {affinity:.2f} kcal/mol，预测具有高亲和力")
                    elif affinity < -7.5:
                        st.info(f"🔵 **中等结合** — 结合能 {affinity:.2f} kcal/mol，预测具有一定亲和力")
                    else:
                        st.warning(f"🟡 **弱结合** — 结合能 {affinity:.2f} kcal/mol，预测亲和力较弱")

                # 3. 3D 可视化
                st.subheader("🧊 3D 构象可视化")
                st.caption("蛋白为卡通模型，不同颜色代表不同对接构象")
                try:
                    html_str = result["view"]._make_html()
                    st.components.v1.html(html_str, height=600)
                except Exception as e:
                    st.warning(f"3D 可视化渲染失败: {e}")

                # 4. 原始输出
                with st.expander("📄 Smina 原始输出"):
                    st.code(result["output_text"], language="text")

                # 5. 下载按钮
                st.download_button(
                    label="💾 下载所有构象 (SDF)",
                    data=result["sdf_data"],
                    file_name="docking_poses.sdf",
                    mime="chemical/x-mdl-sdfile",
                    key="docking_download_sdf",
                )

                # 6. 对接信息
                with st.expander("ℹ️ 对接参数详情"):
                    import json
                    st.json(json.dumps({
                        "target": pdb_id or "上传的PDB文件",
                        "ligand_smiles": ligand_smiles.strip(),
                        "num_poses": num_poses,
                        "exhaustiveness": exhaustiveness,
                        "pose_count": len(result["results"]) if result["results"] else 0,
                    }, indent=2, ensure_ascii=False))

                # 缓存结果以便清空后查看
                cached_result = dict(result)
                cached_result["smiles"] = ligand_smiles.strip()
                st.session_state["docking_last_result"] = cached_result

            except FileNotFoundError:
                st.error("❌ 未找到 Smina 命令，请确保 Smina 已安装并位于 PATH 中")
                st.info(
                    "安装命令：\n"
                    "- macOS: `brew install smina`\n"
                    "- Linux: `wget -O /usr/local/bin/smina https://...`\n"
                    "- conda: `conda install -c conda-forge smina`"
                )
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Smina 执行失败 (退出码 {e.returncode})")
                if hasattr(e, "output") and e.output:
                    with st.expander("查看错误详情"):
                        st.code(e.output, language="text")
            except ImportError as e:
                st.error(f"❌ 缺少依赖: {e}")
                # 从错误信息中提取缺失的模块名，给出精确的安装建议
                err_msg = str(e)
                if "No module named" in err_msg:
                    mod = err_msg.split("No module named")[-1].strip().strip("'").strip('"')
                    st.info(f"请执行: conda activate egfr-md && pip install {mod.split('.')[0]}")
                elif "未安装" in err_msg:
                    st.info(err_msg)  # 错误信息已包含安装建议
                else:
                    st.info("请确认已安装: pyarrow openbabel py3Dmol rpds-py")
            except Exception as e:
                st.error(f"❌ 对接失败: {e}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc(), language="text")

    # ---- 上次结果回显 ----
    elif "docking_last_result" in st.session_state:
        with st.expander("📌 上次对接结果（点击查看）", expanded=False):
            cached = st.session_state["docking_last_result"]
            st.caption(f"共 {len(cached.get('results', []))} 个构象")
            if cached.get("results"):
                df_cached = pd.DataFrame(cached["results"])
                df_cached.columns = ["构象 #", "结合能 (kcal/mol)", "RMSD l.b.", "RMSD u.b."]
                st.dataframe(df_cached, width="stretch", hide_index=True)

            st.download_button(
                label="💾 下载上次结果 (SDF)",
                data=cached["sdf_data"],
                file_name="docking_poses.sdf",
                mime="chemical/x-mdl-sdfile",
                key="docking_download_sdf_cached",
            )

            # KNIME 导出
            if cached.get("results"):
                scores = [r.get("affinity") for r in cached["results"] if isinstance(r, dict) and r.get("affinity") is not None]
                if scores and cached.get("smiles"):
                    import pandas as _pd
                    _df = _pd.DataFrame([{
                        "smiles": cached["smiles"],
                        "docking_score": float(scores[0]),
                    }])
                    knime_export_section(
                        _df,
                        title="分子对接结果",
                        key_prefix="dock_knime",
                    )

    # ---------- 说明与帮助 ----------
    st.divider()
    with st.expander("📘 使用说明与常见问题", expanded=False):
        st.markdown("""
        ### 对接流程
        1. **选择靶点蛋白**: 输入 PDB ID 或上传 PDB 文件（蛋白需含共晶配体以定位口袋）
        2. **输入配体 SMILES**: 画一个分子或从模板中选择
        3. **调整参数**: 设置构象数量和搜索精度
        4. **运行对接**: 点击按钮，等待 1~5 分钟
        5. **查看结果**: 结合能排序、3D 可视化、下载 SDF 文件

        ### 对接参数说明
        - **搜索精度 (exhaustiveness)**: 4 级快速预览，8 级标准计算，16+ 级精细搜索
        - **生成构象数**: 返回 Top-N 个最优构象，越多越好但输出更大
        - **口袋缓冲区**: 共晶配体周围额外搜索空间，默认 5 Å

        ### 常见问题
        | 问题 | 解决方法 |
        |------|---------|
        | `smina: command not found` | 安装 Smina 并加入 PATH |
        | 未找到共晶配体 | 手动输入 `ligand_resname`（如 HETATM 记录的残基名） |
        | OpenBabel 报错 | 确认已安装：`pip install openbabel` |
        | 对接时间过长 | 降低 `exhaustiveness` 或 `num_poses` |

        ### 与蛋白-配体作用分析的区别
        | | 蛋白-配体作用 | 分子对接 |
        |--|-------------|---------|
        | **目的** | 分析已有复合物的已存在相互作用 | 预测配体的结合构象与亲和力 |
        | **工具** | PLIP | Smina (AutoDock Vina) |
        | **输入** | 已有 PDB 复合物 | 蛋白 + 任意 SMILES 配体 |
        | **流程** | 已有复合物 → PLIP 分析相互作用 | 蛋白 + SMILES → Smina 对接 → 结合能排序 |
        """)


@st.cache_resource
def _check_smina() -> bool:
    """检查 Smina 命令行工具是否可用（结果缓存，避免每次 rerun 都起子进程）。

    注意：**不能直接跑裸名字 `smina`** —— 那要求它出现在 PATH 上。实测本机
    `shutil.which("smina")` 返回 None（smina.exe 装在 conda 环境的 `Library\bin`，
    而用某些启动方式（VS Code 直接运行、或启动器直接调用环境里的 python）时
    `Library\bin` 并不在 PATH 上），于是"明明装好了却报未检测到"。
    这里统一用 docking_utils.find_smina_executable() 解析绝对路径，并以输出/退出码判断，
    不能只看"子进程启动成功"（启动成功 ≠ 工具能跑）。
    """
    try:
        from docking_utils import find_smina_executable
    except Exception:
        find_smina_executable = None  # type: ignore[assignment]

    candidates = []
    if find_smina_executable is not None:
        try:
            exe = find_smina_executable()
            if exe:
                candidates.append(exe)
        except Exception:
            pass
    # 兜底：仍然尝试 PATH 上的裸名字（例如用户把 smina 放进系统目录并加入 PATH）
    candidates.append("smina")

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--help"],
                capture_output=True,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0:
            return True
        # smina 在参数不全时会打印用法并以非 0 退出；只要输出里确实是 smina 的用法就算可用
        output = (result.stdout or b"") + (result.stderr or b"")
        lowered = output.lower()
        if b"smina" in lowered and (b"usage" in lowered or b"receptor" in lowered):
            return True
    return False
