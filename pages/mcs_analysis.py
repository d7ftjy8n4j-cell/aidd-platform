# pages/mcs_analysis.py
"""
最大公共子结构 (MCS) 分析页面

计算一组分子的核心骨架，高亮展示。
"""

import io

import pandas as pd
import streamlit as st
from rdkit import Chem

from mcs_utils import compute_mcs, highlight_mcs_in_molecules, get_mcs_smarts_as_mol


# ---------- 常用分子模板 ----------
PRESET_MOLECULES = {
    "EGFR 抑制剂系列": (
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)Cl)NC3=NC=CC(=N3)C4=CN=CN4\n"
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)F)NC3=NC=CC(=N3)C4=CN=CN4\n"
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)Br)NC3=NC=CC(=N3)C4=CN=CN4"
    ),
    "喹唑啉类抑制剂": (
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4\n"
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Br)OCCCN4CCOCC4\n"
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)CF)OCCCN4CCOCC4"
    ),
}


def page_mcs_analysis():
    """最大公共子结构分析主函数"""

    st.title("🧩 最大公共子结构 (MCS) 分析")

    st.markdown("""
    **功能说明**：计算一组分子的最大公共子结构（Maximum Common Substructure），
    用于识别系列化合物的核心骨架、分析构效关系 (SAR)。

    - **输入**：多个分子的 SMILES（每行一个）
    - **输出**：MCS 的 SMARTS 模式、原子/键数、分子高亮图
    """)

    # ---------- 输入区域 ----------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📝 输入分子")

        # 模板选择
        preset = st.selectbox(
            "快速加载模板",
            ["-- 手动输入 --"] + list(PRESET_MOLECULES.keys()),
            help="选择预设分子组，或手动输入 SMILES",
        )
        if preset != "-- 手动输入 --":
            default_text = PRESET_MOLECULES[preset]
            # 模板选择后同步到输入框（st.text_area 用 key 绑定，仅首次默认值不生效）
            if st.session_state.get("mcs_last_preset") != preset:
                st.session_state["mcs_input"] = PRESET_MOLECULES[preset]
                st.session_state["mcs_last_preset"] = preset
        else:
            default_text = (
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)Cl)NC3=NC=CC(=N3)C4=CN=CN4\n"
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)F)NC3=NC=CC(=N3)C4=CN=CN4\n"
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)Br)NC3=NC=CC(=N3)C4=CN=CN4"
            )

        input_text = st.text_area(
            "输入 SMILES（每行一个）",
            default_text,
            height=160,
            key="mcs_input",
            help="每行一个 SMILES 字符串，至少 2 个分子",
        )

        # 文件上传
        uploaded_file = st.file_uploader(
            "或上传 SMILES 文件 (.txt, .csv, .smi)",
            type=["txt", "csv", "smi"],
        )
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                st.error("❌ 文件编码不是 UTF-8，请另存为 UTF-8 后重新上传（Windows 下可用记事本另存为 UTF-8）")
                content = None
            if content is not None:
                if uploaded_file.name.endswith(".csv"):
                    try:
                        df_csv = pd.read_csv(io.StringIO(content))
                    except Exception as e:
                        st.error(f"❌ CSV 解析失败: {e}")
                        df_csv = None
                    if df_csv is not None:
                        if "SMILES" in df_csv.columns:
                            input_text = "\n".join(df_csv["SMILES"].dropna().tolist())
                        else:
                            # 无表头时：若首列名本身是可解析的 SMILES，则按 header=None 重读，避免丢第一行
                            try:
                                if Chem.MolFromSmiles(str(df_csv.columns[0])) is not None:
                                    df_csv = pd.read_csv(io.StringIO(content), header=None)
                            except Exception:
                                pass
                            input_text = "\n".join(df_csv.iloc[:, 0].dropna().astype(str).tolist())
                else:
                    input_text = content

    with col_right:
        st.subheader("⚙️ 参数设置")

        threshold = st.slider(
            "Threshold（共享比例）",
            min_value=0.6,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="值越小 MCS 越大（允许部分分子缺失某些子结构）。1.0 = 必须全部共享。",
        )
        ring_matches_ring = st.checkbox(
            "环必须匹配环",
            value=True,
            help="确保芳香环不会被匹配到脂肪环",
        )
        match_valences = st.checkbox(
            "匹配化合价",
            value=False,
            help="启用后更严格，要求化合价状态匹配",
        )
        timeout = st.number_input(
            "超时时间（秒）",
            min_value=5,
            max_value=120,
            value=30,
            help="分子越多/越大，需要的时间越长",
        )

    # ---------- 执行分析 ----------
    if st.button("🚀 计算 MCS", type="primary", use_container_width=True):
        smiles_list = [
            s.strip()
            for s in input_text.strip().split("\n")
            if s.strip()
        ]

        if len(smiles_list) < 2:
            st.error("至少需要 2 个有效分子的 SMILES")
            st.stop()

        # 预验证 SMILES
        valid_smiles = []
        invalid_indices = []
        for i, smi in enumerate(smiles_list):
            if Chem.MolFromSmiles(smi) is not None:
                valid_smiles.append(smi)
            else:
                invalid_indices.append(i + 1)

        if invalid_indices:
            st.warning(f"以下行号的 SMILES 无效将被跳过: {invalid_indices}")

        if len(valid_smiles) < 2:
            st.error("有效分子不足 2 个，无法计算 MCS")
            st.stop()

        with st.spinner("⏳ 正在计算最大公共子结构（MCS 是 NP 完全问题，可能需要一些时间）..."):
            smarts, num_atoms, num_bonds, error = compute_mcs(
                valid_smiles,
                threshold=threshold,
                ring_matches_ring=ring_matches_ring,
                match_valences=match_valences,
                timeout=timeout,
            )

        if error:
            st.error(f"❌ {error}")
            st.stop()

        # ---------- 结果展示 ----------
        st.success("✅ MCS 计算完成！")

        # 1. 指标卡片
        st.subheader("📊 MCS 摘要")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MCS 原子数", num_atoms)
        c2.metric("MCS 键数", num_bonds)
        c3.metric("输入分子数", len(valid_smiles))
        c4.metric("Threshold", f"{threshold:.2f}")

        # 2. SMARTS + 结构图
        st.subheader("🔬 MCS SMARTS 模式")
        st.code(smarts, language="text")

        mcs_img = get_mcs_smarts_as_mol(smarts)
        if mcs_img:
            st.caption("公共子结构骨架（2D）")
            st.image(mcs_img, width=400)

        # 3. 分子高亮图（核心可视化）
        st.subheader("🎨 各分子中 MCS 高亮")
        st.caption("青色高亮 = 最大公共子结构（每分子展示第一个匹配）")

        images = highlight_mcs_in_molecules(valid_smiles, smarts, size=350)
        if images:
            # 按每行最多 4 个排列
            n_cols = min(4, len(images))
            rows = (len(images) + n_cols - 1) // n_cols

            for r in range(rows):
                cols = st.columns(n_cols)
                for c in range(n_cols):
                    idx = r * n_cols + c
                    if idx < len(images):
                        with cols[c]:
                            st.caption(f"分子 {idx + 1}")
                            st.image(images[idx], use_container_width=True)
                            short_smi = (
                                valid_smiles[idx]
                                if len(valid_smiles[idx]) <= 45
                                else valid_smiles[idx][:42] + "..."
                            )
                            st.caption(f"`{short_smi}`")

        # 4. 分子列表
        with st.expander("📋 查看所有输入分子"):
            df_mols = pd.DataFrame({
                "序号": range(1, len(valid_smiles) + 1),
                "SMILES": valid_smiles,
            })
            st.dataframe(df_mols, use_container_width=True)

        # 5. 下载
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="💾 下载 MCS SMARTS",
                data=smarts,
                file_name="mcs_result.smarts",
                mime="chemical/x-daylight-smiles",
            )
        with col_dl2:
            st.download_button(
                label="📄 下载结果摘要",
                data=(
                    f"MCS SMARTS: {smarts}\n"
                    f"原子数: {num_atoms}\n"
                    f"键数: {num_bonds}\n"
                    f"输入分子数: {len(valid_smiles)}\n"
                    f"Threshold: {threshold:.2f}\n"
                    f"环必须匹配环: {ring_matches_ring}\n"
                ),
                file_name="mcs_summary.txt",
                mime="text/plain",
            )

        # 6. 提示联动
        with st.expander("💡 与平台其他模块联动"):
            st.markdown("""
            | 已有模块 | 联动方式 |
            |----------|----------|
            | 分子聚类 (Butina) | 对每个聚类中的分子计算 MCS，识别各类核心骨架 |
            | 分子预测 | 对高活性分子计算 MCS，发现活性关键子结构 |
            | 药效团设计 | MCS 核心骨架作为药效团设计起点 |
            | 化学依据 | MCS 作为相似性比对的补充维度 |
            """)
