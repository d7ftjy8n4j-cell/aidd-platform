# pages/page_molecular_generation.py
# -*- coding: utf-8 -*-
"""
分子生成页面 (字符级 RNN)

基于字符级 LSTM 自回归生成新 SMILES 分子.
借鉴 REINVENT 架构,演示化学语言模型与迁移学习概念.

参考: TeachOpenCADD T034 (RNN-based molecular property prediction)
"""

import io
import sys
import pandas as pd
import numpy as np
import streamlit as st
import logging

from components.knime_export import knime_export_section

# RDKit 用于 2D 结构渲染
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# PIL/Pillow 用于图片渲染
try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


def page_molecular_generation():
    """分子生成主页面"""

    # 检查 PyTorch 是否可用 (降级策略,与 SHAP 模块风格一致)
    try:
        from utils.molecular_generation_utils import TORCH_AVAILABLE
    except ImportError:
        TORCH_AVAILABLE = False

    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch 未安装,分子生成功能不可用")
        st.markdown("""
        **安装指引**:
        ```bash
        pip install torch rdkit-pypi
        ```
        或使用 conda:
        ```bash
        conda install pytorch -c pytorch
        ```
        """)
        return

    st.title("🧬 分子生成 (SMILES-RNN)")
    st.caption("基于字符级 LSTM 自回归生成全新 EGFR 抑制剂候选分子.")

    with st.popover("🎓 教学点"):
        st.markdown("""
        **AI 驱动的从头分子设计(de novo design)**:

        - **化学语言模型**:将 SMILES 视为一种"化学语言",LSTM 循环神经网络学习字符序列规律
        - **自回归生成**:逐个字符地"写"出新 SMILES----类似手机键盘的预测文本
        - **温度采样** ($T$):控制生成多样性
          - $T \\rightarrow 0$:确定性输出(总是选最高概率字符)
          - $T = 1$:按学习到的分布采样
          - $T > 1$:增加随机性,探索更广阔的化学空间

        **迁移学习**:在 EGFR 抑制剂数据集上对基础模型做少量额外训练,
        使模型"偏向"生成 EGFR 相关的化学结构.

        **借鉴架构**:REINVENT (Olivecrona et al., *J Cheminform* 2017)  
        > 参考:TeachOpenCADD T034
        """)

    # ---------- 初始化 Session State ----------
    for key, default in [
        ("molgen_generator", None),
        ("molgen_trained", False),
        ("molgen_stats", None),
        ("molgen_default_trained", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---------- 侧边栏: 模型训练/加载 ----------
    with st.sidebar:
        st.header("🧠 模型管理")

        # 方式 1: 使用内置数据一键训练
        st.subheader("快速开始")
        if st.button("⚡ 使用内置 EGFR 数据集训练", use_container_width=True, key="molgen_train_default"):
            with st.spinner("正在训练 LSTM 模型 (约需 30-60 秒)..."):
                try:
                    from utils.molecular_generation_utils import MolecularGenerator, DEFAULT_SMILES

                    gen = MolecularGenerator()
                    progress_bar = st.progress(0, text="训练中...")
                    status_text = st.empty()

                    def progress_cb(epoch, loss):
                        total = 30  # 默认 30 epochs
                        progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                        status_text.text(f"Loss: {loss:.4f}")

                    stats = gen.train_many(DEFAULT_SMILES, epochs=30, progress_callback=progress_cb)

                    st.session_state["molgen_generator"] = gen
                    st.session_state["molgen_trained"] = True
                    st.session_state["molgen_stats"] = stats
                    st.session_state["molgen_default_trained"] = True

                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"训练完成！词表={stats['vocab_size']}, "
                               f"最终 loss={stats['final_loss']:.4f}")
                    st.rerun()

                except ImportError as e:
                    st.error(f"模块加载失败: {e}")
                except Exception as e:
                    st.error(f"训练失败: {e}")
                    import traceback
                    with st.expander("错误详情"):
                        st.code(traceback.format_exc())

        st.divider()

        # 方式 2: 上传自定义 SMILES 训练
        st.subheader("自定义训练 / 微调")
        custom_mode = st.radio(
            "训练模式",
            ["从零训练", "微调现有模型"],
            index=0,
            key="molgen_train_mode",
            disabled=not st.session_state["molgen_trained"],
        )

        custom_smiles_text = st.text_area(
            "输入 SMILES (每行一个, 至少 10 个)",
            "",
            height=100,
            key="molgen_custom_smiles",
        )
        custom_epochs = st.slider("训练轮数", 5, 50, 15, key="molgen_custom_epochs")

        if st.button("🔄 开始训练/微调", use_container_width=True, key="molgen_train_custom"):
            smiles = [s.strip() for s in custom_smiles_text.split("\n") if s.strip()]
            if len(smiles) < 5:
                st.error("请至少输入 5 个有效 SMILES")
            else:
                with st.spinner(f"{'微调' if custom_mode == '微调现有模型' else '训练'}中..."):
                    try:
                        from utils.molecular_generation_utils import MolecularGenerator

                        if custom_mode == "从零训练" or not st.session_state["molgen_trained"]:
                            gen = MolecularGenerator()
                            stats = gen.train_many(smiles, epochs=custom_epochs)
                            st.session_state["molgen_default_trained"] = False
                        else:
                            gen = st.session_state["molgen_generator"]
                            stats = gen.fine_tune(smiles, epochs=custom_epochs)

                        st.session_state["molgen_generator"] = gen
                        st.session_state["molgen_trained"] = True
                        st.session_state["molgen_stats"] = stats
                        st.success(f"完成！loss={stats.get('final_loss', stats.get('fine_tune_loss', 0)):.4f}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"失败: {e}")

        st.divider()

        # 方式 3: 上传已训练模型
        st.subheader("加载已保存模型")
        uploaded_model = st.file_uploader(
            "上传 .pt 模型文件",
            type=["pt"],
            key="molgen_upload_model",
        )
        if uploaded_model and st.button("📂 加载模型", key="molgen_load_btn"):
            with st.spinner("加载模型..."):
                try:
                    from utils.molecular_generation_utils import MolecularGenerator

                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        tmp.write(uploaded_model.read())
                        model_path = tmp.name

                    gen = MolecularGenerator()
                    gen.load_model(model_path)
                    os.unlink(model_path)

                    st.session_state["molgen_generator"] = gen
                    st.session_state["molgen_trained"] = True
                    st.session_state["molgen_stats"] = gen.training_stats
                    st.session_state["molgen_default_trained"] = False
                    st.success(f"模型已加载 (vocab_size={gen.vocab_size})")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载失败: {e}")

    # ---------- 主区域: 生成控制 ----------

    if not st.session_state["molgen_trained"]:
        st.info(
            "👈 请先在左侧边栏训练或加载模型."
            "点击「⚡ 使用内置 EGFR 数据集训练」可快速体验."
        )
        with st.expander("📘 页面说明", expanded=False):
            st.markdown("""
            ### 什么是 SMILES-RNN？

            SMILES (Simplified Molecular Input Line Entry System) 是一种用字符串表示
            分子结构的化学语言. RNN (循环神经网络) 通过学习大量已知分子的 SMILES 序列,
            掌握字符间的转换规律,从而能够自回归地生成全新的,化学上合理的 SMILES.

            ### 核心概念

            **1. 字符级语言模型**
            将 "C", "O", "N", "=", "(" 等化学符号视为"字母",RNN 学习
            如何排列这些字母以构成有效的分子结构.

            **2. 自回归生成**
            从一个起始字符 (如 "C") 开始,模型逐字预测下一个最可能出现的
            字符,直到生成终止符 "~".

            **3. 温度采样**
            - `Temperature = 0.5`: 保守,倾向于高频模式,有效性高
            - `Temperature = 1.0`: 标准,按学习到的分布采样
            - `Temperature = 1.5`: 创造性,引入更多随机性

            **4. 迁移学习 (Fine-tuning)**
            在通用化学数据集上预训练后,用特定靶点的抑制剂数据进行微调,
            生成的分子会偏向该靶点的化学空间.

            ### 工作流衔接
            ```
            分子生成 -> 分子预测 (活性评估) -> 药物筛选 (ADME) -> 批量对接
            ```
            """)
        return

    # ---------- 生成参数 ----------
    st.subheader("🎛️ 生成参数")

    col_seed, col_temp, col_count, col_len = st.columns(4)

    with col_seed:
        seed = st.text_input(
            "起始前缀",
            "C",
            help="生成起始字符 (C=碳, c=芳香碳, 留空随机)",
            key="molgen_seed",
        )
    with col_temp:
        temperature = st.slider(
            "温度 (Temperature)",
            0.1, 2.0, 0.8, 0.1,
            help=">1 更多样, <1 更保守, ~0 贪心解码",
            key="molgen_temp",
        )
    with col_count:
        num_samples = st.number_input(
            "生成数量",
            5, 50, 20,
            key="molgen_count",
        )
    with col_len:
        max_len = st.slider(
            "最大长度",
            30, 200, 120,
            key="molgen_maxlen",
        )

    # ---------- 生成按钮 ----------
    if st.button("🚀 生成分子", type="primary", use_container_width=True, key="molgen_generate"):
        gen: "MolecularGenerator" = st.session_state["molgen_generator"]

        with st.spinner(f"自回归采样 {num_samples} 个 SMILES..."):
            results = gen.generate(
                seed=seed.strip() if seed.strip() else "",
                temperature=temperature,
                max_length=max_len,
                num_samples=num_samples,
            )

        valid = [r for r in results if r["valid"]]
        invalid = [r for r in results if not r["valid"]]

        # ---- 统计卡片 ----
        st.subheader("📊 生成统计")
        cols = st.columns(4)
        cols[0].metric("总生成", len(results), border=True)
        cols[1].metric("有效分子", len(valid),
                       f"{len(valid) / max(len(results), 1) * 100:.0f}%", border=True)
        cols[2].metric("无效 SMILES", len(invalid), border=True)
        cols[3].metric("有效率", f"{len(valid) / max(len(results), 1) * 100:.0f}%",
                       delta=None if len(valid) / max(len(results), 1) > 0.5 else "低",
                       delta_color="inverse",
                       border=True)

        # ---- 有效分子展示 ----
        if valid:
            st.subheader(f"✅ 有效分子 ({len(valid)} 个)")

            # 网格展示
            cols_per_row = 4
            for i, item in enumerate(valid):
                if i % cols_per_row == 0:
                    cols = st.columns(cols_per_row)

                with cols[i % cols_per_row]:
                    try:
                        if RDKIT_OK and PIL_OK:
                            mol = Chem.MolFromSmiles(item["canonical"])
                            if mol:
                                img = Draw.MolToImage(mol, size=(200, 150))
                                st.image(img, use_container_width=True)
                    except Exception:
                        pass
                    st.code(item["canonical"][:60], language="text")
                    if item["seed_used"] != "(随机)":
                        st.caption(f"← {item['seed_used']}")

            # 属性统计
            if RDKIT_OK and len(valid) > 1:
                with st.expander("📈 分子属性分析", expanded=False):
                    mw_list, logp_list, hbd_list, hba_list = [], [], [], []
                    for item in valid:
                        try:
                            mol = Chem.MolFromSmiles(item["canonical"])
                            if mol:
                                mw_list.append(Descriptors.MolWt(mol))
                                logp_list.append(Descriptors.MolLogP(mol))
                                hbd_list.append(Descriptors.NumHDonors(mol))
                                hba_list.append(Descriptors.NumHAcceptors(mol))
                        except Exception:
                            pass

                    if mw_list:
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                        axes[0, 0].hist(mw_list, bins=15, color="#2196F3", edgecolor="white")
                        axes[0, 0].set_title("分子量分布")
                        axes[0, 0].set_xlabel("MW (g/mol)")

                        axes[0, 1].hist(logp_list, bins=15, color="#4CAF50", edgecolor="white")
                        axes[0, 1].set_title("LogP 分布")
                        axes[0, 1].set_xlabel("LogP")

                        axes[1, 0].hist(hbd_list, bins=8, color="#FF9800", edgecolor="white")
                        axes[1, 0].set_title("氢键供体")
                        axes[1, 0].set_xlabel("HBD 数量")

                        axes[1, 1].hist(hba_list, bins=8, color="#9C27B0", edgecolor="white")
                        axes[1, 1].set_title("氢键受体")
                        axes[1, 1].set_xlabel("HBA 数量")

                        plt.tight_layout()
                        st.pyplot(fig)

            # SMILES 列表 + 输出到下游
            st.subheader("📋 SMILES 列表 (可复制到预测模块)")
            smiles_text = "\n".join(r["canonical"] for r in valid)
            st.code(smiles_text, language="text")
            st.caption("💡 复制以上 SMILES,粘贴到「分子预测」页面的输入框中,一键评估活性")

            # 下载
            st.download_button(
                "📥 下载 SMILES (TXT)",
                smiles_text,
                f"generated_smiles_{len(valid)}.txt",
                "text/plain",
                key="molgen_dl_txt",
            )

            # KNIME 导出
            if valid:
                import pandas as _pd
                _df = _pd.DataFrame(valid)
                knime_export_section(
                    _df,
                    title="分子生成结果",
                    key_prefix="molgen_knime",
                )

        # ---- 无效 SMILES (折叠) ----
        if invalid:
            with st.expander(f"⚠️ 无效 SMILES ({len(invalid)} 个)", expanded=False):
                for i, item in enumerate(invalid):
                    st.caption(f"#{i + 1}: `{item['smiles']}`")

    # ---------- 模型信息 ----------
    if st.session_state["molgen_stats"]:
        with st.expander("📊 模型训练信息", expanded=False):
            stats = st.session_state["molgen_stats"]
            st.json({
                "词表大小": stats.get("vocab_size", "N/A"),
                "训练分子数": stats.get("n_smiles", "N/A"),
                "训练轮数": stats.get("epochs", "N/A"),
                "最终 Loss": f"{stats.get('final_loss', stats.get('fine_tune_loss', 0)):.4f}",
                "微调轮数": stats.get("fine_tune_epochs", "未微调"),
                "微调分子数": stats.get("fine_tune_smiles", "未微调"),
            })

            if "loss_history" in stats:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 3))
                losses = stats["loss_history"]
                ax.plot(range(1, len(losses) + 1), losses)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("训练 Loss 曲线")
                st.pyplot(fig)
