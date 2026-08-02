#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成药尘光用户手册 .docx 文件"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import datetime

doc = Document()

# ============================================================
# 全局样式设定
# ============================================================
style = doc.styles['Normal']
font = style.font
font.name = '微软雅黑'
font.size = Pt(11)
font.color.rgb = RGBColor(0x33, 0x33, 0x33)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(6)

# 设置页边距
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

# ---------- 标题样式 ----------
for level in range(1, 4):
    h_style = doc.styles[f'Heading {level}']
    h_font = h_style.font
    h_font.name = '微软雅黑'
    h_style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    if level == 1:
        h_font.size = Pt(22)
        h_font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)
        h_font.bold = True
        h_style.paragraph_format.space_before = Pt(24)
        h_style.paragraph_format.space_after = Pt(12)
    elif level == 2:
        h_font.size = Pt(16)
        h_font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
        h_font.bold = True
        h_style.paragraph_format.space_before = Pt(20)
        h_style.paragraph_format.space_after = Pt(8)
    else:
        h_font.size = Pt(13)
        h_font.color.rgb = RGBColor(0x34, 0x49, 0x5E)
        h_font.bold = True
        h_style.paragraph_format.space_before = Pt(14)
        h_style.paragraph_format.space_after = Pt(6)


# ============================================================
# 辅助函数
# ============================================================
def add_para(text, bold=False, alignment=None, size=None, color=None, space_after=None):
    """添加段落"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = '微软雅黑'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    if bold:
        run.bold = True
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if alignment is not None:
        p.alignment = alignment
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def add_bullet(text, level=0):
    """添加项目符号列表"""
    p = doc.add_paragraph(text, style='List Bullet')
    if level > 0:
        p.paragraph_format.left_indent = Cm(1.5 * (level + 1))
    return p


def add_table(headers, rows, col_widths=None):
    """添加格式化表格"""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.name = '微软雅黑'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

    # 数据行
    for r, row_data in enumerate(rows):
        for c, val in enumerate(row_data):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = '微软雅黑'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    # 表后空行
    doc.add_paragraph()
    return table


def add_note_block(text, note_type="info"):
    """添加提示/注意块"""
    colors = {
        "info": RGBColor(0xD6, 0xEA, 0xF8),
        "warning": RGBColor(0xFC, 0xE4, 0xD6),
        "tip": RGBColor(0xD5, 0xF5, 0xE3),
    }
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("💡 " if note_type == "tip" else "⚠️ " if note_type == "warning" else "📌 ")
    run.font.name = '微软雅黑'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    run = p.add_run(text)
    run.font.name = '微软雅黑'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    run.font.color.rgb = RGBColor(0x92, 0x3B, 0x13) if note_type == "warning" else RGBColor(0x24, 0x63, 0x9C)
    run.italic = True
    return p


def add_page_break():
    doc.add_page_break()


# ============================================================
# 封面
# ============================================================
for _ in range(6):
    doc.add_paragraph()

add_para("药尘光", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, size=36,
         color=RGBColor(0x1A, 0x56, 0xDB))
add_para("EGFR 抑制剂智能发现与设计平台", bold=True,
         alignment=WD_ALIGN_PARAGRAPH.CENTER, size=20, color=RGBColor(0x2C, 0x3E, 0x50))

doc.add_paragraph()
doc.add_paragraph()

add_para("用 户 手 册", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
         size=28, color=RGBColor(0x34, 0x49, 0x5E))

for _ in range(4):
    doc.add_paragraph()

add_para(f"软件版本：V2.0.0 (Navigation 重构版)", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=12)
add_para(f"文档版本：V2.0.0", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=12)
add_para(f"更新日期：{datetime.date.today().strftime('%Y年%m月%d日')}", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=12)
add_para("作者：dadamingli", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=12)

add_page_break()

# ============================================================
# 目录页（手动）
# ============================================================
doc.add_heading('目  录', level=1)

toc_items = [
    ("一、系统概述", [
        "项目背景与命名寓意",
        "设计理念：双核驱动，理形相生",
        "核心功能全景",
        "模型性能指标",
        "系统要求",
        "快速上手（3分钟入门）",
    ]),
    ("二、系统安装与启动", [
        "环境配置",
        "启动系统",
        "首次启动检查",
    ]),
    ("三、主界面介绍", [
        "界面布局与导航",
        "侧边栏功能详解",
    ]),
    ("四、功能模块详解", [
        "4.01  首页·系统概览",
        "4.02  数据获取（ChEMBL / PubChem / 文件上传）",
        "4.03  分子预测（RF + GNN 双引擎）",
        "4.04  分子评估（药物筛选 + 化学依据）",
        "4.05  药效团设计",
        "4.06  化学空间（分子聚类 + 公共子结构MCS）",
        "4.07  结构分析（3D可视化 + PLIP相互作用）",
        "4.08  分子对接（单分子 + 批量）",
        "4.09  分子动力学（MD模拟 + MM-GBSA）",
        "4.10  激酶相似性（KLIFS IFP）",
        "4.11  分子生成（LSTM自回归）",
        "4.12  自动化流程（一键串联）",
        "4.13  模型与系统（性能 + 架构 + 关于）",
    ]),
    ("五、常见问题与解决方案", ["启动 / 预测 / 可视化 / 性能 / 导出"]),
    ("六、附录", ["术语表", "快捷键列表", "系统配置参考", "版本更新历史"]),
]

for section_title, items in toc_items:
    p = doc.add_paragraph()
    run = p.add_run(section_title)
    run.bold = True
    run.font.size = Pt(13)
    run.font.name = '微软雅黑'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    run.font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)
    for item in items:
        p2 = doc.add_paragraph()
        p2.paragraph_format.left_indent = Cm(1.5)
        run2 = p2.add_run(item)
        run2.font.size = Pt(11)
        run2.font.name = '微软雅黑'
        run2._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

add_page_break()

# ============================================================
# 一、系统概述
# ============================================================
doc.add_heading('一、系统概述', level=1)

doc.add_heading('1.1  项目背景与命名寓意', level=2)
add_para(
    '"药尘光"——取"药"之疗愈，"尘"之微观，"光"之希望。'
    '它寓意着：在微观分子世界的尘埃中，于无数细微之处探寻治愈疾病的希望之光。'
    '药者仁心，尘者至微，光者向愈，三字相合，寄托了平台"于细微处见真知、以智慧照健康"的初心。'
)
add_para(
    '表皮生长因子受体（Epidermal Growth Factor Receptor, EGFR）是一种跨膜酪氨酸激酶受体，'
    '在非小细胞肺癌、结直肠癌等多种恶性肿瘤的发生与进展中发挥着关键作用。EGFR 抑制剂已成为'
    '精准医疗领域的重要治疗策略。然而，传统药物发现过程往往周期长、成本高、失败率高，'
    '亟需人工智能技术的赋能。基于这一现实需求，"药尘光"应运而生。'
)
add_para(
    '"药尘光"不仅是专业的科研辅助工具，更是一座面向对人工智能与药物发现充满兴趣的学生和'
    '研究者打造的"数字游乐场"。平台以 Web 应用形式部署于云端，用户无需安装任何软件，'
    '只需打开浏览器，即可便捷地体验 AI 驱动的药物设计与分析全流程，'
    '沿着一条完整的 **12 步 AIDD 学习路径**，从零基础进阶到理解现代智能药物发现的核心理念。'
)

doc.add_heading('1.2  设计理念：双核驱动，理形相生', level=2)
add_para('"药尘光"采用独特的"双核驱动，理形相生"设计理念：')

add_table(
    ["维度", "模型", "核心思想", "技术映射"],
    [
        ["理", "随机森林 (RF)", "捕捉「经验之理」— 物理化学经验规则",
         "200+ RDKit描述符 + 100棵决策树"],
        ["形", "图神经网络 (GNN)", "感知「结构之形」— 分子拓扑图模式",
         "3层GCN + 13维原子特征 + 全局池化"],
    ],
    col_widths=[1.5, 3, 5.5, 5.5]
)

add_para(
    '两种模型互为补充、交叉验证：RF 以物理化学经验规则见长，GNN 以分子拓扑结构取胜。'
    '最终通过加权平均与一致性判断给出稳健的综合预测，在可解释性与端到端学习之间形成协同，'
    '既"知其然"，也"知其所以然"。'
)

doc.add_heading('1.3  核心功能全景', level=2)
add_para(
    '"药尘光"V2.0 共包含 **13 个导航页面、20 余个子功能模块**，覆盖从数据获取、活性预测、'
    '成药性评估到分子设计、结构分析与动力学模拟的 AIDD 完整工作流：'
)

add_table(
    ["编号", "模块名称", "核心功能", "关键技术与工具"],
    [
        ["01", "首页", "系统概览、双模型状态、数据集简介", "CSS动画 + 状态指示器"],
        ["02", "数据获取", "ChEMBL检索 / PubChem相似性搜索 / 文件上传", "chembl_webresource_client"],
        ["03", "分子预测", "RF标准预测 / GNN高级预测 / 双模型对比", "scikit-learn + PyTorch Geometric"],
        ["04", "分子评估", "Lipinski五规则 / PAINS / Brenk / 相似性搜索 / 性质计算", "RDKit FilterCatalog + Morgan指纹"],
        ["05", "药效团设计", "特征提取→聚类→3D药效团模型生成", "氢键/疏水/芳香特征检测 + py3Dmol"],
        ["06", "化学空间", "Butina聚类 + UMAP降维 + 公共子结构MCS分析", "rdFMCS + tanimoto距离"],
        ["07", "结构分析", "蛋白质-配体3D可视化 + PLIP相互作用分析", "py3Dmol + PLIP"],
        ["08", "分子对接", "单分子对接 / 批量虚拟筛选 / 构象排名", "Smina (AutoDock Vina分支)"],
        ["09", "分子动力学", "全原子MD模拟 / RMSD/RMSF轨迹分析 / MM-GBSA结合能", "OpenMM + SMIRNOFF + MDTraj"],
        ["10", "激酶相似性", "KLIFS IFP指纹 / Jaccard距离矩阵 / 脱靶分析", "KLIFS REST API + 85位IFP"],
        ["11", "分子生成", "LSTM自回归SMILES生成 / 温度采样 / 属性分布", "PyTorch char-level LSTM"],
        ["12", "自动化流程", "RF→GNN→ADME→PAINS→药效团→相似性 一键串联", "多步骤流水线 + 智能判定"],
        ["13", "模型与系统", "模型性能 / 特征重要性 / 系统架构 / 技术栈", "SHAP + 混淆矩阵 + 训练曲线"],
    ],
    col_widths=[1, 2.5, 5.5, 6.5]
)

doc.add_heading('1.4  模型性能指标', level=2)
add_para('"药尘光"采用双引擎预测架构，模型性能指标如下：')

add_table(
    ["指标", "随机森林 (RF)", "GNN (GCN)"],
    [
        ["AUC", "0.8695", "0.8081"],
        ["准确率", "0.7856", "0.7652"],
        ["训练数据", "13,286 ChEMBL EGFR 化合物", "13,286 ChEMBL EGFR 化合物"],
        ["特征维度", "200+ RDKit分子描述符", "13维原子特征"],
        ["模型架构", "100棵决策树", "3层GCNConv + BatchNorm"],
        ["可解释性", "SHAP特征重要性", "注意力权重（规划中）"],
        ["推理速度", "约0.3秒/分子", "约0.8秒/分子"],
    ],
    col_widths=[3, 6, 6]
)

doc.add_heading('1.5  系统要求', level=2)
add_note_block(
    '本平台已部署至 Streamlit Cloud，推荐直接访问在线版本 '
    '(https://ai-egfr-platform.streamlit.app/)，无需本地配置。', "tip"
)
add_para('如需本地部署或二次开发，请参考以下环境要求：')
add_bullet('操作系统：Windows 10/11, Linux, macOS')
add_bullet('Python 版本：3.10 或更高')
add_bullet('内存：最低 4 GB，推荐 8 GB 以上（RF 模型加载需要约 3 GB 内存）')
add_bullet('硬盘空间：至少 5 GB 可用空间（模型文件约 3.2 GB）')
add_bullet('浏览器：Chrome 90+, Firefox 88+, Edge 90+ 等现代浏览器')
add_bullet('网络：可选（仅在线加载 PDB 结构时需要）')

doc.add_heading('1.6  快速上手（3 分钟入门）', level=2)
add_para('无需任何安装，打开浏览器访问平台即可开始探索。以下是 3 分钟快速体验路径：')

add_table(
    ["步骤", "操作", "预期结果"],
    [
        ["1", "在 🧪 分子预测 输入吉非替尼的 SMILES：\n"
         "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4",
         "RF / GNN 双引擎给出活性概率、置信度与 SHAP 解释"],
        ["2", "在 🧪 分子评估 查看该分子",
         "Lipinski 五规则雷达图、PAINS/Brenk 毒性警报、理化性质一览"],
        ["3", "在 🔬 结构分析 输入 PDB ID：2ITY",
         "观察埃罗替尼-EGFR 复合物 3D 结合模式，自动检测氢键/疏水/π-π 相互作用"],
        ["4", "在 ⚙️ 自动化流程 一键运行全流程",
         "预测 → 成药性 → 毒性 → 药效团 → 相似性，生成汇总报告"],
    ],
    col_widths=[1.2, 6.5, 7.8]
)

add_note_block(
    '每个页面均设有"🎓 教学点"弹窗，点击即可了解该功能背后的药物设计理论，'
    '零基础也能循序渐进。完整学习路径见 3.1 节。', "tip"
)

add_page_break()

# ============================================================
# 二、系统安装与启动
# ============================================================
doc.add_heading('二、系统安装与启动', level=1)

add_note_block(
    '平台已部署至 Streamlit Cloud，通过浏览器访问 '
    'https://ai-egfr-platform.streamlit.app/ 即可使用全部功能，'
    '无需任何安装配置。如需本地运行或二次开发，请参考以下步骤。', "tip"
)

doc.add_heading('2.1  环境配置', level=2)

add_para('步骤 1：克隆项目并进入目录', bold=True)
add_para('git clone <repository-url> && cd ai-egfr-platform', size=10)

add_para('步骤 2：创建并激活 Conda 虚拟环境', bold=True)
add_para('conda env create -f environment_md.yml', size=10)
add_para('conda activate egfr', size=10)

add_para('步骤 3（备选）：使用 pip 安装依赖', bold=True)
add_para('pip install -r requirements.txt', size=10)

add_para('关键依赖版本：', bold=True)
add_table(
    ["依赖包", "版本", "用途"],
    [
        ["streamlit", ">= 1.56.0", "Web 界面框架"],
        ["scikit-learn", ">= 1.3", "随机森林模型"],
        ["torch", "2.1.2", "GNN 深度学习框架"],
        ["torch-geometric", "2.4.0", "图神经网络库"],
        ["rdkit-pypi", "2022.9.5", "化学信息学工具包"],
        ["openmm", "8.1.1", "分子动力学引擎"],
        ["openff-toolkit", "0.15.2", "SMIRNOFF 力场（配体参数化）"],
        ["py3Dmol", "2.0.4", "3D 分子可视化"],
        ["shap", "0.41.0", "模型可解释性"],
        ["umap-learn", "0.5.5", "降维与聚类可视化"],
    ],
    col_widths=[4, 3, 8]
)

add_para('步骤 4：验证模型文件', bold=True)
add_para('确保以下文件存在于项目根目录：')
add_table(
    ["文件", "大小", "说明"],
    [
        ["rf_egfr_model_final.pkl", "24.22 MB", "随机森林模型权重"],
        ["gcn_egfr_best_model.pth", "179 KB", "GNN 模型权重"],
        ["feature_names.json", "186 B", "特征名列表"],
        ["feature_importance.png", "-", "特征重要性图（展示用）"],
        ["gcn_confusion_matrix.png", "-", "GNN 混淆矩阵（展示用）"],
    ],
    col_widths=[5, 3, 7]
)

doc.add_heading('2.2  启动系统', level=2)
add_para('在项目根目录下执行：', bold=True)
add_para('streamlit run app.py', size=11)

add_para('启动成功标志：')
add_bullet('命令行显示：Local URL: http://localhost:8501')
add_bullet('网络访问：http://192.168.x.x:8501（同一局域网内可访问）')
add_bullet('浏览器自动打开系统首页')
add_bullet('首页顶部显示各模块就绪状态（绿色勾选标记）')

doc.add_heading('2.3  首次启动检查', level=2)
add_para('首次启动后，请在首页顶部检查以下系统状态：')
add_bullet('随机森林预测器：就绪（AUC: 0.8695）')
add_bullet('GNN 预测器：就绪（AUC: 0.8081）')
add_bullet('化学洞察模块：就绪')
add_bullet('药物筛选模块：就绪')
add_bullet('药效团模块：就绪')

add_para('如遇模块离线，请依次排查：')
add_bullet('1. 检查对应 Python 文件是否存在（real_predictor.py、gnn_predictor.py 等）')
add_bullet('2. 检查模型文件路径是否正确')
add_bullet('3. 查看命令行输出的详细错误信息')
add_bullet('4. 注意：多级降级策略确保核心功能即使在部分组件失效时仍可使用')

add_page_break()

# ============================================================
# 三、主界面介绍
# ============================================================
doc.add_heading('三、主界面介绍', level=1)

doc.add_heading('3.1  界面布局与导航', level=2)
add_para(
    '"药尘光"采用 Streamlit 经典的侧边栏 + 主内容区布局。V2.0 升级为'
    '**顶部导航栏（共13个标签页）**，解决了旧版侧边栏导航在模块增多后不易查找的问题。'
)

add_table(
    ["界面区域", "功能说明"],
    [
        ["顶部横幅", '品牌标语\u201c双核驱动 \u00b7 理形相生\u201d（CSS呼吸动画）'],
        ["顶部导航栏", "13个标签页，覆盖AIDD完整学习路径"],
        ["侧边栏", "品牌区 + 教学指南 + 模型状态 + 快速操作 + 系统信息"],
        ["主内容区", "动态渲染当前页面的功能面板与结果展示"],
        ["页脚区域", "版权信息 + 版本号 + 教学流程提示"],
    ],
    col_widths=[3.5, 12]
)

add_note_block(
    '侧边栏内的"🎓 教学指南"提供了 12 步 AIDD 学习路径推荐（数据获取→预测→评估→设计→'
    '化学空间→结构分析→对接→MD→激酶分析→生成→自动化→总结），适合初学者循序渐进探索。', "tip"
)

doc.add_heading('3.2  侧边栏功能详解', level=2)
add_para('点击左侧箭头可展开/收起侧边栏，包含以下功能区块：')

add_table(
    ["区块", "内容"],
    [
        ["品牌区", "Logo + 标语 + 版本信息"],
        ["教学指南", "12步AIDD学习路径推荐 & 各功能页面导航引导"],
        ["模型状态", "RF在线/离线 + GNN在线/离线 — 绿色=正常，红色=离线"],
        ["使用统计", "当前会话的累计预测次数，每次预测后自动更新"],
        ["待处理数据", "如从数据获取模块缓存了数据，提示可送入自动化流程"],
        ["系统信息", "Python版本、Streamlit版本、工作目录、模型路径"],
        ["快速操作", "重置会话缓存 / 导出当前结果"],
        ["反馈区", "星级评分反馈"],
    ],
    col_widths=[3, 12]
)

add_page_break()

# ============================================================
# 四、功能模块详解
# ============================================================
doc.add_heading('四、功能模块详解', level=1)

add_para(
    '本章按顶部导航栏的 **13 个页面** 顺序逐一详解。建议初学者沿 4.01 → 4.13 的顺序浏览，'
    '它对应着"AIDD 认知逻辑"：先取数，再预测，继而评估、设计、分析结构与动态，'
    '最后回归模型理解与全局总结。每个小节均说明**功能用途、操作方法、输出结果**三要素。'
)

# --- 4.01 Home ---
doc.add_heading('4.01  首页 · 系统概览', level=2)
add_para(
    '首页是平台的入口仪表盘，展示"药尘光"的核心价值主张与系统健康状态。'
    '包含渐变色品牌标语（CSS呼吸动画）、双模型状态指示器（RF AUC 0.8695 / GNN AUC 0.8081）、'
    '数据集概览（13,286化合物，50.8% 活性比例），以及新手指南弹窗和12步AIDD学习路径。'
)

# --- 4.02 数据获取 ---
doc.add_heading('4.02  数据获取', level=2)
add_para('支持三种数据获取模式：')

add_table(
    ["模式", "数据源", "功能"],
    [
        ["ChEMBL 检索", "ChEMBL 数据库", "按靶点名（如 EGFR）获取活性数据，支持 IC50/EC50/Ki/Kd 过滤与 pIC50 阈值"],
        ["PubChem 相似性搜索", "PubChem", "按 Tanimoto 相似度阈值检索结构类似物"],
        ["文件上传", "本地文件 (CSV/Excel)", "自动识别 SMILES 列并解析化合物数据"],
    ],
    col_widths=[3, 3.5, 9]
)

add_note_block('从任意模式获取的数据可一键送入"自动化流程"进行分析。', "tip")

# --- 4.03 分子预测 ---
doc.add_heading('4.03  分子预测', level=2)
add_para('核心模块，基于"双核驱动"架构实现 EGFR 抑制活性智能预测。支持三种模式：')

add_table(
    ["模式", "技术", "优势", "耗时"],
    [
        ["标准模式 (RF)", "200+ RDKit描述符 + 随机森林", "可解释性强，提供SHAP特征重要性分析", "约0.3秒"],
        ["高级模式 (GNN)", "3层GCN + 13维原子特征", "端到端学习，自动提取结构特征", "约0.8秒"],
        ["双模型对比", "RF + GNN 并行预测", "交叉验证，一致性判断，分歧分析", "约1.2秒"],
    ],
    col_widths=[2.5, 3.5, 5.5, 2]
)

add_para('操作步骤：')
add_bullet('1. 选择预测模式（标准/高级/双模型对比）')
add_bullet('2. 输入待预测分子的 SMILES 表示')
add_bullet('3. 系统自动验证 SMILES 格式并执行预测')
add_bullet('4. 查看结果：活性概率、置信度、SHAP瀑布图、不确定性评估（100棵树标准差）')

add_para('多级降级策略：', bold=True)
add_para(
    'RF预测器设计有三级降级：RealEGFRPredictor → FallbackEGFRPredictor（RDKit规则）'
    '→ MinimalEGFRPredictor（字符串分析），确保在模型文件缺失时也能提供有意义的预测参考。'
)

add_para('常用 EGFR 抑制剂 SMILES 参考：')
add_table(
    ["药物", "SMILES"],
    [
        ["吉非替尼 (Gefitinib)", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4"],
        ["埃罗替尼 (Erlotinib)", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"],
        ["拉帕替尼 (Lapatinib)", "CS(=O)(=O)CCNCc1occc1c2ccc(Cl)cc2F"],
        ["奥希替尼 (Osimertinib)", "CN1CCN(c2ccc(Nc3nccc(-c4ccc5ccccc5n4)n3)cc2)CC1"],
    ],
    col_widths=[4, 12]
)

add_note_block(
    '使用建议：双模型结论一致时可信度最高；若 RF 与 GNN 分歧明显，'
    '请结合"分子评估"的成药性与毒性结果综合判断，或借助 SHAP 瀑布图定位影响预测的关键特征。', "tip"
)

# --- 4.04 分子评估 ---
doc.add_heading('4.04  分子评估（药物筛选 + 化学依据）', level=2)
add_para('合并两个子标签页，一站式评估化合物的成药潜力。')

add_para('药物筛选子模块：', bold=True)
add_table(
    ["筛选规则", "标准", "说明"],
    [
        ["Lipinski 五规则 (Ro5)", "MW≤500, LogP≤5, HBD≤5, HBA≤10", "允许违反一项；雷达图可视化"],
        ["PAINS 检测", "RDKit FilterCatalog 内置规则", "检测泛测定干扰化合物（假阳性风险）"],
        ["Brenk 警报", "约150个不良子结构SMARTS", "Michael受体、硝基、长脂肪链、卤代环等"],
    ],
    col_widths=[3, 4.5, 8]
)
add_para('支持单分子评估和批量筛选（100分子约3-5秒，1000分子约30-40秒）。')

add_para('化学依据子模块：', bold=True)
add_para(
    '计算分子理化性质（LogP, TPSA, 分子量等），基于 Morgan 指纹（ECFP4, Radius=2, '
    '1024 bits）进行相似性搜索，支持 Tanimoto 相似度阈值筛选，并生成6合1综合分析图。'
)

# --- 4.05 药效团设计 ---
doc.add_heading('4.05  药效团设计', level=2)
add_para('从一组活性分子中提取共同药效团特征，生成3D药效团模型以指导分子设计。')

add_para('四步流程：', bold=True)
add_bullet('1. 数据输入：手动SMILES / 上传文件 / 示例数据')
add_bullet('2. 特征分析：提取并统计6类特征（供体/受体/疏水/芳香/正电/负电）')
add_bullet('3. 药效团生成：按特征类型聚类（距离平均），重要性阈值过滤')
add_bullet('4. 结果导出：3D球体可视化（py3Dmol）+ JSON报告下载')

add_para('系统识别的6类药效团特征：')
add_table(
    ["特征类型", "颜色", "检测规则"],
    [
        ["氢键供体 (HBD)", "绿色", "N-H / O-H 基团"],
        ["氢键受体 (HBA)", "红色", "N / O 孤对电子"],
        ["疏水中心", "黄色", "连续脂肪碳/疏水原子团簇"],
        ["芳香环中心", "橙色", "芳香环几何中心"],
        ["正电荷中心", "蓝色", "带正电的N原子（季铵等）"],
        ["负电荷中心", "品红", "带负电的O原子（羧酸根等）"],
    ],
    col_widths=[3.5, 2, 10]
)

add_para('3D视图交互操作：')
add_table(
    ["操作", "效果"],
    [
        ["左键拖动", "旋转视角"],
        ["右键 / Ctrl+左键拖动", "平移视角"],
        ["滚轮", "缩放"],
        ["双击", "重置视角"],
    ],
    col_widths=[4, 11]
)

# --- 4.06 化学空间 ---
doc.add_heading('4.06  化学空间（分子聚类 + 公共子结构MCS）', level=2)

add_para('分子聚类：', bold=True)
add_para(
    '采用 Butina 算法，基于 Morgan/RDKit 指纹计算 Tanimoto 距离，UMAP 降维至2D/3D'
    '可视化。支持距离阈值调节（0.05-1.0）、簇内相似度统计、代表分子筛选，'
    '结果可送入自动化流程。'
)

add_para('公共子结构 (MCS) 分析：', bold=True)
add_para(
    '基于 RDKit rdFMCS 算法计算多个分子之间的最大公共子结构。输出 SMARTS 表示，'
    '高亮显示分子匹配部分，内置预设 EGFR 抑制剂模板（吉非替尼/埃罗替尼/拉帕替尼/奥希替尼）。'
    '支持 threshold 和 ringMatchesRingOnly 等参数调节。'
)

# --- 4.07 结构分析 ---
doc.add_heading('4.07  结构分析（3D可视化 + 相互作用分析）', level=2)

add_para('3D 结构可视化：', bold=True)
add_para(
    '支持从 RCSB PDB 在线加载或上传本地 PDB 文件。'
    '提供4种蛋白显示样式（cartoon/stick/line/sphere）、多种配色方案（spectrum/chain/residue）、'
    '配体显示开关、表面显示及透明度控制。内置暂停实时刷新功能以释放GPU资源。'
)

add_para('推荐的 EGFR 相关 PDB 结构：')
add_table(
    ["PDB ID", "说明", "分辨率"],
    [
        ["1M17", "EGFR + 埃罗替尼", "2.60 Å"],
        ["2ITY", "EGFR L858R + 吉非替尼", "2.98 Å"],
        ["4I22", "EGFR T790M + WZ4002", "2.45 Å"],
        ["4ZAU", "EGFR T790M/L858R + 奥希替尼", "2.40 Å"],
    ],
    col_widths=[3, 8, 3]
)

add_para('PLIP 相互作用分析：', bold=True)
add_para(
    '基于 PLIP (Protein-Ligand Interaction Profiler) 自动检测蛋白-配体之间的非键相互作用：'
    '氢键、疏水接触、盐桥、π-π堆积、π-阳离子作用、卤键等。'
    '以 py3Dmol 3D渲染和类型分布统计图呈现，结果可与 MD 模拟结果自动衔接。'
)

# --- 4.08 分子对接 ---
doc.add_heading('4.08  分子对接（单分子 + 批量）', level=2)
add_para(
    '采用 Smina（AutoDock Vina 分支）进行分子对接。支持单分子对接'
    '（结合口袋自动检测、Top-N构象排名、3D可视化、SDF下载）和批量虚拟筛选'
    '（多配体并行、按结合能排序对比）。内嵌常用EGFR抑制剂模板供快速测试。'
)

add_table(
    ["功能", "说明"],
    [
        ["对接引擎", "Smina (AutoDock Vina 分支)"],
        ["口袋检测", "基于参考配体的自动结合口袋识别"],
        ["构象数量", "可设 Top-N（默认9个）"],
        ["结果输出", "结合能排序表 + 结合能分布图 + 3D构象可视化 + SDF下载"],
        ["模板分子", "吉非替尼 / 埃罗替尼 / 拉帕替尼 / 奥希替尼 SMILES预设"],
    ],
    col_widths=[3, 12]
)

# --- 4.09 分子动力学 ---
doc.add_heading('4.09  分子动力学（MD模拟 + MM-GBSA）', level=2)

add_para('MD 模拟：', bold=True)
add_para('使用 OpenMM 进行全原子分子动力学模拟，3步流程（输入设置 → 执行监控 → 结果展示）：')

add_table(
    ["步骤", "进度", "操作内容", "耗时参考"],
    [
        ["输入设置", "—", "上传蛋白+配体、设定温度(300K)/步长(2fs)/模拟时长", "—"],
        ["系统构建", "15%-38%", "配体拓扑转换→Gasteiger电荷→蛋白-配体合并→力场→溶剂→系统→ligand参数注入",
         "数秒"],
        ["能量最小化", "—", "约束蛋白 → 全系统优化", "数十秒"],
        ["NVT平衡", "—", "Langevin 300K, 约束蛋白重原子", "数分钟"],
        ["NPT平衡", "—", "MonteCarlo 1bar, 逐步释放约束", "数分钟"],
        ["生产模拟", "—", "LangevinMiddleIntegrator, 2fs步长", "取决于设定时长"],
        ["轨迹分析", "—", "RMSD / RMSF 计算与绘图", "数十秒"],
    ],
    col_widths=[2.5, 1.5, 7, 2]
)

add_para('力场配置（V2.0 已优化）：', bold=True)
add_table(
    ["组件", "力场", "参数化方式"],
    [
        ["蛋白", "AMBER ff14SB", "标准残基模板"],
        ["溶剂/离子", "TIP3P + 0.15M离子", "标准水模型参数"],
        ["配体", "SMIRNOFF Sage 2.1.0", "RDKit Gasteiger 电荷（ms级）+ Interchange注入"],
    ],
    col_widths=[2.5, 4, 8.5]
)
add_note_block(
    'V2.0 起配体电荷计算已从 AM1-BCC 半经验量化计算（分钟级）切换为 RDKit Gasteiger 电荷'
    '（毫秒级），去除了对 antechamber 和 OpenEye 的依赖，大幅缩短系统构建时间。', "info"
)

add_para('MM-GBSA：', bold=True)
add_para(
    '从 MD 模拟轨迹中提取代表性构象，使用 GB-Neck2 隐式溶剂模型估算结合自由能 ΔG，'
    '包括真空相互作用能、极性/非极性溶剂化能的分解。'
)

# --- 4.10 激酶相似性 ---
doc.add_heading('4.10  激酶相似性（KLIFS IFP 分析）', level=2)
add_para(
    '基于 KLIFS 数据库的 IFP（Interaction FingerPrint，85位）计算激酶结合口袋相似性。'
    '使用 Jaccard 距离矩阵和热图可视化，辅助评估化合物的激酶选择性谱和潜在脱靶风险。'
)

add_para('预置激酶（12种）：')
add_para('EGFR, ErbB2, ErbB4, CDK2, CDK4, MET, KDR, LCK, SRC, ABL1, BRAF, p38a')

add_para('脱靶风险评估：')
add_table(
    ["Jaccard 距离", "风险等级", "提示"],
    [
        ["< 0.30", "高度相似", "交叉反应风险较高，需关注选择性"],
        ["0.30 - 0.50", "中等相似", "可能有部分交叉反应"],
        ["> 0.50", "差异较大", "激酶选择性较好"],
    ],
    col_widths=[3, 3, 9]
)

# --- 4.11 分子生成 ---
doc.add_heading('4.11  分子生成（LSTM 自回归）', level=2)
add_para(
    '基于字符级 LSTM 的自回归分子生成引擎（借鉴 REINVENT 架构）。'
    '支持内置 EGFR 数据集训练（30 epochs）、自定义数据集训练/微调、'
    '温度采样控制生成多样性、模型保存/加载，以及生成分子的属性分布图'
    '（MW/LogP/HBD/HBA）对比分析。'
)

add_table(
    ["参数", "说明"],
    [
        ["模型架构", "3层字符级 LSTM, 嵌入维度128, 隐藏层512"],
        ["训练模式", "内置数据集 / 自定义SMILES文件 / 模型微调"],
        ["温度参数", "0.2 - 2.0（低温=保守，高温=多样）"],
        ["生成数量", "可自定义一次生成的分子数"],
        ["属性分析", "自动计算MW/LogP/HBD/HBA分布并与训练集对比"],
    ],
    col_widths=[3, 12]
)

# --- 4.12 自动化流程 ---
doc.add_heading('4.12  自动化流程（一键串联）', level=2)
add_para(
    '将多个分析步骤串联为一条自动化流水线，一键完成从预测到筛选的全链路分析。'
    '所有6个步骤均可按需勾选：'
)

add_table(
    ["步骤", "功能", "输出"],
    [
        ["1. RF 预测", "随机森林活性预测", "活性概率 + 类别判定"],
        ["2. GNN 预测", "图神经网络活性预测", "活性概率 + 双模型一致性"],
        ["3. ADME/Ro5", "Lipinski 五规则筛选", "是否符合口服药物标准"],
        ["4. PAINS/Brenk", "不良子结构检测", "假阳性/毒性风险警报"],
        ["5. 药效团匹配", "药效团特征匹配", "药效团符合度评估"],
        ["6. 相似性搜索", "已知EGFR抑制剂相似性", "最相似参考化合物 + Tanimoto分数"],
    ],
    col_widths=[2, 4, 9]
)

add_para('最终输出：')
add_bullet('汇总表：推荐 / 活性但成药性差 / 非活性 / 需人工判断')
add_bullet('Markdown 报告下载')
add_bullet('JSON 原始数据导出')

# --- 4.13 模型与系统 ---
doc.add_heading('4.13  模型与系统（性能 + 架构 + 关于）', level=2)
add_para('三合一标签页，提供对系统技术的全面解读：')

add_table(
    ["子标签", "内容"],
    [
        ["模型性能", "RF/GNN对比指标、特征重要性图（SHAP Top10）、混淆矩阵、训练曲线"],
        ["系统架构", "双引擎预测架构图、技术栈一览表（12+核心技术）"],
        ["关于项目", "教学价值说明、数据来源与引用、开源协议 (MIT License)、致谢"],
    ],
    col_widths=[3, 12]
)

add_page_break()

# ============================================================
# 五、常见问题与解决方案
# ============================================================
doc.add_heading('五、常见问题与解决方案', level=1)

doc.add_heading('5.1  启动与安装', level=2)

faq_startup = [
    ('Q: 启动时提示 "ModuleNotFoundError: No module named \'streamlit\'"',
     'A: 请确保已激活虚拟环境并安装了全部依赖：\n'
     '   conda activate egfr && pip install -r requirements.txt'),
    ('Q: 模型文件加载失败，侧边栏显示"离线"',
     'A: 检查模型文件（rf_egfr_model_final.pkl, gcn_egfr_best_model.pth, feature_names.json）'
     '是否存在于项目根目录。如文件缺失，系统将自动降级到备用预测器，功能受限但仍可使用。'),
    ('Q: 内存不足导致启动失败',
     'A: 随机森林模型加载需要约3 GB内存。解决方案：关闭其他应用程序、增加虚拟内存、'
     '或仅使用GNN模型（内存占用约500 MB）。'),
]
for q, a in faq_startup:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

doc.add_heading('5.2  预测问题', level=2)

faq_pred = [
    ('Q: 提示"无效的SMILES字符串"',
     'A: 请检查：SMILES字符串是否包含非法字符、分子结构是否在RDKit支持范围内、'
     '字符串长度是否超过1000字符限制。'),
    ('Q: GNN预测器显示"离线"或预测失败',
     'A: 可能原因：GNN模型文件缺失、PyTorch未正确安装、系统内存不足。'
     '注意本系统默认使用CPU版本PyTorch。'),
]
for q, a in faq_pred:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

doc.add_heading('5.3  可视化问题', level=2)

faq_viz = [
    ('Q: 3D结构显示空白或"渲染失败"',
     'A: 尝试以下方案：刷新页面、检查浏览器WebGL支持（chrome://gpu）、更换浏览器、'
     '开启暂停实时刷新功能（在结构分析页面提供）、检查网络连接（在线加载PDB时）。'),
]
for q, a in faq_viz:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

doc.add_heading('5.4  性能优化', level=2)

faq_perf = [
    ('Q: 批量筛选大文件时系统卡顿',
     'A: 建议分批处理（每次不超过500个分子）、关闭其他应用程序、增加系统虚拟内存。'),
    ('Q: MD模拟构建阶段卡住',
     'A: V2.0已优化：配体电荷计算由AM1-BCC（分钟级）切换为Gasteiger（毫秒级），'
     '构建时间应缩短至数十秒。如仍有问题，检查OpenMM是否正确识别GPU平台。'),
]
for q, a in faq_perf:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

doc.add_heading('5.5  数据导出', level=2)

faq_export = [
    ('Q: 导出按钮无法点击',
     'A: 需要先完成至少一次预测，系统才会生成可导出的结果。'),
    ('Q: 导出的CSV文件中文乱码',
     'A: 系统使用UTF-8编码导出CSV。如使用Excel打开乱码，请使用"数据→从文本/CSV导入"功能，'
     '选择UTF-8编码。'),
]
for q, a in faq_export:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

doc.add_heading('5.6  高级模块（对接 / 相互作用 / MD）', level=2)

faq_adv = [
    ('Q: 分子对接或 PLIP 相互作用分析提示"依赖不可用"',
     'A: 这两个模块依赖 openbabel（分子格式转换）与 plip 库。Streamlit Cloud 暂不支持编译 '
     'openbabel，请在本地环境安装后使用：\n'
     '   conda install -c conda-forge openbabel smina && pip install plip'),
    ('Q: MD 模拟 / MM-GBSA 提示 openmm 不可用',
     'A: 需在本地安装：conda install -c conda-forge openmm pdbfixer mdtraj openff-toolkit。'
     '系统构建阶段已优化为毫秒级 Gasteiger 电荷；若仍卡住，请检查 OpenMM 是否正确识别 '
     'GPU/CUDA 平台，或切换为 CPU 平台运行。'),
    ('Q: 3D 结构可视化空白或异常',
     'A: 3D 渲染基于 WebGL 与 py3Dmol。请确认浏览器已开启硬件加速（chrome://gpu），'
     '或尝试在"结构分析"页面开启暂停实时刷新后重新渲染。'),
]
for q, a in faq_adv:
    add_para(q, bold=True, space_after=2)
    add_para(a, space_after=10)

add_page_break()

# ============================================================
# 六、附录
# ============================================================
doc.add_heading('六、附录', level=1)

doc.add_heading('附录 A：术语表', level=2)

add_table(
    ["术语", "全称", "说明"],
    [
        ["EGFR", "Epidermal Growth Factor Receptor", "表皮生长因子受体，抗癌药物重要靶点"],
        ["SMILES", "Simplified Molecular Input Line Entry System", "分子结构的线性文本表示法"],
        ["RF", "Random Forest", "随机森林，集成学习算法"],
        ["GNN / GCN", "Graph Neural Network / Graph Convolutional Network", "图神经网络 / 图卷积网络"],
        ["AUC", "Area Under the ROC Curve", "ROC曲线下面积，衡量分类器排序能力"],
        ["Ro5", "Lipinski's Rule of Five", "Lipinski五规则，评估口服药物潜力"],
        ["PAINS", "Pan-Assay Interference Compounds", "泛测定干扰化合物"],
        ["MCS", "Maximum Common Substructure", "最大公共子结构"],
        ["RMSD", "Root-Mean-Square Deviation", "均方根偏差，衡量MD模拟结构稳定性"],
        ["RMSF", "Root-Mean-Square Fluctuation", "均方根波动，衡量残基柔性"],
        ["IFP", "Interaction FingerPrint", "相互作用指纹"],
        ["SHAP", "SHapley Additive exPlanations", "基于Shapley值的模型解释方法"],
        ["Gasteiger", "Gasteiger-Marsili Charge", "基于电负性均衡的快速部分电荷算法"],
        ["SMIRNOFF", "SMIRKS-Native Open Force Field", "基于SMIRKS模式的新型分子力场格式"],
        ["MM-GBSA", "MM-Generalized Born Surface Area", "MM-GBSA结合自由能估算方法"],
        ["LSTM", "Long Short-Term Memory", "长短期记忆网络"],
        ["UMAP", "Uniform Manifold Approximation and Projection", "统一流形逼近与投影（降维算法）"],
    ],
    col_widths=[2, 5.5, 8]
)

doc.add_heading('附录 B：3D 可视化快捷键', level=2)

add_table(
    ["操作", "快捷键 / 方式", "效果"],
    [
        ["旋转视角", "鼠标左键拖动", "围绕分子中心旋转"],
        ["平移视角", "鼠标右键 或 Ctrl+左键", "平移画面"],
        ["缩放", "鼠标滚轮", "放大/缩小"],
        ["重置视角", "双击左键", "恢复默认视角"],
        ["暂停渲染", "页面控件按钮", "暂停实时刷新（节省GPU资源）"],
        ["全屏", "py3Dmol内置按钮", "全屏3D视图"],
    ],
    col_widths=[3, 5, 7]
)

doc.add_heading('附录 C：系统配置参考', level=2)

add_para('推荐运行环境：', bold=True)
add_bullet('操作系统：Windows 10/11 64位 或 Ubuntu 20.04 LTS')
add_bullet('CPU：Intel Core i5 或同等性能（支持 AVX 指令集）')
add_bullet('内存：16 GB RAM')
add_bullet('硬盘：SSD，至少10 GB可用空间')
add_bullet('网络：宽带连接（用于在线加载PDB结构）')

add_para('最低运行环境：', bold=True)
add_bullet('操作系统：Windows 10 64位 或 Ubuntu 18.04')
add_bullet('CPU：Intel Core i3 或同等性能')
add_bullet('内存：4 GB RAM')
add_bullet('硬盘：至少5 GB可用空间')

add_para('云端访问：', bold=True)
add_note_block(
    '推荐使用云端版本 https://ai-egfr-platform.streamlit.app/，无需任何本地配置。'
    '支持 Windows, macOS, Linux 及移动设备浏览器访问。', "tip"
)

doc.add_heading('附录 D：版本更新历史', level=2)

add_table(
    ["版本", "日期", "主要更新内容"],
    [
        ["V2.0.0", "2026-08-01",
         "导航重构（13 页）；新增分子动力学（OpenMM）、MM-GBSA、分子对接（Smina）、"
         "批量虚拟筛选、激酶相似性（KLIFS-IFP）、分子生成（LSTM）、自动化流程、"
         "分子聚类与 MCS 分析；配体电荷改用 Gasteiger 毫秒级方案；引入多级降级容错"],
        ["V0.5", "—",
         "整合双引擎预测（RF + GNN）与 SHAP 可解释性、不确定性估计"],
        ["V0.3", "—",
         "标签页架构成型，加入分子评估、药效团设计、化学空间"],
        ["V0.1-V0.2", "—",
         "平台基础框架、双模型状态与数据获取模块"],
    ],
    col_widths=[2.2, 2.5, 10.8]
)

# ============================================================
# 尾页
# ============================================================
add_page_break()

for _ in range(8):
    doc.add_paragraph()

add_para("药尘光", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, size=28,
         color=RGBColor(0x1A, 0x56, 0xDB))
add_para("EGFR 抑制剂智能发现与设计平台", bold=True,
         alignment=WD_ALIGN_PARAGRAPH.CENTER, size=16, color=RGBColor(0x2C, 0x3E, 0x50))

doc.add_paragraph()
add_para("双核驱动 · 理形相生", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=14,
         color=RGBColor(0x7F, 0x8C, 0x8D))

for _ in range(3):
    doc.add_paragraph()

add_para(f"文档版本：V2.0.0", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=11)
add_para(f"更新日期：{datetime.date.today().strftime('%Y年%m月%d日')}", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=11)
add_para("© 2026 dadamingli  |  MIT License", alignment=WD_ALIGN_PARAGRAPH.CENTER, size=11)

# ============================================================
# 保存
# ============================================================
import os as _os
output_path = r"C:\Users\dadamingli\Desktop\my-egfr-v2\药尘光-用户手册-V2.0.docx"
doc.save(output_path)
print(f"用户手册已生成: {output_path}")

# 同步一份到桌面，便于直接取用
desktop_manual = r"C:\Users\dadamingli\Desktop\药尘光-用户手册-V2.0.docx"
try:
    _os.remove(desktop_manual)
except FileNotFoundError:
    pass
doc.save(desktop_manual)
print(f"已同步到桌面: {desktop_manual}")
