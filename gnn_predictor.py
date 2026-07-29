"""
gnn_predictor.py - GNN预测器 (适配您训练好的EGFR GCN模型)
用于Streamlit应用集成，将SMILES字符串转换为分子图并进行活性预测
作者：dadamingli

模型定义与训练脚本完全一致：
  - 层命名: conv1/conv2/conv3, bn1/bn2 (非 ModuleList)
  - 原子特征数: 14 (非12)
  - 前向传播: ReLU(BN(Conv)) + Dropout 逐层展开
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
import logging
import os
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 模型定义（与训练脚本完全一致）
# ============================================================================
class GCNModel(torch.nn.Module):
    """
    EGFR GCN 分类器，与训练脚本完全一致的架构。

    层命名：conv1, conv2, conv3, bn1, bn2（直接属性，非 ModuleList）
    以匹配保存的 state_dict 键名。
    """

    def __init__(self, num_node_features=14, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = Linear(hidden_dim // 2, 1)

        logger.info(f"初始化GCN模型: {num_node_features} -> {hidden_dim} -> 1")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1: Conv -> BN -> ReLU -> Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2: Conv -> BN -> ReLU -> Dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 3: Conv (无 BN/Dropout，与训练一致)
        x = self.conv3(x, edge_index)

        # 全局平均池化
        x = global_mean_pool(x, batch)

        # 全连接分类头
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x.view(-1, 1)  # [batch_size, 1]


# ============================================================================
# 预测器
# ============================================================================
class GCNPredictor:
    """
    GCN预测器 - 用于Streamlit应用集成
    加载您训练好的模型: gcn_egfr_best_model.pth

    模型结构与训练脚本完全一致，键名无需重映射。
    """

    def __init__(self, model_path=None, device=None):
        """
        参数:
            model_path: 模型文件路径，默认为项目目录下的 gcn_egfr_best_model.pth
            device: 'cpu' 或 'cuda'，默认自动检测
        """
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模型路径
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'gcn_egfr_best_model.pth')

        self.model_path = model_path
        self.num_node_features = 13   # 训练时使用的原子特征数（匹配 conv1.lin.weight[128,13]）
        self.hidden_dim = 128         # 隐藏层维度

        # 初始化并加载权重
        self.model = GCNModel(
            num_node_features=self.num_node_features,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        if os.path.exists(model_path):
            self._load_weights()
        else:
            logger.warning(f"⚠️ 模型文件未找到: {model_path}，使用随机初始化权重")

        self.model.eval()
        logger.info(f"✅ GCN预测器初始化完成，设备: {self.device}")

    def _load_weights(self):
        """加载预训练权重（严格模式，键名完全匹配）"""
        logger.info(f"正在加载模型: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✓ 模型权重加载成功（严格模式），总参数量: {total_params:,}")

    def _smiles_to_graph(self, smiles):
        """
        将SMILES转换为PyTorch Geometric Data对象。
        原子特征与训练模型一致（13维，匹配 conv1.lin.weight 形状 [128,13]）。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES字符串: {smiles}")

        # ---- 构建13维原子特征（匹配训练模型 input_dim=13） ----
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                float(atom.GetAtomicNum()),             # 1. 原子序数
                float(atom.GetDegree()),                # 2. 度（连接数）
                float(atom.GetFormalCharge()),          # 3. 形式电荷
                float(atom.GetHybridization().real),    # 4. 杂化类型
                float(atom.GetIsAromatic()),            # 5. 芳香性
                float(atom.GetTotalNumHs()),            # 6. 总氢原子数
                float(atom.GetImplicitValence()),       # 7. 隐式价
                float(atom.GetNumRadicalElectrons()),   # 8. 自由基电子数
                float(atom.GetIsotope()),               # 9. 同位素
                float(atom.GetMass() / 100.0),          # 10. 原子质量（归一化）
                float(atom.GetTotalValence()),          # 11. 总化合价
                1.0 if atom.GetNumImplicitHs() > 0 else 0.0,  # 12. 氢键供体
                1.0 if atom.GetAtomicNum() in [7, 8] else 0.0, # 13. 氢键受体 (N,O)
            ]
            atom_features.append(feat)

        if not atom_features:
            raise ValueError("分子无原子")

        x = torch.tensor(atom_features, dtype=torch.float)

        # ---- 构建边索引 ----
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        if not edges:
            edges = [[0, 0]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # ---- 构建 Data ----
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.size(0), dtype=torch.long)

        logger.debug(f"图数据: {len(atom_features)}原子, {len(edges)//2}边")
        return data

    def predict(self, smiles, return_details=False):
        """
        预测单个分子的EGFR抑制活性。

        返回:
            dict: {
                "success": bool,
                "prediction": 0/1,
                "prediction_label": "活性"/"非活性",
                "probability_active": float,
                ...
            }
        """
        t_start = time.perf_counter()

        try:
            data = self._smiles_to_graph(smiles)
            data = data.to(self.device)

            with torch.no_grad():
                logit = self.model(data)
                probability = torch.sigmoid(logit).item()
            prediction = 1 if probability > 0.5 else 0

            # 置信度
            if abs(probability - 0.5) > 0.3:
                confidence, confidence_score = "高", 0.9
            elif abs(probability - 0.5) > 0.15:
                confidence, confidence_score = "中", 0.7
            else:
                confidence, confidence_score = "低", 0.5

            inference_time_ms = (time.perf_counter() - t_start) * 1000

            result = {
                "success": True,
                "smiles": smiles,
                "prediction": prediction,
                "prediction_label": "活性" if prediction == 1 else "非活性",
                "probability_active": probability,
                "probability_inactive": 1 - probability,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "model_type": "GCN (图卷积网络)",
                "model_auc": 0.8081,
                "model_accuracy": 0.7652,
                "timestamp": np.datetime64('now'),
            }

            if return_details:
                result.update({
                    "num_atoms": data.x.size(0),
                    "num_bonds": data.edge_index.size(1) // 2,
                    "inference_time_ms": round(inference_time_ms, 2),
                    "device": str(self.device),
                    "model_path": self.model_path,
                })

            logger.info(f"预测: {smiles[:30]}... -> {result['prediction_label']} ({probability:.3f})")
            return result

        except Exception as e:
            logger.error(f"预测失败: {smiles[:30]}... - {e}")
            return {
                "success": False,
                "smiles": smiles,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def batch_predict(self, smiles_list, batch_size=32):
        """
        批量预测多个分子。

        参数:
            smiles_list: SMILES 字符串列表
            batch_size: 批次大小

        返回:
            预测结果列表
        """
        results = []
        total = len(smiles_list)
        logger.info(f"开始批量预测: {total}个分子")

        for i in range(0, total, batch_size):
            batch = smiles_list[i:i + batch_size]
            for smiles in batch:
                results.append(self.predict(smiles))
            processed = min(i + batch_size, total)
            logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")

        return results

    def test_model(self, test_smiles=None):
        """测试模型是否能正常预测"""
        if test_smiles is None:
            test_smiles = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"  # 吉非替尼

        logger.info("🧪 开始模型测试...")
        logger.info(f"   模型类型: {type(self.model).__name__}")
        logger.info(f"   输入维度: {self.num_node_features}")
        logger.info(f"   隐藏层维度: {self.hidden_dim}")

        result = self.predict(test_smiles, return_details=True)
        if result["success"]:
            logger.info(f"   ✅ 测试通过!")
            logger.info(f"   预测结果: {result['prediction_label']}")
            logger.info(f"   活性概率: {result['probability_active']:.4f}")
            logger.info(f"   置信度: {result['confidence']}")
            if "inference_time_ms" in result:
                logger.info(f"   推理时间: {result['inference_time_ms']} ms")
        else:
            logger.error(f"   ❌ 测试失败: {result['error']}")

        return result


# ========== 命令行测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🧬 GNN预测器测试")
    print("=" * 60)

    try:
        predictor = GCNPredictor()
        print("✅ 预测器初始化成功")
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        exit(1)

    test_result = predictor.test_model()

    print("\n📋 示例预测:")
    examples = [
        "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",  # 高活性EGFR抑制剂
        "CC(=O)OC1=CC=CC=C1C(=O)O",           # 阿司匹林 (非活性)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       # 咖啡因 (非活性)
    ]
    for smiles in examples:
        result = predictor.predict(smiles)
        status = "✅" if result["success"] else "❌"
        label = result.get("prediction_label", "错误")
        prob = result.get("probability_active", 0)
        print(f"  {status} {smiles[:30]:30} -> {label:8} ({prob:.3f})")

    print("\n" + "=" * 60)
    print("🚀 GNN预测器准备就绪!")
    print("=" * 60)
