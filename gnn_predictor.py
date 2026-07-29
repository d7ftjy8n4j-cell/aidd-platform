"""
gnn_predictor.py - GNN预测器 (适配您训练好的EGFR GCN模型)
用于Streamlit应用集成，将SMILES字符串转换为分子图并进行活性预测
作者：dadamingli
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Module, ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCNModel(Module):
    """
    与您训练的GCN模型完全一致的架构
    输入维度: 12, 隐藏层维度: 128, 输出维度: 1
    """
    
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=3, dropout=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建GCN卷积层 - 注意这里的层结构
        self.convs = ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 批归一化层
        self.bns = ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm1d(hidden_dim))
        
        # 全连接层
        self.lin1 = Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = Linear(hidden_dim // 2, 1)
        
        logger.info(f"初始化GCN模型: {input_dim} -> {hidden_dim} -> 1")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN层
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        # 全局平均池化
        x = global_mean_pool(x, batch)
        
        # 全连接层
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x

class GCNPredictor:
    """
    GCN预测器 - 用于Streamlit应用集成
    加载您训练好的完整模型: gcn_egfr_complete_model.pth
    """
    
    def __init__(self, model_path=None, device=None):
        """
        初始化预测器

        参数:
            model_path: 模型文件路径，默认为当前目录的模型文件
            device: 设备 ('cpu' 或 'cuda')，自动检测
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 设置模型路径
        if model_path is None:
            # 获取当前文件所在目录，然后相对于它找到模型文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'gcn_egfr_best_model.pth')

        self.model_path = model_path
        self.model = None
        self.input_dim = 12  # 与您的模型输入维度一致
        self.hidden_dim = 128  # 与您的模型隐藏层维度一致

        # 加载模型
        self._load_model()
        logger.info(f"✅ GCN预测器初始化完成，使用设备: {self.device}")
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件未找到: {self.model_path}")

            logger.info(f"正在加载模型: {self.model_path}")

            # 始终使用 state_dict 方式加载，更稳定可靠
            # 先创建模型架构
            self.model = GCNModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=3,
                dropout=0.5
            )

            # 加载 state_dict
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # ---- 键名兼容：旧模型 (conv1/conv2/conv3/bn1/bn2) -> 新模型 (convs.N/bns.N) ----
            remapped = self._remap_state_dict_keys(state_dict)
            try:
                self.model.load_state_dict(remapped, strict=True)
                logger.info("✓ 模型参数加载成功（严格模式）")
            except Exception as e:
                logger.warning(f"严格模式加载失败: {e}")
                try:
                    self.model.load_state_dict(remapped, strict=False)
                    missing, unexpected = self.model.load_state_dict(remapped, strict=False)
                    if missing:
                        logger.warning(f"缺少的键 (将被随机初始化): {missing}")
                    if unexpected:
                        logger.warning(f"多余的键 (已忽略): {unexpected}")
                except Exception as e2:
                    logger.error(f"模型加载完全失败: {e2}")
                    raise

            # 设置模型为评估模式
            self.model.to(self.device)
            self.model.eval()

            # 验证模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"模型总参数量: {total_params:,}")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    @staticmethod
    def _remap_state_dict_keys(state_dict):
        """
        将旧格式键名转换为新格式，确保与保存的权重兼容。

        旧格式: conv1.weight, conv2.bias, bn1.weight, bn2.bias, conv3.lin.weight ...
        新格式: convs.0.weight, convs.1.bias, bns.0.weight, bns.1.bias, convs.2.lin.weight ...

        如果已经是新格式，则原样返回。
        """
        import re

        # 快速检查：如果第一个键已经是新格式则跳过
        sample_key = next(iter(state_dict.keys()), "")
        if re.match(r"^(convs|bns)\.\d+\.", sample_key):
            return state_dict

        new_dict = {}
        for key, value in state_dict.items():
            new_key = key

            # conv1 -> convs.0, conv2 -> convs.1, conv3 -> convs.2
            m = re.match(r"^conv(\d+)\.(.+)$", key)
            if m:
                layer_num = int(m.group(1)) - 1
                new_key = f"convs.{layer_num}.{m.group(2)}"

            # bn1 -> bns.0, bn2 -> bns.1
            m = re.match(r"^bn(\d+)\.(.+)$", key)
            if m:
                layer_num = int(m.group(1)) - 1
                new_key = f"bns.{layer_num}.{m.group(2)}"

            if new_key != key:
                logger.debug(f"键名重映射: {key} -> {new_key}")
            new_dict[new_key] = value

        remapped_count = sum(1 for k, nk in zip(state_dict.keys(), new_dict.keys()) if k != nk)
        if remapped_count > 0:
            logger.info(f"✓ 已重映射 {remapped_count} 个权重键名 (旧格式 -> 新格式)")

        return new_dict

    def _smiles_to_graph(self, smiles):
        """
        将SMILES字符串转换为PyTorch Geometric图数据
        
        参数:
            smiles: SMILES字符串
        
        返回:
            PyTorch Geometric Data对象
        """
        try:
            # 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"无效的SMILES字符串: {smiles}")
            
            # ========== 关键：提取原子特征（必须与训练时完全一致）==========
            # 您的模型输入维度为12，这里提取12个特征
            atom_features = []
            for atom in mol.GetAtoms():
                # 特征列表 - 维度必须为12
                features = [
                    float(atom.GetAtomicNum()),          # 1. 原子序数
                    float(atom.GetDegree()),             # 2. 度（连接数）
                    float(atom.GetFormalCharge()),       # 3. 形式电荷
                    float(atom.GetHybridization().real), # 4. 杂化类型
                    float(atom.GetIsAromatic()),         # 5. 是否是芳香原子
                    float(atom.GetTotalNumHs()),         # 6. 总氢原子数
                    float(atom.GetImplicitValence()),    # 7. 隐式价
                    float(atom.GetNumRadicalElectrons()), # 8. 自由基电子数
                    float(atom.GetIsotope()),            # 9. 同位素
                    float(atom.GetMass() / 100.0),       # 10. 原子质量（归一化）
                    # 补充特征到12维
                    1.0 if atom.GetNumImplicitHs() > 0 else 0.0,  # 11. 氢键供体（简化）
                    1.0 if atom.GetAtomicNum() in [7, 8] else 0.0, # 12. 氢键受体（N, O）
                ]
                
                # 验证特征维度
                if len(features) != self.input_dim:
                    raise ValueError(f"特征维度错误: 期望{self.input_dim}, 实际{len(features)}")
                
                atom_features.append(features)
            
            # 节点特征矩阵
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # ========== 构建边索引 ==========
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                # 无向图，添加双向边
                edge_indices.append([i, j])
                edge_indices.append([j, i])
            
            # 处理单原子分子特殊情况
            if len(edge_indices) == 0:
                edge_indices.append([0, 0])
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # ========== 创建Data对象 ==========
            data = Data(
                x=x,
                edge_index=edge_index,
                smiles=smiles  # 保存SMILES用于调试
            )
            
            # 添加batch维度（单个分子）
            data.batch = torch.zeros(x.size(0), dtype=torch.long)
            
            logger.debug(f"图数据创建成功: {len(atom_features)}个原子, {len(edge_indices)//2}条边")
            return data
            
        except Exception as e:
            logger.error(f"SMILES转图失败: {smiles} - {e}")
            raise
    
    def predict(self, smiles, return_details=False):
        """
        预测单个分子的EGFR抑制活性
        
        参数:
            smiles: SMILES字符串
            return_details: 是否返回详细信息
        
        返回:
            字典格式的预测结果
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        try:
            # 1. 转换为图数据
            data = self._smiles_to_graph(smiles)
            data = data.to(self.device)
            
            # 2. 预测
            with torch.no_grad():
                out = self.model(data)
                probability = torch.sigmoid(out).item()
                prediction = 1 if probability > 0.5 else 0
            
            # 3. 计算置信度
            if abs(probability - 0.5) > 0.3:
                confidence = "高"
                confidence_score = 0.9
            elif abs(probability - 0.5) > 0.15:
                confidence = "中"
                confidence_score = 0.7
            else:
                confidence = "低"
                confidence_score = 0.5
            
            # 4. 计算推理时间
            inference_time = None
            if torch.cuda.is_available() and start_time:
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
            
            # 5. 组装结果
            result = {
                "success": True,
                "smiles": smiles,
                "prediction": prediction,  # 0=非活性, 1=活性
                "prediction_label": "活性" if prediction == 1 else "非活性",
                "probability_active": probability,
                "probability_inactive": 1 - probability,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "model_type": "GCN (图卷积网络)",
                "model_auc": 0.8081,  # 您的测试集AUC
                "model_accuracy": 0.7652,  # 您的测试集准确率
                "timestamp": np.datetime64('now')
            }
            
            # 添加详细信息（如果请求）
            if return_details:
                result.update({
                    "num_atoms": data.x.size(0),
                    "num_bonds": data.edge_index.size(1) // 2,
                    "inference_time_ms": inference_time,
                    "device": str(self.device),
                    "model_path": self.model_path
                })
            
            logger.info(f"预测成功: {smiles[:30]}... -> {result['prediction_label']} ({probability:.3f})")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "smiles": smiles,
                "error": str(e),
                "error_type": type(e).__name__
            }
            logger.error(f"预测失败: {smiles[:30]}... - {e}")
            return error_result
    
    def batch_predict(self, smiles_list, batch_size=32):
        """
        批量预测多个分子
        
        参数:
            smiles_list: SMILES字符串列表
            batch_size: 批处理大小
        
        返回:
            预测结果列表
        """
        results = []
        total = len(smiles_list)
        
        logger.info(f"开始批量预测: {total}个分子")
        
        for i in range(0, total, batch_size):
            batch = smiles_list[i:i+batch_size]
            batch_results = []
            
            for smiles in batch:
                result = self.predict(smiles)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 打印进度
            processed = min(i + batch_size, total)
            logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")
        
        return results
    
    def test_model(self, test_smiles=None):
        """测试模型是否正常工作"""
        if test_smiles is None:
            test_smiles = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"  # 吉非替尼
        
        logger.info("🧪 开始模型测试...")
        
        # 测试1: 模型架构
        logger.info("1. 检查模型架构...")
        logger.info(f"   模型类型: {type(self.model).__name__}")
        logger.info(f"   输入维度: {self.input_dim}")
        logger.info(f"   隐藏层维度: {self.hidden_dim}")
        
        # 测试2: 预测功能
        logger.info("2. 测试预测功能...")
        result = self.predict(test_smiles, return_details=True)
        
        if result["success"]:
            logger.info(f"   ✅ 测试通过!")
            logger.info(f"   测试分子: {test_smiles[:50]}...")
            logger.info(f"   预测结果: {result['prediction_label']}")
            logger.info(f"   活性概率: {result['probability_active']:.4f}")
            logger.info(f"   置信度: {result['confidence']}")
            
            if "inference_time_ms" in result and result["inference_time_ms"] is not None:
                logger.info(f"   推理时间: {result['inference_time_ms']:.1f} ms")
        else:
            logger.error(f"   ❌ 测试失败: {result['error']}")
        
        return result

# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🧬 GNN预测器测试")
    print("=" * 60)
    
    # 初始化预测器
    try:
        predictor = GCNPredictor()
        print("✅ 预测器初始化成功")
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        exit(1)
    
    # 测试模型
    test_result = predictor.test_model()
    
    # 示例预测
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
    print("🚀 GNN预测器准备就绪，可集成到Streamlit应用!")
    print("=" * 60)