"""
真实EGFR抑制剂预测引擎
"""
import joblib
import numpy as np
import json
import os
import sys
import re
import warnings

# sklearn 升级后加载旧版本训练的模型会刷 InconsistentVersionWarning。
# 模型已验证可用（见 tests/），仅静默该提示类警告。
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator .* from version .* when using version",
    category=UserWarning,
)

# Windows GBK 控制台防御：emoji print 在 GBK 编码下会抛 UnicodeEncodeError
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 尝试导入RDKit，如果不可用则设置标志
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit 不可用，将使用备用分子分析方法")

# 添加当前目录到路径，以便导入重建脚本
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimal_predictor import TIER_REAL  # noqa: E402  # 层级标签（P0-03）


class RealEGFRPredictor:
    def __init__(self):
        """直接从当前目录加载模型和特征"""
        self.model = None
        self.feature_names = []
        self.predictor_tier = TIER_REAL
        
        try:
            # 获取当前文件所在目录，然后相对于它找到模型文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"📁 当前目录: {current_dir}")

            # 加载模型（若依赖环境损坏，则走兜底逻辑）
            model_path = os.path.join(current_dir, "rf_egfr_model_final.pkl")
            # 如果存在兼容模型，优先使用兼容模型
            compatible_model_path = os.path.join(current_dir, "rf_egfr_model_compatible.pkl")

            print(f"🔍 检查模型文件...")
            print(f"   原始模型: {model_path} (存在: {os.path.exists(model_path)})")
            print(f"   兼容模型: {compatible_model_path} (存在: {os.path.exists(compatible_model_path)})")

            # 确定使用哪个模型
            if os.path.exists(compatible_model_path):
                model_path = compatible_model_path
                print(f"✅ 使用兼容模型")
            elif not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 加载模型
            print(f"📦 开始加载模型: {model_path}")
            self.model = joblib.load(model_path)
            print(f"✅ 模型加载成功")
            print(f"   模型类型: {type(self.model).__name__}")

            # 加载特征名称
            feature_path = os.path.join(current_dir, "feature_names.json")
            print(f"\n📋 加载特征文件: {feature_path}")
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"特征文件不存在: {feature_path}")

            with open(feature_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            print(f"✅ 加载 {len(self.feature_names)} 个特征")

            # 验证
            if hasattr(self.model, 'n_features_in_'):
                print(f"   模型期望特征数: {self.model.n_features_in_}")
                if len(self.feature_names) != self.model.n_features_in_:
                    print(f"⚠️ 警告：特征数量不匹配！")
                    print(f"   特征文件: {len(self.feature_names)} 个")
                    print(f"   模型期望: {self.model.n_features_in_} 个")

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print(f"\n错误详情:")
            import traceback
            traceback.print_exc()
            
            # P0-03/F-15：不再用合成数据伪造模型（_rebuild_model 已删除）。
            # model 保持 None，app.py 降级链会接管到确定性规则基线并在 UI 显式示警。
            print("\n🚫 模型加载失败：已放弃伪造模型，降级链将使用确定性规则基线并显式示警。")
            self.model = None
            self.feature_names = []
    
    def smiles_to_features(self, smiles):
        """将SMILES转换为模型所需的特征向量"""
        if not self.model:
            return None
        
        # 如果有RDKit，使用RDKit计算特征
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print(f"❌ 无效的SMILES: {smiles}")
                return None
            
            features = []
            for feat_name in self.feature_names:
                try:
                    func = getattr(Descriptors, feat_name)
                    value = func(mol)
                    features.append(float(value) if value is not None and not (isinstance(value, float) and np.isnan(value)) else 0.0)
                except:
                    features.append(0.0)
            
            return np.array(features).reshape(1, -1)
        else:
            # 无RDKit时，使用简单SMILES分析
            return self._smiles_to_features_simple(smiles)
    
    def _smiles_to_features_simple(self, smiles):
        """简单的SMILES特征提取（不依赖RDKit）"""
        length = len(smiles)
        
        # 原子计数（粗略估计）
        c_count = smiles.count('C') - smiles.count('Cl')
        n_count = smiles.count('N')
        o_count = smiles.count('O')
        s_count = smiles.count('S')
        f_count = smiles.count('F')
        cl_count = smiles.count('Cl')
        br_count = smiles.count('Br')
        
        # 键计数
        double_bonds = smiles.count('=')
        triple_bonds = smiles.count('#')
        
        # 分支
        branch_start = smiles.count('(')
        branch_end = smiles.count(')')
        
        # 环
        ring_digits = sum(1 for c in smiles if c.isdigit())
        rings = ring_digits // 2
        
        # 芳香性
        aromatic_c = smiles.count('c')
        aromatic_n = smiles.count('n')
        aromatic_o = smiles.count('o')
        
        # 根据 feature_names 的顺序返回特征
        feature_map = {
            "SMILES长度": length,
            "碳原子数": max(c_count, 0),
            "氮原子数": n_count,
            "氧原子数": o_count,
            "硫原子数": s_count,
            "氟原子数": f_count,
            "氯原子数": cl_count,
            "溴原子数": br_count,
            "双键数": double_bonds,
            "三键数": triple_bonds,
            "分支开始": branch_start,
            "分支结束": branch_end,
            "环数": rings,
            "芳香碳": aromatic_c,
            "芳香氮": aromatic_n,
            "芳香氧": aromatic_o
        }
        
        features = [feature_map.get(name, 0.0) for name in self.feature_names]
        return np.array(features).reshape(1, -1)
    
    def predict(self, smiles):
        """主预测函数"""
        if not self.model:
            return {"error": "模型未加载"}
        
        features = self.smiles_to_features(smiles)
        if features is None:
            return {"error": "SMILES解析失败"}
        
        try:
            # 预测
            proba = self.model.predict_proba(features)[0]
            pred_class = int(proba[1] > 0.5)
            
            # 特征重要性解释
            explanation = None
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                # 取最重要的5个特征
                top5_idx = importances.argsort()[-5:][::-1]
                explanation = {
                    "top_features": [self.feature_names[i] for i in top5_idx],
                    "top_importance": [importances[i] for i in top5_idx],
                    "values": {self.feature_names[i]: features[0][i] for i in top5_idx}
                }
            
            return {
                "success": True,
                "smiles": smiles,
                "prediction": pred_class,  # 0=非活性, 1=活性
                "probability_active": float(proba[1]),
                "confidence": "高" if abs(proba[1]-0.5) > 0.3 else "中",
                "predictor_tier": TIER_REAL,
                "explanation": explanation,
                "features_used": self.feature_names,
                "feature_values": features[0].tolist()
            }
            
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}

# 测试代码
if __name__ == "__main__":
    # 测试
    print("🧪 测试真实预测器...")
    
    # 初始化
    predictor = RealEGFRPredictor()
    
    # 测试吉非替尼
    gefitinib = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
    result = predictor.predict(gefitinib)
    
    if "error" in result:
        print(f"❌ 测试失败: {result['error']}")
    else:
        print(f"✅ 测试成功!")
        print(f"SMILES: {result['smiles'][:50]}...")
        print(f"预测: {'活性' if result['prediction']==1 else '非活性'}")
        print(f"活性概率: {result['probability_active']:.3f}")
        
        if result['explanation']:
            print("\n🔬 最重要的5个特征:")
            for i, (feat, imp) in enumerate(zip(result['explanation']['top_features'], 
                                               result['explanation']['importance']), 1):
                val = result['explanation']['values'][feat]
                print(f"  {i}. {feat}: 值={val:.2f}, 重要性={imp:.4f}")