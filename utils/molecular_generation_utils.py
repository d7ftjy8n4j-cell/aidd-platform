# utils/molecular_generation_utils.py
"""
基于字符级 LSTM 的分子生成模块 (SMILES-RNN)

参考: TeachOpenCADD T034 (RNN-based property prediction) 的 LSTM 架构，
      REINVENT (Olivecrona et al., 2017) 的自回归 SMILES 生成范式。

核心教学概念:
    - 化学语言模型: SMILES 被视为一种"语言"，RNN 学习其"语法"
    - 自回归生成:  逐字符采样，'~' 为终止符
    - 温度采样:    控制生成分子的多样性 vs 有效性
    - 迁移学习:    在 EGFR 抑制剂上微调，展示聚焦库设计

使用方式:
    trainer = MolecularGenerator()
    trainer.train_many(smiles_list, epochs=30)
    generated = trainer.generate("C", temperature=0.8, num_samples=10)
"""

import os
import json
import random
import logging
from typing import List, Optional, Dict, Tuple

import numpy as np

# ---------- PyTorch 降级策略 ----------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None      # type: ignore

from collections import Counter

# RDKit 用于 SMILES 有效性校验
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False


# ---------- 默认训练数据: EGFR 抑制剂 SMILES 子集 ----------

DEFAULT_SMILES = [
    # Gefitinib 系列
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCCCN1CCOCC1",
    "COc1cc2c(cc1OCCCN1CCOCC1)ncnc2Nc1ccc(F)c(Cl)c1",
    # Erlotinib 系列
    "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(C#C)c1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    # Lapatinib 系列
    "CS(=O)(=O)CCNCc1ccc(-c2cc3ncnc(Nc4ccc(Cl)c(OCC5ccccn5)c4)c3cc2F)o1",
    # Osimertinib 系列
    "CN1CCN(c2ccc(Nc3nccc(-c4ccccc4)n3)c(OC)c2)CC1",
    "COc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)ccc1N1CCN(C)CC1",
    # Afatinib 系列
    "CN(C)C=CC(=O)Nc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1O[C@H]1CCOC1",
    # Dacomitinib 系列
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)C=C",
    # 类药性小分子 (EGFR 激酶抑制剂常见骨架)
    "Cn1cnc2c(Nc3ccccc3)ncnc12",
    "O=C(Nc1ccccc1)c1cccc(-c2ccccc2)c1",
    "Nc1ncnc2c1ncn2-c1ccccc1",
    "Cc1cc(Nc2nccc(-c3ccccc3)n2)ccc1OC",
    "Cc1ccc(Nc2nc(-c3ccccc3)cs2)cc1",
    "COc1cccc(NC(=O)C=Cc2ccccc2)c1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "O=C1CC(c2cccc(F)c2)c2ccccc2N1",
    "CN1CCN(C(=O)C=Cc2ccccc2)CC1",
    "CC(C)(C)c1ccc(C(=O)Nc2ccccc2)o1",
    "Cc1cnc(NC(=O)c2ccccc2)s1",
    "O=C(Nc1cccc(Cl)c1)c1ccccn1",
    "CS(=O)(=O)Nc1ccc(Oc2ccccc2)cc1",
    "Cc1noc(C)c1S(=O)(=O)Nc1ccccc1",
]

# 补充更多骨架多样性
_extra_pyrimidine = [
    "Cc1cccc(Nc2ncnc3ccccc23)c1",
    "COc1ccc(Nc2ncnc3ccc(OC)cc23)cc1",
    "Clc1ccc(Nc2ncnc3ccccc23)cc1",
    "Fc1cccc(Nc2ncnc3cc(OCCO)ccc23)c1",
    "OCCNc1ncnc2cc(OCC3CC3)ccc12",
]
_extra_purin = [
    "Nc1nc2c(ncn2C2CCCC2)c(=O)[nH]1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "Cc1nc2c(n1C)nc(N)nc2=O",
]
_extra_quinazoline = [
    "COc1cc2ncnc(Nc3ccc(Br)cc3)c2cc1OC",
    "COc1cc2ncnc(Nc3ccc(OC)c(OC)c3)c2cc1OC",
    "COc1cc2ncnc(Nc3ccc(C)cc3)c2cc1OCCO",
]
DEFAULT_SMILES.extend(_extra_pyrimidine + _extra_purin + _extra_quinazoline)


# ---------- 字符级 LSTM 模型 ----------

# torch 不可用时 nn 为 None，class CharRNN(nn.Module) 会抛 AttributeError。
# 用占位基类保证模块在 torch-less 环境可导入（页面据此显示“PyTorch 未安装”）。
_ModuleBase = nn.Module if TORCH_AVAILABLE else object


class CharRNN(_ModuleBase):
    """
    字符级 LSTM 分子生成器 (REINVENT 架构简化版)

    输入:   (batch, seq_len) 字符索引
    嵌入:   (batch, seq_len, embed_size)
    LSTM:   (batch, seq_len, hidden_size) → 输出 + (h, c)
    全连接: (batch, seq_len, vocab_size) → 下个字符的 logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        x: (batch, seq_len) or (batch, 1) for single-step
        hidden: (h, c) each (num_layers, batch, hidden_size)
        """
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ---------- 训练数据集构建 ----------

def _build_vocab(smiles_list: List[str]) -> List[str]:
    """从 SMILES 列表中提取统一字符集"""
    chars = set()
    for s in smiles_list:
        chars.update(s)
    chars = sorted(chars)
    # 确保 '~' (终止符) 和 ' ' (填充) 在词表中
    vocab = [" ", "~"] + [c for c in chars if c not in (" ", "~")]
    return vocab


def _smiles_to_tensor(
    smiles_list: List[str], vocab: List[str], char_to_idx: Dict[str, int],
    max_len: int = 100,
) -> List[torch.Tensor]:
    """将 SMILES 列表转为索引张量 (含终止符)，超过 max_len 则截断"""
    tensors = []
    for s in smiles_list:
        # 预留终止符位置
        max_body_len = max_len - 1
        if len(s) > max_body_len:
            s = s[:max_body_len]
        indices = [char_to_idx.get(c, 0) for c in s] + [char_to_idx["~"]]
        tensors.append(torch.tensor(indices[:max_len], dtype=torch.long))
    return tensors


def _collate_sequences(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    变长序列批处理: 填充到相同长度，生成 (input, target) 对
    input:  [pad, c1, c2, ..., cn]
    target: [c1, c2, ..., cn, pad]
    """
    max_len = max(len(t) for t in tensors)
    batch_size = len(tensors)
    inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, t in enumerate(tensors):
        # 填充到左侧 (或右侧, 不影响因果性)
        seq_len = len(t)
        inputs[i, :seq_len] = t
        # target 右移一位: 预测下个字符
        targets[i, :seq_len - 1] = t[1:]
        # 忽略最后一个 target (预测终止符之后)

    return inputs, targets


# ---------- 主类 ----------

class MolecularGenerator:
    """
    分子生成器: 训练 + 生成 + 微调

    Attributes
    ----------
    vocab : list[str]
        字符词表 (含填充符 ' ' 和终止符 '~')
    model : CharRNN
        PyTorch LSTM 模型
    trained : bool
        是否已完成训练
    """

    def __init__(self):
        self.vocab: List[str] = []
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.model: Optional[CharRNN] = None
        self.trained: bool = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_stats: Dict = {}

    # ---------- 词表 ----------

    def _init_vocab(self, smiles_list: List[str]):
        self.vocab = _build_vocab(smiles_list)
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    # ---------- 编解码 ----------

    def _encode(self, smiles: str) -> torch.Tensor:
        """SMILES → 索引张量 (含终止符)"""
        indices = [self.char_to_idx.get(c, 0) for c in smiles] + [self.char_to_idx["~"]]
        return torch.tensor(indices, dtype=torch.long)

    def _decode(self, indices: torch.Tensor) -> str:
        """索引张量 → SMILES (截断到终止符)"""
        chars = []
        for i in indices.cpu().numpy():
            c = self.idx_to_char.get(int(i), "")
            if c == "~":
                break
            chars.append(c)
        return "".join(chars)

    # ---------- 训练 ----------

    def train_many(
        self,
        smiles_list: List[str],
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        progress_callback=None,
    ) -> Dict:
        """
        从零训练字符级 LSTM 分子生成模型。

        参数
        ----------
        smiles_list : list[str]
            训练 SMILES 列表 (必须全部通过 RDKit 有效性检查)
        epochs : int
            训练轮数，默认 30
        batch_size : int
            批大小，默认 16
        learning_rate : float
            学习率，默认 0.001
        progress_callback : callable, optional
            callback(epoch, loss)

        返回
        -------
        dict : {"vocab_size": int, "epochs": int, "final_loss": float}
        """
        # 过滤无效 SMILES
        valid_smiles = []
        if RDKIT_OK:
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s.strip())
                if mol is not None:
                    valid_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        else:
            valid_smiles = [s.strip() for s in smiles_list]

        if len(valid_smiles) < 10:
            raise ValueError(
                f"有效 SMILES 不足 ({len(valid_smiles)} < 10)，"
                f"至少需要 10 个有效分子用于训练"
            )

        logging.info(f"训练集: {len(valid_smiles)} 个有效 SMILES")

        # 初始化词表
        self._init_vocab(valid_smiles)
        logging.info(f"词表大小: {self.vocab_size} (字符: {''.join(self.vocab[:30])}...)")

        # 构建模型
        self.model = CharRNN(
            self.vocab_size,
            embed_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.2,
        ).to(self.device)

        # 准备数据
        tensors = _smiles_to_tensor(valid_smiles, self.vocab, self.char_to_idx)

        # 训练循环
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充 token

        n_batches = max(1, (len(tensors) + batch_size - 1) // batch_size)  # 向上取整，与实际循环批次数一致
        losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            random.shuffle(tensors)

            epoch_loss = 0.0
            for i in range(0, len(tensors), batch_size):
                batch_tensors = tensors[i:i + batch_size]
                inputs, targets = _collate_sequences(batch_tensors)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(inputs)  # (batch, seq, vocab)
                logits = logits.view(-1, self.vocab_size)
                targets = targets.view(-1)
                loss = criterion(logits, targets)
                loss.backward()
                # 梯度裁剪防爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if progress_callback:
                progress_callback(epoch, avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                logging.info(f"  Epoch {epoch:3d}/{epochs}  Loss: {avg_loss:.4f}")

        self.model.eval()
        self.trained = True
        self.training_stats = {
            "vocab_size": self.vocab_size,
            "n_smiles": len(valid_smiles),
            "epochs": epochs,
            "final_loss": losses[-1],
            "loss_history": losses,
        }

        return self.training_stats

    # ---------- 微调 (迁移学习) ----------

    def fine_tune(
        self,
        smiles_list: List[str],
        epochs: int = 10,
        learning_rate: float = 0.0003,
        progress_callback=None,
    ) -> Dict:
        """
        在已有模型基础上微调 (迁移学习演示)。
        必须先调用 train_many() 或 load_model()。

        参数同 train_many()，返回更新后的 training_stats。
        """
        if not self.trained or self.model is None:
            raise RuntimeError("请先训练或加载模型，再执行微调")

        valid_smiles = []
        if RDKIT_OK:
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s.strip())
                if mol is not None:
                    valid_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        else:
            valid_smiles = [s.strip() for s in smiles_list]

        if len(valid_smiles) < 5:
            raise ValueError("微调需要至少 5 个有效 SMILES")

        logging.info(f"微调集: {len(valid_smiles)} 个 SMILES (epochs={epochs})")

        # 扩展词表 (如果有新字符)
        new_chars = set()
        for s in valid_smiles:
            new_chars.update(s)
        old_chars = set(self.vocab)
        added = new_chars - old_chars
        if added:
            logging.warning(f"微调数据含新字符: {added}，将扩展词表并重建嵌入层")
            self._extend_vocab(added)

        tensors = _smiles_to_tensor(valid_smiles, self.vocab, self.char_to_idx)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        batch_size = min(16, len(tensors))
        n_batches = max(1, (len(tensors) + batch_size - 1) // batch_size)  # 向上取整，与实际循环批次数一致
        losses = []

        for epoch in range(1, epochs + 1):
            random.shuffle(tensors)
            epoch_loss = 0.0
            for i in range(0, len(tensors), batch_size):
                batch_tensors = tensors[i:i + batch_size]
                inputs, targets = _collate_sequences(batch_tensors)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(inputs)
                loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            if progress_callback:
                progress_callback(epoch, avg_loss)

        self.model.eval()
        self.training_stats["fine_tune_loss"] = losses[-1]
        self.training_stats["fine_tune_smiles"] = len(valid_smiles)
        self.training_stats["fine_tune_epochs"] = epochs

        return self.training_stats

    def _extend_vocab(self, new_chars: set):
        """扩展词表以支持新字符 (用于微调场景)"""
        old_size = self.vocab_size
        for c in sorted(new_chars):
            if c not in self.char_to_idx:
                idx = len(self.vocab)
                self.vocab.append(c)
                self.char_to_idx[c] = idx
                self.idx_to_char[idx] = c

        self.vocab_size = len(self.vocab)
        if self.vocab_size > old_size:
            # 重建嵌入和输出层
            old_embed = self.model.embedding
            old_fc = self.model.fc

            new_embed = nn.Embedding(self.vocab_size, self.model.embed_size, padding_idx=0).to(self.device)
            new_fc = nn.Linear(self.model.hidden_size, self.vocab_size).to(self.device)

            # 复制旧权重
            with torch.no_grad():
                new_embed.weight[:old_size] = old_embed.weight
                new_fc.weight[:old_size] = old_fc.weight
                new_fc.bias[:old_size] = old_fc.bias

            self.model.embedding = new_embed.to(self.device)
            self.model.fc = new_fc.to(self.device)

    # ---------- 生成 ----------

    def generate(
        self,
        seed: str = "",
        temperature: float = 0.8,
        max_length: int = 120,
        num_samples: int = 10,
    ) -> List[Dict]:
        """
        自回归生成 SMILES 分子。

        参数
        ----------
        seed : str
            起始 SMILES 前缀 (空 = 从头生成)
        temperature : float
            采样温度。>1: 更随机多样; <1: 更保守有效; =0.0: 贪心解码
        max_length : int
            生成的 SMILES 最大长度
        num_samples : int
            尝试生成的分子数量

        返回
        -------
        list[dict]
            [{"smiles": str, "valid": bool, "length": int, "seed_used": str}, ...]
            按有效性 + 结构排序
        """
        if not self.trained or self.model is None:
            raise RuntimeError("模型未训练，请先调用 train_many()")
        if num_samples < 1:
            return []

        self.model.eval()
        results = []
        stop_idx = self.char_to_idx["~"]
        space_idx = self.char_to_idx.get(" ", 0)

        for _ in range(num_samples):
            with torch.no_grad():
                if seed:
                    seed_indices = [self.char_to_idx.get(c, space_idx) for c in seed]
                    input_tensor = torch.tensor([seed_indices], dtype=torch.long).to(self.device)
                    # 初始前向只传 seed[:-1]，避免最后一个字符被重复处理两次；
                    # 单字符 seed 时直接置 hidden=None（模型内部会初始化）
                    if len(seed_indices) > 1:
                        _, hidden = self.model(input_tensor[:, :-1])
                    else:
                        hidden = None
                    input_tensor = input_tensor[:, -1:]
                    output_chars = list(seed)
                else:
                    # 起始符 = 空格
                    input_tensor = torch.tensor([[space_idx]], dtype=torch.long).to(self.device)
                    hidden = None
                    output_chars = []

                for _ in range(max_length):
                    logits, hidden = self.model(input_tensor, hidden)
                    raw_logits = logits[0, -1]  # (vocab_size,)

                    if temperature < 1e-6:
                        # 贪心解码: 屏蔽填充/空格 token（index 0）后取 argmax，与采样分支一致
                        next_idx = torch.argmax(raw_logits[1:]).item() + 1
                    else:
                        scaled = raw_logits / temperature
                        probs = torch.softmax(scaled, dim=-1)
                        # 屏蔽填充符号
                        probs[0] = 0.0
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                        else:
                            # 全零回退到均匀分布 (屏蔽填充)
                            probs[1:] = 1.0
                            probs = probs / probs.sum()
                        next_idx = torch.multinomial(probs, 1).item()

                    if next_idx == stop_idx:
                        break

                    next_char = self.idx_to_char[next_idx]
                    output_chars.append(next_char)
                    input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(self.device)

            smiles = "".join(output_chars)
            is_valid = False
            canonical = None

            if RDKIT_OK:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    is_valid = True
                    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            results.append({
                "smiles": smiles,
                "canonical": canonical or smiles,
                "valid": is_valid,
                "length": len(smiles),
                "seed_used": seed if seed else "(随机)",
            })

        # 排序: 有效 → 长度适中 → 无效
        results.sort(key=lambda r: (-r["valid"], abs(r["length"] - 25), r["length"]))
        return results

    # ---------- 序列化 ----------

    def save_model(self, path: str):
        """保存模型权重 + 词表"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "training_stats": self.training_stats,
        }, path)
        logging.info(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型权重 + 词表"""
        # weights_only=True：checkpoint 仅含 state_dict/vocab/int/基本 dict，
        # 避免恶意 .pt 文件 pickle 反序列化执行任意代码 (RCE)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.vocab = checkpoint["vocab"]
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = checkpoint["vocab_size"]
        self.training_stats = checkpoint.get("training_stats", {})

        self.model = CharRNN(
            self.vocab_size,
            embed_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.trained = True
        logging.info(f"模型已加载: {path} (vocab_size={self.vocab_size})")
