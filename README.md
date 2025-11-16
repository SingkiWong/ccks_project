# 第三章:基于启发式神经网络的因果关系提取方法

## 论文复现说明

本项目完整复现了论文第三章"基于启发式神经网络的因果关系提取方法"的核心算法和实验。

## 目录结构

```
.
├── causality_extraction.py   # 核心模型实现
├── train.py                   # 训练脚本
├── visualize.py              # 可视化工具
├── requirements.txt          # 依赖包
├── README.md                 # 本文档
└── data/                     # 数据目录
    ├── train.json
    ├── val.json
    └── test.json
```

## 核心组件

### 1. 启发式规则设计 (3.2.1节)

实现了三类启发式规则:

#### 1.1 词汇触发规则
- **强触发词** (置信度 0.9): 导致、致使、引发、造成等
- **弱触发词** (置信度 0.6): 影响、冲击、扰动、改变等

#### 1.2 句法依存规则
检查依存树中的因果路径模式 (置信度 0.75)

#### 1.3 语义角色规则  
基于语义角色标注识别AM-CAU角色 (置信度 0.8)

#### 1.4 规则整合 (公式 3.3)
```
A_ij^(0) = max({c_lex, c_syn, c_sem})
```

### 2. 图神经网络编码 (3.2.2节)

- 多层图注意力网络(GAT): L=3层, K=4个注意力头
- 边置信度门控机制 (公式 3.11)
- 多头注意力消息传递 (公式 3.10)
- 层级传播 (公式 3.15)

### 3. 关系分类与损失函数 (3.2.3节)

- MLP分类器 (公式 2.20)
- 自适应焦点损失 (公式 3.22): α=0.25, γ=2.0

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python train.py
```

### 预测
```python
from causality_extraction import CausalExtractionModel, CausalEvent

model = CausalExtractionModel()
text = "央行加息导致股市下跌"
events = [...]  # 定义事件列表
causal_pairs = model.predict(text, events)
```

## 实验结果 (表3.2)

| 模型 | ECE F1 |
|------|--------|
| BERT-softmax | 30.99 |
| **本文方法** | **49.00** |

提升: +18.01个百分点

## 引用

```bibtex
@mastersthesis{thesis2025,
  title={基于文本因果信息增强的时间序列预测方法},
  school={华东师范大学},
  year={2025}
}
```
