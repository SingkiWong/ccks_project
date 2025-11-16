# CCKS2021因果关系抽取 - 使用指南

## 📊 数据集信息

### 训练集 (ccks_task2_train.txt)
- **样本总数**: 7,000条
- **因果关系总数**: 7,908对
- **平均每个样本**: 1.13个因果关系

### 测试集 (ccks_task2_eval_data.txt)
- **样本总数**: 1,000条
- **标注**: 无标注(用于最终评估)

### 数据格式

**训练集格式**:
```json
{
    "text_id": "1291633",
    "text": "铁矿：中长期，今年铁矿供需格局明显改善，巴西矿难及飓风对发运的影响，导致铁矿石全年供应走低",
    "result": [
        {
            "reason_type": "台风",
            "reason_product": "",
            "reason_region": "巴西",
            "reason_industry": "",
            "result_type": "供给减少",
            "result_product": "铁矿石",
            "result_region": "",
            "result_industry": ""
        }
    ]
}
```

**测试集格式** (无result字段):
```json
{
    "text_id": "1715813",
    "text": "双焦由于钢厂利润收窄，上周延续跌势..."
}
```

## 🎯 主要事件类型

### 原因类型 (Top 10)
1. 供给减少 (1,199)
2. 市场价格下降 (1,198)
3. 市场价格提升 (1,184)
4. 需求减少 (806)
5. 供给增加 (718)
6. 需求增加 (623)
7. 限产 (229)
8. 猪瘟 (182)
9. 其他贸易摩擦 (163)
10. 干旱 (157)

### 结果类型 (Top 10)
1. 市场价格提升 (2,003)
2. 市场价格下降 (1,699)
3. 供给减少 (902)
4. 产品利润下降 (650)
5. 需求减少 (409)
6. 产品利润增加 (369)
7. 销量（消费）减少 (321)
8. 需求增加 (297)
9. 负向影响 (254)
10. 运营成本提升 (203)

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train_ccks.py
```

训练配置:
- **BERT**: bert-base-chinese
- **GNN层数**: 3
- **注意力头数**: 4
- **学习率**: 2e-5
- **Batch Size**: 4
- **训练轮数**: 10
- **训练/验证划分**: 90% / 10%

### 3. 预测测试集

```bash
python predict_ccks.py \
    --input /mnt/user-data/uploads/ccks_task2_eval_data.txt \
    --output predictions.txt \
    --model checkpoints/best_model.pt \
    --threshold 0.3
```

参数说明:
- `--input`: 输入文件(测试集)
- `--output`: 输出文件
- `--model`: 模型路径
- `--threshold`: 置信度阈值(0-1)

### 4. 查看预测结果

```bash
head -5 predictions.txt
```

输出格式与训练集相同。

## 📁 文件说明

### 核心文件
1. **causality_extraction.py** - 核心模型实现
   - HeuristicRules: 启发式规则
   - CausalGNN: 图神经网络
   - CausalExtractionModel: 完整模型

2. **train_ccks.py** - CCKS训练脚本
   - CCKSCausalityDataset: 数据集类
   - 自动划分训练/验证集
   - 支持早停和模型保存

3. **predict_ccks.py** - CCKS预测脚本
   - CCKSPredictor: 预测器类
   - 生成提交格式结果

4. **ccks_adapter.py** - 数据适配器
   - 数据格式转换
   - 统计分析工具

## 💡 使用技巧

### 调整超参数

修改 `train_ccks.py` 中的参数:

```python
train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    num_epochs=20,        # 增加训练轮数
    learning_rate=1e-5,   # 降低学习率
    save_dir='./checkpoints'
)
```

### 调整模型结构

修改模型初始化:

```python
model = CausalExtractionModel(
    bert_model_name='bert-base-chinese',
    hidden_dim=768,
    num_gnn_layers=4,     # 增加GNN层数
    num_heads=8           # 增加注意力头数
)
```

### 调整预测阈值

```bash
# 提高阈值,减少误报
python predict_ccks.py --threshold 0.5

# 降低阈值,增加召回
python predict_ccks.py --threshold 0.2
```

## 📊 性能优化建议

### 1. 数据增强
- 使用同义词替换
- 回译(中文→英文→中文)
- 随机删除/插入

### 2. 模型改进
- 使用领域特定的预训练模型(FinBERT-Chinese)
- 增加对抗训练
- 集成多个模型

### 3. 后处理
- 规则过滤明显错误的预测
- 基于置信度排序
- 实体类型约束

## 🔍 调试技巧

### 查看单个样本预测

```python
from predict_ccks import CCKSPredictor

predictor = CCKSPredictor('checkpoints/best_model.pt')

sample = {
    'text_id': 'test_001',
    'text': '原油价格上涨导致化工产品成本增加'
}

result = predictor.predict_sample(sample)
print(result)
```

### 分析错误案例

```python
# 对比预测和真实标签
def analyze_errors(predictions, ground_truth):
    # 计算精确率、召回率、F1
    # 分析错误类型
    pass
```

## ⚠️ 注意事项

1. **内存占用**
   - BERT模型约400MB
   - 建议使用GPU训练
   - Batch size根据显存调整

2. **训练时间**
   - 7000样本约需2-4小时(CPU)
   - GPU可减少到30-60分钟

3. **事件识别**
   - 当前使用简化的关键词匹配
   - 建议使用专门的事件识别模型
   - 可以手动标注事件提高准确率

4. **数据特点**
   - 金融领域文本
   - 因果关系较隐含
   - 需要领域知识

## 📈 预期结果

根据论文,模型应达到:
- **Precision**: 46%+
- **Recall**: 52%+
- **F1 Score**: 49%+

实际结果取决于:
- 事件识别质量
- 训练轮数
- 超参数选择
- 数据增强策略

## 🎓 进阶使用

### 集成多个模型

```python
# 训练多个模型
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    model = CausalExtractionModel(...)
    train_model(...)
    
# 集成预测
predictions = ensemble_predict(models, test_data)
```

### 使用外部知识

```python
# 加载金融知识图谱
kg = load_financial_kg()

# 增强事件表示
event_features = enhance_with_kg(events, kg)
```

## 📧 问题反馈

如遇问题,请检查:
1. PyTorch版本 >= 2.0
2. Transformers版本 >= 4.30
3. 数据文件编码为UTF-8
4. 模型文件完整性

## 📚 参考资料

- 论文: 《基于文本因果信息增强的时间序列预测方法》第三章
- CCKS2021: http://sigkg.cn/ccks2021/
- BERT: https://github.com/google-research/bert
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
