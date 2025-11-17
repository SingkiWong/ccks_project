# CLAUDE.md - AI Assistant Guide

This document provides comprehensive guidance for AI assistants (like Claude) working on this codebase. It explains the project structure, architecture, conventions, and workflows.

---

## Project Overview

**Project Name**: CCKS2021 Causality Extraction System
**Domain**: Natural Language Processing (NLP) - Financial Text Analysis
**Language**: Chinese
**Framework**: PyTorch, Transformers, PyTorch Geometric
**Task**: Extract causal relationships from Chinese financial text

### Academic Context

This project is an implementation of **Chapter 3** from the master's thesis:
**"基于文本因果信息增强的时间序列预测方法"** (Time Series Prediction Enhanced by Textual Causal Information)
**Institution**: 华东师范大学 (East China Normal University)
**Year**: 2025

The implementation focuses on **heuristic neural network-based causal relation extraction** using Graph Neural Networks (GNN) with BERT embeddings.

---

## Repository Structure

```
ccks_project/
├── causality_extraction.py       # Core model implementation (838 lines)
├── train_ccks.py                 # Training script for CCKS dataset (389 lines)
├── predict_ccks_improved.py      # Improved prediction script (319 lines)
├── ccks_adapter.py               # Data format adapter (309 lines)
├── evaluate_predictions.py       # Evaluation metrics script (418 lines)
├── requirements.txt              # Python dependencies
├── README.md                     # Academic paper reproduction guide
├── CCKS_README.md                # CCKS competition usage guide
├── CCKS数据集使用说明.txt        # Chinese dataset instructions
├── 使用说明.txt                  # Chinese usage guide
├── ccks_task2_train.txt          # Training data (7,000 samples)
├── ccks_task2_eval_data.txt      # Evaluation data (1,000 samples)
└── checkpoints/                  # Model checkpoints (created during training)
    └── best_model.pt
```

---

## Core Architecture

### 1. Model Components

The system implements a **three-stage pipeline**:

#### Stage 1: Heuristic Rules (`HeuristicRules` class)
- **Location**: `causality_extraction.py:40-191`
- **Purpose**: Initialize causal graph edges using linguistic rules
- **Three rule types**:
  1. **Lexical Trigger Rules** (词汇触发规则)
     - Strong triggers (confidence 0.9): 导致, 致使, 引发, 造成, etc.
     - Weak triggers (confidence 0.6): 影响, 冲击, 扰动, etc.
  2. **Syntactic Dependency Rules** (句法依存规则) - confidence 0.75
  3. **Semantic Role Rules** (语义角色规则) - confidence 0.8
- **Formula 3.3**: `A_ij^(0) = max({c_lex, c_syn, c_sem})`

#### Stage 2: Graph Neural Network (`CausalGNN` class)
- **Location**: `causality_extraction.py:297-432`
- **Architecture**:
  - Multi-layer Graph Attention Network (GAT)
  - Default: L=3 layers, K=4 attention heads
  - Edge confidence gating mechanism (Formula 3.11)
  - Multi-head attention message passing (Formula 3.10)
  - Layer normalization and residual connections
- **Input**: Node features + Edge features (11-dimensional)
- **Edge Features** (`EdgeFeatureExtractor`):
  1. Token distance (log-normalized)
  2. Sentence distance
  3. Temporal order
  4. Cosine similarity (Formula 3.12)
  5. Connective features (5-dim)
  6. Rule confidence score

#### Stage 3: Relation Classification (`CausalRelationClassifier`)
- **Location**: `causality_extraction.py:434-512`
- **Architecture**: MLP with 3 layers
- **Output Classes**:
  - 0: i causes j
  - 1: j causes i
  - 2: no causal relation
- **Loss Function**: Adaptive Focal Loss (Formula 3.22)
  - α (alpha) = 0.25
  - γ (gamma) = 2.0
  - Addresses class imbalance

### 2. Data Structures

#### `CausalEvent` (dataclass)
```python
text: str          # Event description text
start_pos: int     # Start position in document
end_pos: int       # End position in document
event_type: str    # Event category (e.g., "价格变动", "市场变化")
arguments: Dict    # Arguments (product, region, industry, type)
```

#### `CausalPair` (dataclass)
```python
cause: CausalEvent      # Cause event
effect: CausalEvent     # Effect event
confidence: float       # Prediction confidence [0-1]
```

---

## CCKS Dataset Format

### Training Data Structure
Each line is a JSON object:
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

### Event Type Taxonomy

**Top 10 Cause Types**:
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

**Top 10 Effect Types**:
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

---

## Key Files Deep Dive

### 1. `causality_extraction.py` - Core Model

**Main Classes**:
- `HeuristicRules`: Implements linguistic rules for initial graph construction
- `EdgeFeatureExtractor`: Extracts 11-dimensional edge features
- `CausalGNN`: Graph neural network with attention gating
- `CausalRelationClassifier`: Final classification layer
- `CausalExtractionModel`: Complete end-to-end model

**Key Methods**:
- `encode_events()`: BERT-based event encoding (uses [CLS] token)
- `build_initial_graph()`: Applies heuristic rules to create edges
- `forward()`: Full pipeline execution
- `predict()`: Inference with confidence threshold

**Important Constants**:
- Default BERT: `bert-base-chinese`
- Hidden dimension: 768
- Edge feature dimension: 11
- Confidence threshold: 0.5 (default), 0.3 (recommended for CCKS)

### 2. `train_ccks.py` - Training Pipeline

**Main Classes**:
- `CCKSCausalityDataset`: PyTorch Dataset wrapper for CCKS data

**Training Configuration** (lines 187-203):
```python
num_epochs: 20 (default 10)
learning_rate: 2e-5
batch_size: 4
optimizer: AdamW (weight_decay=0.01)
scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
train/val split: 90% / 10%
```

**Data Processing**:
- Automatically parses CCKS JSON format
- Constructs event pairs (all combinations)
- Generates labels: 0 (i→j), 1 (j→i), 2 (no relation)
- Uses focal loss to handle class imbalance

### 3. `predict_ccks_improved.py` - Improved Inference

**Enhancement**: Intelligent event extraction from raw text

**Event Extraction Strategy**:
1. **Keyword-based matching**: Uses financial domain keywords
2. **Sentence splitting**: Falls back to sentence-level events
3. **Deduplication**: Removes redundant events
4. **Limit**: Maximum 10 events per sample

**Important**: The original CCKS data doesn't provide event boundaries, so this script uses heuristics to identify events from text.

### 4. `evaluate_predictions.py` - Metrics Calculation

**Metrics**:
- Precision (精确率)
- Recall (召回率)
- F1 Score

**Evaluation Method**:
- Exact match on (reason_type, reason_product, result_type, result_product)
- Sample-level and relation-level statistics
- Per-type breakdown

---

## Development Workflows

### Workflow 1: Training a New Model

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
# Ensure ccks_task2_train.txt is in the project root

# 3. Train model
python train_ccks.py

# Model will be saved to ./checkpoints/best_model.pt
# Training logs show loss, validation metrics
```

**Expected Output**:
- Best validation loss < 0.5
- Training time: 2-4 hours (CPU), 30-60 minutes (GPU)
- Model size: ~400MB (BERT + GNN)

### Workflow 2: Making Predictions

```bash
# Using the improved predictor
python predict_ccks_improved.py \
    --input ccks_task2_eval_data.txt \
    --output predictions.txt \
    --model checkpoints/best_model.pt \
    --threshold 0.3
```

**Parameters**:
- `--threshold`: Lower values increase recall, higher values increase precision
  - Recommended: 0.3 for CCKS competition
  - Default: 0.5

### Workflow 3: Evaluating Results

```bash
# Basic evaluation
python evaluate_predictions.py \
    --pred predictions.txt \
    --gold ccks_task2_train.txt \
    --detailed

# With error analysis
python evaluate_predictions.py \
    --pred predictions.txt \
    --gold ccks_task2_train.txt \
    --errors \
    --num-errors 20
```

**Output**:
- Precision, Recall, F1 scores
- Per-type statistics
- False positive/negative examples
- Metrics saved to `predictions_metrics.json`

---

## Code Conventions

### 1. Naming Conventions

**Classes**: PascalCase
- `CausalExtractionModel`, `HeuristicRules`, `CCKSCausalityDataset`

**Functions**: snake_case
- `train_step()`, `evaluate_predictions()`, `extract_events_from_text()`

**Variables**: snake_case
- `event_pairs`, `edge_features`, `confidence`

**Constants**: UPPER_SNAKE_CASE
- `EVENT_TYPES`, `EVENT_TYPE_MAP`

### 2. Docstring Style

Uses **Google-style** docstrings:
```python
def compute_initial_confidence(self, text: str, event_i: CausalEvent,
                              event_j: CausalEvent) -> float:
    """
    公式 3.3: 计算初始边权重
    A_ij^(0) = max({c_lex, c_syn, c_sem})

    Args:
        text: 文本
        event_i: 事件i
        event_j: 事件j

    Returns:
        初始置信度
    """
```

### 3. Type Hints

**Always use** type hints for function signatures:
```python
from typing import List, Dict, Tuple, Optional

def predict(self, text: str, events: List[CausalEvent],
            threshold: float = 0.5) -> List[CausalPair]:
```

### 4. Error Handling

**Training scripts**: Use try-except with continue
```python
try:
    loss = train_step(...)
except Exception as e:
    print(f"\n训练出错: {e}")
    continue
```

**Prediction scripts**: Graceful degradation
```python
if len(events) < 2:
    return {'text_id': text_id, 'text': text, 'result': []}
```

---

## Important Implementation Details

### 1. BERT Encoding Strategy

- Uses **[CLS] token** representation: `outputs.last_hidden_state[:, 0, :]`
- Events encoded **independently** (not in context)
- **No gradient** for BERT during event extraction: `with torch.no_grad()`
- BERT **fine-tuned** during training via `model.bert.parameters()`

### 2. Graph Construction

**Initial edge creation**:
- Only creates edges with confidence > 0.3 (line 608)
- This threshold is **critical** - too high misses edges, too low adds noise
- Empty graph handling: returns empty tensors if no edges

**Edge features** are **concatenated** with attention outputs:
```python
GATConv(..., edge_dim=edge_feature_dim)
```

### 3. Training Label Generation

For each sample with events [e0, e1, e2, ...]:
- Generate **all pairs**: (e0,e1), (e0,e2), (e1,e0), (e1,e2), ...
- Labels based on ground truth:
  - `0`: if (i, j) in true_pairs → i causes j
  - `1`: if (j, i) in true_pairs → j causes i
  - `2`: otherwise → no relation

**Class imbalance**: Most pairs are class 2 (no relation)
- **Solution**: Focal loss with α=0.25, γ=2.0

### 4. Prediction Threshold Tuning

The `threshold` parameter in `predict()` affects precision/recall trade-off:
```python
if pred_class < 2 and confidence > threshold:
    causal_pairs.append(pair)
```

**Guidelines**:
- threshold=0.3: High recall (good for CCKS)
- threshold=0.5: Balanced
- threshold=0.7: High precision

---

## Common Tasks for AI Assistants

### Task 1: Improving Event Extraction

**Current limitation**: Event extraction is heuristic-based (keyword matching)

**File to modify**: `predict_ccks_improved.py:48-119`

**Suggested improvements**:
1. Add NER (Named Entity Recognition) for products/regions
2. Use dependency parsing for event boundaries
3. Train a separate event detection model
4. Use semantic similarity to merge similar events

### Task 2: Hyperparameter Tuning

**Key hyperparameters**:
- GNN layers (3): `num_gnn_layers` in `train_ccks.py:369`
- Attention heads (4): `num_heads` in `train_ccks.py:370`
- Learning rate (2e-5): `learning_rate` in `train_ccks.py:382`
- Focal loss α (0.25), γ (2.0): in `train_step()` at line 272

**How to experiment**:
```python
# In train_ccks.py, modify:
model = CausalExtractionModel(
    bert_model_name='bert-base-chinese',
    hidden_dim=768,
    num_gnn_layers=4,  # Try 2, 3, 4, 5
    num_heads=8        # Try 4, 8, 12
)
```

### Task 3: Adding New Event Types

**Files to modify**:
1. `train_ccks.py`: Update `EVENT_TYPE_MAP` (lines 32-45)
2. `ccks_adapter.py`: Update `EVENT_TYPES` (lines 37-63)
3. `predict_ccks_improved.py`: Update `EVENT_TYPES` (lines 22-27)

**Consistency**: Ensure all three dictionaries are synchronized

### Task 4: Implementing Cross-Validation

**Current**: 90/10 train/val split (single split)

**Enhancement**: K-fold cross-validation
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # Train separate model for each fold
    # Average results
```

### Task 5: Model Ensemble

**Current**: Single model prediction

**Enhancement**: Combine multiple models
```python
# Train models with different seeds
models = []
for seed in [42, 123, 456, 789]:
    torch.manual_seed(seed)
    model = train_model(...)
    models.append(model)

# Ensemble prediction (majority voting or averaging)
```

---

## Performance Expectations

### Academic Benchmark (Table 3.2 in thesis)

| Model | ECE F1 |
|-------|--------|
| BERT-softmax | 30.99% |
| **This Method** | **49.00%** |

**Improvement**: +18.01 percentage points

### CCKS Competition Expectations

Based on training data statistics:
- **Precision**: 46%+
- **Recall**: 52%+
- **F1 Score**: 49%+

**Factors affecting results**:
1. Event identification quality (major bottleneck)
2. Training epochs (10-20 recommended)
3. Threshold tuning
4. Data augmentation

---

## Troubleshooting Guide

### Issue 1: CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
1. Reduce batch size: `batch_size=2` or `batch_size=1`
2. Reduce GNN layers: `num_gnn_layers=2`
3. Use gradient checkpointing (not implemented)
4. Move to CPU (slower): Don't use `.cuda()`

### Issue 2: Empty Predictions

**Symptoms**: All samples return `result: []`

**Debugging**:
1. Check threshold: Try lower values (0.2, 0.3)
2. Verify model loaded: Check checkpoint exists
3. Check event extraction: Print `events` in predictor
4. Verify graph construction: Check `edge_index.shape`

**Common cause**: Events not detected → no graph → no predictions

### Issue 3: Low Recall

**Symptoms**: F1 low due to missing many true relations

**Solutions**:
1. Lower prediction threshold: `--threshold 0.2`
2. Improve event extraction: Add more keywords
3. Reduce initial graph threshold: Line 608, change `0.3` to `0.2`
4. Check heuristic rules: Add more trigger words

### Issue 4: Low Precision

**Symptoms**: Many false positives

**Solutions**:
1. Raise prediction threshold: `--threshold 0.5`
2. Increase focal loss γ: `gamma=3.0` or `gamma=4.0`
3. Add post-processing filters
4. Train longer (more epochs)

### Issue 5: Training Loss Not Decreasing

**Symptoms**: Loss stays high or fluctuates

**Debugging**:
1. Check data loading: Print batch samples
2. Verify labels: Ensure mix of 0, 1, 2 classes
3. Reduce learning rate: `1e-5` instead of `2e-5`
4. Check BERT loading: Ensure `bert-base-chinese` downloads correctly

---

## Dependencies and Environment

### Python Version
- **Required**: Python 3.8+
- **Recommended**: Python 3.9 or 3.10

### Core Dependencies (from requirements.txt)

```
torch>=2.0.0              # PyTorch framework
torch-geometric>=2.3.0    # Graph neural networks
transformers>=4.30.0      # BERT models
numpy>=1.24.0            # Numerical operations
matplotlib>=3.7.0        # Visualization (unused in main code)
networkx>=3.1            # Graph utilities
tqdm>=4.65.0             # Progress bars
scikit-learn>=1.3.0      # Metrics and utilities
```

### Installing Dependencies

```bash
# Standard installation
pip install -r requirements.txt

# If torch-geometric installation fails:
pip install torch torchvision
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For GPU support (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Model Downloads

**BERT model** (`bert-base-chinese`) auto-downloads from HuggingFace:
- Size: ~400MB
- Location: `~/.cache/huggingface/transformers/`
- First run will download (requires internet)

**Offline setup**: Download model manually and specify local path:
```python
model = CausalExtractionModel(
    bert_model_name='/path/to/bert-base-chinese',
    ...
)
```

---

## Testing and Validation

### Unit Testing (Not Implemented)

**Recommendation**: Add pytest tests for:
1. `HeuristicRules.compute_initial_confidence()`
2. `EdgeFeatureExtractor.extract_edge_features()`
3. `CCKSDatasetAdapter.parse_ccks_sample()`
4. Model forward pass with dummy data

**Example test**:
```python
# test_causality_extraction.py
def test_heuristic_rules():
    rules = HeuristicRules()
    text = "价格上涨导致需求下降"
    event1 = CausalEvent(text="价格上涨", start_pos=0, end_pos=4, ...)
    event2 = CausalEvent(text="需求下降", start_pos=6, end_pos=10, ...)

    confidence = rules.compute_initial_confidence(text, event1, event2)
    assert confidence >= 0.9  # "导致" is a strong trigger
```

### Integration Testing

**Manual validation script** (create this):
```python
# test_end_to_end.py
from causality_extraction import CausalExtractionModel, CausalEvent

model = CausalExtractionModel()
text = "油价上涨导致化工成本增加"
events = [
    CausalEvent(text="油价上涨", start_pos=0, end_pos=4,
                event_type="价格变动", arguments={}),
    CausalEvent(text="化工成本增加", start_pos=6, end_pos=12,
                event_type="成本变化", arguments={})
]

pairs = model.predict(text, events, threshold=0.3)
print(f"Found {len(pairs)} causal relations")
for pair in pairs:
    print(f"{pair.cause.text} → {pair.effect.text} ({pair.confidence:.2f})")
```

---

## Git and Version Control

### Branch Strategy (if using Git)

- `main`: Stable, reproducible thesis results
- `develop`: Active development
- `feature/*`: New features (e.g., `feature/improved-event-extraction`)
- `experiment/*`: Hyperparameter experiments

### .gitignore Recommendations

```gitignore
# Checkpoints
checkpoints/
*.pt
*.pth

# Data (if large)
ccks_task2_train.txt
ccks_task2_eval_data.txt
predictions*.txt

# Python
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# HuggingFace cache
.cache/

# Jupyter
.ipynb_checkpoints/
```

### Commit Message Convention

```
feat: Add cross-validation to training pipeline
fix: Correct edge feature dimension mismatch
docs: Update CLAUDE.md with new event types
refactor: Extract event detection to separate module
perf: Optimize BERT encoding with batch processing
```

---

## Advanced Topics

### 1. Curriculum Learning

**Idea**: Train on easier examples first

**Implementation**:
```python
# Sort samples by number of events (easier = fewer events)
sorted_data = sorted(dataset, key=lambda x: len(x['events']))

# Train in stages
for stage in range(3):
    stage_data = sorted_data[stage*2000:(stage+1)*2000]
    train(stage_data)
```

### 2. Active Learning

**Idea**: Select most uncertain samples for annotation

**Implementation**:
```python
# Predict with confidence
predictions = model.predict(unlabeled_data)

# Select low-confidence samples
uncertain = [p for p in predictions if 0.4 < p.confidence < 0.6]

# Request human annotation for these
```

### 3. Multi-task Learning

**Idea**: Train on event extraction + relation extraction jointly

**Modification**: Add event detection head to `CausalExtractionModel`

### 4. Domain Adaptation

**Challenge**: Model trained on financial text may not generalize

**Solutions**:
1. Fine-tune on target domain data
2. Use domain-specific BERT (e.g., FinBERT-Chinese)
3. Add domain-agnostic features

---

## FAQ for AI Assistants

### Q1: How do I add a new heuristic rule?

**A**: Edit `causality_extraction.py`, lines 48-62:
```python
self.strong_triggers = {
    '导致': 0.9,
    '你的新触发词': 0.9,  # Add here
    ...
}
```

### Q2: Can I use a different BERT model?

**A**: Yes! Modify model initialization:
```python
model = CausalExtractionModel(
    bert_model_name='hfl/chinese-roberta-wwm-ext',  # Example
    ...
)
```

Supported models: Any HuggingFace Chinese BERT variant

### Q3: How do I save training logs?

**A**: Add logging to `train_ccks.py`:
```python
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# In training loop:
logging.info(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Q4: Can I train on English data?

**A**: Yes, but requires modifications:
1. Change BERT model to `bert-base-uncased`
2. Update heuristic rules with English trigger words
3. Modify event type taxonomy

### Q5: How do I visualize the causal graph?

**A**: Add visualization code using NetworkX:
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for pair in causal_pairs:
    G.add_edge(pair.cause.text, pair.effect.text,
               weight=pair.confidence)

nx.draw(G, with_labels=True)
plt.savefig('causal_graph.png')
```

### Q6: What's the maximum number of events per sample?

**A**: No hard limit in the model, but:
- `predict_ccks_improved.py` limits to 10 events (line 119)
- Computational complexity is O(n²) where n = number of events
- Recommended max: 15-20 events for performance

---

## Performance Optimization

### 1. Batch Processing (Not Fully Implemented)

**Current**: Events encoded one-by-one
**Optimization**: Batch encode all events in a sample
```python
# In encode_events():
all_texts = [event.text for event in events]
tokens = self.tokenizer(all_texts, padding=True,
                        truncation=True, return_tensors='pt')
outputs = self.bert(**tokens)
```

### 2. Caching Event Embeddings

**Idea**: Cache BERT embeddings to avoid recomputation
```python
# Add to CausalExtractionModel:
self.event_cache = {}

def encode_events(self, text, events):
    cache_keys = [event.text for event in events]
    uncached = [e for e in events if e.text not in self.event_cache]

    # Encode only uncached
    if uncached:
        ...
        self.event_cache.update(...)

    return [self.event_cache[e.text] for e in events]
```

### 3. Mixed Precision Training

**Benefit**: Faster training, less memory

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(...)
    loss = ...

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Citation and References

### Citing This Implementation

```bibtex
@mastersthesis{thesis2025,
  title={基于文本因果信息增强的时间序列预测方法},
  school={华东师范大学},
  year={2025}
}
```

### Related Papers

1. **Graph Neural Networks**: Veličković et al., "Graph Attention Networks" (ICLR 2018)
2. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
3. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

### External Resources

- CCKS2021 Competition: http://sigkg.cn/ccks2021/
- PyTorch Geometric Docs: https://pytorch-geometric.readthedocs.io/
- Transformers Docs: https://huggingface.co/docs/transformers/
- BERT Chinese: https://github.com/google-research/bert

---

## Changelog and Version History

### Current Version (as of this CLAUDE.md)

**Version**: 1.0 (Thesis Implementation)
**Date**: 2025-01
**Status**: Stable, reproducible results

**Known Limitations**:
1. Event extraction is heuristic-based (not learned)
2. No dependency parsing (simplified syntactic rules)
3. No semantic role labeling (simplified semantic rules)
4. Single-language support (Chinese only)
5. No multi-document support

**Future Work** (suggested):
1. End-to-end event detection
2. Cross-lingual causality extraction
3. Temporal reasoning (event ordering)
4. Multi-hop causality chains
5. Causality explanation generation

---

## Contact and Support

For questions about the **codebase**:
- Check this CLAUDE.md file first
- Review code comments in `causality_extraction.py`
- Consult `README.md` and `CCKS_README.md`

For questions about the **thesis**:
- Refer to Chapter 3 of the thesis document
- Check academic references in Bibliography

For **CCKS competition**:
- Official site: http://sigkg.cn/ccks2021/
- Dataset description in `CCKS_README.md`

---

## Summary Checklist for AI Assistants

Before modifying this codebase, ensure you understand:

- [ ] The three-stage pipeline (Heuristics → GNN → Classification)
- [ ] CCKS data format (JSON with reason/result fields)
- [ ] Event vs. CausalPair data structures
- [ ] Heuristic rule types and confidences
- [ ] GNN architecture (layers, heads, edge features)
- [ ] Focal loss and class imbalance handling
- [ ] Training workflow (train_ccks.py)
- [ ] Prediction workflow (predict_ccks_improved.py)
- [ ] Evaluation metrics (evaluate_predictions.py)
- [ ] Hyperparameter locations and defaults

**Good luck with your development!** 🚀

---

*This CLAUDE.md was generated by an AI assistant to help future AI assistants understand and work with this codebase effectively.*
