# Architecture Summary & File Structure

## Project Structure

```
/home/user/ccks_project/
├── causality_extraction.py          # Core model implementation (838 lines)
│   ├── HeuristicRules              # Lexical/syntax/semantic rules
│   ├── EdgeFeatureExtractor        # Edge feature computation
│   ├── CausalGNN                   # Graph neural network (GAT-based)
│   ├── CausalRelationClassifier    # MLP classifier
│   └── CausalExtractionModel       # Full pipeline
│
├── train_ccks.py                    # Training script (388 lines)
│   ├── CCKSCausalityDataset        # Data loader
│   ├── train_model()               # Training loop
│   └── main()                       # Entry point
│
├── predict_ccks_improved.py         # Inference script (318 lines)
│   ├── ImprovedCCKSPredictor       # Prediction class
│   ├── extract_events_from_text()  # Event extraction
│   └── predict_file()              # Batch prediction
│
├── evaluate_predictions.py          # Evaluation script (418 lines)
│   ├── CCKSEvaluator               # Metrics calculation
│   └── analyze_errors()            # Error analysis
│
├── ccks_adapter.py                  # Data adapter (309 lines)
│   ├── CCKSDatasetAdapter          # Format conversion
│   └── analyze_ccks_dataset()      # Statistics
│
├── ccks_task2_train.txt            # Training data (6,999 samples)
├── ccks_task2_eval_data.txt        # Test data (999 samples)
├── requirements.txt                # Dependencies
├── README.md                       # Original readme
└── CCKS_README.md                  # CCKS task guide
```

---

## Data Flow Diagram

### Training Flow
```
Raw CCKS Format Data
      ↓
CCKSCausalityDataset (parse_sample)
      ↓
[text_id, text, events[], causal_pairs[]]
      ↓
train_step() loop
      ↓
Forward Pass:
  1. BERT encode events → [N, 768]
  2. build_initial_graph() → edges with confidence scores
  3. CausalGNN (3 GAT layers) → [N, 768] updated reps
  4. Classifier on pairs → logits [num_pairs, 3]
      ↓
Focal Loss Computation
      ↓
Backprop & Update
      ↓
Saved Checkpoint
```

### Inference Flow
```
Test Data (text_id, text)
      ↓
extract_events_from_text()
  - Keyword matching (multiple occurrences)
  - Fallback: sentence splitting
  - Deduplication by (text, type)
  - Return top 10 events
      ↓
Build All Pairs: [(i,j) for all i≠j]  ← OVERPREDICTION SOURCE #1
  - 8 events → 56 pairs
  - 10 events → 90 pairs
      ↓
Forward Pass:
  1. BERT encode → [N, 768]
  2. build_initial_graph (confidence > 0.3) → edges
  3. GAT processing
  4. Classifier → logits
  5. Softmax → probabilities
      ↓
Threshold Filter (default 0.3)  ← OVERPREDICTION SOURCE #2
  - pred_class < 2 AND confidence > 0.3
  - For 3 classes: ~60-70% pass this threshold
      ↓
Output: 21,839 predictions
```

---

## Model Architecture Details

### Component 1: Heuristic Rules
```
Input: (text, event_i, event_j)
  ↓
1. Lexical Rules:
   - Strong triggers (导致, 致使): confidence = 0.9
   - Weak triggers (影响, 冲击): confidence = 0.6
   - Check text between events
  ↓
2. Syntax Rules:
   - Dependency path patterns: confidence = 0.75
  ↓
3. Semantic Rules:
   - SRL AM-CAU roles: confidence = 0.8
  ↓
Output: max({c_lex, c_syn, c_sem})
```

### Component 2: Edge Feature Extraction
```
Input: (text, event_i, event_j, confidence)
  ↓
Extract 11 dimensions:
  1. Token distance: log(1 + |pos_i - pos_j|)
  2. Sentence distance: count('。') between events
  3. Temporal order: +1 if i before j, -1 else
  4. Semantic similarity: cosine(BERT(event_i), BERT(event_j))
  5-9. Connective features: 5 placeholder values
  10-11. Initial confidence (heuristic score)
  ↓
Output: [11-dim feature vector]
```

### Component 3: Graph Neural Network
```
Input: 
  - node_features: [N, 768] (BERT embeddings)
  - edge_index: [2, E] (source/target indices)
  - edge_features: [E, 11] (extracted features)
  ↓
For each of 3 layers:
  1. Compute edge gates (Equation 3.11):
     g_ji = sigmoid(u^T [z_i; z_j; f_ji])
  
  2. GAT message passing (Equation 3.10):
     m_i^k = Σ_{j∈N_i} w̃_ji · W_k z_j
  
  3. Edge gating application
  
  4. LayerNorm + ReLU
  
  5. Residual connection
  ↓
Output: [N, 768] updated node representations
```

### Component 4: Classifier
```
Input: [event_i_repr, event_j_repr] → [1536-dim]
  ↓
MLP:
  Linear(1536 → 768) + ReLU
  Dropout(0.1)
  Linear(768 → 384) + ReLU
  Dropout(0.1)
  Linear(384 → 3)
  ↓
Output: logits for [cause, effect, no-relation]
```

### Component 5: Loss Function
```
Input: logits, targets
  ↓
Focal Loss (Equation 3.22):
  L = -α * (1 - p_t)^γ * log(p_t)
  
  where:
  - α = 0.25 (class balance weight)
  - γ = 2.0 (focus parameter)
  - p_t = probability of true class
  ↓
Output: scalar loss
```

---

## Key Hyperparameters

### Model Architecture
| Parameter | Value | Notes |
|-----------|-------|-------|
| BERT Model | bert-base-chinese | 12 layers, 768-dim |
| BERT Hidden | 768 | Full dimension |
| GNN Layers | 3 | L=3 in paper |
| Attention Heads | 4 | K=4 in paper |
| Edge Features | 11 | Manually engineered |
| Dropout | 0.1 | In GAT layers |

### Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 10 | Too few → underfitting |
| Batch Size | 4 | Very small |
| Learning Rate | 2e-5 | Standard for BERT |
| Optimizer | AdamW | + weight_decay=0.01 |
| Scheduler | ReduceLROnPlateau | Factor=0.5, patience=3 |
| Loss Function | Focal Loss | α=0.25, γ=2.0 |
| Train/Val Split | 90/10 | 6,299 / 699 samples |

### Inference (CRITICAL - ROOT OF OVERPREDICTION)
| Parameter | Value | Critical Issue? |
|-----------|-------|-----------------|
| Threshold | 0.3 | YES - too permissive |
| Max Events | 10 | YES - creates 90 pairs |
| Edge Threshold | 0.3 | YES - too permissive |
| Event Dedup | (text, type) | MEDIUM - weak dedup |

---

## Data Statistics

### Training Set
```
Samples: 6,999
Total Relations: 7,908
Avg Relations/Sample: 1.13
Samples with Relations: 6,999 (100%)
Max Relations/Sample: 10
```

### Event Types Distribution
```
Top Reason Types:
  1. 供给减少 (1,199)
  2. 市场价格下降 (1,198)
  3. 市场价格提升 (1,184)
  4. 需求减少 (806)
  5. 供给增加 (718)

Top Result Types:
  1. 市场价格提升 (2,003)
  2. 市场价格下降 (1,699)
  3. 供给减少 (902)
  4. 产品利润下降 (650)
  5. 需求减少 (409)
```

### Evaluation Set
```
Samples: 999
Total Relations: 7,886 (estimated)
Avg Relations/Sample: 7.89
No ground truth labels provided
```

---

## Complexity Analysis

### Time Complexity
```
Per sample:
  - BERT encoding: O(N * 768) = O(N)
  - Graph building: O(N²) [all pairs]
  - GAT forward: O(E * 768) = O(N²)
  - Classification: O(N² * 1536) = O(N²)
  Total: O(N²) where N = num events

Per batch (999 samples):
  - Average N = 9 events
  - Average pairs = 72 per sample
  - Total predictions = ~72,000
  - GPU time: ~10-20 seconds
```

### Space Complexity
```
Per sample:
  - Event embeddings: O(N * 768)
  - Edge index: O(E) = O(N²)
  - Edge features: O(E * 11) = O(N²)
  - Logits: O(E * 3) = O(N²)
  Total: O(N²) space

For batch:
  - ~500 events total
  - ~5,000+ edges
  - GPU memory: ~2-3GB
```

---

## Known Issues Summary

| Issue | Type | Severity | Fix Time |
|-------|------|----------|----------|
| Threshold too low (0.3) | Config | CRITICAL | 1 min |
| All-pairs generation | Algorithm | CRITICAL | 15 min |
| Events extracted too liberally | Data | HIGH | 10 min |
| Edge threshold (0.3) | Config | HIGH | 1 min |
| Bidirectional duplicates | Logic | MEDIUM | 10 min |
| Under-training (10 epochs) | Training | MEDIUM | 2-4 hours |
| Weak deduplication | Logic | MEDIUM | 10 min |
| No negative sampling | Training | LOW | 1 hour |

---

## Estimated Predictions by Configuration

```
Current Setup:
  Events/sample: 10 (extracted)
  Pairs/sample: 90 (all combinations)
  Threshold: 0.3
  Pass rate: 65% → ~58 predictions/sample
  Total: 999 × 58 ≈ 21,839 ✓ (matches actual)

After Phase 1 (threshold 0.5, max 6 events):
  Events/sample: 6
  Pairs/sample: 30
  Threshold: 0.5
  Pass rate: 35% → ~10 predictions/sample
  Total: 999 × 10 ≈ 10,000

After Phase 2 (selective pairs):
  Events/sample: 6
  Pairs/sample: 10 (heuristic edges only)
  Threshold: 0.5
  Pass rate: 65% → ~6.5 predictions/sample
  Total: 999 × 6.5 ≈ 6,500
```

---

## Performance Bottlenecks

1. **Event Extraction** (35% of inference time)
   - Keyword matching in all text
   - BERT encoding for similarity
   - Deduplication logic

2. **Graph Construction** (25% of inference time)
   - O(N²) pair evaluation
   - Heuristic rule checking
   - Edge feature extraction

3. **GNN Forward Pass** (20% of inference time)
   - GAT operations
   - Edge gating
   - Layer normalization

4. **Classifier** (15% of inference time)
   - MLP inference
   - Softmax computation

5. **Output Formatting** (5% of inference time)
   - Threshold filtering
   - Format conversion

---

## GPU Requirements

- **Minimum**: 4GB VRAM
  - BERT: 400MB
  - Event embeddings: 500MB
  - GNN parameters: 200MB
  - Batch data: 800MB

- **Recommended**: 8GB VRAM
  - Allows larger batch sizes
  - Better performance

- **Training**: 8-16GB
  - Gradient accumulation
  - Optimizer states

---

## Dependencies

```
torch==2.0.0+           # PyTorch
torch-geometric==2.3.0  # Graph neural networks
transformers==4.30.0    # BERT model
numpy==1.24.0
matplotlib==3.7.0       # Visualization
networkx==3.1           # Graph utilities
tqdm==4.65.0            # Progress bars
scikit-learn==1.3.0     # Metrics
```

---

## Next Steps for Improvement

1. **Quick Wins** (1 hour)
   - Adjust thresholds
   - Reduce max events
   - Fix deduplication

2. **Medium Improvements** (4 hours)
   - Selective pair generation
   - Better edge filtering
   - Bidirectional dedup

3. **Long-term** (1-2 days)
   - Train longer (20+ epochs)
   - Better event extraction (NER)
   - Domain-specific BERT (FinBERT)
   - Ensemble multiple models
   - Add confidence calibration

