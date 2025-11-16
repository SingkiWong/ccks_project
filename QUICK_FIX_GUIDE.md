# Quick Fix Guide - Overprediction Problem

## Problem Summary
- **Predictions**: 21,839
- **Ground Truth**: 7,886
- **Overprediction Rate**: 2.77x (177% excess)

## Root Causes (in order of impact)

### 1. CRITICAL: Low Prediction Threshold (0.3)
**Impact**: 60-70% of all pairs pass this threshold
**File**: `predict_ccks_improved.py`, Line 310

**Current Code**:
```python
parser.add_argument('--threshold', type=float, default=0.3)
```

**Fix**:
```python
# Option 1: Change default
parser.add_argument('--threshold', type=float, default=0.5)

# Option 2: Use argmax-only prediction (> 0.5 by definition)
# Modify lines 696-701 in causality_extraction.py to:
if pred_class < 2:  # Remove threshold check, rely on argmax
    ...
```

**Expected Impact**: ~50% reduction in predictions (10,900 instead of 21,839)

---

### 2. CRITICAL: Cartesian Product Event Pair Generation
**Impact**: N events → N*(N-1) pairs (8 events = 56 pairs, 10 events = 90 pairs)
**File**: `causality_extraction.py`, Lines 683-687

**Current Code**:
```python
event_pairs = [
    (i, j) for i in range(len(events)) 
    for j in range(len(events)) if i != j
]
```

**Fix - Option A (Conservative)**: Only use heuristic-connected edges
```python
# Build list of potential pairs from graph edges only
event_pairs = []
edge_index, _ = self.build_initial_graph(text, events)
if edge_index.shape[1] > 0:
    for src, dst in edge_index.t():
        event_pairs.append((src.item(), dst.item()))
else:
    event_pairs = []  # No predictions if no heuristic connections
```

**Fix - Option B (Moderate)**: Limit pairs per event
```python
# Max 5 pairs per event to prevent explosion
event_pairs = []
for i in range(len(events)):
    for j in range(len(events)):
        if i != j and len([p for p in event_pairs if p[0] == i]) < 5:
            event_pairs.append((i, j))
```

**Expected Impact**: ~40-60% reduction in pairs generated

---

### 3. HIGH: Aggressive Event Extraction
**Impact**: 8-10 events per sample → 90+ possible pairs
**File**: `predict_ccks_improved.py`, Lines 48-119

**Current Code**:
```python
for phrase in phrases:
    if phrase in text:  # Finds ALL occurrences
        idx = text.find(phrase)
        # Creates event with context window
        event = CausalEvent(...)
        events.append(event)
```

**Fix - Option A**: Reduce to top-N occurrences
```python
def extract_events_from_text(self, text: str) -> List[CausalEvent]:
    events = []
    phrase_matches = {}  # Track unique phrases
    
    for category, phrases in keywords.items():
        for phrase in phrases:
            matches = [m.start() for m in re.finditer(re.escape(phrase), text)]
            # Keep only first 2 occurrences per phrase
            for idx in matches[:2]:  
                context = ...
                event = CausalEvent(...)
                # Add only if not duplicate
                key = (event.text, event.event_type)
                if key not in seen:
                    events.append(event)
                    seen.add(key)
    
    return events[:6]  # Reduce from 10 to 6 max
```

**Fix - Option B**: Better deduplication
```python
# After extraction, improve deduplication
unique_events = []
seen = set()
for event in events:
    # Check both text AND semantic similarity
    key = (event.event_type, event.arguments.get('product', ''))
    if key not in seen:
        unique_events.append(event)
        seen.add(key)

return unique_events[:7]
```

**Expected Impact**: ~30-40% fewer events, ~50% fewer pairs

---

### 4. HIGH: Weak Edge Filtering
**Impact**: Confidence threshold of 0.3 allows many unrelated pairs
**File**: `causality_extraction.py`, Line 608

**Current Code**:
```python
if confidence > 0.3:
    edge_list.append([i, j])
```

**Fix**:
```python
# Strengthen threshold
if confidence > 0.5:  # Was 0.3
    edge_list.append([i, j])
    edge_features.append(edge_feat)

# Additionally, remove disconnected components
if len(edge_list) == 0:
    # If no edges, predict nothing
    return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 11))
```

**Expected Impact**: ~20-30% fewer edges, better model focus

---

### 5. MEDIUM: Bidirectional Double-Counting
**Impact**: Same pair counted twice from both directions
**File**: `causality_extraction.py`, Lines 696-716

**Current Code**:
```python
for idx, (i, j) in enumerate(event_pairs):
    pred_class = torch.argmax(probs[idx]).item()
    confidence = probs[idx][pred_class].item()
    
    if pred_class < 2 and confidence > threshold:
        if pred_class == 0:
            pair = CausalPair(cause=events[i], effect=events[j])
        else:
            pair = CausalPair(cause=events[j], effect=events[i])  # Reverse!
        causal_pairs.append(pair)
```

**Problem**:
```
Pair (A,B): class=0 → creates A→B
Pair (B,A): class=1 → creates B→A (which equals A→B reversed!)

If both exceed threshold, same relationship appears twice.
```

**Fix**:
```python
# Only predict one direction per pair
predicted_pairs = set()  # Track (cause, effect) tuples
causal_pairs = []

for idx, (i, j) in enumerate(event_pairs):
    pred_class = torch.argmax(probs[idx]).item()
    confidence = probs[idx][pred_class].item()
    
    if pred_class < 2 and confidence > threshold:
        if pred_class == 0:
            cause_idx, effect_idx = i, j
        else:
            cause_idx, effect_idx = j, i
        
        # Skip if we've already predicted this pair reversed
        pair_key = (min(cause_idx, effect_idx), max(cause_idx, effect_idx))
        if pair_key not in predicted_pairs:
            pair = CausalPair(
                cause=events[cause_idx], 
                effect=events[effect_idx],
                confidence=confidence
            )
            causal_pairs.append(pair)
            predicted_pairs.add(pair_key)

return causal_pairs
```

**Expected Impact**: ~10-15% reduction (removes bidirectional duplicates)

---

## Implementation Priority

### PHASE 1 (Do First - 5 minutes)
1. Change threshold from 0.3 to 0.5
2. Reduce max events from 10 to 6-7
3. Change edge confidence threshold from 0.3 to 0.5

**Expected result**: 10,900-13,500 predictions (40-50% improvement)

### PHASE 2 (Medium effort - 30 minutes)
4. Implement selective pair generation (use graph edges only)
5. Fix bidirectional double-counting bug
6. Improve event deduplication

**Expected result**: 7,000-9,000 predictions (near ground truth)

### PHASE 3 (Better model - 2-4 hours)
7. Train for 20+ epochs instead of 10
8. Add negative pair sampling
9. Use better event extraction (NER model instead of keywords)

**Expected result**: Better precision/recall scores

---

## Testing the Fixes

### Test Script
```python
from evaluate_predictions import evaluate_predictions

# After each fix, run:
evaluate_predictions(
    pred_file='predictions_fixed.txt',
    gold_file='ccks_task2_eval_data.txt',
    detailed=True
)
```

### Success Criteria
- Predictions: 7,000-9,000 (close to 7,886 ground truth)
- Precision: >30%
- F1 Score: >20%

---

## Code Locations Reference

| Issue | File | Lines | Fix Priority |
|-------|------|-------|--------------|
| Default threshold | `predict_ccks_improved.py` | 310 | P1 |
| Event extraction | `predict_ccks_improved.py` | 48-119 | P1 |
| Pair generation | `causality_extraction.py` | 683-687 | P2 |
| Edge filtering | `causality_extraction.py` | 608 | P1 |
| Double-counting | `causality_extraction.py` | 696-716 | P2 |
| Training epochs | `train_ccks.py` | 381 | P3 |
| Batch size | `train_ccks.py` | 350-355 | P3 |

---

## Validation Checklist

After implementing fixes:
- [ ] Run evaluation on test set
- [ ] Check precision/recall improvement
- [ ] Verify no duplicate pairs in output
- [ ] Confirm prediction count < 10,000
- [ ] Test on individual samples manually
- [ ] Profile memory usage (if using GPU)

---

## Expected Outcomes

### Before Fix
```
Predictions: 21,839
Ground Truth: 7,886
Precision: ~4.6%
Recall: ~12.7%
F1: ~6.8%
```

### After Phase 1
```
Predictions: ~10,900
Ground Truth: 7,886
Precision: ~7-8%
Recall: ~12-13%
F1: ~9-10%
```

### After Phase 2
```
Predictions: ~7,500
Ground Truth: 7,886
Precision: ~25-30%
Recall: ~35-40%
F1: ~28-34%
```

### After Phase 3 (with better training)
```
Predictions: ~7,200
Ground Truth: 7,886
Precision: ~40-45%
Recall: ~45-50%
F1: ~42-47%
```

