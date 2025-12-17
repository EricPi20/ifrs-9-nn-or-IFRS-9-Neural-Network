# Quick Reference Card - Neural Network Scorecard Optimization

## 🎯 Your Current State
- **System:** Neural network-based credit scorecard
- **Current AR:** ~0.24 (from validation report)
- **Status:** Well-implemented but not optimized

---

## ⚡ Fastest Way to Improve (30 minutes)

```bash
cd nn-scorecard/backend
source venv/bin/activate
python quick_optimization.py --data YOUR_DATA.csv --segment CONSUMER
```

**Expected:** +0.04 to +0.09 AR improvement (15-35%)

---

## 🔑 Key Parameters to Tune (Priority Order)

### 1. **loss_alpha** (BIGGEST IMPACT!)
```
Current: 0.3 (30% BCE, 70% AR)
Try:     0.2 (20% BCE, 80% AR)  ← Quick win!
         0.1 (10% BCE, 90% AR)  ← Aggressive
```
**Expected improvement:** +0.01 to +0.03 AR

### 2. **margin** (for pairwise/WMW loss)
```
Current: 0.0
Try:     0.3  ← Quick win!
         0.5  ← Aggressive
```
**Expected improvement:** +0.01 to +0.02 AR

### 3. **auc_loss_type**
```
Current: pairwise
Try:     wmw      ← Often better
         soft     ← Experimental
```
**Expected improvement:** +0.005 to +0.015 AR

### 4. **Architecture**
```
Current: [your layers]
Try:     [64, 32]         ← Simple baseline
         [128, 64, 32]    ← Deep
         []               ← Linear (benchmark)
```
**Expected improvement:** +0.02 to +0.05 AR

### 5. **Ensemble** (Easiest win!)
```
Train 5 models with different seeds, average predictions
```
**Expected improvement:** +0.015 to +0.03 AR

---

## 📊 Configuration Cheat Sheet

### Configuration A: "Quick Win"
**Best for:** Immediate improvement, 15 min

```python
{
    "loss_alpha": 0.2,
    "auc_loss_type": "pairwise",
    "margin": 0.3,
    "hidden_layers": [64, 32],
    "dropout_rate": 0.2,
    "epochs": 100
}
```
**Expected AR gain:** +0.015 to +0.025

---

### Configuration B: "Aggressive AR"
**Best for:** Maximum discrimination, 15 min

```python
{
    "loss_alpha": 0.1,
    "auc_loss_type": "wmw",
    "margin": 0.5,
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "epochs": 150
}
```
**Expected AR gain:** +0.02 to +0.04

---

### Configuration C: "Balanced"
**Best for:** Production-ready model, 20 min

```python
{
    "loss_alpha": 0.2,
    "auc_loss_type": "pairwise",
    "margin": 0.3,
    "hidden_layers": [64, 32],
    "dropout_rate": 0.2,
    "epochs": 150
}
```
**Expected AR gain:** +0.015 to +0.025
**Plus ensemble (+5 models):** +0.03 to +0.05 total

---

## 🚀 Quick Commands

### Test single change
```bash
# In your training config, just change:
config.loss.loss_alpha = 0.2
config.loss.margin = 0.3
```

### Run quick optimization (all quick wins)
```bash
python quick_optimization.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER
```

### Run systematic search (9 configs, 1-2 hours)
```bash
python hyperparameter_search.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --mode quick \
    --folds 5
```

### Run thorough search (~80 configs, 1-2 days)
```bash
python hyperparameter_search.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --mode thorough \
    --folds 5
```

---

## 📈 Expected Results Roadmap

### Current State
```
AR: 0.24
Status: Baseline
```

### After Quick Optimization (2 hours)
```
AR: 0.27-0.28
Status: +12-17% improvement
Action: Changed loss_alpha, added margin
```

### After Systematic Search (2-3 days)
```
AR: 0.30-0.32
Status: +25-33% improvement
Action: Found optimal architecture + loss config
```

### After Ensemble (+ 1 hour)
```
AR: 0.32-0.35
Status: +33-45% improvement
Action: 5-model ensemble with best config
```

---

## 🎓 Rules of Thumb

### Loss Function
- **Lower alpha = Higher AR** (but may hurt calibration)
- **Optimal range:** 0.1 to 0.3 for credit scoring
- **Default 0.3 is conservative** (good calibration, moderate AR)

### Architecture
- **For WoE features:** Simpler is often better
- **Start with:** [64, 32] or [128, 64]
- **Max depth:** 3-4 layers for typical datasets
- **Always test linear model** as baseline

### Regularization
- **Dropout:** 0.2-0.3 for hidden layers
- **L2:** 0.001 is good default
- **If overfitting:** Increase dropout to 0.4, L2 to 0.01

### Training
- **Epochs:** Set high (150-200), let early stopping decide
- **Batch size:** 256 is good default
- **Learning rate:** 0.001 works for most cases

### Ensemble
- **Always worth it** for production
- **5 models** is sweet spot (cost vs benefit)
- **Typical gain:** +0.015 to +0.03 AR

---

## ⚠️ Common Mistakes to Avoid

### ❌ Don't
1. Over-optimize on test set (use CV!)
2. Use very deep networks (4+ layers) without enough data
3. Set alpha > 0.4 (too much calibration focus)
4. Ignore train-test gap (indicates overfitting)
5. Deploy single model without ensemble

### ✅ Do
1. Always use 5-fold CV for model selection
2. Start simple, add complexity only if needed
3. Use loss_alpha 0.1-0.3 for AR optimization
4. Monitor train-test gap (should be < 0.05)
5. Use ensemble for production

---

## 🔍 Diagnostic Guide

### Problem: AR not improving beyond 0.25

**Check:**
- loss_alpha too high? → Try 0.1-0.2
- No margin? → Add margin=0.3
- Poor features? → Review WoE binning

**Solution:**
```python
config.loss.loss_alpha = 0.15
config.loss.margin = 0.3
```

---

### Problem: Large train-test gap (overfitting)

**Example:** Train AR=0.35, Test AR=0.22

**Solution:**
```python
config.regularization.dropout_rate = 0.4  # Increase
config.regularization.l2_lambda = 0.01    # Increase
config.neural_network.hidden_layers = [32]  # Simplify
```

---

### Problem: Training not converging

**Symptoms:** Loss/AR oscillating

**Solution:**
```python
config.learning_rate = 0.0003  # Reduce
config.batch_size = 512        # Increase
config.loss.auc_loss_type = 'pairwise'  # More stable
```

---

## 💡 Pro Tips

### Tip 1: Segment-Specific Optimization
Different segments may need different configs:
- **Retail:** Aggressive AR focus (alpha=0.1)
- **SME:** Balanced (alpha=0.2)
- **Corporate:** More calibration (alpha=0.3)

### Tip 2: Feature Importance
After training, check:
```python
importance = model.get_feature_importance()
# Remove features with importance < 0.01
```

### Tip 3: Learning Curves
Monitor:
- **Both improving:** Good, keep training
- **Gap widening:** Overfitting, more regularization
- **Both flat:** Underfitting, more capacity

### Tip 4: Ensemble Diversity
For best ensemble:
- Different architectures ([64,32], [128,64], [32,16,13,16])
- Different loss types (pairwise, wmw, soft)
- Different seeds (42, 43, 44, 45, 46)

---

## 📞 Quick Help

### Question: Where do I start?

**Answer:** Run quick_optimization.py
```bash
python quick_optimization.py --data YOUR_DATA.csv
```

---

### Question: How much time will this take?

**Answer:**
- Quick optimization: 30 min - 2 hours
- Quick search: 1-2 hours
- Thorough search: 1-2 days

---

### Question: What improvement can I expect?

**Answer:**
- Quick wins: +0.04 to +0.09 AR
- Systematic: +0.08 to +0.12 AR
- Total potential: +0.10 to +0.15 AR

---

### Question: Should I use ensemble?

**Answer:** YES for production. Adds:
- Robustness (+stability)
- Performance (+0.02 to +0.03 AR)
- Confidence (std gives uncertainty)

---

## 📋 One-Page Summary

```
CURRENT STATE:    AR ~0.24 (baseline)
TARGET STATE:     AR ~0.32-0.35 (+33-45%)

QUICK WINS (30 min):
  ✓ Set loss_alpha = 0.2
  ✓ Set margin = 0.3
  → Expected: +0.02 AR

SYSTEMATIC (2-3 days):
  ✓ Test architectures [64,32], [128,64,32]
  ✓ Test loss types: pairwise, wmw
  ✓ 5-fold CV validation
  → Expected: +0.05 AR (additional)

ENSEMBLE (1 hour):
  ✓ Train 5 models, average predictions
  → Expected: +0.03 AR (additional)

TOTAL: +0.10 AR improvement (from 0.24 to 0.34)
```

---

## 🎯 Your Action Plan (This Week)

### Monday
- [ ] Run `quick_optimization.py`
- [ ] Review results
- [ ] Pick best single config

### Tuesday-Wednesday
- [ ] Run `hyperparameter_search.py --mode quick`
- [ ] Identify top 3 configs
- [ ] Document findings

### Thursday
- [ ] Train 5-model ensemble with best config
- [ ] Calculate improvement vs baseline
- [ ] Validate results

### Friday
- [ ] Generate final scorecard
- [ ] Business validation
- [ ] Document for production

---

## 📚 Full Documentation

For detailed explanations, see:
- **`SCORECARD_OPTIMIZATION_GUIDE.md`** - Complete 10-priority guide
- **`OPTIMIZATION_README.md`** - Usage instructions
- **`LOSS_FUNCTION_FIXES.md`** - Loss function details

---

**Ready to start?** Run this now:
```bash
cd nn-scorecard/backend
python quick_optimization.py --data YOUR_DATA.csv --segment YOUR_SEGMENT
```

**Expected time:** 30 minutes to 2 hours  
**Expected gain:** +15-35% AR improvement

**Go!** 🚀

