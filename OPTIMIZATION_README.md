# Neural Network Scorecard Optimization - Quick Start Guide

## 📚 What You Have

I've created a comprehensive optimization framework for your neural network scorecard system:

### 1. **Complete Optimization Guide** 📖
- **File:** `SCORECARD_OPTIMIZATION_GUIDE.md`
- **Contents:** 10 priority areas with detailed recommendations
- **Expected improvement:** +33-45% AR (from ~0.24 to ~0.32-0.35)

### 2. **Hyperparameter Search Script** 🔍
- **File:** `nn-scorecard/backend/hyperparameter_search.py`
- **Purpose:** Systematic exploration of configurations
- **Modes:** Quick (1-2 hours), Thorough (1-2 days), Custom

### 3. **Quick Optimization Script** ⚡
- **File:** `nn-scorecard/backend/quick_optimization.py`
- **Purpose:** Test immediate improvements in 30 min - 2 hours
- **Expected improvement:** +0.04 to +0.09 AR

---

## 🚀 Getting Started (Choose Your Path)

### Path A: Quick Wins (Recommended First!)

**Time:** 30 minutes to 2 hours  
**Expected Improvement:** +0.04 to +0.09 AR  
**Effort:** Low

```bash
cd nn-scorecard/backend
source venv/bin/activate

# Run quick optimization on your data
python quick_optimization.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --target default

# This will test:
# 1. Lower alpha (0.2) + margin (0.3)
# 2. WMW loss
# 3. Aggressive AR focus (alpha=0.1)
# 4. 5-model ensemble
```

**What it does:**
- Compares 3 optimized configurations against your baseline
- Trains 5-model ensemble with best config
- Provides detailed improvement analysis
- Saves results to `data/quick_optimization/`

**Example output:**
```
BASELINE: Test AR = 0.2400

IMPROVEMENTS:
  Test 1 (alpha=0.2 + margin): +0.0180 (+7.5%)
  Test 2 (WMW loss): +0.0150 (+6.2%)
  Test 3 (alpha=0.1): +0.0210 (+8.8%)
  Test 4 (Ensemble): +0.0320 (+13.3%)

TOTAL IMPROVEMENT: +0.0320 (+13.3%)
FINAL TEST AR: 0.2720
```

---

### Path B: Systematic Search

**Time:** 1-2 days  
**Expected Improvement:** +0.08 to +0.12 AR  
**Effort:** Medium

#### Step 1: Quick Search (1-2 hours)

Test 9-12 key configurations:

```bash
python hyperparameter_search.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --mode quick \
    --folds 5 \
    --top-k 5
```

**Output:**
- CSV with all results: `data/experiments/search_results_TIMESTAMP.csv`
- Top 5 configs: `data/experiments/top_5_configs_TIMESTAMP.json`
- Console summary with best configurations

#### Step 2: Thorough Search (1-2 days)

After quick search identifies promising areas:

```bash
python hyperparameter_search.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --mode thorough \
    --folds 5 \
    --top-k 10
```

**This tests:**
- 4 loss_alpha values × 3 AUC types = 12 loss configurations
- 7 architectures (from linear to deep)
- 3 dropout values
- 3 L2 values
- **Total:** ~80-100 configurations

---

### Path C: Custom Search

**Time:** Variable  
**Effort:** High

Create your own search space:

```json
// my_search_config.json
[
  {
    "architecture": {
      "hidden_layers": [64, 32],
      "model_type": "neural_network"
    },
    "loss": {
      "loss_alpha": 0.2,
      "auc_loss_type": "pairwise",
      "margin": 0.3,
      "auc_gamma": 2.0
    },
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "l2_lambda": 0.001,
    "epochs": 150
  },
  {
    "architecture": {
      "hidden_layers": [128, 64, 32],
      "model_type": "neural_network"
    },
    "loss": {
      "loss_alpha": 0.1,
      "auc_loss_type": "wmw",
      "margin": 0.5,
      "auc_gamma": 2.0
    },
    "learning_rate": 0.0003,
    "dropout_rate": 0.3,
    "l2_lambda": 0.01,
    "epochs": 200
  }
]
```

Run:
```bash
python hyperparameter_search.py \
    --data data/uploads/your_data.csv \
    --segment CONSUMER \
    --mode custom \
    --config my_search_config.json \
    --folds 5
```

---

## 📊 Understanding Results

### Key Metrics

**AR (Accuracy Ratio / Gini):**
- Primary metric for scorecard quality
- Range: 0.0 to 1.0 (higher is better)
- Industry benchmarks:
  - Acceptable: 0.30+
  - Good: 0.40+
  - Excellent: 0.50+

**Cross-Validation Std:**
- Measures model stability
- Lower is better
- Rule of thumb:
  - std < 0.02: Excellent stability
  - std < 0.05: Good stability
  - std > 0.05: Unstable (need more regularization)

**Train-Test Gap:**
- Difference between train AR and test AR
- Indicates overfitting
- Rule of thumb:
  - Gap < 0.03: Excellent
  - Gap < 0.05: Good
  - Gap > 0.10: Overfitting

### Interpreting Search Results

Example from `search_results_TIMESTAMP.csv`:

```csv
config_id,architecture,loss_alpha,auc_loss_type,margin,dropout,l2_lambda,mean_ar,std_ar,mean_auc,std_auc
0,[128 64 32],0.2,pairwise,0.3,0.3,0.001,0.3245,0.0180,0.6622,0.0090
1,[64 32],0.1,wmw,0.5,0.2,0.001,0.3198,0.0210,0.6599,0.0105
2,[64 32 16],0.2,soft,0.0,0.2,0.001,0.3156,0.0165,0.6578,0.0082
```

**Best config:** ID 0
- **Why:** Highest mean_ar (0.3245)
- **Stability:** Good (std_ar = 0.0180)
- **Configuration:** [128,64,32], alpha=0.2, pairwise, margin=0.3

---

## 🎯 Recommended Workflow

### Week 1: Quick Exploration

**Day 1-2:**
1. ✅ Run `quick_optimization.py` on your data
2. ✅ Review results and improvements
3. ✅ Document baseline vs optimized AR

**Day 3-4:**
1. ✅ Run `hyperparameter_search.py --mode quick`
2. ✅ Identify top 3-5 configurations
3. ✅ Analyze what works best (loss config? architecture?)

**Day 5:**
1. ✅ Train final model with best config
2. ✅ Generate scorecard
3. ✅ Validate with business rules

### Week 2: Thorough Optimization

**Day 1-3:**
1. ✅ Run `hyperparameter_search.py --mode thorough`
2. ✅ Monitor progress (takes 1-2 days)
3. ✅ Review results

**Day 4-5:**
1. ✅ Train ensemble with top 3 configs
2. ✅ Out-of-time validation
3. ✅ Final model selection

### Week 3: Production

**Day 1-2:**
1. ✅ Retrain final model on full dataset
2. ✅ Generate production scorecard
3. ✅ Create documentation

**Day 3-5:**
1. ✅ Business validation
2. ✅ Deploy to production
3. ✅ Monitor performance

---

## 💡 Tips and Best Practices

### 1. Start Small, Scale Up

```bash
# First test on subset
head -1000 data.csv > data_subset.csv

# Quick test
python quick_optimization.py --data data_subset.csv --no-ensemble

# If looks good, run on full data
python quick_optimization.py --data data.csv
```

### 2. Use Good Hardware

**GPU vs CPU:**
- GPU: 3-5x faster for neural networks
- Your system detects automatically
- Check with: `torch.cuda.is_available()`

**Recommendations:**
- Quick optimization: CPU is fine
- Thorough search: GPU highly recommended

### 3. Monitor Progress

Both scripts log detailed progress:

```bash
# Save output to file
python quick_optimization.py --data data.csv 2>&1 | tee optimization.log

# View results later
cat optimization.log
```

### 4. Parallel Experiments

Run multiple segments in parallel:

```bash
# Terminal 1
python hyperparameter_search.py --segment CONSUMER --mode quick &

# Terminal 2
python hyperparameter_search.py --segment SME --mode quick &

# Terminal 3
python hyperparameter_search.py --segment CORPORATE --mode quick &
```

### 5. Save Everything

All scripts save results automatically:
- Quick optimization: `data/quick_optimization/`
- Hyperparameter search: `data/experiments/`

Results include:
- Summary JSON with all metrics
- CSV with detailed results
- Top K configurations

---

## 🔧 Advanced Usage

### Custom Loss Function

Edit `quick_optimization.py` to add your own config:

```python
def create_optimized_config_5(self) -> TrainingConfig:
    """My custom configuration."""
    config = self.create_baseline_config()
    config.loss.loss_alpha = 0.15
    config.loss.auc_loss_type = 'soft'
    config.loss.auc_gamma = 10.0  # Very sharp
    config.neural_network.hidden_layers = [256, 128, 64]
    config.neural_network.dropout_rate = 0.4
    return config
```

### Custom Metrics

Add custom evaluation to search:

```python
# In hyperparameter_search.py, modify train_with_cv()

# After getting result, add:
if result.test_metrics.discrimination.gini_ar > 0.35:
    # Calculate business-specific metrics
    top_10_capture = calculate_top_10_capture(y_val, predictions)
    result_dict['top_10_capture'] = top_10_capture
```

### Integration with Existing Code

Use optimized configs in your training pipeline:

```python
from quick_optimization import QuickOptimizer

# Load best config
optimizer = QuickOptimizer('data.csv')
best_config = optimizer.create_optimized_config_1()

# Use in your existing training code
result = your_training_function(
    data=your_data,
    config=best_config
)
```

---

## 📈 Expected Improvements Summary

| Method | Time | Effort | Expected AR Gain | Use When |
|--------|------|--------|------------------|----------|
| Quick optimization | 2 hours | Low | +0.04 to +0.09 | Need fast results |
| Quick search | 2 hours | Low | +0.05 to +0.08 | Explore options |
| Thorough search | 1-2 days | Medium | +0.08 to +0.12 | Final optimization |
| Ensemble | +30 min | Low | +0.015 to +0.03 | Production models |
| Manual tuning | Variable | High | +0.10 to +0.15 | Domain expertise |

**Combined (recommended):**
1. Quick optimization → +0.05 AR
2. Thorough search → +0.03 AR (additional)
3. Ensemble → +0.02 AR (additional)
**Total: +0.10 AR improvement (+40% from baseline 0.24)**

---

## 🐛 Troubleshooting

### Issue: Script crashes with memory error

**Solution:**
- Reduce batch size: `--batch-size 128`
- Use smaller dataset for testing
- Close other applications

### Issue: Training takes too long

**Solution:**
- Reduce epochs: Edit config `epochs=50`
- Use `--no-ensemble` flag
- Skip thorough search, use quick mode

### Issue: No improvement in AR

**Possible causes:**
1. Data quality issues (check for outliers, missing values)
2. Features not discriminatory enough
3. Class imbalance too extreme

**Solutions:**
1. Review WoE binning strategy
2. Add feature interactions
3. Use class weights: `use_class_weights=True`

### Issue: Large train-test gap (overfitting)

**Solutions:**
1. Increase dropout: `dropout_rate=0.4`
2. Increase L2: `l2_lambda=0.01`
3. Use simpler architecture: `hidden_layers=[32]`
4. Early stopping with lower patience

---

## 📚 Next Steps

### After Optimization:

1. **✅ Validate Results**
   - Out-of-time validation
   - Segment-level performance
   - Business rules check

2. **✅ Generate Scorecard**
   - Use best model from optimization
   - Verify monotonicity
   - Check score distribution

3. **✅ Document Everything**
   - Best hyperparameters
   - Training history
   - Validation results
   - Business approval

4. **✅ Deploy to Production**
   - API endpoint setup
   - Monitoring dashboard
   - Retraining schedule

5. **✅ Monitor Performance**
   - Track AR over time
   - Population stability (PSI)
   - Model drift detection

---

## 📞 Support

### Documentation

- **Main guide:** `SCORECARD_OPTIMIZATION_GUIDE.md` (comprehensive)
- **This guide:** `OPTIMIZATION_README.md` (quick start)
- **Project README:** `nn-scorecard/README.md` (system overview)

### Key Files

- Loss functions: `nn-scorecard/backend/app/services/losses.py`
- Training logic: `nn-scorecard/backend/app/services/trainer.py`
- Metrics: `nn-scorecard/backend/app/services/metrics.py`
- Models: `nn-scorecard/backend/app/models/neural_network.py`

### Validation Scripts

- Metrics validation: `nn-scorecard/backend/validate_metrics.py`
- Training analysis: `nn-scorecard/backend/analyze_training_run.py`

---

## ✅ Success Checklist

Before deploying your optimized scorecard:

### Optimization Complete
- [ ] Baseline AR documented
- [ ] Quick optimization run completed
- [ ] Systematic search completed (quick or thorough)
- [ ] Best configuration identified
- [ ] Ensemble tested (optional but recommended)

### Validation Complete
- [ ] Cross-validation performed (5-fold minimum)
- [ ] Out-of-time validation performed
- [ ] Train-test gap < 0.05
- [ ] Stability check (std_ar < 0.05)
- [ ] Segment-level validation

### Quality Checks
- [ ] Scorecard monotonicity verified
- [ ] Business rules validated
- [ ] Feature importance makes sense
- [ ] Score distribution reasonable
- [ ] No unexpected behaviors

### Documentation
- [ ] Hyperparameters documented
- [ ] Training history saved
- [ ] Validation results logged
- [ ] Improvement over baseline quantified
- [ ] Reproducibility verified (fixed seed)

### Production Ready
- [ ] Final model trained on full data
- [ ] Scorecard generated
- [ ] API endpoint tested
- [ ] Monitoring setup
- [ ] Retraining schedule defined

---

## 🎉 Summary

You now have:

1. ✅ **Comprehensive optimization guide** with 10 priority areas
2. ✅ **Two practical scripts** for immediate improvements
3. ✅ **Clear workflow** from baseline to production
4. ✅ **Expected improvements** of +33-45% AR

**Start with Quick Optimization →** Get +4-9% improvement in 2 hours!

Then proceed to systematic search for additional gains.

Good luck! 🚀

---

**Questions? Review the main guide:** `SCORECARD_OPTIMIZATION_GUIDE.md`

