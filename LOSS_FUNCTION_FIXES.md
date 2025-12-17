# Loss Function Review and Fixes

## Summary
Reviewed and fixed issues with the loss function implementation to ensure proper usage and full configurability through the API.

## Issues Found and Fixed

### 1. ✅ Missing API Configuration Parameters

**Problem:** The `LossConfig` schema was missing two important parameters:
- `auc_loss_type`: Allows users to choose which AUC surrogate to use in CombinedLoss ('pairwise', 'soft', or 'wmw')
- `margin`: Controls the margin parameter for pairwise/WMW losses (enforces stricter separation)

**Fix:** Added both fields to `LossConfig` in `app/models/schemas.py`:

```python
class LossConfig(BaseModel):
    loss_type: str = Field(default='combined', ...)
    loss_alpha: float = Field(default=0.3, ...)
    auc_gamma: float = Field(default=2.0, ...)
    auc_loss_type: str = Field(
        default='pairwise',
        description="AUC surrogate for combined loss: 'pairwise', 'soft', or 'wmw'"
    )
    margin: float = Field(
        default=0.0,
        ge=0.0,
        description="Margin for pairwise/WMW losses"
    )
```

**Impact:** Users can now fully configure the loss function through the API.

---

### 2. ✅ Added Validation for `auc_loss_type`

**Problem:** No validation existed for the `auc_loss_type` parameter.

**Fix:** Added field validator:

```python
@field_validator('auc_loss_type')
@classmethod
def validate_auc_loss_type(cls, v):
    """Validate AUC loss type is supported."""
    allowed = ['pairwise', 'soft', 'wmw']
    if v not in allowed:
        raise ValueError(f'AUC loss type must be one of {allowed}')
    return v
```

---

### 3. ✅ Added Warning for Plain BCE Loss

**Problem:** When users select plain `'bce'` loss, they get NO AR/Gini optimization, which might not be obvious since the primary goal is AR optimization.

**Fix:** Added logging warning in `create_loss_function()`:

```python
if loss_type == 'bce':
    logger.warning(
        "Using plain BCE loss: This optimizes for probability calibration only. "
        "For AR/Gini optimization, consider using 'combined', 'pairwise_auc', 'soft_auc', or 'wmw'."
    )
    return nn.BCELoss()
```

---

### 4. ✅ Enhanced Documentation

**Problem:** Loss function documentation didn't clearly explain the implications of each choice.

**Fix:** Updated docstring in `create_loss_function()`:

```python
Important:
    - 'bce' optimizes for probability calibration only (no AR/Gini optimization)
    - For AR/Gini optimization, use 'combined', 'pairwise_auc', 'soft_auc', or 'wmw'
    - 'combined' balances calibration (BCE) and discrimination (AR), recommended for most cases
```

---

### 5. ✅ Added Informative Logging for CombinedLoss

**Problem:** Users couldn't easily see which AUC surrogate was being used.

**Fix:** Added logging when CombinedLoss is created:

```python
logger.info(
    f"Creating CombinedLoss: alpha={loss_alpha:.2f} (BCE weight), "
    f"auc_type={auc_loss_type}, margin={margin:.3f}"
)
```

---

### 6. ✅ Updated All Configuration Examples

**Fix:** Updated all example configurations in schemas to include the new fields:

```python
"loss": {
    "loss_type": "combined",
    "loss_alpha": 0.3,
    "auc_gamma": 2.0,
    "auc_loss_type": "pairwise",
    "margin": 0.0
}
```

---

### 7. ✅ Added Comprehensive Tests

**Added Tests:**
1. `test_factory_combined_with_margin` - Tests margin parameter
2. `test_factory_combined_different_auc_types` - Tests all AUC types
3. `test_factory_pairwise_with_margin` - Tests pairwise with margin
4. `test_factory_wmw_with_margin` - Tests WMW with margin
5. `test_loss_config_defaults` - Updated to test new defaults
6. `test_loss_config_invalid_auc_loss_type` - Validates auc_loss_type

---

## Test Results

### Loss Function Tests
```bash
✅ 45 tests passed (test_losses.py)
```

### Schema Tests
```bash
✅ 44 tests passed (test_schemas.py)
```

---

## What Was Already Working Well

1. **✅ Proper AUC Surrogate Implementation**
   - All three AUC surrogates correctly implement differentiable approximations
   - Edge cases handled correctly (all positive/negative batches)
   - Gradient flow maintained

2. **✅ CombinedLoss Design**
   - Correctly combines BCE with AUC surrogate
   - Formula: `Total = α * BCE + (1-α) * AUC` is mathematically sound
   - Default α=0.3 (70% weight to AR) is appropriate

3. **✅ Trainer Integration**
   - Properly handles both CombinedLoss (tuple return) and other losses (tensor return)
   - Special `isinstance` check prevents errors

---

## Usage Examples

### Example 1: Combined Loss with Soft AUC
```python
config = LossConfig(
    loss_type='combined',
    loss_alpha=0.3,
    auc_loss_type='soft',
    auc_gamma=2.0
)
```

### Example 2: WMW Loss with Margin
```python
config = LossConfig(
    loss_type='wmw',
    margin=0.5  # Stricter separation requirement
)
```

### Example 3: Combined Loss with Pairwise AUC and Margin
```python
config = LossConfig(
    loss_type='combined',
    loss_alpha=0.2,  # 80% weight to AR
    auc_loss_type='pairwise',
    margin=0.3
)
```

---

## Files Modified

1. `nn-scorecard/backend/app/models/schemas.py`
   - Added `auc_loss_type` and `margin` fields to `LossConfig`
   - Added validation for `auc_loss_type`
   - Updated examples

2. `nn-scorecard/backend/app/services/losses.py`
   - Added logging import
   - Added warning for plain BCE
   - Added info logging for CombinedLoss
   - Enhanced docstring

3. `nn-scorecard/backend/tests/test_losses.py`
   - Added 3 new test methods
   - Updated existing tests

4. `nn-scorecard/backend/tests/test_schemas.py`
   - Added validation test for `auc_loss_type`
   - Updated default tests
   - Fixed pre-existing test bug (early_stopping default)

---

## Conclusion

The loss functions are now:
- ✅ Fully configurable through the API
- ✅ Properly validated
- ✅ Well documented
- ✅ Comprehensively tested
- ✅ User-friendly with helpful warnings and logging

All mathematical implementations were already correct. The fixes primarily focused on improving configurability, validation, and user experience.

