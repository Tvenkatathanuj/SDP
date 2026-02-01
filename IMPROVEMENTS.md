# Model Improvements to Reduce False Positives

## Problem
The handwriting model was incorrectly classifying healthy spirals as Parkinson's disease, indicating high false positive rate.

## Solutions Implemented

### 1. **Class Weighting & Balanced Training**
- Added class weights (1.5 for Healthy, 1.0 for Parkinson's) to prioritize correct healthy classifications
- Changed loss function: Focal Loss with higher gamma (2.5) to focus on harder examples
- Tracking balanced accuracy and specificity metrics during training

### 2. **Test-Time Augmentation (TTA)**
- Predict on multiple augmented versions of the same image:
  - Original image
  - Horizontally flipped image
- Average predictions for more robust results
- Reduces variance and improves generalization

### 3. **Temperature Scaling**
- Apply temperature scaling (T=1.5) to soften overconfident predictions
- Better probability calibration
- Reduces false positives from overconfident model outputs

### 4. **Higher Classification Threshold**
- Increased decision threshold from 0.5 to 0.55 for Parkinson's classification
- Requires higher confidence before classifying as Parkinson's
- Provides warning for borderline cases (0.4-0.55 range)

### 5. **Improved Optimizer & Learning Rate Schedule**
- OneCycleLR scheduler with lower peak learning rate (3e-4)
- Stronger weight decay (1e-3) for better regularization
- Warmup period before applying aggressive augmentation

### 6. **Enhanced Training Metrics**
- Monitor specificity (correct healthy predictions)
- Monitor sensitivity (correct Parkinson's predictions)  
- Use balanced metric (60% accuracy + 40% balanced_acc) for model selection
- Prioritize models that perform well on both classes

## Expected Results

### Before:
- High false positive rate
- Healthy spirals classified as Parkinson's
- Overconfident predictions

### After:
- ✅ Reduced false positives through multiple techniques
- ✅ More balanced predictions between classes
- ✅ Better calibrated confidence scores
- ✅ Borderline cases flagged for monitoring
- ✅ More robust predictions with TTA

## How to Retrain

1. Open `handwriting_parkinsons_detection.ipynb`
2. Run all cells sequentially (the improvements are already integrated)
3. The notebook will:
   - Use class-balanced training
   - Apply temperature scaling
   - Use TTA during validation
   - Save the best model based on balanced metrics

## Usage in Gradio App

The `app.py` has been updated to:
- Use TTA during inference (averages predictions)
- Apply temperature scaling (T=1.5)
- Use threshold of 0.55 instead of 0.5
- Provide warnings for borderline cases (0.4-0.55)

## Additional Recommendations

1. **Collect More Data**: Especially more healthy samples if imbalanced
2. **Cross-Validation**: Use k-fold CV for more robust evaluation
3. **Ensemble Models**: Train multiple models with different seeds and average
4. **External Validation**: Test on external dataset to verify generalization
5. **Clinical Validation**: Work with medical professionals to validate predictions

## Technical Details

### Temperature Scaling Formula
```
softmax(logits / T) where T = 1.5
```
- T > 1: Softens probabilities (less confident)
- T < 1: Sharpens probabilities (more confident)

### Test-Time Augmentation
```
final_prediction = mean([pred_original, pred_flipped])
```

### Balanced Metric
```
balanced_metric = 0.6 * accuracy + 0.4 * balanced_accuracy
```

This ensures the model performs well on both Healthy and Parkinson's classes.

---

**Version:** 2.0  
**Last Updated:** February 2026  
**Status:** Ready for Retraining
