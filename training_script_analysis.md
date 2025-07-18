# Training Script Analysis: Potential High Loss Issues

## Overview
This analysis examines the provided training script for a multi-phase object detection model using RPN (Region Proposal Network) and detection heads. Several issues could contribute to high loss during training.

## Critical Issues Identified

### 1. **Gradient Scaling and Mixed Precision Issues**
```python
# Current implementation:
with autocast():
    pooled = roi_align(input=c4, boxes=[proposals], output_size=(7,7), spatial_scale=1/8, sampling_ratio=2)
    cls_det, reg_det = det_head(pooled)
    det_cls, det_reg = detection_loss(...)
    loss = rpn_cls + rpn_reg + det_cls + det_reg
```

**Problems:**
- Only detection head forward pass is under `autocast()`, but RPN losses are computed outside
- Mixed precision scaling may cause gradient underflow/overflow
- Loss components from different precision contexts are being added

**Solution:**
```python
with autocast():
    # Move ALL forward passes under autocast
    feats = backbone(torch.stack(images))
    c4 = feats['c4']
    logits, deltas = rpn_head(c4)
    # ... rest of forward pass
    loss = rpn_cls + rpn_reg + det_cls + det_reg
```

### 2. **Anchor Generation and Scaling Issues**
```python
anchors = AnchorGenerator(base_size=16, scales=scales, ratios=ratios, stride=8)
```

**Problems:**
- Base size (16) vs stride (8) mismatch - typically base_size should equal stride
- For 128x128 input images, feature map is 16x16, so stride should be 128/16 = 8
- Anchor scales [0.5, 1.0, 2.0] might be too large for small objects in Quick Draw dataset

**Recommended fixes:**
```python
# Better anchor configuration for 128x128 images
anchors = AnchorGenerator(base_size=8, scales=[0.25, 0.5, 1.0], ratios=ratios, stride=8)
```

### 3. **Learning Rate and Optimization Issues**
```python
lr = 2e-4     # start slightly higher when fine-tuning pretrained backbone
optimizer = torch.optim.SGD(opt_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
```

**Problems:**
- Learning rate might be too high for fine-tuning, especially with partial backbone freezing
- SGD might be too aggressive for detection tasks
- No warmup schedule for transfer learning

**Recommended fixes:**
```python
# Lower learning rate for fine-tuning
lr = 5e-5 to 1e-4
# Consider Adam optimizer
optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
# Add warmup
```

### 4. **Backbone Freezing Strategy Issues**
```python
# Unfreeze layers 2, 3, 4
for name, p in backbone.named_parameters():
    p.requires_grad = any(stage in name for stage in ["layer2","layer3","layer4"])
```

**Problems:**
- Conv1 and layer1 are frozen, but these might be important for domain adaptation
- Batch normalization layers might still be updating running statistics even when frozen
- No gradual unfreezing strategy

**Recommended fixes:**
```python
# Gradual unfreezing or unfreeze more layers
for name, p in backbone.named_parameters():
    if any(stage in name for stage in ["layer1", "layer2", "layer3", "layer4"]):
        p.requires_grad = True
    # Or implement gradual unfreezing by epoch
```

### 5. **Detection Head Architecture Issues**
From the DetectionHead class:
```python
# BBox regression layer: 4 coords per class (including background)
self.bbox_pred = nn.Linear(hidden_dim, 4 * (num_classes + 1))
```

**Problems:**
- Class-specific bbox regression is unusual and computationally expensive
- Should typically be class-agnostic: `4 * 1` outputs
- This creates a massive output layer for many classes (e.g., 4 * 376 = 1504 outputs for phase 4)

**Recommended fix:**
```python
# Class-agnostic bbox regression
self.bbox_pred = nn.Linear(hidden_dim, 4)
```

### 6. **Loss Balancing Issues**
```python
loss = rpn_cls + rpn_reg + det_cls + det_reg
```

**Problems:**
- No loss weighting - different loss components may have vastly different magnitudes
- RPN and detection losses might be imbalanced
- No loss normalization

**Recommended fixes:**
```python
# Add loss weights
loss = (
    1.0 * rpn_cls + 
    1.0 * rpn_reg + 
    2.0 * det_cls + 
    1.0 * det_reg
)
```

### 7. **ROI Align Configuration Issues**
```python
pooled = roi_align(input=c4, boxes=[proposals], output_size=(7,7), spatial_scale=1/8, sampling_ratio=2)
```

**Problems:**
- `spatial_scale=1/8` assumes stride=8, but this should match actual backbone stride
- Small sampling_ratio=2 might cause aliasing
- Output size (7,7) might be too small for detailed features

**Recommended fixes:**
```python
# Verify spatial_scale matches actual backbone stride
# Consider larger output size or higher sampling ratio
pooled = roi_align(input=c4, boxes=[proposals], output_size=(7,7), spatial_scale=1/8, sampling_ratio=4)
```

### 8. **Data and Target Issues**
**Potential problems:**
- Quick Draw dataset might have inconsistent annotation quality
- Bounding box coordinates might be in wrong format (absolute vs relative)
- Class imbalance across phases
- Small sample sizes in later phases (15-20 samples per class)

### 9. **Proposal Layer Configuration**
```python
proposal_layer = ProposalLayer(
    pre_nms_top_n=3000,
    post_nms_top_n=1000,
    nms_thresh=0.7,
    min_size=8
)
```

**Problems:**
- Very high number of proposals (1000) for small 128x128 images
- NMS threshold 0.7 might be too high
- Min size 8 pixels might be too small

## Recommended Debugging Steps

1. **Add loss component logging:**
```python
print(f"RPN cls: {rpn_cls:.4f}, RPN reg: {rpn_reg:.4f}, Det cls: {det_cls:.4f}, Det reg: {det_reg:.4f}")
```

2. **Check gradient norms:**
```python
total_norm = torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
print(f"Gradient norm: {total_norm:.4f}")
```

3. **Validate anchor-target matching:**
- Ensure positive/negative anchor ratios are reasonable
- Check if any positive anchors are being generated

4. **Monitor proposal quality:**
- Log number of proposals generated
- Check proposal-ground truth IoU distributions

5. **Verify data loading:**
- Ensure targets are in correct format and coordinate system
- Check for NaN/inf values in inputs

## Priority Fixes

1. **Fix detection head bbox regression** (class-agnostic)
2. **Improve anchor configuration** for small objects
3. **Add loss component weighting and logging**
4. **Lower learning rate** and consider Adam optimizer
5. **Move all forward passes under autocast()**
6. **Implement gradual backbone unfreezing**

These changes should significantly improve training stability and reduce high loss issues.