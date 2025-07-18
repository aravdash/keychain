import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import roi_align
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet50

from utils.quickdraw_iterable_dataset import QuickDrawIterableDataset
from model.backbone import ResNetBackbone
from model.rpn import RPNHead
from model.detection_head import DetectionHead

from utils.anchors import AnchorGenerator
from utils.rpn_utils import reshape_rpn_outputs, decode_boxes
from utils.proposals import ProposalLayer
from utils.rpn_loss import rpn_loss
from utils.detection_utils import detection_loss

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def expand_detection_head(old_ckpt: dict, old_num: int, c4_ch: int, new_num: int, device: torch.device) -> DetectionHead:
    """
    Build a new DetectionHead for `new_num` classes,
    copying weights from the first `old_num` classes.
    """
    new_head = DetectionHead(in_channels=c4_ch, num_classes=new_num).to(device)
    old_sd = old_ckpt['head']

    # Copy classification weights & bias
    w_cls = old_sd['cls_score.weight']
    b_cls = old_sd['cls_score.bias']
    new_head.cls_score.weight.data[:old_num+1] = w_cls
    new_head.cls_score.bias.data[:old_num+1]   = b_cls

    # Copy bbox regression weights & bias (now class-agnostic)
    if 'bbox_pred.weight' in old_sd:
        w_bbox = old_sd['bbox_pred.weight']
        b_bbox = old_sd['bbox_pred.bias']
        # Only copy the first 4 outputs (class-agnostic)
        new_head.bbox_pred.weight.data[:4] = w_bbox[:4]
        new_head.bbox_pred.bias.data[:4]   = b_bbox[:4]

    return new_head

def get_learning_rate_schedule(phase, base_lr=5e-5):
    """Get phase-appropriate learning rate"""
    if phase == 1:
        return base_lr * 2  # Slightly higher for initial training
    else:
        return base_lr  # Lower for fine-tuning

def gradual_unfreeze_backbone(backbone, epoch, phase):
    """Gradually unfreeze backbone layers"""
    if phase == 1:
        # Phase 1: Unfreeze more aggressively
        if epoch <= 5:
            # First 5 epochs: only layer4
            for name, p in backbone.named_parameters():
                p.requires_grad = "layer4" in name
        elif epoch <= 10:
            # Next 5 epochs: layer3 and layer4
            for name, p in backbone.named_parameters():
                p.requires_grad = any(stage in name for stage in ["layer3", "layer4"])
        else:
            # After epoch 10: layer2, layer3, layer4
            for name, p in backbone.named_parameters():
                p.requires_grad = any(stage in name for stage in ["layer2", "layer3", "layer4"])
    else:
        # Later phases: unfreeze more layers from start
        for name, p in backbone.named_parameters():
            p.requires_grad = any(stage in name for stage in ["layer1", "layer2", "layer3", "layer4"])

def main():
    # --------- Phase config ----------
    PHASE = int(os.getenv('TRAIN_PHASE', '1'))

    if PHASE == 1:
        data_dir = "data/keychain_50_phase1"   # 50 classes
        sample_per_class = 80
        resume_ckpt = None
        chkpt_name = "phase1.pth"
    elif PHASE == 2:
        data_dir = "data/keychain_50_phase2"   # 100 classes
        sample_per_class = 30
        resume_ckpt = "checkpoints/phase1.pth"
        chkpt_name = "phase2.pth"
    elif PHASE == 3:
        data_dir = "data/keychain_50_phase3"   # 150 classes
        sample_per_class = 20
        resume_ckpt = "checkpoints/phase2.pth"
        chkpt_name = "phase3.pth"
    elif PHASE == 4:
        data_dir = "data/keychain_50_phase4"   # up to 375 classes
        sample_per_class = 15
        resume_ckpt = "checkpoints/phase3.pth"
        chkpt_name = "phase4.pth"
    else:
        raise ValueError(f"Unsupported TRAIN_PHASE={PHASE}. Must be 1-4.")

    chkpt_dir = "checkpoints"
    os.makedirs(chkpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, TRAIN_PHASE={PHASE}")

    # Improved hyperparameters
    num_epochs    = 25  # Increased for better convergence
    batch_size    = 1
    lr            = get_learning_rate_schedule(PHASE)
    weight_decay  = 1e-4  # Reduced weight decay
    step_size     = 7     # Longer step size
    gamma         = 0.5   # Less aggressive decay
    max_grad_norm = 1.0   # Reduced gradient clipping

    print(f"Learning rate: {lr:.1e}")

    # Data transforms with improved augmentations
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
        transforms.RandomAffine(degrees=10, translate=(0.05,0.05)),  # Reduced augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Dataset and DataLoader
    train_ds = QuickDrawIterableDataset(
        ndjson_dir=data_dir,
        split="train",
        split_ratio=(0.7,0.2,0.1),
        sample_per_class=sample_per_class,
        img_size=128,
        transform=transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Reduced for stability
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Set seed and load pretrained ResNet50
    torch.manual_seed(42)  # Changed seed for reproducibility
    pretrained = resnet50(pretrained=True)

    # Build custom backbone and load pretrained weights
    backbone = ResNetBackbone([3,4,6,3]).to(device)
    backbone.conv1.load_state_dict(pretrained.conv1.state_dict())
    backbone.bn1.load_state_dict(pretrained.bn1.state_dict())
    backbone.layer1.load_state_dict(pretrained.layer1.state_dict())
    backbone.layer2.load_state_dict(pretrained.layer2.state_dict())
    backbone.layer3.load_state_dict(pretrained.layer3.state_dict())
    backbone.layer4.load_state_dict(pretrained.layer4.state_dict())

    # Initial backbone freezing (will be gradually unfrozen)
    for p in backbone.parameters():
        p.requires_grad = False

    # Warm up dummy forward to get c4 channels
    with torch.no_grad():
        dummy = torch.zeros(1,3,128,128, device=device)
        c4_ch = backbone(dummy)["c4"].shape[1]

    # Improved RPN head configuration
    scales = [0.25, 0.5, 1.0]  # Smaller scales for Quick Draw objects
    ratios = [0.5, 1.0, 2.0]   # Reduced ratios for simplicity
    num_anchors = len(scales) * len(ratios)
    rpn_head = RPNHead(in_channels=c4_ch, num_anchors=num_anchors).to(device)

    # Improved proposal layer configuration
    proposal_layer = ProposalLayer(
        pre_nms_top_n=2000,   # Reduced
        post_nms_top_n=512,   # Reduced for 128x128 images
        nms_thresh=0.5,       # Lower threshold
        min_size=4            # Smaller min size
    )

    # Detection head init or expand
    ndjson_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".ndjson"))
    new_num = len(ndjson_files)
    scaler = GradScaler()

    print(f"Training with {new_num} classes")

    # Build or expand detection head & warm-start RPN if resuming
    if resume_ckpt is None:
        det_head = DetectionHead(in_channels=c4_ch, num_classes=new_num).to(device)
    else:
        print(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        old_num = ckpt['head']['cls_score.bias'].numel() - 1
        print(f"Expanding from {old_num} to {new_num} classes")
        det_head = expand_detection_head(ckpt, old_num, c4_ch, new_num, device)
        
        # Load RPN weights if available
        if 'rpn' in ckpt:
            rpn_state = ckpt['rpn']
            # Only load weights that match current architecture
            rpn_dict = rpn_head.state_dict()
            filtered_state = {k: v for k, v in rpn_state.items() 
                            if k in rpn_dict and v.shape == rpn_dict[k].shape}
            rpn_head.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)} RPN parameters")

    det_head = det_head.to(device)

    # Improved anchor configuration
    anchor_generator = AnchorGenerator(
        base_size=8,      # Match stride
        scales=scales, 
        ratios=ratios, 
        stride=8
    )

    # Assemble optimizer params with different learning rates
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    rpn_params = list(rpn_head.parameters())
    det_params = list(det_head.parameters())

    # Use AdamW optimizer with parameter groups
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': rpn_params, 'lr': lr},
        {'params': det_params, 'lr': lr}
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Loss weights
    loss_weights = {
        'rpn_cls': 1.0,
        'rpn_reg': 1.0,
        'det_cls': 2.0,  # Higher weight for detection classification
        'det_reg': 1.0
    }

    # Training loop
    print("Starting training...")
    for epoch in range(1, num_epochs+1):
        # Gradual backbone unfreezing
        gradual_unfreeze_backbone(backbone, epoch, PHASE)
        
        # Update optimizer param groups after unfreezing
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        optimizer.param_groups[0]['params'] = backbone_params

        backbone.train()
        rpn_head.train()
        det_head.train()
        
        total_loss = 0.0
        total_rpn_cls = 0.0
        total_rpn_reg = 0.0
        total_det_cls = 0.0
        total_det_reg = 0.0
        iters = 0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Check for NaN/inf in inputs
                if torch.isnan(torch.stack(images)).any() or torch.isinf(torch.stack(images)).any():
                    print(f"Warning: NaN/inf detected in input images at batch {batch_idx}")
                    continue

                optimizer.zero_grad()

                # All forward passes under autocast for consistency
                with autocast():
                    # 1) Backbone forward
                    feats = backbone(torch.stack(images))
                    c4 = feats['c4']

                    # 2) RPN forward + loss
                    logits, deltas = rpn_head(c4)
                    cls_logits, bbox_deltas = reshape_rpn_outputs(logits, deltas, num_anchors)
                    anchors = anchor_generator.grid_anchors(c4.shape[-2:]).to(device)
                    rpn_cls, rpn_reg = rpn_loss(cls_logits, bbox_deltas, anchors, targets[0]['boxes'])

                    # 3) Proposals
                    scores = cls_logits.softmax(-1)[0,:,1]
                    raw_props = decode_boxes(anchors, bbox_deltas[0])
                    proposals = proposal_layer(anchors, scores, raw_props, images[0].shape[-2:])

                    # Skip if no proposals
                    if proposals.numel() == 0:
                        continue

                    # 4) ROI + detection head
                    pooled = roi_align(
                        input=c4, 
                        boxes=[proposals], 
                        output_size=(7,7), 
                        spatial_scale=1/8, 
                        sampling_ratio=4  # Increased sampling ratio
                    )
                    
                    cls_det, reg_det = det_head(pooled)
                    det_cls, det_reg = detection_loss(
                        cls_det, reg_det, proposals,
                        targets[0]['boxes'], targets[0]['labels'], new_num
                    )

                    # Weighted loss combination
                    loss = (
                        loss_weights['rpn_cls'] * rpn_cls + 
                        loss_weights['rpn_reg'] * rpn_reg + 
                        loss_weights['det_cls'] * det_cls + 
                        loss_weights['det_reg'] * det_reg
                    )

                    # Check for NaN/inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/inf loss detected at batch {batch_idx}")
                        continue

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(rpn_head.parameters()) + list(det_head.parameters()),
                    max_grad_norm
                )
                
                scaler.step(optimizer)
                scaler.update()

                # Accumulate losses for logging
                total_loss += loss.item()
                total_rpn_cls += rpn_cls.item()
                total_rpn_reg += rpn_reg.item()
                total_det_cls += det_cls.item()
                total_det_reg += det_reg.item()
                iters += 1

                # Log every 100 batches
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f}, GradNorm={total_norm:.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if iters == 0:
            print(f"Warning: No valid batches in epoch {epoch}")
            continue

        scheduler.step()
        epoch_time = (time.time() - start_time) / 60
        avg_loss = total_loss / iters
        avg_rpn_cls = total_rpn_cls / iters
        avg_rpn_reg = total_rpn_reg / iters
        avg_det_cls = total_det_cls / iters
        avg_det_reg = total_det_reg / iters

        print(f"[Phase {PHASE}] [Epoch {epoch}/{num_epochs}]")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  RPN cls: {avg_rpn_cls:.4f}, RPN reg: {avg_rpn_reg:.4f}")
        print(f"  Det cls: {avg_det_cls:.4f}, Det reg: {avg_det_reg:.4f}")
        print(f"  Time: {epoch_time:.1f} min, LR: {optimizer.param_groups[0]['lr']:.1e}")
        print(f"  Trainable params: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'phase': PHASE,
            'rpn': rpn_head.state_dict(),
            'head': det_head.state_dict(),
            'backbone': backbone.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'loss': avg_loss,
            'num_classes': new_num
        }
        
        torch.save(checkpoint, os.path.join(chkpt_dir, chkpt_name))
        
        # Save best model
        if epoch == 1 or avg_loss < getattr(main, 'best_loss', float('inf')):
            main.best_loss = avg_loss
            torch.save(checkpoint, os.path.join(chkpt_dir, f"best_{chkpt_name}"))
            print(f"  New best model saved!")

    print(f"Phase {PHASE} training complete!")
    print(f"Best loss: {getattr(main, 'best_loss', 'N/A'):.4f}")

if __name__ == "__main__":
    main()