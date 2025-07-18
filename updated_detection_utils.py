# utils/detection_utils.py

import torch
import torch.nn.functional as F
from utils.box_utils import box_iou, encode_boxes

def match_proposals_to_gt(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assigns each of N proposals a classification label and regression target.
    Args:
      proposals: [N,4] float Tensor (x1,y1,x2,y2)
      gt_boxes:  [M,4] float Tensor
      gt_labels: [M]   int64 Tensor in {1..num_classes}
    Returns:
      labels:       [N] int64 Tensor with values:
                      -1 = ignore,
                       0 = background (neg),
                      1..K = object class
      bbox_targets: [N,4] float Tensor of (dx,dy,dw,dh) for positives, zeros elsewhere
    """
    N = proposals.size(0)
    M = gt_boxes.size(0)

    # 1) IoU matrix [N,M]
    ious = box_iou(proposals, gt_boxes)

    # 2) Best GT for each proposal
    max_iou, argmax_iou = ious.max(dim=1)  # [N]

    # 3) For each GT, ensure its best proposal is marked positive
    max_iou_per_gt, _ = ious.max(dim=0)    # [M]
    # anchors (proposal indices) that hit each gt
    gt_best_inds = (ious == max_iou_per_gt).nonzero()[:,0]

    # 4) Initialize labels to -1 (ignore)
    labels = proposals.new_full((N,), -1, dtype=torch.int64)

    # 5) Negative: IoU <= neg threshold
    labels[max_iou <= neg_iou_thresh] = 0

    # 6) Positive: IoU >= pos threshold
    labels[max_iou >= pos_iou_thresh] = gt_labels[argmax_iou[max_iou >= pos_iou_thresh]]
    # and force the best for each GT
    labels[gt_best_inds] = gt_labels[argmax_iou[gt_best_inds]]

    # 7) Compute bbox regression targets for positives
    bbox_targets = proposals.new_zeros((N,4))
    pos_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
    if pos_inds.numel() > 0:
        matched_gt = gt_boxes[argmax_iou[pos_inds]]   # [num_pos,4]
        bbox_targets[pos_inds] = encode_boxes(
            proposals[pos_inds], matched_gt
        )

    return labels, bbox_targets


def detection_loss(
    class_logits: torch.Tensor,  # [N, num_classes+1]
    bbox_preds:    torch.Tensor, # [N, 4] - CLASS-AGNOSTIC (CHANGED!)
    proposals:     torch.Tensor, # [N,4]
    gt_boxes:      torch.Tensor, # [M,4]
    gt_labels:     torch.Tensor, # [M]
    num_classes:   int,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes ROI‐Head classification + bbox‐regression losses.
    Updated for CLASS-AGNOSTIC bbox regression.
    Returns (cls_loss, reg_loss).
    """
    device = class_logits.device

    # 1) Match proposals → (labels, bbox_targets)
    labels, bbox_targets = match_proposals_to_gt(
        proposals, gt_boxes, gt_labels, pos_iou_thresh, neg_iou_thresh
    )
    labels = labels.to(device)
    bbox_targets = bbox_targets.to(device)

    # 2) Classification loss (ignore_label = -1)
    cls_loss = F.cross_entropy(
        class_logits,
        labels,
        ignore_index=-1
    )

    # 3) Regression loss: CLASS-AGNOSTIC - only for positive proposals
    #    bbox_preds is now [N, 4] instead of [N, 4*(num_classes+1)]
    pos_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
    if pos_inds.numel() > 0:
        # Direct use of bbox predictions (no need to select by class)
        pred_deltas = bbox_preds[pos_inds]      # [num_pos, 4]
        target_deltas = bbox_targets[pos_inds]  # [num_pos, 4]
        reg_loss = F.smooth_l1_loss(pred_deltas, target_deltas, beta=1.0)
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return cls_loss, reg_loss


# OPTIONAL: Keep the old function for backward compatibility
def detection_loss_class_specific(
    class_logits: torch.Tensor,  # [N, num_classes+1]
    bbox_preds:    torch.Tensor, # [N, 4*(num_classes+1)]
    proposals:     torch.Tensor, # [N,4]
    gt_boxes:      torch.Tensor, # [M,4]
    gt_labels:     torch.Tensor, # [M]
    num_classes:   int,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Original implementation for class-specific bbox regression.
    Use this if you want to keep the old DetectionHead.
    """
    device = class_logits.device

    # 1) Match proposals → (labels, bbox_targets)
    labels, bbox_targets = match_proposals_to_gt(
        proposals, gt_boxes, gt_labels, pos_iou_thresh, neg_iou_thresh
    )
    labels = labels.to(device)
    bbox_targets = bbox_targets.to(device)

    # 2) Classification loss (ignore_label = -1)
    cls_loss = F.cross_entropy(
        class_logits,
        labels,
        ignore_index=-1
    )

    # 3) Regression loss: only for positive proposals
    #    First reshape bbox_preds to [N, num_classes+1, 4]
    N = bbox_preds.size(0)
    preds = bbox_preds.view(N, num_classes+1, 4)

    pos_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
    if pos_inds.numel() > 0:
        # select the predicted deltas for the true class of each positive
        # gather per-row: preds[pos_inds, labels[pos_inds], :]
        pred_deltas = preds[pos_inds, labels[pos_inds]]
        target_deltas = bbox_targets[pos_inds]
        reg_loss = F.smooth_l1_loss(pred_deltas, target_deltas, beta=1.0)
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return cls_loss, reg_loss