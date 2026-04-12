import torch
import torch.nn as nn
import torch.nn.functional as F

class HSPNetLoss(nn.Module):
    """Composite loss for HSPNet, including WCE, Topology, and Offset losses."""
    def __init__(self, pos_weight=50.0, topology_weight=0.1, offset_weight=1.0):
        super(HSPNetLoss, self).__init__()
        # Weighted Cross-Entropy for Stage 1 & 2
        self.wce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
        
        # MSE Loss for Stage 3 Offset Regression
        self.offset_loss = nn.MSELoss(reduction='none') # Use 'none' to manually mask
        
        self.topology_weight = topology_weight
        self.offset_weight = offset_weight

    def forward(self, stage1_logits, stage2_logits, stage3_offsets, labels, features):
        """
        Args:
            stage1_logits, stage2_logits: Model classification outputs.
            stage3_offsets: Predicted XYZ offsets.
            labels: Ground truth labels (0 or 1).
            features: The original 10-dim input features, to get Theta.
        """
        # --- 1. Weighted Cross-Entropy Loss --- #
        loss_s1 = self.wce_loss(stage1_logits, labels)
        loss_s2 = self.wce_loss(stage2_logits, labels)

        # --- 2. Topology Loss (Theta Continuity) --- #
        theta = features[:, 9, :] # Extract Theta dimension (B, N)
        preds_s2 = F.softmax(stage2_logits, dim=1)[:, 1, :] # Get probability of being a boundary point
        
        # Get indices of points predicted as boundary
        boundary_preds = (preds_s2 > 0.5).float()
        
        # Sort theta values for predicted boundary points
        sorted_theta, sort_indices = torch.sort(theta * boundary_preds, dim=1)
        
        # Calculate the difference between adjacent sorted theta values
        theta_diff = sorted_theta[:, 1:] - sorted_theta[:, :-1]
        
        # Penalize large gaps. We only care about gaps between valid points.
        valid_points_mask = (sorted_theta[:, :-1] > 0) & (sorted_theta[:, 1:] > 0)
        topology_loss = torch.mean(theta_diff[valid_points_mask] ** 2)
        if torch.isnan(topology_loss):
            topology_loss = 0.0 # Handle cases with no predicted boundary points

        # --- 3. Offset Regression Loss --- #
        # Only calculate loss for ground truth boundary points (label == 1)
        gt_mask = labels.unsqueeze(1).expand_as(stage3_offsets).bool() # (B, 3, N)
        
        # Target is zero vector, as we want points to move to the true edge
        gt_offsets = torch.zeros_like(stage3_offsets)
        
        # Calculate MSE loss for all points
        raw_offset_loss = self.offset_loss(stage3_offsets, gt_offsets)
        
        # Apply mask and calculate the mean over the valid points
        masked_offset_loss = torch.sum(raw_offset_loss[gt_mask]) / (torch.sum(gt_mask) + 1e-8)

        # --- 4. Composite Loss --- #
        total_loss = (loss_s1 + loss_s2) + (self.topology_weight * topology_loss) + (self.offset_weight * masked_offset_loss)
        
        return total_loss, loss_s1, loss_s2, topology_loss, masked_offset_loss
