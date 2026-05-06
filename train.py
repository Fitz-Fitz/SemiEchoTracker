import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import albumentations as A
import numpy as np
import os
import random
import logging
import argparse
from tqdm import tqdm
import shutil

from models.SemiEchoTracker import SemiEchoTracker
from models.dinov2 import get_dinov2_model
from dataset.echo_dataset import EchoDataset


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description='SemiEchoTracker')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Path to the dataset')
    parser.add_argument('--data_type', type=str, default='PLAX', help='Data type')
    parser.add_argument('--image_size', type=tuple, default=(224, 224), help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--frame_num', type=int, default=10, help='Number of frames')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--tracker_lr_multiplier', type=float, default=1.0,
                        help='Learning-rate multiplier for tracker-only parameters')
    parser.add_argument('--detector_lr_multiplier', type=float, default=1.0,
                        help='Learning-rate multiplier for detector/shared parameters')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # loss weight
    parser.add_argument('--velocity', type=float, default=0.1, help='Weight of velocity loss')
    parser.add_argument('--tracking', type=float, default=0.1, help='Weight of tracking loss')
    parser.add_argument('--tracking_supervised', type=float, default=1.0, help='Weight of supervised tracker endpoint loss')
    parser.add_argument('--mutual_consistency', type=float, default=0.1, help='Weight of mutual consistency loss')
    parser.add_argument('--endpoint_supervision_weight', type=float, default=1.0,
                        help='Weight for detector endpoint supervision in the total loss')
    parser.add_argument('--agent_num', type=int, default=32, help='number of agent tokens')
    parser.add_argument('--attention_layers', type=int, default=3,
                        help='Number of trainable EchoAgentAttention blocks after frozen DINO')
    parser.add_argument('--feature_branch_mode', type=str, default='shared',
                        choices=['shared', 'split_attention'],
                        help='shared uses one attention branch; split_attention gives detector/tracker separate attention branches')
    parser.add_argument('--coord_head', type=str, default='conv_mlp_keypoint_cnn_gcn_noxy',
                        choices=['pooled_gcn', 'fullres_gcn', 'query_attn',
                                 'conv_mlp', 'conv_mlp_gcn', 'conv_mlp_spatial_gcn',
                                 'conv_mlp_keypoint_gcn', 'conv_mlp_keypoint_cnn_gcn',
                                 'conv_mlp_keypoint_cnn_gcn_noxy'],
                        help='Direct coordinate head; fullres_gcn keeps the native 16x16 feature grid')
    parser.add_argument('--graph_mode', type=str, default='split', choices=['split', 'shared'],
                        help='Use separate detector/tracker landmark graphs or share one graph')
    parser.add_argument('--training_mode', type=str, default='semi', choices=['semi', 'detector_only'],
                        help='semi uses detector+tracker losses; detector_only trains only the frame-wise detector on endpoints')
    parser.add_argument('--mode', type=str, default=None, choices=['semi', 'detector_only'],
                        help='Alias for --training_mode, useful for ablation scripts.')
    parser.add_argument('--detector_only', action='store_true',
                        help='Alias for --training_mode detector_only')
    parser.add_argument('--aug_mode', type=str, default='standard', choices=['standard', 'weak', 'none'],
                        help='Training augmentation strength')
    parser.add_argument('--rotation_limit', type=float, default=30.0,
                        help='Clip-level rotation limit in degrees for training')
    parser.add_argument('--rotation_prob', type=float, default=0.5,
                        help='Clip-level rotation probability for training')
    
    # training settings
    parser.add_argument('--warmup_epochs', type=int, default=30, help='Number of warmup epochs')
    parser.add_argument('--cons_epoch', type=int, default=0, help='Epoch to start consistency loss between detector and tracker')
    parser.add_argument('--consistency_ramp_epochs', type=int, default=0,
                        help='Linearly ramp detector-tracker consistency weight over this many epochs after --cons_epoch')
    parser.add_argument('--detach_tracker_teacher', action='store_true',
                        help='Detach tracker coordinates in detector-tracker consistency so tracker teaches detector')
    parser.add_argument('--detach_tracker_inputs', action='store_true',
                        help='Detach shared features, detector seeds, and shared adjacency before the tracker branch')
    parser.add_argument('--consistency_gate', type=str, default='none', choices=['none', 'fb'],
                        help='Mask detector-tracker consistency by tracker reliability')
    parser.add_argument('--consistency_fb_threshold', type=float, default=8.0,
                        help='Pixel threshold for forward/backward tracker agreement when --consistency_gate fb')
    parser.add_argument('--consistency_mask_normalization', type=str, default='selected',
                        choices=['selected', 'total'],
                        help='Normalize masked consistency by selected points or by all candidate points')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short results directory name for long ablation configs')
    parser.add_argument('--best_metric', type=str, default=None,
                        choices=['err_detection_endpoint', 'err_init', 'full_sequence_tracking', 'err_tracking'],
                        help='Optional validation metric used to save best_model.pth')
    parser.add_argument('--num_workers', type=int, default=6, help='DataLoader worker processes')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Accumulate gradients over this many mini-batches before each optimizer step')
    args = parser.parse_args()
    if args.detector_only:
        args.training_mode = 'detector_only'
    return args


def process_features(dinov2_model, inputs, B, T, H, W):
    """Extract DINOv2 patch tokens + CLS token.

    Args:
        dinov2_model: frozen DINOv2 backbone
        inputs: [B*T, C, H, W] image tensor
    Returns:
        extra_feature: [B, T, C_dino, H//14, W//14]
        cls_token:     [B, T, C_dino]
    """
    with torch.no_grad():
        extra_feature_dict = dinov2_model.forward_features(inputs)
        extra_feature = extra_feature_dict['x_norm_patchtokens']
        cls_token = extra_feature_dict['x_norm_clstoken']           # [B*T, C]
        cls_token = cls_token.reshape(B, T, -1)

        extra_feature = extra_feature.reshape(B * T, H // 14, W // 14, -1)
        extra_feature = extra_feature.permute(0, 3, 1, 2).contiguous()   # [B*T, C, H//14, W//14]
        extra_feature = extra_feature.reshape(B, T, -1, H // 14, W // 14)
    return extra_feature, cls_token


def soft_wing_loss(pred, target, omega=5.0, epsilon=2.0):
    """Soft Wing Loss implementation with numerical stability improvements"""
    diff = torch.abs(pred - target)
    # Add gradient clipping to prevent explosion
    diff = torch.clamp(diff, max=100.0)
    # Add small epsilon to prevent log(0)
    delta = omega * (1 - torch.log(1 + diff/(epsilon + 1e-8)))
    loss = torch.where(diff < omega, delta, diff - omega/2)
    return torch.mean(torch.clamp(loss, min=-100.0, max=100.0))


def coordinate_loss(pred, target):
    return soft_wing_loss(pred, target)


def coordinate_loss_elementwise(pred, target):
    diff = torch.clamp(torch.abs(pred - target), max=100.0)
    delta = 5.0 * (1 - torch.log(1 + diff / (2.0 + 1e-8)))
    return torch.clamp(torch.where(diff < 5.0, delta, diff - 2.5), min=-100.0, max=100.0)


def masked_coordinate_loss(pred, target, point_weights, args):
    if point_weights is None:
        return coordinate_loss(pred, target)
    weights = point_weights.to(device=pred.device, dtype=pred.dtype)
    while weights.dim() < pred.dim():
        weights = weights.unsqueeze(-1)
    elem_loss = coordinate_loss_elementwise(pred, target)
    if args.consistency_mask_normalization == 'total':
        denom = torch.as_tensor(
            weights.numel() * pred.shape[-1],
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        denom = weights.sum() * pred.shape[-1]
    if denom <= 0:
        return pred.sum() * 0.0
    return (elem_loss * weights).sum() / torch.clamp(denom, min=1.0)


def tracker_agreement_weights(forward_coords, backward_coords, args):
    if args.consistency_gate == 'none':
        return None
    agreement = torch.norm(forward_coords.detach() - backward_coords.detach(), dim=-1)
    if args.consistency_gate == 'fb':
        return (agreement <= args.consistency_fb_threshold).float()
    raise ValueError(f'Unknown consistency_gate: {args.consistency_gate}')


def migrate_split_graph_state_dict(state_dict):
    """Map older checkpoints with one shared graph to split graph params."""
    state_dict = dict(state_dict)
    old_key = 'sttgnn.adj'
    detector_key = 'sttgnn.detector_adj'
    tracker_key = 'sttgnn.tracker_adj'
    if old_key in state_dict:
        old_adj = state_dict.pop(old_key)
        state_dict.setdefault(detector_key, old_adj.clone())
        state_dict.setdefault(tracker_key, old_adj.clone())
    return state_dict


def is_tracker_parameter(name):
    tracker_prefixes = (
        'tracker_feature_module.',
        'tracker_attention_modules.',
        'sttgnn.gc1.',
        'sttgnn.gc2.',
        'sttgnn.gc3.',
        'sttgnn.tracker_adj',
        'sttgnn.pred_head.',
        'sttgnn.gru_projector.',
        'sttgnn.gru_forward.',
        'sttgnn.gru_backward.',
        'sttgnn.gru_deprojector.',
    )
    return name.startswith(tracker_prefixes)



def train_or_validate(model, dinov2_model, data_loader, optimizer, device, epoch, 
                     writer, scaler, args, is_training=True):
    """Training/validation loop
    
    Args:
        model: Main model
        dinov2_model: Feature extraction model
        data_loader: Data loader
        optimizer: Optimizer
        device: Computing device
        epoch: Current epoch
        writer: Tensorboard writer
        scaler: Gradient scaler for mixed precision
        args: Training arguments
        is_training: Whether in training mode
        
    Returns:
        Dictionary of metrics
    """
    if is_training:
        model.train()
    else:
        model.eval()
    dinov2_model.eval()
    error_init_landmark_all, error_tracking_landmark_all, error_init_all = [], [], []
    error_detection_endpoint_all, error_full_sequence_tracking_all = [], []
    total_loss = 0
    total_loss_landmarks = 0
    total_loss_consistency = 0
    total_loss_tracking = 0
    total_loss_velocity = 0
    total_loss_tracking_supervised = 0
    total_consistency_mask_fraction = 0.0
    consistency_mask_count = 0
    desc = 'Training' if is_training else 'Validating'
    pbar = tqdm(data_loader, desc=f'{desc} Epoch {epoch}')
    spacing = 0.6 if args.data_type == 'PLAX' else 0.67
    detector_only = args.training_mode == 'detector_only' or not is_training
    grad_accum_steps = max(1, args.grad_accum_steps) if is_training else 1
    
    for batch_idx, batch_data in enumerate(pbar):
        loss_mutual_consistency = torch.tensor(0.0, device=device)
        loss_tracking_supervised = torch.tensor(0.0, device=device)
        loss_velocity = torch.tensor(0.0, device=device)
        loss_tracking = torch.tensor(0.0, device=device)
        consistency_mask_fraction = None
        if is_training and batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()
        # Get batch data
        inputs = batch_data['video'].to(device).float()
        landmarks = batch_data['landmarks'].to(device).float()
        rescale_x = batch_data['rescale_x'].cpu().numpy() # B
        rescale_y = batch_data['rescale_y'].cpu().numpy() # B
        B, T, c, H, W = inputs.shape
        inputs = inputs.view(B*T, c, H, W)
        # Extract features
        extra_feature, cls_token = process_features(dinov2_model, inputs, B, T, H, W)
        del inputs
        # Forward pass
        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast('cuda') if is_training else torch.no_grad():
                if detector_only:
                    spatial_init = model(extra_feature, cls_token, infer_mode=True)
                    pred_coords_forward = None
                    pred_coords_backward = None
                else:
                    # Get detector and bidirectional tracker predictions.
                    # During training seed the tracker from GT endpoints for a clean
                    # supervision signal; during validation use detector predictions.
                    if is_training:
                        tracker_seed_coords = torch.stack(
                            [landmarks[:, 0], landmarks[:, -1]], dim=1
                        ).clone()
                        tracker_seed_coords[..., 0] = tracker_seed_coords[..., 0].clamp(0, args.image_size[1] - 1) / (args.image_size[1] - 1)
                        tracker_seed_coords[..., 1] = tracker_seed_coords[..., 1].clamp(0, args.image_size[0] - 1) / (args.image_size[0] - 1)
                    else:
                        tracker_seed_coords = None
                    pred_coords_forward, pred_coords_backward, pred_offset_forward, pred_offset_backward, spatial_init = model(
                        extra_feature,
                        cls_token,
                        tracker_seed_coords=tracker_seed_coords,
                        detach_tracker_inputs=args.detach_tracker_inputs,
                    )

                # Supervised loss for the detector at the labelled endpoint frames.
                loss_landmarks_init = coordinate_loss(spatial_init[:, 0], landmarks[:, 0]) + \
                                      coordinate_loss(spatial_init[:, -1], landmarks[:, -1])
                loss = loss_landmarks_init * args.endpoint_supervision_weight

                if not detector_only:
                    loss_tracking_supervised = coordinate_loss(pred_coords_forward[:, -1], landmarks[:, -1]) + \
                                               coordinate_loss(pred_coords_backward[:, 0], landmarks[:, 0])
                    loss_velocity = coordinate_loss(
                        torch.norm(pred_offset_forward, dim=-1),
                        torch.norm(pred_offset_backward, dim=-1),
                    )
                    loss_tracking = coordinate_loss(pred_coords_forward, pred_coords_backward)

                    invalid_loss = False
                    for loss_name, loss_value in (
                        ('loss_landmarks_init', loss_landmarks_init),
                        ('loss_tracking_supervised', loss_tracking_supervised),
                        ('loss_velocity', loss_velocity),
                        ('loss_tracking', loss_tracking),
                    ):
                        if torch.isnan(loss_value) or torch.isinf(loss_value):
                            logging.warning(f"NaN/Inf detected in {loss_name} at epoch {epoch}")
                            invalid_loss = True
                    if invalid_loss:
                        continue

                    loss = loss + loss_tracking_supervised * args.tracking_supervised
                    loss = loss + loss_velocity * args.velocity
                    loss = loss + loss_tracking * args.tracking

                    # Detector-tracker co-training on unlabeled middle frames.
                    if epoch >= args.cons_epoch:
                        tracker_forward_teacher = pred_coords_forward[:, 1:-1]
                        tracker_backward_teacher = pred_coords_backward[:, 1:-1]
                        detector_student = spatial_init[:, 1:-1]
                        if args.detach_tracker_teacher:
                            tracker_forward_teacher = tracker_forward_teacher.detach()
                            tracker_backward_teacher = tracker_backward_teacher.detach()
                        if tracker_forward_teacher.numel() > 0:
                            consistency_weights = tracker_agreement_weights(
                                tracker_forward_teacher,
                                tracker_backward_teacher,
                                args,
                            )
                            if consistency_weights is not None:
                                consistency_mask_fraction = consistency_weights.float().mean()
                            tracker_teacher = 0.5 * (tracker_forward_teacher + tracker_backward_teacher)
                            loss_mutual_consistency = masked_coordinate_loss(
                                tracker_teacher,
                                detector_student,
                                consistency_weights,
                                args,
                            )
                            consistency_weight_scale = 1.0
                            if args.consistency_ramp_epochs > 0:
                                ramp_progress = (epoch - args.cons_epoch + 1) / max(1, args.consistency_ramp_epochs)
                                consistency_weight_scale = max(0.0, min(1.0, ramp_progress))
                            if not torch.isnan(loss_mutual_consistency) and not torch.isinf(loss_mutual_consistency):
                                loss += loss_mutual_consistency * args.mutual_consistency * consistency_weight_scale
                else:
                    if torch.isnan(loss_landmarks_init) or torch.isinf(loss_landmarks_init):
                        logging.warning(f"NaN/Inf detected in loss_landmarks_init at epoch {epoch}")
                        continue

        # Backward pass
        if is_training and not torch.isnan(loss):
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(data_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        
        # Calculate prediction errors in millimetres (same metric for train & val).
        # For training we only have ground-truth at the first/last frame; for
        # validation every frame is annotated. Both branches produce errors in mm.
        if is_training:
            frame_indices = [0, -1]
        else:
            frame_indices = list(range(landmarks.shape[1]))

        rx = float(rescale_x[0])
        ry = float(rescale_y[0])

        # Vectorised error over all frames / landmarks for sample 0 in the batch.
        target_xy = landmarks[0, frame_indices]                              # [F, N, 2]
        scale = torch.tensor([rx, ry], device=target_xy.device, dtype=target_xy.dtype)

        def _pixel_err_mm_pair(pred_xy, target_xy):
            diff = (pred_xy - target_xy) * scale                              # [F, N, 2] in original-image pixels
            return (diff.pow(2).sum(dim=-1).sqrt() * spacing).detach().cpu().numpy()  # mm

        def _pixel_err_mm(pred):
            return _pixel_err_mm_pair(pred[0, frame_indices], target_xy)

        err_init_matrix = _pixel_err_mm(spatial_init)               # [F, N]
        endpoint_pred_indices = [0, -1] if spatial_init.shape[1] > 1 else [0]
        endpoint_target_indices = [0, -1] if landmarks.shape[1] > 1 else [0]
        err_detection_endpoint_matrix = _pixel_err_mm_pair(
            spatial_init[0, endpoint_pred_indices],
            landmarks[0, endpoint_target_indices],
        )
        if detector_only:
            err_fwd_matrix = err_init_matrix
            err_bwd_matrix = err_init_matrix
        else:
            err_fwd_matrix = _pixel_err_mm(pred_coords_forward)         # [F, N]
            err_bwd_matrix = _pixel_err_mm(pred_coords_backward)        # [F, N]

        # Keep the original naming semantics:
        #   error_init        - errors of the global detector (spatial_init)
        #   error_init_landmark - errors on the "seeding" end of each direction
        #                         (forward at frame 0, backward at frame -1)
        #   error_tracking_landmark - errors on the tracked-to end
        #                         (forward at frame -1, backward at frame 0)
        # For validation (all frames), aggregate forward/backward separately.
        if is_training:
            # Training reports both frames symmetrically
            error_init = err_init_matrix.reshape(-1).tolist()
            if detector_only:
                error_init_landmark = error_init
                error_tracking_landmark = error_init
                full_sequence_tracking = error_init
            else:
                error_init_landmark = np.concatenate([
                    err_fwd_matrix[0],      # forward seed @ frame 0
                    err_bwd_matrix[-1],     # backward seed @ frame -1
                ]).tolist()
                error_tracking_landmark = np.concatenate([
                    err_fwd_matrix[-1],     # forward tracked to frame -1
                    err_bwd_matrix[0],      # backward tracked to frame 0
                ]).tolist()
                full_sequence_tracking = error_tracking_landmark
        else:
            # Validation reports across all frames
            error_init = err_init_matrix.reshape(-1).tolist()
            error_init_landmark = err_fwd_matrix.reshape(-1).tolist()
            error_tracking_landmark = err_bwd_matrix.reshape(-1).tolist()
            if detector_only:
                full_sequence_tracking = error_init
            else:
                full_sequence_tracking = np.concatenate([
                    err_fwd_matrix.reshape(-1),
                    err_bwd_matrix.reshape(-1),
                ]).tolist()

        
        # Update metrics
        error_init_landmark_all.append(np.mean(error_init_landmark))
        error_tracking_landmark_all.append(np.mean(error_tracking_landmark))
        error_init_all.append(np.mean(error_init))
        error_detection_endpoint_all.append(np.mean(err_detection_endpoint_matrix))
        error_full_sequence_tracking_all.append(np.mean(full_sequence_tracking))
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss_land': f'{loss_landmarks_init.item():.2f}',
            'loss_cons': f'{loss_mutual_consistency.item():.2f}',
            'loss_velo': f'{loss_velocity.item():.2f}',
            'loss_track': f'{loss_tracking.item():.2f}',
            'loss_track_sup': f'{loss_tracking_supervised.item():.2f}',
            'err_init': f'{np.mean(error_init):.2f}',
            'err_det_end': f'{np.mean(err_detection_endpoint_matrix):.2f}',
            'full_seq': f'{np.mean(full_sequence_tracking):.2f}',
            'err_init_t': f'{np.mean(error_init_landmark):.2f}',
            'err_tracking': f'{np.mean(error_tracking_landmark):.2f}',
        })
        
        total_loss_landmarks += loss_landmarks_init.item()
        total_loss_consistency += loss_mutual_consistency.item()
        total_loss_velocity += loss_velocity.item()
        total_loss_tracking += loss_tracking.item()
        total_loss_tracking_supervised += loss_tracking_supervised.item()
        if consistency_mask_fraction is not None:
            total_consistency_mask_fraction += consistency_mask_fraction.item()
            consistency_mask_count += 1
    # Calculate final metrics
    metrics = {
        'loss': total_loss / len(data_loader),
        'loss_landmarks': total_loss_landmarks / len(data_loader),
        'loss_consistency': total_loss_consistency / len(data_loader),
        'loss_velocity': total_loss_velocity / len(data_loader),
        'loss_tracking': total_loss_tracking / len(data_loader),
        'loss_tracking_supervised': total_loss_tracking_supervised / len(data_loader),
        'consistency_mask_fraction': (
            total_consistency_mask_fraction / consistency_mask_count
            if consistency_mask_count > 0 else 0.0
        ),
        'err_init_t': np.mean(error_init_landmark_all),
        'err_tracking': np.mean(error_tracking_landmark_all),
        'err_init': np.mean(error_init_all),
        'err_detection_endpoint': np.mean(error_detection_endpoint_all),
        'full_sequence_tracking': np.mean(error_full_sequence_tracking_all),
    }
    
    # Log metrics
    split = 'train' if is_training else 'val'
    writer.add_scalar(f'Loss/{split}', metrics['loss'], epoch)
    writer.add_scalar(f'Loss/{split}_landmarks', metrics['loss_landmarks'], epoch)
    writer.add_scalar(f'Loss/{split}_consistency', metrics['loss_consistency'], epoch)
    writer.add_scalar(f'Loss/{split}_velocity', metrics['loss_velocity'], epoch)
    writer.add_scalar(f'Loss/{split}_tracking', metrics['loss_tracking'], epoch)
    writer.add_scalar(f'Loss/{split}_tracking_supervised', metrics['loss_tracking_supervised'], epoch)
    writer.add_scalar(f'Consistency/{split}_mask_fraction', metrics['consistency_mask_fraction'], epoch)
    writer.add_scalar(f'Error/{split}_init', metrics['err_init'], epoch)
    writer.add_scalar(f'Error/{split}_detection_endpoint', metrics['err_detection_endpoint'], epoch)
    writer.add_scalar(f'Error/{split}_full_sequence_tracking', metrics['full_sequence_tracking'], epoch)
    writer.add_scalar(f'Error/{split}_init_t', metrics['err_init_t'], epoch)
    writer.add_scalar(f'Error/{split}_tracking', metrics['err_tracking'], epoch)
    
    logging.info(f'Epoch {epoch} {split.capitalize()} - '
                f'Total Loss: {metrics["loss"]:.4f} - '
                f'Landmarks: {metrics["loss_landmarks"]:.4f} - '
                f'Consistency: {metrics["loss_consistency"]:.4f} - '
                f'Velocity: {metrics["loss_velocity"]:.4f} - '
                f'Tracking: {metrics["loss_tracking"]:.4f} - '
                f'Tracking Supervised: {metrics["loss_tracking_supervised"]:.4f} - '
                f'Consistency Mask: {metrics["consistency_mask_fraction"]:.4f} - '
                f'Err Init: {metrics["err_init"]:.4f} - '
                f'Err Detection Endpoint: {metrics["err_detection_endpoint"]:.4f} - '
                f'Full Sequence Tracking: {metrics["full_sequence_tracking"]:.4f} - '
                f'Err Init T: {metrics["err_init_t"]:.4f} - '
                f'Err Tracking: {metrics["err_tracking"]:.4f}')
    
    return metrics


def train(model, dinov2_model, train_loader, optimizer, device, epoch, writer, scaler, args):
    """Training function wrapper"""
    return train_or_validate(model, dinov2_model, train_loader, optimizer, device, 
                           epoch, writer, scaler, args, is_training=True)


def validate(model, dinov2_model, val_loader, device, epoch, writer, args):
    """Validation function wrapper"""
    return train_or_validate(model, dinov2_model, val_loader, None, device,
                           epoch, writer, None, args, is_training=False)


def get_augmentations(args):
    """Get data augmentation transforms.

    Train uses `A.ReplayCompose` so that the same random augmentation is applied
    to every frame in a clip (consistent temporal appearance); the first frame
    records the replay params and subsequent frames replay them.
    """
    if args.aug_mode == 'standard':
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(p=0.2),
        ]
    elif args.aug_mode == 'weak':
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        ]
    else:
        train_transforms = [
            A.Resize(args.image_size[0], args.image_size[1]),
        ]

    return {
        "train": A.ReplayCompose(train_transforms),
        "val": A.Compose([
            A.Resize(args.image_size[0], args.image_size[1]),
        ])
    }


def main():
    """Main training function"""
    # Setup
    args = parse_args()
    if args.mode is not None:
        args.training_mode = args.mode
    if args.detector_only:
        args.training_mode = 'detector_only'
    seed = 33
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    number_of_points = 11 if args.data_type == 'PLAX' else 3
    
    # Create run name and directory
    tracker_cfg = ''
    if args.training_mode != 'detector_only':
        tracker_cfg = (
            f"_tdetach{int(args.detach_tracker_teacher)}"
            f"_tinpdetach{int(args.detach_tracker_inputs)}"
            f"_gate{args.consistency_gate}{args.consistency_fb_threshold}"
            f"_mnorm{args.consistency_mask_normalization}"
            f"_epw{args.endpoint_supervision_weight}"
        )
    run_name = (f"bs{args.batch_size}_"
                f"ep{args.epochs}_"
                f"lr{args.lr}_"
                f"mode{args.training_mode}_"
                f"coord{args.coord_head}_"
                f"graph{args.graph_mode}_"
                f"aug{args.aug_mode}_"
                f"rot{args.rotation_limit}p{args.rotation_prob}_"
                f"wd{args.weight_decay}_"
                f"warm{args.warmup_epochs}_"
                f"cons{args.cons_epoch}_"
                f"fr{args.frame_num}_"
                f'agent{args.agent_num}_'
                f'attn{args.attention_layers}_'
                f'fbranch{args.feature_branch_mode}_'
                f"sz{args.image_size[0]}x{args.image_size[1]}_"
                f"velocity{args.velocity}_"
                f"tracking{args.tracking}_"
                f"mutual{args.mutual_consistency}"
                f"{tracker_cfg}")
    if args.run_tag is not None:
        run_name = args.run_tag
    results_dir = os.path.join(f'./results/{args.data_type}', run_name)
    # Snapshot the current code folder for reproducibility.
    # Use an absolute path anchored to this file so it works regardless of cwd.
    code_src = os.path.dirname(os.path.abspath(__file__))
    code_dst = os.path.join(results_dir, 'code')
    if os.path.exists(code_dst):
        shutil.rmtree(code_dst)
    shutil.copytree(code_src, code_dst)
    # Setup logging and tensorboard
    setup_logging(results_dir)
    writer = SummaryWriter(log_dir=results_dir)
    logging.info(f'Starting training with arguments: {args}')
    
    # Setup data
    aug = get_augmentations(args)
    train_dataset = EchoDataset(root_dir=args.data_dir, transform=aug['train'],
                               splits='train', frame_num=args.frame_num,
                               target_size=args.image_size, data_type=args.data_type,
                               rotation_limit=args.rotation_limit,
                               rotation_prob=args.rotation_prob)
    val_dataset = EchoDataset(root_dir=args.data_dir, transform=aug['val'],
                             splits='val', frame_num=args.frame_num,
                             target_size=args.image_size, data_type=args.data_type)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=max(1, args.num_workers // 2),
                          pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    
    # Setup models
    model = SemiEchoTracker(
        agent_num=args.agent_num,
        nodes_num=number_of_points,
        image_size=args.image_size[0],
        coord_head=args.coord_head,
        graph_mode=args.graph_mode,
        attention_layers=args.attention_layers,
        feature_branch_mode=args.feature_branch_mode,
    ).to(args.device)
    logging.info('start loading dinov2 model...')
    dinov2_model = get_dinov2_model().to(args.device)
    logging.info('dinov2 model loaded')
    
    # Freeze DINOv2 model
    dinov2_model.eval()
    for param in dinov2_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer - remove dinov2_model parameters since they're frozen.
    detector_params, tracker_params = [], []
    for name, param in model.named_parameters():
        if is_tracker_parameter(name):
            tracker_params.append(param)
        else:
            detector_params.append(param)
    optimizer = optim.AdamW(
        [
            {
                'params': detector_params,
                'lr': args.lr * args.detector_lr_multiplier,
                'name': 'detector_shared',
            },
            {
                'params': tracker_params,
                'lr': args.lr * args.tracker_lr_multiplier,
                'name': 'tracker',
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logging.info(
        f'Optimizer lr groups: detector_shared={args.lr * args.detector_lr_multiplier:.6g}, '
        f'tracker={args.lr * args.tracker_lr_multiplier:.6g}'
    )
    
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                  total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer,
                                          T_max=max(1, args.epochs - args.warmup_epochs))
        scheduler = SequentialLR(optimizer,
                               schedulers=[warmup_scheduler, main_scheduler],
                               milestones=[args.warmup_epochs])
    elif args.warmup_epochs > 0:
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=max(1, args.epochs))
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Trainable parameters: {trainable_params}/{total_params}')
    best_val_error = float('inf')
    best_save_name = None
    best_records = {
        'endpoint': {'value': float('inf'), 'filename': None, 'alias': 'best_endpoint_model.pth'},
        'detector_fullseq': {'value': float('inf'), 'filename': None, 'alias': 'best_detector_fullseq_model.pth'},
    }
    # Training loop
    for epoch in range(args.epochs):
        train_metrics = train(model, dinov2_model, train_loader, optimizer,
                            args.device, epoch, writer, scaler, args)
        val_metrics = validate(model, dinov2_model, val_loader,
                             args.device, epoch, writer, args)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f'Epoch {epoch}/{args.epochs} finished, LR: {current_lr:.6f}')
        
        # Save checkpoints (DINOv2 is frozen, no need to save it every epoch)
        checkpoint = {
            'epoch': epoch,
            'args': vars(args),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        # Save latest model
        torch.save(checkpoint, os.path.join(results_dir, 'latest_model.pth'))

        # Save best model. Detector-only baselines are selected by endpoint
        # detection because that is the supervised detection metric.
        best_metric_name = args.best_metric
        if best_metric_name is None:
            best_metric_name = 'err_detection_endpoint' if args.training_mode == 'detector_only' else 'err_init'
        avg_error = val_metrics[best_metric_name]
        if avg_error < best_val_error:
            if best_save_name is not None:
                prev_path = os.path.join(results_dir, best_save_name)
                if os.path.exists(prev_path):
                    os.remove(prev_path)
            best_val_error = avg_error
            best_save_name = f'best_model_{best_val_error:.4f}_epoch{epoch}.pth'
            torch.save(checkpoint, os.path.join(results_dir, best_save_name))
            torch.save(checkpoint, os.path.join(results_dir, 'best_model.pth'))
            logging.info(f'Best model saved by {best_metric_name}: {best_val_error:.4f}')

        auxiliary_scores = {
            'endpoint': val_metrics['err_detection_endpoint'],
            'detector_fullseq': val_metrics['err_init'],
        }
        for label, score in auxiliary_scores.items():
            record = best_records[label]
            if score < record['value']:
                if record['filename'] is not None:
                    prev_path = os.path.join(results_dir, record['filename'])
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                record['value'] = score
                record['filename'] = f'best_{label}_{score:.4f}_epoch{epoch}.pth'
                torch.save(checkpoint, os.path.join(results_dir, record['filename']))
                torch.save(checkpoint, os.path.join(results_dir, record['alias']))
                logging.info(f'Aux best {label} saved: {score:.4f}')
    
    writer.close()
    logging.info('Training completed.')


if __name__ == '__main__':
    main()
