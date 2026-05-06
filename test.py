import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import imageio.v2 as imageio
import glob
from models.SemiEchoTracker import SemiEchoTracker
from models.dinov2 import get_dinov2_model
from dataset.echo_dataset import EchoDataset
from train import migrate_split_graph_state_dict, process_features


def parse_args():
    parser = argparse.ArgumentParser(description='SemiEchoTracker Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a run directory or a .pth checkpoint file')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory name/path to save results')
    parser.add_argument('--image_size', type=tuple, default=(224, 224), help='Image size')
    parser.add_argument('--frame_num', type=int, default=10, help='Number of frames')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split to use')
    parser.add_argument('--data_type', type=str, default='PLAX', choices=['PLAX', 'A4C'], help='Data type')
    parser.add_argument('--agent_num', type=int, default=None, help='Override agent number when it cannot be parsed from checkpoint path')
    parser.add_argument('--attention_layers', type=int, default=None,
                        help='Override attention layer count when it cannot be read from checkpoint args')
    parser.add_argument('--feature_branch_mode', type=str, default='auto',
                        choices=['auto', 'shared', 'split_attention'],
                        help='Feature branch mode to instantiate; auto reads it from checkpoint args')
    parser.add_argument('--coord_head', type=str, default='auto',
                        choices=['auto', 'pooled_gcn', 'fullres_gcn', 'query_attn',
                                 'conv_mlp', 'conv_mlp_gcn', 'conv_mlp_spatial_gcn',
                                 'conv_mlp_keypoint_gcn', 'conv_mlp_keypoint_cnn_gcn',
                                 'conv_mlp_keypoint_cnn_gcn_noxy'],
                        help='Coordinate head to instantiate; auto reads it from checkpoint args')
    parser.add_argument('--graph_mode', type=str, default='auto', choices=['auto', 'split', 'shared'],
                        help='Graph mode to instantiate; auto reads it from checkpoint args')
    parser.add_argument('--no_visuals', action='store_true', help='Skip per-frame jpg/gif visualization export')
    parser.add_argument('--temporal_ema_alpha', type=float, default=None,
                        help='Optional bidirectional EMA smoothing alpha for detector sequence outputs')
    parser.add_argument('--temporal_ema_keep_endpoints', action='store_true',
                        help='Keep raw detector endpoint predictions when temporal EMA smoothing is enabled')
    return parser.parse_args()


def infer_agent_num(checkpoint_path, fallback=32):
    if 'agent' not in checkpoint_path:
        return fallback
    try:
        return int(checkpoint_path.split('agent')[-1].split('_')[0])
    except (ValueError, IndexError):
        return fallback


def as_numpy_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def checkpoint_arg(checkpoint_args, name, default=None):
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def resolve_checkpoint(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        checkpoint_dir = checkpoint_path
        checkpoint_file = os.path.join(checkpoint_dir, 'best_model.pth')
    else:
        checkpoint_file = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_file) or '.'
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')
    return checkpoint_file, checkpoint_dir


def error_matrix_mm(pred, target, rescale_x, rescale_y, spacing):
    """Return per-frame/per-landmark Euclidean error in mm."""
    scale = np.array([rescale_x, rescale_y], dtype=np.float32)
    diff = (pred - target) * scale
    return np.sqrt(np.sum(diff ** 2, axis=-1)) * spacing


def summarize_errors(errors, thresholds):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'count': 0,
            'sdr': {f'{t}mm': float('nan') for t in thresholds},
        }
    return {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'count': int(errors.size),
        'sdr': {f'{t}mm': float(np.mean(errors <= t) * 100.0) for t in thresholds},
    }


def write_metrics_txt(path, metrics):
    with open(path, 'w') as f:
        f.write(f"Checkpoint: {metrics['checkpoint']}\n")
        f.write(f"Split: {metrics['split']}\n")
        f.write(f"Prediction source: {metrics['prediction_source']}\n\n")
        for section_name in ('endpoint_detection', 'full_sequence_tracking'):
            section = metrics[section_name]
            f.write(f"{section_name}:\n")
            f.write(f"  Mean Error: {section['mean']:.4f} mm\n")
            f.write(f"  Std Error: {section['std']:.4f} mm\n")
            f.write(f"  Count: {section['count']}\n")
            f.write("  Success Detection Rate (SDR):\n")
            for threshold, value in section['sdr'].items():
                f.write(f"    SDR@{threshold}: {value:.2f}%\n")
            f.write("\n")



def bidirectional_ema_smooth(coords, alpha, keep_endpoints=False):
    if alpha is None:
        return coords
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f'temporal_ema_alpha must be in (0, 1], got {alpha}')
    forward = []
    current = coords[0].copy()
    for t in range(coords.shape[0]):
        current = alpha * coords[t] + (1.0 - alpha) * current
        forward.append(current.copy())
    backward = []
    current = coords[-1].copy()
    for t in range(coords.shape[0] - 1, -1, -1):
        current = alpha * coords[t] + (1.0 - alpha) * current
        backward.append(current.copy())
    smoothed = 0.5 * (np.stack(forward, axis=0) + np.stack(backward[::-1], axis=0))
    if keep_endpoints:
        smoothed[0] = coords[0]
        smoothed[-1] = coords[-1]
    return smoothed

def main():
    args = parse_args()
    checkpoint_file, checkpoint_dir = resolve_checkpoint(args.checkpoint)
    if os.path.isabs(args.output_dir):
        save_dir_base = os.path.join(args.output_dir, args.split)
    else:
        save_dir_base = os.path.join(checkpoint_dir, args.output_dir, args.split)
    os.makedirs(save_dir_base, exist_ok=True)
    number_of_points = 11 if args.data_type == 'PLAX' else 3
    spacing = 0.6 if args.data_type == 'PLAX' else 0.67
    checkpoint = torch.load(checkpoint_file, map_location=args.device)
    checkpoint_args = checkpoint.get('args', {})
    agent_num = args.agent_num if args.agent_num is not None else checkpoint_arg(
        checkpoint_args, 'agent_num', infer_agent_num(args.checkpoint))
    coord_head = checkpoint_arg(checkpoint_args, 'coord_head', 'conv_mlp_keypoint_cnn_gcn_noxy')
    if args.coord_head != 'auto':
        coord_head = args.coord_head
    graph_mode = checkpoint_arg(checkpoint_args, 'graph_mode', 'split')
    if args.graph_mode != 'auto':
        graph_mode = args.graph_mode
    attention_layers = args.attention_layers if args.attention_layers is not None else checkpoint_arg(
        checkpoint_args, 'attention_layers', 3)
    feature_branch_mode = checkpoint_arg(checkpoint_args, 'feature_branch_mode', 'shared')
    if args.feature_branch_mode != 'auto':
        feature_branch_mode = args.feature_branch_mode
    # Setup model
    model = SemiEchoTracker(
        nodes_num=number_of_points,
        agent_num=agent_num,
        image_size=args.image_size[0],
        coord_head=coord_head,
        graph_mode=graph_mode,
        attention_layers=attention_layers,
        feature_branch_mode=feature_branch_mode,
    ).to(args.device)
    # DINOv2 backbone is frozen in training; load it directly from the pretraining
    # folder instead of from a per-run checkpoint (which we no longer save).
    dinov2_model = get_dinov2_model().to(args.device)
    dinov2_model.eval()

    # Load checkpoint. Older checkpoints do not contain parameters for newer
    # optional coordinate heads; fall back only for missing/unexpected keys.
    try:
        model.load_state_dict(migrate_split_graph_state_dict(checkpoint['model_state_dict']))
    except RuntimeError:
        load_info = model.load_state_dict(
            migrate_split_graph_state_dict(checkpoint['model_state_dict']),
            strict=False,
        )
        print(
            'model loaded with strict=False '
            f'(missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)})'
        )
    model.eval()
    print('model loaded!')
    # Setup dataset
    transform = A.Compose([
        A.Resize(args.image_size[0], args.image_size[1]),
    ])
    
    dataset = EchoDataset(root_dir=args.data_dir, transform=transform, splits=args.split, 
                            frame_num=args.frame_num, target_size=args.image_size, data_type=args.data_type)
    print(f'dataset loaded!')
    # Inference loop
    point_size = 20
    
    # Initialize error metrics. Detection is evaluated on endpoint frames only;
    # full_sequence_tracking follows the project convention and evaluates the
    # detector's sequence predictions over all frames used for tracking.
    endpoint_detection_errors = []
    full_sequence_tracking_errors = []
    per_patient_metrics = []
    sdr_thresholds = [1, 2, 3, 4]  # mm thresholds for SDR
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing'):
            data = dataset[idx]
            video = data['video'].unsqueeze(0).to(args.device).float() # B T C H W
            patient_name = data['patient_name']
            rescale_x = data['rescale_x']
            rescale_y = data['rescale_y']
            gt_landmarks = data['landmarks']
            
            B, T, c, H, W = video.shape
            video_input = video.view(B*T, c, H, W)
            extra_feature, cls_token = process_features(dinov2_model, video_input, B, T, H, W)
            spatial_init = model(extra_feature, cls_token, infer_mode=True)
            save_dir = os.path.join(save_dir_base, f'{patient_name}')
            os.makedirs(save_dir, exist_ok=True)
            spatial_init = spatial_init.cpu().numpy()[0]
            spatial_init = bidirectional_ema_smooth(
                spatial_init,
                args.temporal_ema_alpha,
                keep_endpoints=args.temporal_ema_keep_endpoints,
            )
            # Convert predictions to numpy
            landmarks = as_numpy_array(gt_landmarks) # T N 2
            
            errors = error_matrix_mm(spatial_init, landmarks, rescale_x, rescale_y, spacing)
            endpoint_indices = [0, T - 1] if T > 1 else [0]
            endpoint_errors = errors[endpoint_indices]
            endpoint_detection_errors.extend(endpoint_errors.reshape(-1).tolist())
            full_sequence_tracking_errors.extend(errors.reshape(-1).tolist())
            per_patient_metrics.append({
                'patient_name': patient_name,
                'endpoint_detection': summarize_errors(endpoint_errors, sdr_thresholds),
                'full_sequence_tracking': summarize_errors(errors, sdr_thresholds),
            })

            # Save visualizations of spatial init
            if not args.no_visuals:
                for t in range(T):
                    frame = video[0, t, 0].cpu().numpy()
                    plt.imshow(frame, cmap='gray')
                    plt.scatter(spatial_init[t][:, 0], spatial_init[t][:, 1], color='green', s=point_size, marker='o', label='pred')
                    plt.scatter(landmarks[t][:, 0], landmarks[t][:, 1], color='red', s=point_size, marker='x', label='gt')
                    plt.legend()
                    plt.axis('off')
                    plt.savefig(os.path.join(save_dir, f'frame_{t:03d}.jpg'), bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()
                spatial_list = glob.glob(os.path.join(save_dir, '*.jpg'))
                spatial_list.sort()
                spatial_reverse_list = spatial_list[::-1]
                spatial_list = spatial_list + spatial_reverse_list
                imageio.mimsave(os.path.join(save_dir, 'result_spatial.gif'), [imageio.imread(image) for image in spatial_list], duration=0.1, quality=100)

            np.savez(os.path.join(save_dir, 'pred_data.npz'), pred=spatial_init, gt=landmarks,
                     error_mm=errors, endpoint_indices=np.asarray(endpoint_indices),
                     rescale_x=rescale_x, rescale_y=rescale_y)
    
    metrics = {
        'checkpoint': checkpoint_file,
        'split': args.split,
        'prediction_source': (
            'detector' if args.temporal_ema_alpha is None
            else f'detector+bidirectional_ema_alpha_{args.temporal_ema_alpha}'
        ),
        'endpoint_detection': summarize_errors(endpoint_detection_errors, sdr_thresholds),
        'full_sequence_tracking': summarize_errors(full_sequence_tracking_errors, sdr_thresholds),
        'per_patient': per_patient_metrics,
    }

    print("\nEndpoint Detection Metrics:")
    print(f"Mean Error: {metrics['endpoint_detection']['mean']:.4f} mm")
    print(f"Std Error: {metrics['endpoint_detection']['std']:.4f} mm")
    for threshold, value in metrics['endpoint_detection']['sdr'].items():
        print(f"SDR@{threshold}: {value:.2f}%")

    print("\nFull-Sequence Tracking Metrics:")
    print(f"Mean Error: {metrics['full_sequence_tracking']['mean']:.4f} mm")
    print(f"Std Error: {metrics['full_sequence_tracking']['std']:.4f} mm")
    for threshold, value in metrics['full_sequence_tracking']['sdr'].items():
        print(f"SDR@{threshold}: {value:.2f}%")

    metrics_file = os.path.join(save_dir_base, 'metrics.txt')
    write_metrics_txt(metrics_file, metrics)
    with open(os.path.join(save_dir_base, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        os.path.join(save_dir_base, 'metrics_summary.npz'),
        endpoint_detection_errors_mm=np.asarray(endpoint_detection_errors, dtype=np.float32),
        full_sequence_tracking_errors_mm=np.asarray(full_sequence_tracking_errors, dtype=np.float32),
        patient_names=np.asarray([item['patient_name'] for item in per_patient_metrics], dtype=object),
    )


if __name__ == '__main__':
    main()
