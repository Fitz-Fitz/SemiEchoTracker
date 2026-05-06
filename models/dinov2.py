import os
import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _candidate_pretrain_dirs():
    env_dir = os.environ.get('SEMI_ECHO_PRETRAIN_DIR')
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'pretraining')),
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', 'pretraining')),
        os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..', 'pretraining')),
    ])
    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            yield path


def _resolve_pretrain_dir():
    for path in _candidate_pretrain_dirs():
        if os.path.isdir(path):
            return path
    checked = '\n'.join(_candidate_pretrain_dirs())
    raise FileNotFoundError(
        'Cannot find DINOv2 pretraining directory. Set SEMI_ECHO_PRETRAIN_DIR. '
        f'Checked:\n{checked}'
    )


def get_dinov2_model():
    hub_dir = _resolve_pretrain_dir()
    ckpt_path = os.path.join(hub_dir, 'teacher_checkpoint.pth')

    if not os.path.isdir(hub_dir):
        raise FileNotFoundError(f'DINOv2 hub directory not found: {hub_dir}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'DINOv2 teacher checkpoint not found: {ckpt_path}')

    model_guided = torch.hub.load(hub_dir, 'dinov2_vits14', source='local').cuda()
    pretrained = torch.load(ckpt_path, weights_only=True)
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            continue
        new_state_dict[key.replace('backbone.', '')] = value
    model_guided.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model_guided.load_state_dict(new_state_dict, strict=True)
    print(f'DINOv2 loaded from {ckpt_path}')
    return model_guided
