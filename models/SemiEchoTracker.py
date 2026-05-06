import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.GCN import STTGCN
from models.agent_attention import EchoAgentAttention

class SemiEchoTracker(nn.Module):
    """Semi-supervised Echo Tracking Network
    
    This network combines agent attention mechanism with graph convolutional networks
    for semi-supervised tracking in echo data.
    
    Args:
        in_channels (int): Number of input channels
        nodes_num (int): Number of nodes in the graph
        refine_num (int): Number of refinement iterations
        affinity_ratio (float): Ratio for affinity calculation
        attention_heads (int): Number of attention heads
        agent_num (int): Number of agent tokens
        attention_dropout (float): Dropout rate for attention
    """
    def __init__(self,
                 in_channels=384,
                 nodes_num=11,
                 attention_heads=8,
                 agent_num=36,
                 attention_dropout=0.3,
                 feature_hw=16,
                 image_size=224,
                 coord_head='conv_mlp_keypoint_cnn_gcn_noxy',
                 graph_mode='split',
                 attention_layers=3,
                 feature_branch_mode='shared'):
        super().__init__()
        self.nodes_num = nodes_num
        self.feature_hw = feature_hw
        self.image_size = image_size
        if attention_layers < 1:
            raise ValueError(f'attention_layers must be >= 1, got {attention_layers}')
        if feature_branch_mode not in ('shared', 'split_attention'):
            raise ValueError(f'Unsupported feature_branch_mode: {feature_branch_mode}')
        self.coord_head = coord_head
        self.graph_mode = graph_mode
        self.attention_layers = attention_layers
        self.feature_branch_mode = feature_branch_mode

        # Feature extraction module
        self.feature_module = nn.ModuleDict({
            f'norm_{i + 1}': nn.LayerNorm(in_channels)
            for i in range(attention_layers)
        })

        # Agent attention modules
        attention_params = {
            'dim': in_channels,
            'num_heads': attention_heads,
            'agent_num': agent_num,
            'attn_drop': attention_dropout
        }
        self.attention_modules = nn.ModuleList([
            EchoAgentAttention(**attention_params) for _ in range(attention_layers)
        ])
        if self.feature_branch_mode == 'split_attention':
            self.tracker_feature_module = nn.ModuleDict({
                f'norm_{i + 1}': nn.LayerNorm(in_channels)
                for i in range(attention_layers)
            })
            self.tracker_attention_modules = nn.ModuleList([
                EchoAgentAttention(**attention_params) for _ in range(attention_layers)
            ])

        # Graph neural network module. STTGCN outputs coords in TARGET pixel
        # space (image_size x image_size) — matching the labels — while
        # internally operating on the feature_hw x feature_hw feature map.
        self.sttgnn = STTGCN(
            feat_dim=in_channels,
            num_vertices=nodes_num,
            H=image_size,
            W=image_size,
            feature_hw=feature_hw,
            coord_head=coord_head,
            graph_mode=graph_mode,
        )
        logging.info('SemiEchoTracker initialization successful')
    
    def _fuse_features(self, x, cls_token, feature_module=None, attention_modules=None):
        """Apply agent-attention blocks and produce [B, T, H, W, C] features.

        The previous implementation bilinearly upsampled features from the 16x16
        patch grid to 224x224 before feeding STTGCN. That is unnecessary: the
        downstream `grid_sample` already does bilinear interpolation from the
        source patch grid, and `global_init_conv` can run directly on 16x16.
        Removing the upsample reduces peak memory by ~1.5 GB and speeds up
        each step considerably.
        """
        if feature_module is None:
            feature_module = self.feature_module
        if attention_modules is None:
            attention_modules = self.attention_modules

        B, T, C, H, W = x.shape
        input_tokens = x.reshape(B * T, C, H * W)                       # [B*T, C, HW]
        cls_token = cls_token.reshape(B * T, -1).unsqueeze(-1)          # [B*T, C, 1]
        input_tokens = torch.cat([cls_token, input_tokens], dim=2)      # [B*T, C, HW+1]
        input_tokens = input_tokens.permute(0, 2, 1)                    # [B*T, HW+1, C]

        for i, attention_module in enumerate(attention_modules):
            norm = feature_module[f'norm_{i + 1}']
            input_tokens = attention_module(norm(input_tokens), B=B)

        input_tokens = input_tokens.permute(0, 2, 1)                    # [B*T, C, HW+1]
        input_tokens = input_tokens.reshape(B, T, C, H * W + 1)
        input_tokens = input_tokens[:, :, :, 1:]                         # drop cls_token
        input_tokens = input_tokens.reshape(B, T, C, H, W)

        # Residual + feature normalization along channel dimension
        features = x + F.normalize(input_tokens, p=2, dim=2)             # [B, T, C, H, W]
        features = features.permute(0, 1, 3, 4, 2).contiguous()          # [B, T, H, W, C]
        return features

    def forward(self, x, cls_token, infer_mode=False, tracker_seed_coords=None, detach_tracker_inputs=False):
        """
        x: [B, T, C, H, W]  - DINOv2 patch feature map
        cls_token: [B*T, C] - DINOv2 CLS token
        """
        if self.feature_branch_mode == 'shared':
            features = self._fuse_features(x, cls_token)
            sttgnn_features = features
        else:
            detector_features = self._fuse_features(x, cls_token, self.feature_module, self.attention_modules)
            if infer_mode:
                return self.sttgnn(detector_features, spatial_only=True)
            tracker_features = self._fuse_features(
                x, cls_token, self.tracker_feature_module, self.tracker_attention_modules
            )
            features = detector_features
            sttgnn_features = (detector_features, tracker_features)

        if infer_mode:
            spatial_init = self.sttgnn(sttgnn_features, spatial_only=True)
            return spatial_init
        return self.sttgnn(
            sttgnn_features,
            tracker_seed_coords=tracker_seed_coords,
            detach_tracker_inputs=detach_tracker_inputs,
        )
