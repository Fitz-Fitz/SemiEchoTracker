import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class STTGCN(nn.Module):
    """Spatial-Temporal Tracking Graph Convolutional Network.

    Two resolution concepts are tracked here:
      * `H`, `W`         - target output pixel space that matches the labels
                           (typically the resized image size, e.g. 224x224).
                           All returned coordinates / offsets are expressed in
                           this space so the loss can compare them with the
                           ground-truth landmarks directly.
      * `feature_hw`     - the actual spatial size of the feature map fed in
                           (e.g. 16 for DINOv2-S/14 on 224x224 input). Used to
                           size the internal `global_gcn_1` layer only.

    Feeding the native 16x16 patch grid (instead of the previous 224x224
    bilinear upsample) cuts memory / compute by ~200x without loss of accuracy
    because `grid_sample` already does bilinear interpolation internally.
    """
    def __init__(self, feat_dim=384, hidden_channels=256, num_vertices=11, use_Tanh=False,
                 H=224, W=224, feature_hw=16, coord_head='conv_mlp_keypoint_cnn_gcn_noxy', graph_mode='split'):
        super(STTGCN, self).__init__()
        if coord_head not in (
            'pooled_gcn', 'fullres_gcn', 'query_attn',
            'conv_mlp', 'conv_mlp_gcn', 'conv_mlp_spatial_gcn',
            'conv_mlp_keypoint_gcn', 'conv_mlp_keypoint_cnn_gcn',
            'conv_mlp_keypoint_cnn_gcn_noxy',
        ):
            raise ValueError(f'Unsupported coord_head: {coord_head}')
        self.num_vertices = num_vertices
        self.hidden_channels = hidden_channels
        self.H = H
        self.W = W
        self.feature_hw = feature_hw
        if graph_mode not in ('split', 'shared'):
            raise ValueError(f'Unsupported graph_mode: {graph_mode}')
        self.coord_head = coord_head
        self.graph_mode = graph_mode

        # feat_dim (384) + 2*(num_vertices) for pairwise distances + 2 for coordinates
        input_dim = feat_dim + 2*num_vertices + 2

        # global spatial initialization (Conv -> BN -> ReLU, not the reversed order)
        logging.info(
            f'loading global spatial initialization '
            f'(coord_head={coord_head}, feature_hw={feature_hw}, target HxW={H}x{W})...'
        )
        self.global_init_conv1 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim//2),
            nn.ReLU(inplace=True),
        )
        self.global_init_conv2 = nn.Sequential(
            nn.Conv2d(feat_dim//2, self.num_vertices, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_vertices),
            nn.ReLU(inplace=True),
        )
        # pooled_gcn keeps the original 4x4 bottleneck. fullres_gcn preserves
        # the native DINO patch grid and still directly regresses coordinates.
        if coord_head == 'fullres_gcn':
            gcn1_in = feature_hw * feature_hw
        else:
            gcn1_in = (feature_hw // 4) * (feature_hw // 4)
        self.global_gcn_1 = ImprovedGraphConvolution(gcn1_in, hidden_channels)
        self.global_gcn_2 = ImprovedGraphConvolution(hidden_channels, hidden_channels//2)
        self.global_gcn_3 = ImprovedGraphConvolution(hidden_channels//2, hidden_channels)
        self.global_projector = nn.Sequential(
            nn.Linear(hidden_channels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim//2, 2),
            nn.Sigmoid()
        )
        self.query_token_proj = nn.Sequential(
            nn.LayerNorm(feat_dim + 2),
            nn.Linear(feat_dim + 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.query_embed = nn.Parameter(torch.empty(self.num_vertices, hidden_channels))
        nn.init.normal_(self.query_embed, std=0.02)
        self.query_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(hidden_channels)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.query_norm2 = nn.LayerNorm(hidden_channels)
        self.query_projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 2),
            nn.Sigmoid(),
        )
        self.conv_mlp_head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
        )
        self.conv_mlp_projector = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear((hidden_channels // 2) * feature_hw * feature_hw, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, self.num_vertices * 2),
            nn.Sigmoid(),
        )
        self.register_buffer(
            'spatial_grid_adj',
            self._make_grid_adjacency(feature_hw, feature_hw),
            persistent=False,
        )
        self.spatial_gcn_1 = ImprovedGraphConvolution(hidden_channels // 2, hidden_channels // 2, dropout=0.1)
        self.spatial_gcn_2 = ImprovedGraphConvolution(hidden_channels // 2, hidden_channels // 2, dropout=0.1)
        self.spatial_gcn_norm = nn.LayerNorm(hidden_channels // 2)
        self.spatial_gcn_scale = nn.Parameter(torch.tensor(0.1))
        self.conv_gcn_channels = hidden_channels // 4
        self.conv_gcn_context = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                (hidden_channels // 2) * feature_hw * feature_hw,
                self.num_vertices * self.conv_gcn_channels,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.conv_gcn_node_embed = nn.Parameter(torch.empty(self.num_vertices, self.conv_gcn_channels))
        nn.init.normal_(self.conv_gcn_node_embed, std=0.02)
        self.conv_gcn_1 = ImprovedGraphConvolution(self.conv_gcn_channels, self.conv_gcn_channels)
        self.conv_gcn_2 = ImprovedGraphConvolution(self.conv_gcn_channels, self.conv_gcn_channels)
        self.conv_gcn_projector = nn.Sequential(
            nn.Linear(self.conv_gcn_channels, self.conv_gcn_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.conv_gcn_channels, 2),
            nn.Sigmoid(),
        )
        self.keypoint_gcn_channels = hidden_channels // 2
        self.keypoint_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_channels // 2 + 2),
            nn.Linear(hidden_channels // 2 + 2, self.keypoint_gcn_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.keypoint_query_embed = nn.Parameter(torch.empty(self.num_vertices, self.keypoint_gcn_channels))
        nn.init.normal_(self.keypoint_query_embed, std=0.02)
        self.keypoint_cross_attn = nn.MultiheadAttention(
            embed_dim=self.keypoint_gcn_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.keypoint_flat_context = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                (hidden_channels // 2) * feature_hw * feature_hw,
                self.num_vertices * self.keypoint_gcn_channels,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.keypoint_norm1 = nn.LayerNorm(self.keypoint_gcn_channels)
        self.keypoint_ffn = nn.Sequential(
            nn.Linear(self.keypoint_gcn_channels, self.keypoint_gcn_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.keypoint_gcn_channels * 2, self.keypoint_gcn_channels),
        )
        self.keypoint_norm2 = nn.LayerNorm(self.keypoint_gcn_channels)
        self.keypoint_gcn_1 = ImprovedGraphConvolution(self.keypoint_gcn_channels, self.keypoint_gcn_channels)
        self.keypoint_gcn_2 = ImprovedGraphConvolution(self.keypoint_gcn_channels, self.keypoint_gcn_channels)
        self.keypoint_gcn_norm = nn.LayerNorm(self.keypoint_gcn_channels)
        self.keypoint_gcn_scale = nn.Parameter(torch.tensor(0.1))
        self.keypoint_projector = nn.Sequential(
            nn.Linear(self.keypoint_gcn_channels, self.keypoint_gcn_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.keypoint_gcn_channels, 2),
            nn.Sigmoid(),
        )
        self.keypoint_cnn_agg = nn.Sequential(
            nn.Conv2d(
                hidden_channels // 2 + 2,
                self.num_vertices * self.keypoint_gcn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.num_vertices * self.keypoint_gcn_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                self.num_vertices * self.keypoint_gcn_channels,
                self.num_vertices * self.keypoint_gcn_channels,
                kernel_size=1,
                groups=self.num_vertices,
            ),
            nn.GELU(),
        )
        self.keypoint_cnn_node_embed = nn.Parameter(
            torch.empty(self.num_vertices, self.keypoint_gcn_channels)
        )
        nn.init.normal_(self.keypoint_cnn_node_embed, std=0.02)
        self.keypoint_cnn_norm = nn.LayerNorm(self.keypoint_gcn_channels)
        self.keypoint_cnn_agg_noxy = nn.Sequential(
            nn.Conv2d(
                hidden_channels // 2,
                self.num_vertices * self.keypoint_gcn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.num_vertices * self.keypoint_gcn_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                self.num_vertices * self.keypoint_gcn_channels,
                self.num_vertices * self.keypoint_gcn_channels,
                kernel_size=1,
                groups=self.num_vertices,
            ),
            nn.GELU(),
        )
        # local tracking gcn
        logging.info(f'loading local tracking gcn...')
        self.gc1 = ImprovedGraphConvolution(input_dim, hidden_channels)
        self.gc2 = ImprovedGraphConvolution(hidden_channels, hidden_channels//2)
        self.gc3 = ImprovedGraphConvolution(hidden_channels//2, hidden_channels)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
            nn.Tanh() if use_Tanh else nn.Identity()
        )

        # Separate learnable graphs for direct detection and temporal tracking.
        # They can start from the same anatomical prior but should not be forced
        # to share gradients: detector GCN models landmark identity/spatial
        # relations, while tracker GCN models motion coupling under endpoint
        # seeds.
        self.detector_adj = nn.Parameter(torch.ones(num_vertices, num_vertices))
        self.tracker_adj = nn.Parameter(torch.ones(num_vertices, num_vertices))
        self.dropout = nn.Dropout(0.3)
        logging.info(f'loading gru...')

        # Project GCN node features to a shared GRU hidden state across landmarks
        self.gru_projector = nn.Linear(hidden_channels * num_vertices, hidden_channels)

        gru_input_size = hidden_channels
        logging.info(f'Initializing GRU with input size: {gru_input_size}')
        self.gru_forward = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_input_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        self.gru_backward = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_input_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        # Deproject back to per-landmark hidden states after GRU
        self.gru_deprojector = nn.Linear(hidden_channels, hidden_channels * num_vertices)
        logging.info('GCN initialization successful')

    @staticmethod
    def _make_grid_adjacency(grid_h, grid_w):
        num_nodes = grid_h * grid_w
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        for y in range(grid_h):
            for x in range(grid_w):
                idx = y * grid_w + x
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < grid_h and 0 <= nx < grid_w:
                        adj[idx, ny * grid_w + nx] = 1.0
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1.0), -0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(0)

    def _position_grid(self, H, W, device, dtype):
        ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=-1).reshape(1, H * W, 2)

    def _query_attn_coords(self, feature_map, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        tokens = feature_map.reshape(BT, H * W, c)
        pos = self._position_grid(H, W, feature_map.device, feature_map.dtype).expand(BT, -1, -1)
        tokens = self.query_token_proj(torch.cat([tokens, pos], dim=-1))
        queries = self.query_embed.unsqueeze(0).expand(BT, -1, -1)
        attn_out, _ = self.query_attn(queries, tokens, tokens, need_weights=False)
        queries = self.query_norm1(queries + attn_out)
        queries = self.query_norm2(queries + self.query_ffn(queries))
        coords = self.query_projector(queries)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_coords(self, feature_map, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        coords = self.conv_mlp_projector(self.conv_mlp_head(frame_feats))
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_spatial_gcn_coords(self, feature_map, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        f = self.conv_mlp_head(frame_feats)  # [BT, C', H, W]
        _, channels, feat_h, feat_w = f.shape
        tokens = f.flatten(2).transpose(1, 2).contiguous()  # [BT, HW, C']
        if feat_h == self.feature_hw and feat_w == self.feature_hw:
            spatial_adj = self.spatial_grid_adj
        else:
            spatial_adj = self._make_grid_adjacency(feat_h, feat_w).to(device=f.device)
        spatial_adj = spatial_adj.to(device=f.device, dtype=tokens.dtype)
        g = F.relu(self.spatial_gcn_1(tokens, spatial_adj))
        g = F.relu(self.spatial_gcn_2(g, spatial_adj))
        tokens = self.spatial_gcn_norm(tokens + self.spatial_gcn_scale * g)
        f = tokens.transpose(1, 2).reshape(BT, channels, feat_h, feat_w).contiguous()
        coords = self.conv_mlp_projector(f)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_gcn_coords(self, feature_map, adj, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        node_features = self.conv_gcn_context(self.conv_mlp_head(frame_feats))
        node_features = node_features.reshape(BT, self.num_vertices, self.conv_gcn_channels)
        node_features = node_features + self.conv_gcn_node_embed.unsqueeze(0)
        node_features = F.relu(self.conv_gcn_1(node_features, adj))
        node_features = F.relu(self.conv_gcn_2(node_features, adj))
        coords = self.conv_gcn_projector(node_features)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_keypoint_gcn_coords(self, feature_map, adj, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        f = self.conv_mlp_head(frame_feats)  # [BT, C', H, W]
        _, _channels, feat_h, feat_w = f.shape

        tokens = f.flatten(2).transpose(1, 2).contiguous()  # [BT, HW, C']
        pos = self._position_grid(feat_h, feat_w, f.device, f.dtype).expand(BT, -1, -1)
        tokens = self.keypoint_token_proj(torch.cat([tokens, pos], dim=-1))

        queries = self.keypoint_query_embed.unsqueeze(0).expand(BT, -1, -1)
        attn_nodes, _ = self.keypoint_cross_attn(queries, tokens, tokens, need_weights=False)
        flat_nodes = self.keypoint_flat_context(f).reshape(
            BT, self.num_vertices, self.keypoint_gcn_channels
        )
        node_features = self.keypoint_norm1(queries + attn_nodes + flat_nodes)
        node_features = self.keypoint_norm2(node_features + self.keypoint_ffn(node_features))

        graph_features = F.relu(self.keypoint_gcn_1(node_features, adj))
        graph_features = F.relu(self.keypoint_gcn_2(graph_features, adj))
        node_features = self.keypoint_gcn_norm(
            node_features + self.keypoint_gcn_scale * graph_features
        )
        coords = self.keypoint_projector(node_features)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_keypoint_cnn_gcn_coords(self, feature_map, adj, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        f = self.conv_mlp_head(frame_feats)  # [BT, C', H, W]
        _, _channels, feat_h, feat_w = f.shape

        pos = self._position_grid(feat_h, feat_w, f.device, f.dtype)
        pos = pos.reshape(1, feat_h, feat_w, 2).permute(0, 3, 1, 2).expand(BT, -1, -1, -1)
        node_maps = self.keypoint_cnn_agg(torch.cat([f, pos], dim=1))
        node_maps = node_maps.reshape(
            BT, self.num_vertices, self.keypoint_gcn_channels, feat_h, feat_w
        )
        node_features = node_maps.mean(dim=(-1, -2))
        flat_nodes = self.keypoint_flat_context(f).reshape(
            BT, self.num_vertices, self.keypoint_gcn_channels
        )
        node_features = self.keypoint_cnn_norm(
            node_features + flat_nodes + self.keypoint_cnn_node_embed.unsqueeze(0)
        )
        node_features = self.keypoint_norm2(node_features + self.keypoint_ffn(node_features))

        graph_features = F.relu(self.keypoint_gcn_1(node_features, adj))
        graph_features = F.relu(self.keypoint_gcn_2(graph_features, adj))
        node_features = self.keypoint_gcn_norm(
            node_features + self.keypoint_gcn_scale * graph_features
        )
        coords = self.keypoint_projector(node_features)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def _conv_mlp_keypoint_cnn_gcn_noxy_coords(self, feature_map, adj, batch_size, time_steps, H, W, c):
        BT = batch_size * time_steps
        frame_feats = feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()
        f = self.conv_mlp_head(frame_feats)  # [BT, C', H, W]
        _, _channels, feat_h, feat_w = f.shape

        node_maps = self.keypoint_cnn_agg_noxy(f)
        node_maps = node_maps.reshape(
            BT, self.num_vertices, self.keypoint_gcn_channels, feat_h, feat_w
        )
        node_features = node_maps.mean(dim=(-1, -2))
        flat_nodes = self.keypoint_flat_context(f).reshape(
            BT, self.num_vertices, self.keypoint_gcn_channels
        )
        node_features = self.keypoint_cnn_norm(
            node_features + flat_nodes + self.keypoint_cnn_node_embed.unsqueeze(0)
        )
        node_features = self.keypoint_norm2(node_features + self.keypoint_ffn(node_features))

        graph_features = F.relu(self.keypoint_gcn_1(node_features, adj))
        graph_features = F.relu(self.keypoint_gcn_2(graph_features, adj))
        node_features = self.keypoint_gcn_norm(
            node_features + self.keypoint_gcn_scale * graph_features
        )
        coords = self.keypoint_projector(node_features)
        return coords.reshape(batch_size, time_steps, self.num_vertices, 2)

    def extract_features(self, feature_map, coords):
        """
        Extract features from feature map at given coordinates
        feature_map: [B, H, W, C]
        coords: [B, N, 2] normalized coordinates in range [0,1]
        """
        B, H, W, C = feature_map.shape
        
        # Scale coordinates to pixel space
        coords_scaled = coords.clone()
        coords_scaled[..., 0] = coords_scaled[..., 0] * (W - 1)
        coords_scaled[..., 1] = coords_scaled[..., 1] * (H - 1)
        
        # Convert coords to grid sample format [-1, 1]
        coords_grid = coords_scaled.clone()
        coords_grid[..., 0] = coords_grid[..., 0] / (W-1) * 2 - 1
        coords_grid[..., 1] = coords_grid[..., 1] / (H-1) * 2 - 1
        
        # Calculate pairwise distances between landmarks
        dis_x = coords[..., 0].unsqueeze(-1) - coords[..., 0].unsqueeze(-2)  # [B, N, N]
        dis_y = coords[..., 1].unsqueeze(-1) - coords[..., 1].unsqueeze(-2)  # [B, N, N]

        # Sample features using bilinear interpolation
        features = F.grid_sample(
            feature_map.permute(0, 3, 1, 2),  # [B, C, H, W]
            coords_grid.unsqueeze(1),  # [B, 1, N, 2]
            mode='bilinear',
            align_corners=True
        ).squeeze(2).transpose(1, 2)  # [B, N, C]
        
        # Concatenate features with pairwise distances
        dis = torch.cat([dis_x, dis_y], dim=-1)  # [B, N, N, 2]
        dis = dis.reshape(B, self.num_vertices, 2*self.num_vertices)  # [B, N, 2*N]
        features = torch.cat([features, dis], dim=-1)  # [B, N, C+2*N]
        
        return features

    def tracking(self, feature_map, curr_coords=None, adj=None, forward=True):
        """Bidirectional GRU-based tracking over a video clip.

        All coordinates returned are in the target pixel space (self.H, self.W),
        regardless of the feature-map spatial size, so they can be compared
        directly with ground-truth landmarks in the resized image space.
        """
        hidden_states_list, offset_list, refined_coords_list = [], [], []
        batch_size, time_steps, _feat_h, _feat_w, c = feature_map.shape
        # target pixel-space scale (matches labels)
        out_w, out_h = self.W - 1, self.H - 1
        if not forward:
            feature_map = torch.flip(feature_map, dims=[1])  # reverse feature map
        for t in range(time_steps):
            # Extract features for current time step
            curr_feature_map = feature_map[:, t]  # [B, feat_h, feat_w, C]
            point_features = self.extract_features(curr_feature_map, curr_coords)
            # Concatenate coordinates with features
            node_features = torch.cat([curr_coords, point_features], dim=-1)
            # Graph convolution operations
            h = F.relu(self.gc1(node_features, adj, None))
            h1 = self.dropout(h)
            h = F.relu(self.gc2(h1, adj, None))
            h2 = self.dropout(h)
            h = F.relu(self.gc3(h2, adj, None))
            h_reshaped = h.view(batch_size, 1, -1)  # [B, 1, hidden_channels*num_vertices]

            # Project to lower dimension for GRU
            h_projected = self.gru_projector(h_reshaped)  # [B, 1, hidden_channels]

            if t == 0:  # use t0 as initial hidden state
                hidden_state = h_projected.permute(1, 0, 2)  # [1, B, hidden_channels]
                hidden_state = hidden_state.repeat(2, 1, 1)  # [2, B, hidden_channels]
                refined_coords_list.append(torch.stack([
                    curr_coords[..., 0] * out_w,
                    curr_coords[..., 1] * out_h,
                ], dim=-1))
                hidden_states_list.append(h)
            else:
                if forward:
                    h_out, hidden_state = self.gru_forward(h_projected, hidden_state)
                else:
                    h_out, hidden_state = self.gru_backward(h_projected, hidden_state)

                # Deproject back to original dimension
                h_deprojected = self.gru_deprojector(h_out)  # [B, 1, hidden_channels*num_vertices]
                h = h_deprojected.view(batch_size, self.num_vertices, self.hidden_channels)

                # Predict normalised offset in [0, 1] coord space
                offset = self.pred_head(h)
                offset_list.append(torch.stack([offset[..., 0] * out_w, offset[..., 1] * out_h], dim=-1))
                # Apply offset to normalized coordinates (still in [0, 1] space)
                curr_coords = curr_coords + offset
                refined_coords_list.append(torch.stack([
                    curr_coords[..., 0] * out_w,
                    curr_coords[..., 1] * out_h,
                ], dim=-1))
                hidden_states_list.append(h)
        refined_coords_list = torch.stack(refined_coords_list, dim=1)
        offset_list = torch.stack(offset_list, dim=1)
        hidden_states_list = torch.stack(hidden_states_list, dim=1)
        if forward:
            return refined_coords_list, offset_list, hidden_states_list
        else:
            return torch.flip(refined_coords_list, dims=[1]), torch.flip(offset_list, dims=[1]), torch.flip(hidden_states_list, dims=[1])

    def _normalize_landmark_adj(self, adj_param):
        adj = adj_param + torch.eye(self.num_vertices, device=adj_param.device, dtype=adj_param.dtype)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    def forward(self, feature_map, spatial_only=False, tracker_seed_coords=None, detach_tracker_inputs=False):
        """
        Input:
            feature_map: [B, T, H, W, C] dense feature map of time sequence
        """
        if isinstance(feature_map, (tuple, list)):
            detector_feature_map, tracker_feature_map = feature_map
            split_feature_map = True
        else:
            detector_feature_map = feature_map
            tracker_feature_map = feature_map
            split_feature_map = False
        batch_size, time_steps, H, W, c = detector_feature_map.shape
        detector_adj = self._normalize_landmark_adj(self.detector_adj)
        if self.graph_mode == 'shared':
            tracker_adj = detector_adj
        else:
            tracker_adj = self._normalize_landmark_adj(self.tracker_adj)

        if self.coord_head == 'query_attn':
            base_curr_coords = self._query_attn_coords(detector_feature_map, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp':
            base_curr_coords = self._conv_mlp_coords(detector_feature_map, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp_keypoint_gcn':
            base_curr_coords = self._conv_mlp_keypoint_gcn_coords(detector_feature_map, detector_adj, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp_keypoint_cnn_gcn':
            base_curr_coords = self._conv_mlp_keypoint_cnn_gcn_coords(detector_feature_map, detector_adj, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp_keypoint_cnn_gcn_noxy':
            base_curr_coords = self._conv_mlp_keypoint_cnn_gcn_noxy_coords(detector_feature_map, detector_adj, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp_spatial_gcn':
            base_curr_coords = self._conv_mlp_spatial_gcn_coords(detector_feature_map, batch_size, time_steps, H, W, c)
        elif self.coord_head == 'conv_mlp_gcn':
            base_curr_coords = self._conv_mlp_gcn_coords(detector_feature_map, detector_adj, batch_size, time_steps, H, W, c)
        else:
            # -------- Vectorised global spatial initialisation --------
            # Old code ran a Python loop over time; we fuse (B, T) and process in
            # a single forward pass. This is exactly equivalent numerically but
            # ~T-times faster and plays well with batchnorm statistics.
            BT = batch_size * time_steps
            frame_feats = detector_feature_map.reshape(BT, H, W, c).permute(0, 3, 1, 2).contiguous()  # [BT, C, H, W]
            f = self.global_init_conv1(frame_feats)         # [BT, C/2, H, W]
            if self.coord_head == 'pooled_gcn':
                f = F.avg_pool2d(f, kernel_size=2)           # [BT, C/2, H/2, W/2]
                f = self.global_init_conv2(f)                # [BT, N, H/2, W/2]
                f = F.avg_pool2d(f, kernel_size=2)           # [BT, N, H/4, W/4]
            else:
                f = self.global_init_conv2(f)                # [BT, N, H, W]
            f = f.reshape(BT, self.num_vertices, -1)         # [BT, N, spatial tokens]
            h = self.global_gcn_1(f, detector_adj)
            h = self.dropout(h)
            h = self.global_gcn_2(h, detector_adj)
            h = self.dropout(h)
            h = self.global_gcn_3(h, detector_adj)
            base_curr_coords = self.global_projector(h)      # [BT, N, 2] normalised in [0, 1]
            base_curr_coords = base_curr_coords.reshape(batch_size, time_steps, self.num_vertices, 2)

        # Scale to TARGET pixel coordinates (self.H, self.W), NOT feature coords.
        # This is critical: labels live in the resized image space (e.g. 224x224)
        # even though the feature map may be much smaller (e.g. 16x16).
        spatial_init_list = torch.stack(
            [base_curr_coords[..., 0] * (self.W - 1),
             base_curr_coords[..., 1] * (self.H - 1)],
            dim=-1
        )  # [B, T, N, 2]

        if spatial_only:
            return spatial_init_list

        if detach_tracker_inputs and not split_feature_map:
            tracker_feature_map = tracker_feature_map.detach()
        # In split-graph mode the tracker graph is a tracker-owned parameter and
        # should still learn during tracker warmup. In shared mode, detach it to
        # keep tracker losses from updating the detector graph.
        if detach_tracker_inputs and self.graph_mode == 'shared':
            tracker_adj = tracker_adj.detach()

        # Training: seed from GT endpoints for a clean supervision signal.
        # Validation/inference: seed from detector-predicted endpoints.
        if tracker_seed_coords is None:
            curr_coords_forward = base_curr_coords[:, 0].contiguous()
            curr_coords_backward = base_curr_coords[:, -1].contiguous()
        else:
            curr_coords_forward = tracker_seed_coords[:, 0].contiguous()
            curr_coords_backward = tracker_seed_coords[:, -1].contiguous()

        if detach_tracker_inputs:
            curr_coords_forward = curr_coords_forward.detach()
            curr_coords_backward = curr_coords_backward.detach()

        refined_coords_list_forward, offset_list_forward, _ = self.tracking(
            tracker_feature_map, curr_coords_forward, tracker_adj, forward=True)
        refined_coords_list_backward, offset_list_backward, _ = self.tracking(
            tracker_feature_map, curr_coords_backward, tracker_adj, forward=False)
        return refined_coords_list_forward, refined_coords_list_backward, offset_list_forward, offset_list_backward, spatial_init_list

class ImprovedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ImprovedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 分离自环和邻居的转换
        self.weight_self = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_neighbor = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(in_features * 2, 1),
            nn.Sigmoid()
        )
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_self)
        nn.init.kaiming_uniform_(self.weight_neighbor)
        
    def forward(self, x, adj, edge_weight=None):
        # 自环转换
        self_transform = torch.matmul(x, self.weight_self)
        
        # 邻居转换
        neighbor_transform = torch.matmul(x, self.weight_neighbor)
        if edge_weight is not None:
            adj = adj * edge_weight
        neighbor_agg = torch.matmul(adj, neighbor_transform)
        
        # 计算门控权重
        gate_input = torch.cat([x, torch.matmul(adj, x)], dim=-1)
        gate_weight = self.gate(gate_input)
        
        # 组合自环和邻居信息
        output = gate_weight * self_transform + (1 - gate_weight) * neighbor_agg
        
        # 归一化和dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output
