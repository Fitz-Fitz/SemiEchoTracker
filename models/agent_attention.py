import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class EchoAgentAttention(nn.Module):
    """PAST attention block used after frozen DINOv2 features.

    The block keeps the stable projected-query variant used by the best run:
    CLS-to-patch cosine affinity selects top-K perception tokens per frame, the
    selected tokens are modeled across the sequence, then dense patch tokens
    attend to the perception-aware tokens through agent attention.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.3, proj_drop=0.,
                 agent_num=32, window=6, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        max_seq_len = 256 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        trunc_normal_(self.pos_embed, std=.02)

        time_seq = 10
        self.tem_embed = nn.Parameter(torch.zeros(1, time_seq, dim))
        trunc_normal_(self.tem_embed, std=.02)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _temporal_embed(self, T):
        if T <= self.tem_embed.shape[1]:
            return self.tem_embed[:, :T, :]
        return F.interpolate(
            self.tem_embed.transpose(1, 2), size=T, mode='linear', align_corners=False
        ).transpose(1, 2)

    def _get_perception_tokens(self, x):
        b, n, c = x.shape
        cls_token = x[:, 0, :]
        image_tokens = x[:, 1:, :]

        affinity = torch.einsum(
            'bnc,bc->bn',
            F.normalize(image_tokens, dim=-1),
            F.normalize(cls_token, dim=-1),
        )
        topk_indices = torch.topk(affinity, k=self.agent_num, dim=-1).indices

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        batch_idx = torch.arange(b, device=x.device).unsqueeze(1).expand(-1, self.agent_num)
        perception_tokens = q[batch_idx, topk_indices + 1]
        return perception_tokens, q, k, v

    def _process_frame(self, frame_agent_tokens, frame_q, frame_k, frame_v, B, n, h, w, c):
        frame_agent_attn = self.softmax((frame_agent_tokens * self.scale) @ frame_k.transpose(-2, -1))
        frame_agent_attn = self.attn_drop(frame_agent_attn)
        frame_agent_v = frame_agent_attn @ frame_v

        frame_q_attn = self.softmax((frame_q * self.scale) @ frame_agent_tokens.transpose(-2, -1))
        frame_q_attn = self.attn_drop(frame_q_attn)
        frame_x = frame_q_attn @ frame_agent_v

        frame_x = frame_x.permute(0, 2, 1, 3).reshape(B, n, c)
        frame_v_reshaped = frame_v.permute(0, 2, 1, 3).reshape(B, n, c)
        frame_v_ = frame_v_reshaped[:, 1:, :].reshape(B, h, w, c).permute(0, 3, 1, 2)
        frame_dwc = self.dwc(frame_v_).permute(0, 2, 3, 1).reshape(B, n - 1, c)
        frame_x[:, 1:, :] = frame_x[:, 1:, :] + frame_dwc
        return frame_x

    def forward(self, x, B):
        """
        Args:
            x: token features with shape [B*T, 1+H*W, C]. The first token is CLS.
            B: sequence batch size.
        """
        identity = x
        b, n, c = x.shape
        if b % B != 0:
            raise ValueError(f'Flattened batch size {b} is not divisible by B={B}')
        h = int((n - 1) ** 0.5)
        w = h
        assert h * w == n - 1, f'Input sequence length {n - 1} is not a perfect square'
        assert self.agent_num <= n - 1, (
            f'agent_num ({self.agent_num}) cannot be larger than patch token count ({n - 1})'
        )

        num_heads = self.num_heads
        head_dim = c // num_heads
        T = b // B

        x = x + self.pos_embed[:, :n, :]
        perception_tokens, q, k, v = self._get_perception_tokens(x)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        perception_tokens = perception_tokens.reshape(B, T, self.agent_num, c)
        perception_tokens = perception_tokens + self._temporal_embed(T).reshape(1, T, 1, c)
        perception_tokens = perception_tokens.reshape(B, T * self.agent_num, c)
        perception_tokens = perception_tokens.transpose(0, 1)
        perception_tokens = self.self_attention(perception_tokens, perception_tokens, perception_tokens)[0]
        perception_tokens = perception_tokens.transpose(0, 1)

        perception_tokens = perception_tokens.reshape(b, self.agent_num, c)
        perception_tokens = perception_tokens.reshape(b, self.agent_num, num_heads, head_dim)
        perception_tokens = perception_tokens.reshape(B, T, self.agent_num, num_heads, head_dim).permute(0, 1, 3, 2, 4)
        q = q.reshape(B, T, num_heads, n, head_dim)
        k = k.reshape(B, T, num_heads, n, head_dim)
        v = v.reshape(B, T, num_heads, n, head_dim)

        outputs = []
        for i in range(T):
            outputs.append(self._process_frame(
                perception_tokens[:, i], q[:, i], k[:, i], v[:, i], B, n, h, w, c
            ))
        x = torch.stack(outputs, dim=1).reshape(b, n, c)

        x = self.norm1(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + identity
        x = self.norm2(x)
        return x
