"""
Transformer Encoder / Decoder 模块

用于序列级的时间建模和未来轨迹生成。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TrajectoryTransformerEncoder(nn.Module):
    """
    Transformer Encoder: 编码观测序列的时空关系

    输入: (B, T_obs, d_model)
    输出: (B, T_obs, d_model) — memory for decoder
    """

    def __init__(self, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) — 融合后的观测特征
        Returns:
            memory: (B, T, d_model)
        """
        return self.encoder(x)


class TrajectoryTransformerDecoder(nn.Module):
    """
    Transformer Decoder: 自回归生成未来轨迹

    输入:
      - memory: (B, T_obs, d_model) — Encoder 输出
      - z: (B, latent_dim) — CVAE 采样的 latent code
      - (可选) skel_context: (B, T_obs, d_model) — 骨骼上下文
    输出:
      - pred_traj: (B, T_pred, 2) — 预测的未来轨迹
    """

    def __init__(self, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, pred_len=12,
                 latent_dim=32, output_dim=2, skeleton_in_decoder=False,
                 z_scale_init=1.0):
        super().__init__()

        self.pred_len = pred_len
        self.d_model = d_model
        self.skeleton_in_decoder = skeleton_in_decoder

        # Latent code 投影到 decoder 输入；z_scale 提高梯度通向 latent，减轻 decoder 只靠 cross-attn 忽略 z
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.z_scale = nn.Parameter(torch.tensor(float(z_scale_init)))

        # 未来时间步 embedding (可学习)
        self.future_embed = nn.Embedding(pred_len, d_model)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 骨骼上下文融合 (如果 skeleton_in_decoder=True)
        if skeleton_in_decoder:
            self.skel_cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.skel_norm = nn.LayerNorm(d_model)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, output_dim),
        )

        # Causal mask
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(pred_len)
        )

    def _decode_hidden(self, memory, z, skel_context=None):
        """Decoder 隐层特征 (B, T_pred, d_model)，供 probe 头使用。"""
        B = memory.shape[0]
        device = memory.device

        z_emb = self.z_proj(z).unsqueeze(1)
        time_idx = torch.arange(self.pred_len, device=device)
        time_emb = self.future_embed(time_idx).unsqueeze(0).expand(B, -1, -1)
        tgt = self.z_scale * z_emb + time_emb

        causal_mask = self.causal_mask[:self.pred_len, :self.pred_len].to(device)
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
        )

        if self.skeleton_in_decoder and skel_context is not None:
            skel_attn, _ = self.skel_cross_attn(
                query=dec_out, key=skel_context, value=skel_context
            )
            dec_out = self.skel_norm(dec_out + skel_attn)

        return dec_out

    def forward(self, memory, z, skel_context=None, return_hidden=False):
        """
        Args:
            memory: (B, T_obs, d_model) — Encoder 输出
            z: (B, latent_dim) — CVAE latent code
            skel_context: (B, T_obs, d_model) — 骨骼上下文 (可选)
            return_hidden: 为 True 时额外返回 dec_out
        Returns:
            pred: (B, T_pred, 2) 或 (pred, dec_out)
        """
        dec_out = self._decode_hidden(memory, z, skel_context)
        pred = self.output_proj(dec_out)
        if return_hidden:
            return pred, dec_out
        return pred
