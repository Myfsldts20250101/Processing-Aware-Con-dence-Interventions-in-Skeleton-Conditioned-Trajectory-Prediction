"""
跨模态融合模块

四种融合策略:
1. ConcatFusion:  直接拼接 + 线性投影
2. CrossAttentionFusion: 轨迹 query 骨骼 + 逐维 Sigmoid 门控
3. AdaptiveGateFusion: 交叉注意力 + 输入自适应标量/逐通道门控 (推荐改进)
4. DualStreamFusion: 双流交叉注意力 (最重)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _sanitize_key_padding_mask(key_padding_mask):
    """
    nn.MultiheadAttention: 若某一 batch 行的 mask 全为 True（所有 key 被屏蔽），
    该行的输出为 NaN。无骨骼/置信度全低时 skel_mask 可能全 0，会触发此情况。
    对「全屏蔽」行放开第 0 个时间步的 key（特征接近 0，等价于无骨骼时的安全占位）。
    """
    if key_padding_mask is None:
        return None
    all_masked = key_padding_mask.all(dim=1)
    if all_masked.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, 0] = False
    return key_padding_mask


class ConcatFusion(nn.Module):
    """
    方案 A: 特征拼接融合

    traj_feat ∥ skel_feat → Linear → fused
    """

    def __init__(self, d_traj, d_skel, d_out, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_traj + d_skel, d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, traj_feat, skel_feat, skel_mask=None):
        """
        Args:
            traj_feat: (B, T, d_traj)
            skel_feat: (B, T, d_skel)
            skel_mask: (B, T) — 骨骼有效帧 mask, 1=有效 0=缺失
        Returns:
            fused: (B, T, d_out)
        """
        # 缺失骨骼帧用零替代
        if skel_mask is not None:
            skel_feat = skel_feat * skel_mask.unsqueeze(-1)

        x = torch.cat([traj_feat, skel_feat], dim=-1)  # (B, T, d_traj+d_skel)
        fused = self.proj(x)
        return self.norm(fused)


class CrossAttentionFusion(nn.Module):
    """
    方案 B: 交叉注意力融合 (推荐)

    轨迹特征作为 Query, 骨骼特征作为 Key/Value
    让模型学习: 哪些时刻的骨骼姿态对轨迹预测最有帮助
    """

    def __init__(self, d_model, nhead=4, dropout=0.1, gate_bias_init=None):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # 门控机制: 学习骨骼信息的融入比例
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        if gate_bias_init is not None:
            nn.init.constant_(
                self.gate[0].bias,
                float(gate_bias_init),
            )

    def forward(self, traj_feat, skel_feat, skel_mask=None):
        """
        Args:
            traj_feat: (B, T, d_model)
            skel_feat: (B, T, d_model)
            skel_mask: (B, T) — 骨骼有效帧 mask
        Returns:
            fused: (B, T, d_model)
        """
        # 构建 attention mask (True = 忽略)
        if skel_mask is not None:
            key_padding_mask = _sanitize_key_padding_mask((skel_mask == 0))
        else:
            key_padding_mask = None

        # Cross-Attention: traj queries skeleton
        attn_out, _ = self.cross_attn(
            query=traj_feat,
            key=skel_feat,
            value=skel_feat,
            key_padding_mask=key_padding_mask,
        )

        # 门控融合
        gate = self.gate(torch.cat([traj_feat, attn_out], dim=-1))
        x = traj_feat + gate * attn_out
        x = self.norm1(x)

        # FFN
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x


class AdaptiveGateFusion(nn.Module):
    """
    方案 B+: 交叉注意力 + 输入自适应门控

    相对 CrossAttentionFusion，门控 MLP 同时观测 traj / skel / attn_out / mask，
    输出可学习的 α∈(0,1)（逐时刻标量或逐通道），控制骨骼信号注入强度:

        fused = traj + α · attn_out

    缺失骨骼帧 (skel_mask=0) 时强制 α=0，避免无效 cross-attn 噪声。

    gate logits 经 clamp(±4) 再 sigmoid，下界 ≈0.018，防止训练中 α≡0 绝对塌缩。
    """

    def __init__(
        self,
        d_model,
        nhead=4,
        dropout=0.1,
        gate_hidden=None,
        per_channel=False,
        gate_bias_init=1.0,
    ):
        super().__init__()
        self.per_channel = per_channel
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        gate_in_dim = d_model * 3 + 1
        gate_hidden = gate_hidden or max(d_model // 2, 32)
        gate_out_dim = d_model if per_channel else 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, gate_out_dim),
        )
        # 与 cross_attn 对齐：sigmoid(1.0)≈0.731，训练初期强烈倾向开启骨骼注入
        self.gate_mlp[-1].bias.data.fill_(float(gate_bias_init))

        self.last_gate_alpha = None

    def forward(self, traj_feat, skel_feat, skel_mask=None):
        if skel_mask is not None:
            key_padding_mask = _sanitize_key_padding_mask((skel_mask == 0))
            mask_f = skel_mask.unsqueeze(-1).to(traj_feat.dtype)
        else:
            key_padding_mask = None
            mask_f = torch.ones(
                traj_feat.shape[:-1] + (1,),
                device=traj_feat.device,
                dtype=traj_feat.dtype,
            )

        attn_out, _ = self.cross_attn(
            query=traj_feat,
            key=skel_feat,
            value=skel_feat,
            key_padding_mask=key_padding_mask,
        )

        gate_in = torch.cat([traj_feat, skel_feat, attn_out, mask_f], dim=-1)
        gate_logits = self.gate_mlp(gate_in)
        # 防止 logits 极端负值导致 α≡0 塌缩；sigmoid(±4)≈0.018/0.982
        gate_logits = torch.clamp(gate_logits, min=-4.0, max=4.0)
        alpha = torch.sigmoid(gate_logits)
        if not self.per_channel:
            alpha = alpha.expand_as(attn_out)
        alpha = alpha * mask_f

        self.last_gate_alpha = alpha.detach()
        x = traj_feat + alpha * attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class DualStreamFusion(nn.Module):
    """
    方案 C: 双流交叉注意力

    两个模态互相 attend, 然后融合。
    参数量最大, 在大数据集上效果可能最好。
    """

    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # 自注意力
                'traj_self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'skel_self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                # 交叉注意力
                'traj_cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'skel_cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                # LayerNorm
                'traj_norm1': nn.LayerNorm(d_model),
                'traj_norm2': nn.LayerNorm(d_model),
                'skel_norm1': nn.LayerNorm(d_model),
                'skel_norm2': nn.LayerNorm(d_model),
                # FFN
                'traj_ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
                ),
                'skel_ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
                ),
                'traj_norm3': nn.LayerNorm(d_model),
                'skel_norm3': nn.LayerNorm(d_model),
            }))

        # 最终融合
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, traj_feat, skel_feat, skel_mask=None):
        """
        Returns: fused: (B, T, d_model)
        """
        t = traj_feat
        s = skel_feat

        key_padding_mask = (
            _sanitize_key_padding_mask((skel_mask == 0))
            if skel_mask is not None
            else None
        )

        for layer in self.layers:
            # Self-attention
            t_sa, _ = layer['traj_self_attn'](t, t, t)
            t = layer['traj_norm1'](t + t_sa)

            s_sa, _ = layer['skel_self_attn'](s, s, s, key_padding_mask=key_padding_mask)
            s = layer['skel_norm1'](s + s_sa)

            # Cross-attention
            t_ca, _ = layer['traj_cross_attn'](t, s, s, key_padding_mask=key_padding_mask)
            t = layer['traj_norm2'](t + t_ca)

            s_ca, _ = layer['skel_cross_attn'](s, t, t)
            s = layer['skel_norm2'](s + s_ca)

            # FFN
            t = layer['traj_norm3'](t + layer['traj_ffn'](t))
            s = layer['skel_norm3'](s + layer['skel_ffn'](s))

        # 融合两个流
        fused = torch.cat([t, s], dim=-1)  # (B, T, 2*d_model)
        fused = self.output_proj(fused)
        return self.output_norm(fused)


def build_fusion(
    fusion_type,
    d_model,
    nhead=4,
    dropout=0.1,
    cross_attn_gate_bias_init=None,
    fusion_cfg=None,
):
    """工厂函数: 根据配置构建融合模块"""
    fusion_cfg = fusion_cfg or {}
    if fusion_type == "concat":
        return ConcatFusion(d_model, d_model, d_model, dropout)
    elif fusion_type == "cross_attn":
        return CrossAttentionFusion(
            d_model, nhead, dropout,
            gate_bias_init=cross_attn_gate_bias_init,
        )
    elif fusion_type == "adaptive_gate":
        return AdaptiveGateFusion(
            d_model,
            nhead,
            dropout,
            gate_hidden=fusion_cfg.get("adaptive_gate_hidden"),
            per_channel=fusion_cfg.get("adaptive_gate_per_channel", False),
            gate_bias_init=fusion_cfg.get("adaptive_gate_bias_init", 1.0),
        )
    elif fusion_type == "dual_stream":
        return DualStreamFusion(d_model, nhead, num_layers=2, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
