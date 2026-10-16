"""
CVAE (Conditional Variational Autoencoder) 模块

用于建模轨迹预测的多模态不确定性。

训练时:
  Recognition Network: q(z | x_past, x_future, skel) → 近似后验
  Prior Network:       p(z | x_past, skel)            → 先验
  KL Loss = KL(q ∥ p)

推理时:
  从 Prior 采样 z ~ p(z | x_past, skel)
  Decoder 生成多条候选轨迹
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def diagonal_gaussian_kl_terms(mu_q, logvar_q, mu_p, logvar_p):
    """
    逐维 KL( q || p )，q、p 为对角高斯。
    返回 (..., D)，对最后一维求和得到每个样本的总 KL。
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    )


class GaussianEncoder(nn.Module):
    """将输入特征映射到高斯分布的 μ 和 log(σ²)"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class TrajectoryCVAE(nn.Module):
    """
    轨迹预测 CVAE

    提供:
    - Recognition Network (训练)
    - Prior Network (训练+推理)
    - 重参数化采样
    - KL 散度计算
    """

    def __init__(
        self,
        d_model=128,
        latent_dim=32,
        pred_len=12,
        logvar_min=-8.0,
        logvar_max=4.0,
        free_bits=0.0,
        free_bits_weight=1.0,
        hidden_dim=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.pred_len = pred_len
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.free_bits = float(free_bits)
        self.free_bits_weight = float(free_bits_weight)
        # 默认 d_model*2；A_wide 等可加宽以匹配解耦 z 总参
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else int(d_model * 2)

        # Prior: p(z | x_past)
        # 输入: Encoder memory 的聚合
        self.prior_net = GaussianEncoder(
            input_dim=d_model,
            hidden_dim=self.hidden_dim,
            latent_dim=latent_dim,
        )

        # Recognition: q(z | x_past, x_future)
        # 输入: Encoder memory 聚合 + 未来轨迹编码
        self.future_encoder = nn.Sequential(
            nn.Linear(pred_len * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.recog_net = GaussianEncoder(
            input_dim=d_model * 2,  # memory + future
            hidden_dim=self.hidden_dim,
            latent_dim=latent_dim,
        )

    def _aggregate_memory(self, memory):
        """聚合 Encoder memory 为固定维度"""
        # 使用均值池化
        return memory.mean(dim=1)  # (B, d_model)

    def reparameterize(self, mu, logvar, eps=None):
        """重参数化技巧。eps 非空时注入固定噪声（形状需与 mu 广播兼容）。"""
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, memory, future_traj=None, num_z_samples=1):
        """
        Args:
            memory: (B, T_obs, d_model) — Transformer Encoder 的输出
            future_traj: (B, T_pred, 2) — GT 未来轨迹 (仅训练时)
            num_z_samples: 训练时从 q(z|·) 独立重参数采样次数；>1 时返回 z (B, K, D)，KL 仍算一次
        Returns:
            z: (B, latent_dim) 或 (B, K, latent_dim)
            kl_loss: scalar — KL 散度 (仅训练时, 推理时为 0)
        """
        mem_agg = self._aggregate_memory(memory)  # (B, d_model)

        prior_mu, prior_logvar = self.prior_net(mem_agg)
        prior_logvar = prior_logvar.clamp(self.logvar_min, self.logvar_max)

        if self.training and future_traj is not None:
            B = future_traj.shape[0]
            future_flat = future_traj.reshape(B, -1)  # (B, T_pred * 2)
            future_feat = self.future_encoder(future_flat)  # (B, d_model)

            recog_input = torch.cat([mem_agg, future_feat], dim=-1)  # (B, 2*d_model)
            recog_mu, recog_logvar = self.recog_net(recog_input)
            recog_logvar = recog_logvar.clamp(self.logvar_min, self.logvar_max)

            kl_d = diagonal_gaussian_kl_terms(
                recog_mu, recog_logvar, prior_mu, prior_logvar
            )
            kl_loss = kl_d.sum(dim=-1).mean()
            # Free bits：惩罚「逐维、batch 均值 KL」过低，缓解 posterior collapse（q≈p）
            if self.free_bits > 0.0:
                kl_dim_mean = kl_d.mean(dim=0)
                fb_pen = F.relu(self.free_bits - kl_dim_mean).sum()
                kl_loss = kl_loss + self.free_bits_weight * fb_pen

            if num_z_samples <= 1:
                z = self.reparameterize(recog_mu, recog_logvar)
                return z, kl_loss

            D = recog_mu.shape[1]
            std = torch.exp(0.5 * recog_logvar).unsqueeze(1)
            eps = torch.randn(
                B, num_z_samples, D, device=memory.device, dtype=memory.dtype
            )
            z = recog_mu.unsqueeze(1) + std * eps
            return z, kl_loss

        z = self.reparameterize(prior_mu, prior_logvar)
        return z, torch.tensor(0.0, device=memory.device)

    def sample_prior(self, memory, num_samples=20, mode='sample', eps=None, return_eps=False):
        """
        推理时: 从 Prior 采样或取 most-likely (z = μ)

        Args:
            memory: (B, T_obs, d_model)
            num_samples: 采样次数（mode='most_likely' 时强制为 1）
            mode: 'sample' | 'most_likely'
            eps: 可选，固定重参数噪声；形状 (B*K, latent) 或 (B, K, latent)
            return_eps: True 时额外返回实际使用的 eps（most_likely 时为 None）
        Returns:
            z_samples: (B * num_samples, latent_dim)
            memory_expanded: (B * num_samples, T_obs, d_model)
            [eps_out]: 仅 return_eps=True
        """
        B, T, D = memory.shape
        mem_agg = self._aggregate_memory(memory)
        prior_mu, prior_logvar = self.prior_net(mem_agg)
        prior_logvar = prior_logvar.clamp(self.logvar_min, self.logvar_max)

        if mode == 'most_likely':
            num_samples = 1
            z = prior_mu
            memory_expanded = memory
            if return_eps:
                return z, memory_expanded, None
            return z, memory_expanded

        if mode != 'sample':
            raise ValueError(f"sample_prior mode must be 'sample' or 'most_likely', got {mode!r}")

        # 扩展到 num_samples
        prior_mu = prior_mu.unsqueeze(1).expand(-1, num_samples, -1)      # (B, K, latent)
        prior_logvar = prior_logvar.unsqueeze(1).expand(-1, num_samples, -1)
        prior_mu = prior_mu.reshape(B * num_samples, -1)
        prior_logvar = prior_logvar.reshape(B * num_samples, -1)

        if eps is None:
            eps_flat = torch.randn_like(prior_mu)
        else:
            if eps.dim() == 3:
                eps_flat = eps.reshape(B * num_samples, -1)
            else:
                eps_flat = eps
            if eps_flat.shape != prior_mu.shape:
                raise ValueError(
                    f"eps shape {tuple(eps_flat.shape)} != prior {tuple(prior_mu.shape)}"
                )

        z = self.reparameterize(prior_mu, prior_logvar, eps=eps_flat)  # (B*K, latent)

        # 扩展 memory
        memory_expanded = memory.unsqueeze(1).expand(-1, num_samples, -1, -1)
        memory_expanded = memory_expanded.reshape(B * num_samples, T, D)

        if return_eps:
            return z, memory_expanded, eps_flat
        return z, memory_expanded
