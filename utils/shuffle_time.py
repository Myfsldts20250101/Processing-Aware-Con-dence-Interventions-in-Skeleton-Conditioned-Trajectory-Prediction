"""
特征时序打乱 — probe 输入侧控制（§4.6 Temporal Structure Ablation）

在【Decoder 隐层 → Probe 头】之间打乱时间维，不改主干与 probe 结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalShuffle(nn.Module):
    """
    输入: (B, T, D) 或更高维（时间维固定为 dim=1）
    输出: 同形状，T 维被打乱（mode='none' 为恒等）
    """

    def __init__(self, mode: str = 'none', seed: int | None = None):
        super().__init__()
        assert mode in ('none', 'within', 'fixed', 'reverse'), f'Unknown mode: {mode}'
        self.mode = mode
        self.seed = seed
        self._fixed_perm: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return x

        t_dim = 1
        t_len = x.size(t_dim)

        if self.mode == 'reverse':
            return x.flip(dims=[t_dim])

        if self.mode == 'fixed':
            if self._fixed_perm is None or self._fixed_perm.numel() != t_len:
                g = torch.Generator()
                if self.seed is not None:
                    g.manual_seed(int(self.seed))
                self._fixed_perm = torch.randperm(t_len, generator=g)
            perm = self._fixed_perm.to(x.device)
            return x.index_select(t_dim, perm)

        if self.mode == 'within':
            b_len = x.size(0)
            perms = torch.stack([torch.randperm(t_len) for _ in range(b_len)]).to(x.device)
            expand_shape = (b_len, t_len) + x.shape[2:]
            perms_exp = perms.view(b_len, t_len, *([1] * (x.dim() - 2))).expand(expand_shape)
            return x.gather(t_dim, perms_exp)

        raise RuntimeError(f'Unreachable mode: {self.mode}')


def apply_temporal_shuffle(h: torch.Tensor, shuffler: TemporalShuffle | None) -> torch.Tensor:
    """支持 probe 隐层 (B,K,T,D) 或 (B,T,D)。"""
    if shuffler is None or shuffler.mode == 'none':
        return h
    if h.dim() == 4:
        b, k, t, d = h.shape
        flat = h.reshape(b * k, t, d)
        flat = shuffler(flat)
        return flat.reshape(b, k, t, d)
    if h.dim() == 3:
        return shuffler(h)
    raise ValueError(f'apply_temporal_shuffle: unsupported shape {tuple(h.shape)}')


def configure_probe_shuffler(model, mode: str = 'none', seed: int | None = 42):
    shuffler = TemporalShuffle(mode=mode, seed=seed)
    model.set_probe_feature_shuffler(shuffler)
    return shuffler


def shuffle_settings_from_cfg(cfg: dict) -> tuple[str, int]:
    block = cfg.get('probe') or cfg.get('joint_probe') or {}
    mode = str(block.get('shuffle_mode', 'none'))
    seed = int(block.get('shuffle_seed', 42))
    return mode, seed


def assert_shuffle_works(shuffler: TemporalShuffle, feat_shape=(2, 8, 16), tol: float = 1e-6):
    torch.manual_seed(0)
    x = torch.randn(*feat_shape)
    y = shuffler(x)
    assert y.shape == x.shape, f'Shape changed: {x.shape} -> {y.shape}'
    if shuffler.mode == 'none':
        assert torch.allclose(x, y, atol=tol), "mode='none' must be identity"
    elif feat_shape[1] > 1:
        diff = (x - y).abs().sum().item()
        assert diff > tol, f"mode='{shuffler.mode}' did NOT shuffle (diff={diff})"
        x_sorted, _ = x.sort(dim=1)
        y_sorted, _ = y.sort(dim=1)
        assert torch.allclose(x_sorted, y_sorted, atol=tol), \
            f"mode='{shuffler.mode}' lost content (not a permutation)"


def run_diagnostics():
    print('=' * 60)
    print('TemporalShuffle 诊断')
    print('=' * 60)
    for mode in ('none', 'within', 'fixed', 'reverse'):
        s = TemporalShuffle(mode=mode, seed=42)
        assert_shuffle_works(s)
        print(f"  ✓ mode='{mode}': ok")
    flat = torch.randn(2, 3, 8, 128)
    out = apply_temporal_shuffle(flat, TemporalShuffle(mode='within', seed=0))
    assert out.shape == flat.shape
    print('  ✓ 4D probe hidden (B,K,T,D): ok')
    print('✓ 所有模式通过 sanity check')


if __name__ == '__main__':
    run_diagnostics()
