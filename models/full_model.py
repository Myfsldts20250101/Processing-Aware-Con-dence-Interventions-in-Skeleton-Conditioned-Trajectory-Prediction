"""
CTR-GCN-Transformer-CVAE 完整模型

组装所有子模块，提供训练和推理接口。
"""

import torch
import torch.nn as nn

from .ctr_gcn import CTRGCN, SkeletonPreprocessor
from .trajectory_encoder import TrajectoryEncoder
from .fusion import build_fusion
from .transformer_blocks import TrajectoryTransformerEncoder, TrajectoryTransformerDecoder
from .cvae import TrajectoryCVAE
from .probe_head import UpperLimbProbeHead, FullSkeletonProbeHead, SkeletonProbeHead
from utils.shuffle_time import apply_temporal_shuffle
from utils.modality_dropout import apply_modality_dropout


class CTRGCNTrajPredictor(nn.Module):
    """
    完整的 CTR-GCN-Transformer-CVAE 轨迹预测模型

    支持三种运行模式:
    1. 完整模型 (轨迹 + 骨骼)
    2. 纯轨迹 baseline (no_skeleton=True)
    3. 可配置的融合策略和 CVAE 设计
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg['model']
        skel_cfg = cfg['skeleton']
        data_cfg = cfg['data']

        self.obs_len = data_cfg['obs_len']
        self.pred_len = data_cfg['pred_len']
        self.d_model = model_cfg['d_model']
        self.no_skeleton = model_cfg.get('no_skeleton', False)
        self.skeleton_in_decoder = model_cfg.get('skeleton_in_decoder', False)
        self.skel_dim = int(model_cfg.get('skel_dim', self.d_model))
        self.modality_dropout_prob = float(model_cfg.get('modality_dropout_prob', 0.0))
        self.modality_dropout_mode = str(
            model_cfg.get('modality_dropout_mode', 'random')
        )

        # ========================
        # 1. 轨迹编码器
        # ========================
        self.traj_encoder = TrajectoryEncoder(
            input_dim=model_cfg['traj_input_dim'],
            d_model=self.d_model,
            dropout=model_cfg['dropout'],
            use_velocity=True,
        )

        # ========================
        # 2. 骨骼特征提取器 (CTR-GCN)
        # ========================
        if not self.no_skeleton:
            in_ch = skel_cfg['in_channels']
            if skel_cfg.get('use_confidence', False):
                in_ch += 1  # 加置信度通道

            self.skeleton_preprocessor = SkeletonPreprocessor(
                num_joints=skel_cfg['num_joints'],
                use_confidence=skel_cfg.get('use_confidence', True),
            )

            self.skeleton_encoder = CTRGCN(
                in_channels=in_ch,
                num_joints=skel_cfg['num_joints'],
                gcn_channels=model_cfg['gcn_channels'],
                temporal_kernel=model_cfg['gcn_temporal_kernel'],
                d_out=self.skel_dim,
                pretrained_path=skel_cfg.get('pretrained_weights', None),
            )

            skel_drop = float(model_cfg.get('skel_bottleneck_dropout', 0.0))
            if self.skel_dim != self.d_model:
                layers = []
                if skel_drop > 0:
                    layers.append(nn.Dropout(skel_drop))
                layers.append(nn.Linear(self.skel_dim, self.d_model))
                self.skel_proj = nn.Sequential(*layers)
            elif skel_drop > 0:
                self.skel_proj = nn.Dropout(skel_drop)
            else:
                self.skel_proj = None

            # ========================
            # 3. 融合模块
            # ========================
            fusion_cfg = cfg.get('fusion') or {}
            gate_bias = fusion_cfg.get('cross_attn_gate_bias_init', None)
            self.fusion = build_fusion(
                fusion_type=model_cfg['fusion_type'],
                d_model=self.d_model,
                nhead=model_cfg['nhead'] // 2,  # 融合用较少的头
                dropout=model_cfg['dropout'],
                cross_attn_gate_bias_init=gate_bias,
                fusion_cfg=fusion_cfg,
            )

        # ========================
        # 4. Transformer Encoder
        # ========================
        self.transformer_encoder = TrajectoryTransformerEncoder(
            d_model=self.d_model,
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_encoder_layers'],
            dim_feedforward=model_cfg['dim_feedforward'],
            dropout=model_cfg['dropout'],
        )

        # ========================
        # 5. CVAE
        # ========================
        self.decoupled_z = bool(model_cfg.get('decoupled_z', False))
        # A_cap：与解耦 z 同参（多一套 CVAE），但训练仍用共享 z（cvae_simple 不参与前向/损失）
        self.capacity_matched_cvae = bool(
            model_cfg.get('capacity_matched_cvae', False)
        )
        if self.decoupled_z and self.capacity_matched_cvae:
            raise ValueError(
                "decoupled_z 与 capacity_matched_cvae 互斥："
                "前者独立采样，后者仅参数量匹配的共享 z"
            )
        cvae_kwargs = dict(
            d_model=self.d_model,
            latent_dim=model_cfg['latent_dim'],
            pred_len=self.pred_len,
            logvar_min=model_cfg.get('cvae_logvar_min', -8.0),
            logvar_max=model_cfg.get('cvae_logvar_max', 4.0),
            free_bits=model_cfg.get('cvae_free_bits', 0.0),
            free_bits_weight=model_cfg.get('cvae_free_bits_weight', 1.0),
            hidden_dim=model_cfg.get('cvae_hidden_dim', None),
        )
        self.cvae = TrajectoryCVAE(**cvae_kwargs)
        if self.decoupled_z or self.capacity_matched_cvae:
            self.cvae_simple = TrajectoryCVAE(**cvae_kwargs)

        # ========================
        # 6. Transformer Decoder
        # ========================
        self.transformer_decoder = TrajectoryTransformerDecoder(
            d_model=self.d_model,
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_decoder_layers'],
            dim_feedforward=model_cfg['dim_feedforward'],
            dropout=model_cfg['dropout'],
            pred_len=self.pred_len,
            latent_dim=model_cfg['latent_dim'],
            output_dim=model_cfg['traj_input_dim'],
            skeleton_in_decoder=self.skeleton_in_decoder,
            z_scale_init=model_cfg.get('decoder_z_scale_init', 1.0),
        )

    def _project_skel_feat(self, skel_feat):
        """GCN 输出 skel_dim → fusion/decoder 使用的 d_model。"""
        if self.skel_proj is None:
            return skel_feat
        return self.skel_proj(skel_feat)

    def _encode(self, obs_traj, skeleton_kp=None, skeleton_conf=None):
        """
        编码观测序列

        Returns:
            memory: (B, T_obs, d_model) — Transformer Encoder 输出
            skel_context: (B, T_obs, d_model) — 骨骼上下文 (若 skeleton_in_decoder)
        """
        # 轨迹编码
        traj_feat = self.traj_encoder(obs_traj)  # (B, T_obs, d_model)

        skel_context = None

        if not self.no_skeleton and skeleton_kp is not None:
            # 骨骼预处理
            skel_input, skel_mask = self.skeleton_preprocessor(
                skeleton_kp, skeleton_conf
            )
            # CTR-GCN 特征提取 → 可选 bottleneck 投影到 d_model
            skel_raw = self.skeleton_encoder(skel_input)  # (B, T_obs, skel_dim)
            skel_feat = self._project_skel_feat(skel_raw)  # (B, T_obs, d_model)

            if self.training and self.modality_dropout_prob > 0:
                traj_feat, skel_feat = apply_modality_dropout(
                    traj_feat,
                    skel_feat,
                    self.modality_dropout_prob,
                    mode=self.modality_dropout_mode,
                )

            # 融合
            fused = self.fusion(traj_feat, skel_feat, skel_mask)  # (B, T_obs, d_model)

            if self.skeleton_in_decoder:
                skel_context = skel_feat
        else:
            fused = traj_feat

        # Transformer Encoder
        memory = self.transformer_encoder(fused)  # (B, T_obs, d_model)

        return memory, skel_context

    def _encode_progressive(self, obs_traj, skeleton_kp=None, skeleton_conf=None):
        """
        双分支编码：简单特征（纯轨迹）与复杂特征（Cross-Attn 融合）。

        Returns:
            memory_simple: (B, T_obs, d_model) — 仅轨迹编码
            memory_fusion: (B, T_obs, d_model) — 融合后编码
            skel_context: 骨骼上下文（若 skeleton_in_decoder）
        """
        traj_feat = self.traj_encoder(obs_traj)
        skel_context = None

        if self.no_skeleton or skeleton_kp is None:
            memory = self.transformer_encoder(traj_feat)
            return memory, memory, skel_context

        skel_input, skel_mask = self.skeleton_preprocessor(
            skeleton_kp, skeleton_conf
        )
        skel_raw = self.skeleton_encoder(skel_input)
        skel_feat = self._project_skel_feat(skel_raw)

        if self.training and self.modality_dropout_prob > 0:
            traj_feat, skel_feat = apply_modality_dropout(
                traj_feat,
                skel_feat,
                self.modality_dropout_prob,
                mode=self.modality_dropout_mode,
            )

        fusion_out = self.fusion(traj_feat, skel_feat, skel_mask)
        memory_simple = self.transformer_encoder(traj_feat)
        memory_fusion = self.transformer_encoder(fusion_out)

        if self.skeleton_in_decoder:
            skel_context = skel_feat

        return memory_simple, memory_fusion, skel_context

    def _decode_with_z(self, memory, z, skel_context=None):
        """共享 z 的双分支解码。"""
        if z.dim() == 2:
            return self.transformer_decoder(memory, z, skel_context)

        B, K, _ = z.shape
        Tobs = memory.size(1)
        mem_exp = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(
            B * K, Tobs, self.d_model
        )
        z_flat = z.reshape(B * K, -1)
        sk_exp = None
        if skel_context is not None:
            sk_exp = skel_context.unsqueeze(1).expand(-1, K, -1, -1).reshape(
                B * K, Tobs, self.d_model
            )
        pred_flat = self.transformer_decoder(mem_exp, z_flat, sk_exp)
        return pred_flat.reshape(B, K, self.pred_len, 2)

    def forward_progressive(
        self,
        obs_traj,
        future_traj=None,
        skeleton_kp=None,
        skeleton_conf=None,
        train_num_samples=1,
    ):
        """
        渐进式融合训练前向：返回简单/融合双分支预测，共享同一组 z。

        Returns:
            pred_simple: (B, T_pred, 2) 或 (B, K, T_pred, 2)
            pred_fusion: 同上
            kl_loss: scalar
        """
        memory_simple, memory_fusion, skel_context = self._encode_progressive(
            obs_traj, skeleton_kp, skeleton_conf
        )

        z, kl_loss = self.cvae(
            memory_fusion, future_traj, num_z_samples=train_num_samples
        )
        pred_fusion = self._decode_with_z(memory_fusion, z, skel_context)
        pred_simple = self._decode_with_z(memory_simple, z, skel_context)
        return pred_simple, pred_fusion, kl_loss

    def forward_decoupled_z(
        self,
        obs_traj,
        future_traj=None,
        skeleton_kp=None,
        skeleton_conf=None,
        train_num_samples=1,
    ):
        """
        解耦 z：简单/融合分支各自独立的后验 q(z|memory_branch, future)。

        Returns:
            pred_simple, pred_fusion, kl_loss (kl_simple + kl_fusion)
        """
        if not self.decoupled_z:
            raise RuntimeError("decoupled_z=False; call forward_progressive() for shared z")

        memory_simple, memory_fusion, skel_context = self._encode_progressive(
            obs_traj, skeleton_kp, skeleton_conf
        )

        z_simple, kl_simple = self.cvae_simple(
            memory_simple, future_traj, num_z_samples=train_num_samples
        )
        z_fusion, kl_fusion = self.cvae(
            memory_fusion, future_traj, num_z_samples=train_num_samples
        )
        pred_simple = self._decode_with_z(memory_simple, z_simple, skel_context)
        pred_fusion = self._decode_with_z(memory_fusion, z_fusion, skel_context)
        kl_loss = kl_simple + kl_fusion
        return pred_simple, pred_fusion, kl_loss, kl_simple, kl_fusion

    def forward(self, obs_traj, future_traj=None,
                skeleton_kp=None, skeleton_conf=None, train_num_samples=1):
        """
        Args:
            obs_traj: (B, T_obs, 2) — 观测轨迹
            future_traj: (B, T_pred, 2) — GT 未来轨迹 (训练时)
            skeleton_kp: (B, T_obs, V, 2) — 骨骼关键点
            skeleton_conf: (B, T_obs, V) — 关键点置信度
            train_num_samples: 训练时从 q 采样的轨迹条数 K（Best-of-K）；推理忽略
        Returns:
            pred_traj: (B, T_pred, 2) 或 (B, K, T_pred, 2)
            kl_loss: scalar — KL 散度
        """
        # 编码
        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)

        # CVAE：KL 只算一次；可采 K 个 z
        z, kl_loss = self.cvae(memory, future_traj, num_z_samples=train_num_samples)

        if z.dim() == 2:
            pred_traj = self.transformer_decoder(memory, z, skel_context)
        else:
            B, K, _ = z.shape
            Tobs = memory.size(1)
            mem_exp = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(
                B * K, Tobs, self.d_model
            )
            z_flat = z.reshape(B * K, -1)
            sk_exp = None
            if skel_context is not None:
                sk_exp = skel_context.unsqueeze(1).expand(-1, K, -1, -1).reshape(
                    B * K, Tobs, self.d_model
                )
            pred_flat = self.transformer_decoder(mem_exp, z_flat, sk_exp)
            pred_traj = pred_flat.reshape(B, K, self.pred_len, 2)

        return pred_traj, kl_loss

    @torch.no_grad()
    def predict(self, obs_traj, skeleton_kp=None, skeleton_conf=None,
                num_samples=20, predict_mode='sample', eps=None, return_eps=False):
        """
        推理: 生成多条候选轨迹

        Args:
            obs_traj: (B, T_obs, 2)
            skeleton_kp: (B, T_obs, V, 2)
            skeleton_conf: (B, T_obs, V)
            num_samples: 采样次数（predict_mode='most_likely' 时忽略，恒 K=1）
            predict_mode: 'sample'（prior 重参数采样）| 'most_likely'（z=prior μ）
            eps: 可选，固定重参数噪声（仅 sample 模式）
            return_eps: True 时返回 (pred_trajs, eps_out)
        Returns:
            pred_trajs: (B, K, T_pred, 2) — K 条候选轨迹
        """
        self.eval()
        B = obs_traj.shape[0]

        # 编码
        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)

        if predict_mode == 'most_likely':
            num_samples = 1

        # CVAE: 从 Prior 多次采样或取 most-likely
        if return_eps and predict_mode == 'sample':
            z_samples, memory_expanded, eps_out = self.cvae.sample_prior(
                memory, num_samples, mode=predict_mode, eps=eps, return_eps=True,
            )
        else:
            z_samples, memory_expanded = self.cvae.sample_prior(
                memory, num_samples, mode=predict_mode, eps=eps,
            )
            eps_out = None

        # 扩展骨骼上下文
        skel_ctx_expanded = None
        if skel_context is not None:
            skel_ctx_expanded = skel_context.unsqueeze(1).expand(
                -1, num_samples, -1, -1
            ).reshape(B * num_samples, self.obs_len, self.d_model)

        # 解码所有样本
        pred_all = self.transformer_decoder(
            memory_expanded, z_samples, skel_ctx_expanded
        )  # (B*K, T_pred, 2)

        # Reshape
        pred_trajs = pred_all.reshape(B, num_samples, self.pred_len, 2)

        if return_eps:
            return pred_trajs, eps_out
        return pred_trajs

    def attach_probe_head(self, hidden=64, n_joints=6):
        """挂载关节 probe 头（不影响已有质心预测路径）。"""
        HeadCls = FullSkeletonProbeHead if n_joints == 17 else UpperLimbProbeHead
        self.joint_head = HeadCls(
            d_model=self.d_model,
            n_joints=n_joints,
            hidden=hidden,
        )
        # attach 常在 model.to(device) 之后调用，新子模块默认在 CPU，需与主干对齐
        backbone_params = list(self.parameters())
        if backbone_params:
            self.joint_head.to(backbone_params[0].device)
        n_params = self.joint_head.param_count()
        tag = 'FullSkeleton' if n_joints == 17 else 'UpperLimb'
        print(f"[Probe] {tag}ProbeHead attached: {n_params} params, n_joints={n_joints}")
        return self.joint_head

    def attach_full_joint_head(self, hidden=128):
        """挂载全 17 点骨架 probe 头。"""
        return self.attach_probe_head(hidden=hidden, n_joints=17)

    def attach_direct_upper_limb_head(self, hidden=64, n_joints=4):
        """挂载 Direct Joint 高危关节预测头：肘+腕 4 点（K=1 单模态）。"""
        return self.attach_probe_head(hidden=hidden, n_joints=n_joints)

    def freeze_backbone_for_probe(self):
        """冻结全部主干，仅 joint_head 可训练。"""
        for p in self.parameters():
            p.requires_grad = False
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("Call attach_probe_head() before freeze_backbone_for_probe()")
        for p in self.joint_head.parameters():
            p.requires_grad = True
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Probe] Backbone frozen ({frozen} param tensors); trainable={trainable}")

    def set_probe_feature_shuffler(self, shuffler):
        """Probe 输入处时序打乱（§4.6）；shuffler.mode='none' 为恒等。"""
        self._probe_feature_shuffler = shuffler
        mode = getattr(shuffler, 'mode', 'none')
        print(f"[Probe] TemporalShuffle mode={mode}")

    def _shuffle_probe_hidden(self, hidden):
        shuffler = getattr(self, '_probe_feature_shuffler', None)
        return apply_temporal_shuffle(hidden, shuffler)

    def unfreeze_backbone_for_direct_joint(self):
        """解冻全部参数（含 joint_head），用于 Direct Joint 端到端微调。"""
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("Call attach_direct_upper_limb_head() first")
        for p in self.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[DirectJoint] Backbone unfrozen; trainable={trainable}/{total}")

    def _decode_multi_with_hidden(self, memory, z, skel_context=None):
        """解码并返回 (pred_traj, dec_hidden)，支持 K 模态。"""
        if z.dim() == 2:
            pred, hidden = self.transformer_decoder(
                memory, z, skel_context, return_hidden=True,
            )
            return pred, hidden

        B, K, _ = z.shape
        Tobs = memory.size(1)
        mem_exp = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(
            B * K, Tobs, self.d_model,
        )
        z_flat = z.reshape(B * K, -1)
        sk_exp = None
        if skel_context is not None:
            sk_exp = skel_context.unsqueeze(1).expand(-1, K, -1, -1).reshape(
                B * K, Tobs, self.d_model,
            )
        pred_flat, hidden_flat = self.transformer_decoder(
            mem_exp, z_flat, sk_exp, return_hidden=True,
        )
        pred = pred_flat.reshape(B, K, self.pred_len, 2)
        hidden = hidden_flat.reshape(B, K, self.pred_len, self.d_model)
        return pred, hidden

    def forward_probe(
        self,
        obs_traj,
        future_traj=None,
        skeleton_kp=None,
        skeleton_conf=None,
        train_num_samples=1,
    ):
        """
        Probe 训练前向：主干特征 → 关节偏移。

        Returns:
            joint_delta: (B, K, T_pred, 6, 2)
            pred_traj: (B, K, T_pred, 2) 或 (B, T_pred, 2)
            dec_hidden: (B, K, T_pred, d_model)
            kl_loss: scalar
        """
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("joint_head not attached; call attach_probe_head() first")

        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)
        z, kl_loss = self.cvae(memory, future_traj, num_z_samples=train_num_samples)
        pred_traj, dec_hidden = self._decode_multi_with_hidden(memory, z, skel_context)
        dec_hidden = self._shuffle_probe_hidden(dec_hidden)
        joint_delta = self.joint_head(dec_hidden)
        return joint_delta, pred_traj, dec_hidden, kl_loss

    @torch.no_grad()
    def predict_probe(
        self,
        obs_traj,
        skeleton_kp=None,
        skeleton_conf=None,
        num_samples=20,
    ):
        """
        Probe 推理：返回质心多模态预测与关节偏移。

        Returns:
            pred_trajs: (B, K, T_pred, 2)
            joint_deltas: (B, K, T_pred, 6, 2)
        """
        self.eval()
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("joint_head not attached")

        B = obs_traj.shape[0]
        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)
        z_samples, memory_expanded = self.cvae.sample_prior(memory, num_samples)

        skel_ctx_expanded = None
        if skel_context is not None:
            skel_ctx_expanded = skel_context.unsqueeze(1).expand(
                -1, num_samples, -1, -1,
            ).reshape(B * num_samples, self.obs_len, self.d_model)

        pred_flat, hidden_flat = self.transformer_decoder(
            memory_expanded, z_samples, skel_ctx_expanded, return_hidden=True,
        )
        pred_trajs = pred_flat.reshape(B, num_samples, self.pred_len, 2)
        dec_hidden = hidden_flat.reshape(B, num_samples, self.pred_len, self.d_model)
        dec_hidden = self._shuffle_probe_hidden(dec_hidden)
        joint_deltas = self.joint_head(dec_hidden)
        return pred_trajs, joint_deltas

    def forward_direct_joint(
        self,
        obs_traj,
        future_traj=None,
        skeleton_kp=None,
        skeleton_conf=None,
    ):
        """
        单模态上肢关节预测（K=1，无 CVAE 多采样 / 无 Best-of-K）。

        Returns:
            joint_delta: (B, T_pred, n_joints, 2)  默认 n_joints=6 上肢
            pred_traj: (B, T_pred, 2)
            kl_loss: scalar（训练时仍走 recognition，但 z 仅 1 条）
        """
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("joint_head not attached; call attach_probe_head() first")

        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)
        z, kl_loss = self.cvae(memory, future_traj, num_z_samples=1)
        pred_traj, dec_hidden = self.transformer_decoder(
            memory, z, skel_context, return_hidden=True,
        )
        dec_hidden = self._shuffle_probe_hidden(dec_hidden)
        joint_delta = self.joint_head(dec_hidden)
        return joint_delta, pred_traj, kl_loss

    @torch.no_grad()
    def predict_direct_joints(
        self,
        obs_traj,
        skeleton_kp=None,
        skeleton_conf=None,
    ):
        """
        单模态推理：z = prior μ，输出 1 条未来上肢/关节轨迹。

        Returns:
            joint_delta: (B, T_pred, n_joints, 2)
            pred_traj: (B, T_pred, 2)
        """
        self.eval()
        if not hasattr(self, 'joint_head'):
            raise RuntimeError("joint_head not attached")

        memory, skel_context = self._encode(obs_traj, skeleton_kp, skeleton_conf)
        z, memory_expanded = self.cvae.sample_prior(memory, num_samples=1, mode='most_likely')
        sk_exp = skel_context
        pred_traj, dec_hidden = self.transformer_decoder(
            memory_expanded, z, sk_exp, return_hidden=True,
        )
        dec_hidden = self._shuffle_probe_hidden(dec_hidden)
        joint_delta = self.joint_head(dec_hidden)
        return joint_delta, pred_traj

    def freeze_skeleton(self):
        """冻结 CTR-GCN 参数"""
        if hasattr(self, 'skeleton_encoder'):
            for param in self.skeleton_encoder.parameters():
                param.requires_grad = False
            print("[Model] CTR-GCN frozen")

    def unfreeze_skeleton(self, lr_scale=0.1):
        """解冻 CTR-GCN 参数，返回分组参数用于优化器"""
        if hasattr(self, 'skeleton_encoder'):
            for param in self.skeleton_encoder.parameters():
                param.requires_grad = True
            print(f"[Model] CTR-GCN unfrozen with lr_scale={lr_scale}")

    def get_param_groups(self, base_lr, skeleton_lr_scale=0.1):
        """获取分组学习率参数"""
        skeleton_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'skeleton_encoder' in name or 'skel_proj' in name:
                skeleton_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr': base_lr},
            {'params': skeleton_params, 'lr': base_lr * skeleton_lr_scale},
        ]


def build_model(cfg, verbose=True):
    """从配置构建模型

    Args:
        cfg: 模型与数据配置字典
        verbose: 为 True 时打印参数量（训练入口默认开启；剖析脚本可关闭以避免重复输出）
    """
    model = CTRGCNTrajPredictor(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[Model] Total params: {total_params / 1e6:.2f}M")
        print(f"[Model] Trainable params: {trainable_params / 1e6:.2f}M")
        if not cfg['model'].get('no_skeleton', False):
            skel_dim = cfg['model'].get('skel_dim', cfg['model']['d_model'])
            gcn_ch = cfg['model'].get('gcn_channels', [])
            print(
                f"[Model] Skeleton: gcn_channels={gcn_ch} skel_dim={skel_dim} "
                f"→ d_model={cfg['model']['d_model']}"
            )

    # 冻结骨骼分支
    if cfg['training'].get('freeze_skeleton', False):
        model.freeze_skeleton()

    return model
