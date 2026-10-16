"""
CTR-GCN: Channel-wise Topology Refinement Graph Convolution Network
用于从骨骼关键点序列中提取行人动作特征。

输入: (B, C_in, T, V)  — C_in=2或3, T=obs_len, V=num_joints
输出: (B, T, d_out)     — 时间维度对齐的特征序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_coco17_graph():
    """COCO-17 关键点的骨骼连接图"""
    # 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
    # 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
    # 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
    # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
        (0, 5), (0, 6),                         # 肩膀
        (5, 7), (7, 9),                          # 左臂
        (6, 8), (8, 10),                         # 右臂
        (5, 11), (6, 12),                        # 躯干
        (11, 12),                                # 髋部
        (11, 13), (13, 15),                      # 左腿
        (12, 14), (14, 16),                      # 右腿
    ]
    num_nodes = 17
    center = 0  # nose as center (也可以用 hip 的中点)

    # 构建邻接矩阵
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 自连接
    for i in range(num_nodes):
        A[i, i] = 1.0

    return A, edges, center


class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution 单元"""

    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels <= 16:
            rel_reduction = 1
            mid_reduction = 1

        self.rel_channels = in_channels // rel_reduction
        self.mid_channels = in_channels // mid_reduction

        # 通道拓扑精炼
        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        # 聚合
        self.bn = nn.BatchNorm2d(self.out_channels)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, A):
        """
        x: (B, C_in, T, V)
        A: (V, V) 邻接矩阵
        """
        # 通道拓扑精炼
        x1 = self.conv1(x).mean(dim=-2, keepdim=True)   # (B, rel_c, 1, V)
        x2 = self.conv2(x).mean(dim=-2, keepdim=True)   # (B, rel_c, 1, V)

        # 计算精炼的拓扑 (channel-wise)
        x1 = x1.permute(0, 1, 3, 2)  # (B, rel_c, V, 1)
        x2 = x2                       # (B, rel_c, 1, V)
        M = self.tanh(x1 - x2)        # (B, rel_c, V, V)
        M = self.conv4(M)             # (B, out_c, V, V)

        # 图卷积: 结合原始拓扑和精炼拓扑
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, V, V)
        agg = torch.einsum('bctv,bwvk->bctk', self.conv3(x), A + M)

        return self.bn(agg)


class TemporalConv(nn.Module):
    """时间维度卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTCN(nn.Module):
    """多尺度时间卷积网络"""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilations=[1, 2], residual=True):
        super().__init__()

        branch_channels = out_channels // (len(dilations) + 2)
        # 确保整除
        if out_channels - branch_channels * (len(dilations) + 2) != 0:
            branch_channels = out_channels // (len(dilations) + 2)
            remaining = out_channels - branch_channels * (len(dilations) + 2)
        else:
            remaining = 0

        # 各分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size, stride, d),
            )
            for d in dilations
        ])

        # MaxPool 分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))

        # 1×1 分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + remaining, kernel_size=1),
            nn.BatchNorm2d(branch_channels + remaining),
        ))

        # 残差
        self.residual = residual
        if residual:
            if in_channels == out_channels and stride == 1:
                self.res = nn.Identity()
            else:
                self.res = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        if self.residual:
            out = out + self.res(x)
        return self.relu(out)


class CTRGCNBlock(nn.Module):
    """单个 CTR-GCN Block = CTRGC (空间) + MultiScaleTCN (时间)"""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 temporal_kernel=9):
        super().__init__()
        self.gcn = CTRGC(in_channels, out_channels)
        self.tcn = MultiScaleTCN(
            out_channels, out_channels,
            kernel_size=temporal_kernel, stride=stride,
        )
        self.relu = nn.ReLU(inplace=True)

        if residual:
            if in_channels == out_channels and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.residual = lambda x: 0

        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A)
        x = self.tcn(x) + res
        return self.relu(x)


class CTRGCN(nn.Module):
    """
    完整的 CTR-GCN 骨骼特征提取器

    输入: (B, C_in, T, V) — 骨骼关键点序列
    输出: (B, T, d_out)   — 时间维度对齐的语义特征
    """

    def __init__(self, in_channels=2, num_joints=17, gcn_channels=[64, 64, 128],
                 temporal_kernel=9, d_out=128, pretrained_path=None):
        super().__init__()

        A, _, _ = get_coco17_graph()
        self.num_joints = num_joints
        self.d_out = d_out

        # 输入 BN
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # 构建各阶段
        channels = [in_channels] + gcn_channels
        self.blocks = nn.ModuleList()
        for i in range(len(gcn_channels)):
            self.blocks.append(
                CTRGCNBlock(
                    channels[i], channels[i + 1], A,
                    stride=1, residual=(i > 0),
                    temporal_kernel=temporal_kernel,
                )
            )

        # 输出投影: 将 (B, C_last, T, V) → (B, T, d_out)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 池化掉 V 维度
        self.proj = nn.Linear(gcn_channels[-1], d_out)

        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """加载预训练权重，跳过不匹配的层"""
        state_dict = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"[CTR-GCN] Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained params")

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T, V) 骨骼关键点序列
               C_in: 坐标通道 (2=xy 或 3=xyc 含置信度)
               T: 时间帧数
               V: 关节点数 (17 for COCO)
        Returns:
            feat: (B, T, d_out) 每帧的骨骼语义特征
        """
        B, C, T, V = x.shape

        # Input BN
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T * V * C, 1, 1)
        # 改用简单 reshape 做 BN
        x_flat = x.view(B, C * V, T).contiguous()
        x_flat = self.data_bn(x_flat)
        x = x_flat.view(B, C, T, V)

        # GCN blocks
        for block in self.blocks:
            x = block(x)  # (B, C_i, T, V)

        # 池化 V 维度 → (B, C_last, T, 1)
        # 不用 AdaptiveAvgPool2d 因为要保留 T 维度
        x = x.mean(dim=-1)  # (B, C_last, T)
        x = x.permute(0, 2, 1)  # (B, T, C_last)

        # 投影
        feat = self.proj(x)  # (B, T, d_out)

        return feat


class SkeletonPreprocessor(nn.Module):
    """
    骨骼数据预处理模块

    处理:
    1. 缺失关键点的 mask
    2. 相对坐标转换 (以 hip 中点为原点)
    3. 置信度通道整合
    """

    def __init__(self, num_joints=17, use_confidence=True, center_joint=None):
        super().__init__()
        self.num_joints = num_joints
        self.use_confidence = use_confidence
        # 默认用左右 hip 的中点
        self.center_joint = center_joint  # None = 用 hip 中点

    def forward(self, keypoints, confidence=None):
        """
        Args:
            keypoints: (B, T, V, 2) — 关键点 xy 坐标
            confidence: (B, T, V) — 检测置信度 (可选)
        Returns:
            x: (B, C, T, V) — CTR-GCN 输入格式
            mask: (B, T) — 有效帧 mask (至少有 3 个有效关键点)
        """
        B, T, V, _ = keypoints.shape

        # 1. 中心化: 以 hip 中点为原点
        if self.center_joint is not None:
            center = keypoints[:, :, self.center_joint:self.center_joint + 1, :]
        else:
            # COCO: 11=left_hip, 12=right_hip
            center = (keypoints[:, :, 11:12, :] + keypoints[:, :, 12:13, :]) / 2.0
        keypoints = keypoints - center

        # 2. 处理缺失关键点 (坐标为 0 且置信度低)
        if confidence is not None:
            valid = (confidence > 0.1).float()  # (B, T, V)
            keypoints = keypoints * valid.unsqueeze(-1)
            # 帧级 mask: 至少 3 个有效关键点
            mask = (valid.sum(dim=-1) >= 3).float()  # (B, T)
        else:
            mask = torch.ones(B, T, device=keypoints.device)

        # 3. 转换为 CTR-GCN 输入格式 (B, C, T, V)
        x = keypoints.permute(0, 3, 1, 2)  # (B, 2, T, V)

        if self.use_confidence and confidence is not None:
            conf = confidence.unsqueeze(1)  # (B, 1, T, V)
            x = torch.cat([x, conf], dim=1)  # (B, 3, T, V)

        return x, mask
