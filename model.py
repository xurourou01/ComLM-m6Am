# model.py
#增加注意力权重

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 子网络定义 ====================

class SubNetwork(nn.Module):
    """完整子网络：CNN + Multi-Head Attention + 全局平均池化"""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: 输入张量 (batch, seq_len, input_dim)
            return_attention: 是否返回注意力权重
        Returns:
            如果 return_attention=True: (output, attention_weights)
            否则: output
        """
        if self.verbose:
            print(f"    [子网络] 输入: {x.shape}")

        # CNN
        x = x.transpose(1, 2)
        if self.verbose:
            print(f"    [CNN] 转置后: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        if self.verbose:
            print(f"    [CNN] 卷积+BN+ReLU后: {x.shape}")
        x = x.transpose(1, 2)
        if self.verbose:
            print(f"    [CNN] 转置回: {x.shape}")

        # Multi-head Attention (need_weights=True 获取注意力权重)
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        if self.verbose:
            print(f"    [Attention] 输出: {x.shape}")
            print(f"    [Attention] 权重形状: {attn_weights.shape}")

        # Feed-forward
        ff_out = self.linear(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        if self.verbose:
            print(f"    [FFN] 输出: {x.shape}")

        # Global average pooling
        x = x.mean(dim=1)
        if self.verbose:
            print(f"    [GlobalAvgPool] 输出: {x.shape}")

        if return_attention:
            return x, attn_weights
        return x


class CNNOnlySubNetwork(nn.Module):
    """仅 CNN 的子网络"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if self.verbose:
            print(f"    [子网络] 输入: {x.shape}")

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.mean(dim=2)  # 全局平均池化
        x = self.dropout(x)

        if self.verbose:
            print(f"    [CNN] 输出: {x.shape}")

        # CNN 子网络没有注意力权重
        if return_attention:
            return x, None
        return x


class AttentionOnlySubNetwork(nn.Module):
    """仅 Multi-Head Attention 的子网络"""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if self.verbose:
            print(f"    [子网络] 输入: {x.shape}")

        x = self.input_proj(x)
        if self.verbose:
            print(f"    [投影] 输出: {x.shape}")

        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.linear(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        x = x.mean(dim=1)
        if self.verbose:
            print(f"    [Attention] 输出: {x.shape}")

        if return_attention:
            return x, attn_weights
        return x


class NoSubNetwork(nn.Module):
    """无子网络：仅全局平均池化 + 线性映射"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if self.verbose:
            print(f"    [子网络] 输入: {x.shape}")

        x = x.mean(dim=1)  # 全局平均池化
        if self.verbose:
            print(f"    [GlobalAvgPool] 输出: {x.shape}")

        x = self.linear(x)
        x = self.dropout(x)

        if self.verbose:
            print(f"    [Linear] 输出: {x.shape}")

        # 无子网络没有注意力权重
        if return_attention:
            return x, None
        return x


# ==================== 多模型分类器 ====================

class MultiModelClassifier(nn.Module):
    """集成多个子网络，拼接后分类"""

    def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1,
                 sub_type='full', verbose=False):
        super().__init__()
        self.verbose = verbose
        self.sub_type = sub_type

        self.subnets = nn.ModuleDict()
        for name, dim in model_dims.items():
            if sub_type == 'none':
                subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
            elif sub_type == 'cnn':
                subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
            elif sub_type == 'attention':
                subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
                                                 num_heads=num_heads, dropout=dropout, verbose=verbose)
            else:  # 'full' 默认
                subnet = SubNetwork(dim, hidden_dim=sub_hidden,
                                    num_heads=num_heads, dropout=dropout, verbose=verbose)
            self.subnets[name] = subnet

        total_dim = sub_hidden * len(model_dims)
        if verbose:
            print(f"[MultiModelClassifier] 拼接后维度: {total_dim}")

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def extract_features(self, features_dict):
        """提取拼接后的特征（不经过分类器）"""
        outputs = []
        for name, x in features_dict.items():
            if self.verbose:
                print(f"\n  处理模型: {name}")
            out = self.subnets[name](x)
            outputs.append(out)
        concat = torch.cat(outputs, dim=-1)
        if self.verbose:
            print(f"\n[特征提取] 拼接后形状: {concat.shape}")
        return concat

    def forward(self, features_dict):
        concat = self.extract_features(features_dict)
        logits = self.classifier(concat)
        return logits

    def forward_with_attention(self, features_dict):
        """
        前向传播并返回注意力权重
        Returns:
            logits: 分类输出 (batch, 2)
            attention_weights: dict {model_name: attention_weights}
                attention_weights 形状: (batch, num_heads, seq_len, seq_len)
        """
        outputs = []
        attention_weights = {}

        for name, x in features_dict.items():
            if self.verbose:
                print(f"\n  处理模型: {name}")
            out, attn = self.subnets[name](x, return_attention=True)
            outputs.append(out)
            if attn is not None:
                attention_weights[name] = attn
            if self.verbose and attn is not None:
                print(f"    {name} 注意力权重形状: {attn.shape}")

        concat = torch.cat(outputs, dim=-1)
        if self.verbose:
            print(f"\n[特征提取] 拼接后形状: {concat.shape}")

        logits = self.classifier(concat)
        return logits, attention_weights


class SingleModelClassifier(nn.Module):
    """仅使用单个模型的子网络 + 分类层"""

    def __init__(self, model_name, input_dim, sub_hidden=128, num_heads=4,
                 dropout=0.1, sub_type='full', verbose=False):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose

        if sub_type == 'none':
            self.subnet = NoSubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
        elif sub_type == 'cnn':
            self.subnet = CNNOnlySubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
        elif sub_type == 'attention':
            self.subnet = AttentionOnlySubNetwork(input_dim, hidden_dim=sub_hidden,
                                                  num_heads=num_heads, dropout=dropout, verbose=verbose)
        else:  # 'full' 默认
            self.subnet = SubNetwork(input_dim, hidden_dim=sub_hidden,
                                     num_heads=num_heads, dropout=dropout, verbose=verbose)

        self.classifier = nn.Sequential(
            nn.Linear(sub_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def extract_features(self, features_dict):
        """提取子网络输出的特征（不经过分类器）"""
        x = features_dict[self.model_name]
        out = self.subnet(x)
        if self.verbose:
            print(f"[SingleModelClassifier] {self.model_name} 特征形状: {out.shape}")
        return out

    def forward(self, features_dict):
        out = self.extract_features(features_dict)
        logits = self.classifier(out)
        return logits

    def forward_with_attention(self, features_dict):
        """前向传播并返回注意力权重"""
        x = features_dict[self.model_name]
        out, attn = self.subnet(x, return_attention=True)
        logits = self.classifier(out)
        return logits, {self.model_name: attn}


class GatedFusionClassifier(nn.Module):
    """门控融合分类器"""

    def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1,
                 sub_type='full', verbose=False):
        super().__init__()
        self.verbose = verbose
        self.sub_type = sub_type

        self.subnets = nn.ModuleDict()
        for name, dim in model_dims.items():
            if sub_type == 'none':
                subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
            elif sub_type == 'cnn':
                subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
            elif sub_type == 'attention':
                subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
                                                 num_heads=num_heads, dropout=dropout, verbose=verbose)
            else:
                subnet = SubNetwork(dim, hidden_dim=sub_hidden,
                                    num_heads=num_heads, dropout=dropout, verbose=verbose)
            self.subnets[name] = subnet

        num_models = len(model_dims)
        self.gate = nn.Linear(sub_hidden * num_models, num_models)
        self.classifier = nn.Sequential(
            nn.Linear(sub_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def extract_features(self, features_dict):
        """提取门控融合后的特征（不经过分类器）"""
        outputs = [self.subnets[name](x) for name, x in features_dict.items()]
        concat = torch.cat(outputs, dim=-1)
        gates = torch.sigmoid(self.gate(concat))
        fused = sum(gates[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs)))
        if self.verbose:
            print(f"[GatedFusion] 融合后特征形状: {fused.shape}")
        return fused

    def forward(self, features_dict):
        fused = self.extract_features(features_dict)
        logits = self.classifier(fused)
        return logits

    def forward_with_attention(self, features_dict):
        """前向传播并返回注意力权重"""
        outputs = []
        attention_weights = {}

        for name, x in features_dict.items():
            out, attn = self.subnets[name](x, return_attention=True)
            outputs.append(out)
            if attn is not None:
                attention_weights[name] = attn

        concat = torch.cat(outputs, dim=-1)
        gates = torch.sigmoid(self.gate(concat))
        fused = sum(gates[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs)))
        logits = self.classifier(fused)
        return logits, attention_weights


#无 注意力权重
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # ==================== 子网络定义 ====================
#
# class SubNetwork(nn.Module):
#     """完整子网络：CNN + Multi-Head Attention + 全局平均池化"""
#
#     def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1, verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
#                                kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                                num_heads=num_heads,
#                                                 dropout=dropout,
#                                                batch_first=True)
#         self.linear = nn.Linear(hidden_dim, hidden_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         if self.verbose:
#             print(f"    [子网络] 输入: {x.shape}")
#
#         # CNN
#         x = x.transpose(1, 2)
#         if self.verbose:
#             print(f"    [CNN] 转置后: {x.shape}")
#         x = F.relu(self.bn1(self.conv1(x)))
#         if self.verbose:
#             print(f"    [CNN] 卷积+BN+ReLU后: {x.shape}")
#         x = x.transpose(1, 2)
#         if self.verbose:
#             print(f"    [CNN] 转置回: {x.shape}")
#
#         # Multi-head Attention
#         attn_out, _ = self.attention(x, x, x)
#         x = x + self.dropout(attn_out)
#         x = self.norm1(x)
#         if self.verbose:
#             print(f"    [Attention] 输出: {x.shape}")
#
#         # Feed-forward
#         ff_out = self.linear(x)
#         x = x + self.dropout(ff_out)
#         x = self.norm2(x)
#         if self.verbose:
#             print(f"    [FFN] 输出: {x.shape}")
#
#         # Global average pooling
#         x = x.mean(dim=1)
#         if self.verbose:
#             print(f"    [GlobalAvgPool] 输出: {x.shape}")
#
#         return x
#
#
# class CNNOnlySubNetwork(nn.Module):
#     """仅 CNN 的子网络"""
#
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.1, verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
#                                kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         if self.verbose:
#             print(f"    [子网络] 输入: {x.shape}")
#
#         x = x.transpose(1, 2)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = x.mean(dim=2)  # 全局平均池化
#         x = self.dropout(x)
#
#         if self.verbose:
#             print(f"    [CNN] 输出: {x.shape}")
#         return x
#
#
# class AttentionOnlySubNetwork(nn.Module):
#     """仅 Multi-Head Attention 的子网络"""
#
#     def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1, verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                                num_heads=num_heads,
#                                                dropout=dropout,
#                                                batch_first=True)
#         self.linear = nn.Linear(hidden_dim, hidden_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         if self.verbose:
#             print(f"    [子网络] 输入: {x.shape}")
#
#         x = self.input_proj(x)
#         if self.verbose:
#             print(f"    [投影] 输出: {x.shape}")
#
#         attn_out, _ = self.attention(x, x, x)
#         x = x + self.dropout(attn_out)
#         x = self.norm1(x)
#
#         ff_out = self.linear(x)
#         x = x + self.dropout(ff_out)
#         x = self.norm2(x)
#
#         x = x.mean(dim=1)
#         if self.verbose:
#             print(f"    [Attention] 输出: {x.shape}")
#         return x
#
#
# class NoSubNetwork(nn.Module):
#     """无子网络：仅全局平均池化 + 线性映射"""
#
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.1, verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         if self.verbose:
#             print(f"    [子网络] 输入: {x.shape}")
#
#         x = x.mean(dim=1)  # 全局平均池化
#         if self.verbose:
#             print(f"    [GlobalAvgPool] 输出: {x.shape}")
#
#         x = self.linear(x)
#         x = self.dropout(x)
#
#         if self.verbose:
#             print(f"    [Linear] 输出: {x.shape}")
#         return x
#
#
# # ==================== 多模型分类器 ====================
#
# class MultiModelClassifier(nn.Module):
#     """集成多个子网络，拼接后分类"""
#
#     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1,
#                  sub_type='full', verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.sub_type = sub_type
#
#         self.subnets = nn.ModuleDict()
#         for name, dim in model_dims.items():
#             if sub_type == 'none':
#                 subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#             elif sub_type == 'cnn':
#                 subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#             elif sub_type == 'attention':
#                 subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
#                                                  num_heads=num_heads, dropout=dropout, verbose=verbose)
#             else:  # 'full' 默认
#                 subnet = SubNetwork(dim, hidden_dim=sub_hidden,
#                                     num_heads=num_heads, dropout=dropout, verbose=verbose)
#             self.subnets[name] = subnet
#
#         total_dim = sub_hidden * len(model_dims)
#         if verbose:
#             print(f"[MultiModelClassifier] 拼接后维度: {total_dim}")
#
#         self.classifier = nn.Sequential(
#             nn.Linear(total_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 2)
#         )
#
#     def extract_features(self, features_dict):
#         """提取拼接后的特征（不经过分类器）"""
#         outputs = []
#         for name, x in features_dict.items():
#             if self.verbose:
#                 print(f"\n  处理模型: {name}")
#             out = self.subnets[name](x)
#             outputs.append(out)
#         concat = torch.cat(outputs, dim=-1)
#         if self.verbose:
#             print(f"\n[特征提取] 拼接后形状: {concat.shape}")
#         return concat
#
#     def forward(self, features_dict):
#         concat = self.extract_features(features_dict)
#         logits = self.classifier(concat)
#         return logits
#
#
# class SingleModelClassifier(nn.Module):
#     """仅使用单个模型的子网络 + 分类层"""
#
#     def __init__(self, model_name, input_dim, sub_hidden=128, num_heads=4,
#                  dropout=0.1, sub_type='full', verbose=False):
#         super().__init__()
#         self.model_name = model_name
#         self.verbose = verbose
#
#         if sub_type == 'none':
#             self.subnet = NoSubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#         elif sub_type == 'cnn':
#             self.subnet = CNNOnlySubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#         elif sub_type == 'attention':
#             self.subnet = AttentionOnlySubNetwork(input_dim, hidden_dim=sub_hidden,
#                                                   num_heads=num_heads, dropout=dropout, verbose=verbose)
#         else:  # 'full' 默认
#             self.subnet = SubNetwork(input_dim, hidden_dim=sub_hidden,
#                                      num_heads=num_heads, dropout=dropout, verbose=verbose)
#
#         self.classifier = nn.Sequential(
#             nn.Linear(sub_hidden, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)
#         )
#
#     def extract_features(self, features_dict):
#         """提取子网络输出的特征（不经过分类器）"""
#         x = features_dict[self.model_name]
#         out = self.subnet(x)
#         if self.verbose:
#             print(f"[SingleModelClassifier] {self.model_name} 特征形状: {out.shape}")
#         return out
#
#     def forward(self, features_dict):
#         out = self.extract_features(features_dict)
#         logits = self.classifier(out)
#         return logits
#
#
# class GatedFusionClassifier(nn.Module):
#     """门控融合分类器"""
#
#     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1,
#                  sub_type='full', verbose=False):
#         super().__init__()
#         self.verbose = verbose
#         self.sub_type = sub_type
#
#         self.subnets = nn.ModuleDict()
#         for name, dim in model_dims.items():
#             if sub_type == 'none':
#                 subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#             elif sub_type == 'cnn':
#                 subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout, verbose=verbose)
#             elif sub_type == 'attention':
#                 subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
#                                                  num_heads=num_heads, dropout=dropout, verbose=verbose)
#             else:
#                 subnet = SubNetwork(dim, hidden_dim=sub_hidden,
#                                     num_heads=num_heads, dropout=dropout, verbose=verbose)
#             self.subnets[name] = subnet
#
#         num_models = len(model_dims)
#         self.gate = nn.Linear(sub_hidden * num_models, num_models)
#         self.classifier = nn.Sequential(
#             nn.Linear(sub_hidden, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)
#         )
#         self.dropout = nn.Dropout(dropout)
#
#     def extract_features(self, features_dict):
#         """提取门控融合后的特征（不经过分类器）"""
#         outputs = [self.subnets[name](x) for name, x in features_dict.items()]
#         concat = torch.cat(outputs, dim=-1)
#         gates = torch.sigmoid(self.gate(concat))
#         fused = sum(gates[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs)))
#         if self.verbose:
#             print(f"[GatedFusion] 融合后特征形状: {fused.shape}")
#         return fused
#
#     def forward(self, features_dict):
#         fused = self.extract_features(features_dict)
#         logits = self.classifier(fused)
#         return logits





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # #正则化:在 SubNetwork 中添加更多 Dropout 和 LayerNorm
# # class SubNetwork(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
# #         super().__init__()
# #         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
# #                                kernel_size=3, padding=1)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.dropout1 = nn.Dropout(dropout)          # 新增
# #         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
# #                                                 num_heads=num_heads,
# #                                                 dropout=dropout,
# #                                                 batch_first=True)
# #         self.dropout2 = nn.Dropout(dropout)          # 新增
# #         self.linear = nn.Linear(hidden_dim, hidden_dim)
# #         self.dropout3 = nn.Dropout(dropout)          # 新增
# #         self.norm1 = nn.LayerNorm(hidden_dim)
# #         self.norm2 = nn.LayerNorm(hidden_dim)
# #
# #     def forward(self, x):
# #         x = x.transpose(1, 2)
# #         x = F.relu(self.bn1(self.conv1(x)))
# #         x = self.dropout1(x)                         # 卷积后 dropout
# #         x = x.transpose(1, 2)
# #
# #         attn_out, _ = self.attention(x, x, x)
# #         x = x + self.dropout2(attn_out)
# #         x = self.norm1(x)
# #
# #         ff_out = self.linear(x)
# #         x = x + self.dropout3(ff_out)
# #         x = self.norm2(x)
# #
# #         x = x.mean(dim=1)
# #         return x
#
# class SubNetwork(nn.Module):
#     """每个模型的子网络：CNN + Multi-Head Attention + 全局平均池化"""
#     def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
#                                kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                                 num_heads=num_heads,
#                                                 dropout=dropout,
#                                                 batch_first=True)
#         self.linear = nn.Linear(hidden_dim, hidden_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = x.transpose(1, 2)                # (batch, input_dim, seq_len)
#         x = F.relu(self.bn1(self.conv1(x)))  # (batch, hidden_dim, seq_len)
#         x = x.transpose(1, 2)                # (batch, seq_len, hidden_dim)
#
#         # Multi-head attention with residual
#         attn_out, _ = self.attention(x, x, x)   # (batch, seq_len, hidden_dim)
#         x = x + self.dropout(attn_out)
#         x = self.norm1(x)
#
#         # Feed-forward with residual
#         ff_out = self.linear(x)                 # (batch, seq_len, hidden_dim)
#         x = x + self.dropout(ff_out)
#         x = self.norm2(x)
#
#         # Global average pooling
#         x = x.mean(dim=1)                        # (batch, hidden_dim)
#         return x
#
#
# #模型实验
# # 1. 无子网络：仅对输入序列做全局平均池化，然后线性映射到 sub_hidden 维度
# class NoSubNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = x.mean(dim=1)          # (batch, input_dim)
#         x = self.linear(x)         # (batch, hidden_dim)
#         x = self.dropout(x)
#         return x
#
# # 2. 仅 CNN：卷积 + 批归一化 + ReLU + 全局平均池化
# class CNNOnlySubNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
#                                kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = x.transpose(1, 2)                     # (batch, input_dim, seq_len)
#         x = F.relu(self.bn1(self.conv1(x)))       # (batch, hidden_dim, seq_len)
#         x = x.mean(dim=2)                         # (batch, hidden_dim)
#         x = self.dropout(x)
#         return x
#
# # 3. 仅 Multi-Head Attention（无CNN）
# class AttentionOnlySubNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, hidden_dim)  # 将输入映射到 hidden_dim
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                                num_heads=num_heads,
#                                                dropout=dropout,
#                                                batch_first=True)
#         self.linear = nn.Linear(hidden_dim, hidden_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = self.input_proj(x)                    # (batch, seq_len, hidden_dim)
#         attn_out, _ = self.attention(x, x, x)     # (batch, seq_len, hidden_dim)
#         x = x + self.dropout(attn_out)
#         x = self.norm1(x)
#         ff_out = self.linear(x)
#         x = x + self.dropout(ff_out)
#         x = self.norm2(x)
#         x = x.mean(dim=1)                         # (batch, hidden_dim)
#         return x
#
#
# class MultiModelClassifier(nn.Module):
#     """集成多个子网络，拼接后分类"""
#     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1, sub_type='full'):
#         super().__init__()
#         self.subnets = nn.ModuleDict()
#         for name, dim in model_dims.items():
#             if sub_type == 'none':
#                 subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout)
#             elif sub_type == 'cnn':
#                 subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout)
#             elif sub_type == 'attention':
#                 subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
#                                                   num_heads=num_heads, dropout=dropout)
#             else:  # 'full' 默认
#                 subnet = SubNetwork(dim, hidden_dim=sub_hidden,
#                                     num_heads=num_heads, dropout=dropout)
#             self.subnets[name] = subnet
#
#         total_dim = sub_hidden * len(model_dims)
#         self.classifier = nn.Sequential(
#             nn.Linear(total_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 2)
#         )
#
#     def forward(self, features_dict):
#         outputs = []
#         for name, x in features_dict.items():
#             out = self.subnets[name](x)
#             outputs.append(out)
#         concat = torch.cat(outputs, dim=-1)
#         logits = self.classifier(concat)
#         return logits
#
#
# class GatedFusionClassifier(nn.Module):
#     """门控融合分类器"""
#     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1, sub_type='full'):
#         super().__init__()
#         self.subnets = nn.ModuleDict()
#         for name, dim in model_dims.items():
#             if sub_type == 'none':
#                 subnet = NoSubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout)
#             elif sub_type == 'cnn':
#                 subnet = CNNOnlySubNetwork(dim, hidden_dim=sub_hidden, dropout=dropout)
#             elif sub_type == 'attention':
#                 subnet = AttentionOnlySubNetwork(dim, hidden_dim=sub_hidden,
#                                                   num_heads=num_heads, dropout=dropout)
#             else:
#                 subnet = SubNetwork(dim, hidden_dim=sub_hidden,
#                                     num_heads=num_heads, dropout=dropout)
#             self.subnets[name] = subnet
#
#         num_models = len(model_dims)
#         self.gate = nn.Linear(sub_hidden * num_models, num_models)
#         self.classifier = nn.Sequential(
#             nn.Linear(sub_hidden, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)
#         )
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, features_dict):
#         outputs = [self.subnets[name](x) for name, x in features_dict.items()]
#         concat = torch.cat(outputs, dim=-1)
#         gates = torch.sigmoid(self.gate(concat))
#         fused = sum(gates[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs)))
#         logits = self.classifier(fused)
#         return logits
#
#
# class SingleModelClassifier(nn.Module):
#     def __init__(self, model_name, input_dim, sub_hidden=128, num_heads=4, dropout=0.1, sub_type='full'):
#         super().__init__()
#         self.model_name = model_name
#         if sub_type == 'none':
#             self.subnet = NoSubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout)
#         elif sub_type == 'cnn':
#             self.subnet = CNNOnlySubNetwork(input_dim, hidden_dim=sub_hidden, dropout=dropout)
#         elif sub_type == 'attention':
#             self.subnet = AttentionOnlySubNetwork(input_dim, hidden_dim=sub_hidden,
#                                                   num_heads=num_heads, dropout=dropout)
#         else:  # 'full' 默认
#             self.subnet = SubNetwork(input_dim, hidden_dim=sub_hidden,
#                                      num_heads=num_heads, dropout=dropout)
#         self.classifier = nn.Sequential(
#             nn.Linear(sub_hidden, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)
#         )
#
#     def forward(self, features_dict):
#         x = features_dict[self.model_name]
#         out = self.subnet(x)
#         logits = self.classifier(out)
#         return logits
#
#
# # class MultiModelClassifier(nn.Module):
# #     """集成多个子网络，拼接后分类"""
# #     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1):
# #         super().__init__()
# #         self.subnets = nn.ModuleDict()
# #         for name, dim in model_dims.items():
# #             self.subnets[name] = SubNetwork(dim, hidden_dim=sub_hidden,
# #                                             num_heads=num_heads, dropout=dropout)
# #         total_dim = sub_hidden * len(model_dims)
# #         self.classifier = nn.Sequential(
# #             nn.Linear(total_dim, 256),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(256, 2)   # 二分类
# #         )
# #
# #     def forward(self, features_dict):
# #         """
# #         features_dict: {model_name: tensor (batch, seq_len, dim)}
# #         """
# #         outputs = []
# #         for name, x in features_dict.items():
# #             # 如果输入包含特殊token（如RNAFM长度为43），可选择截取中间41个
# #             # if name == 'rnafm':
# #             #     x = x[:, 1:-1, :]   # 去掉首尾特殊token
# #             out = self.subnets[name](x)   # (batch, sub_hidden)
# #             outputs.append(out)
# #         concat = torch.cat(outputs, dim=-1)  # (batch, total_dim)
# #         logits = self.classifier(concat)
# #         return logits
# #
# #
# # #新增门控融合分类器,通过config中的开关决定是否要用门控
# # class GatedFusionClassifier(nn.Module):
# #     """
# #     门控融合分类器：每个模型独立经过子网络后，通过门控机制加权融合，再经小型MLP分类
# #     """
# #     def __init__(self, model_dims, sub_hidden=128, num_heads=4, dropout=0.1):
# #         super().__init__()
# #         self.subnets = nn.ModuleDict()
# #         for name, dim in model_dims.items():
# #             self.subnets[name] = SubNetwork(dim, hidden_dim=sub_hidden,
# #                                             num_heads=num_heads, dropout=dropout)
# #         num_models = len(model_dims)
# #
# #         # 门控网络：输入为所有子网络输出的拼接，输出为每个模型的权重（sigmoid）
# #         self.gate = nn.Linear(sub_hidden * len(model_dims), len(model_dims))
# #         #self.gate = nn.Linear(sub_hidden * num_models, num_models)
# #
# #         # 小型分类头
# #         self.classifier = nn.Sequential(
# #             nn.Linear(sub_hidden, 64),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(64, 2)
# #         )
# #         self.dropout = nn.Dropout(dropout)
# #
# #     def forward(self, features_dict):
# #         # 1. 每个模型经过子网络
# #         outputs = [self.subnets[name](x) for name, x in features_dict.items()]  # list of (batch, sub_hidden)
# #         # 2. 拼接
# #         concat = torch.cat(outputs, dim=-1)  # (batch, sub_hidden * num_models)
# #         # 3. 计算门控权重（sigmoid）
# #         gates = torch.sigmoid(self.gate(concat))  # (batch, num_models)
# #         # 4. 加权融合
# #         fused = sum(gates[:, i].unsqueeze(-1) * outputs[i] for i in range(len(outputs)))  # (batch, sub_hidden)
# #         # 5. 分类
# #         logits = self.classifier(fused)
# #         return logits
# #
# # class SingleModelClassifier(nn.Module):
# #     """仅使用单个模型的子网络 + 分类层（接收字典输入）"""
# #     def __init__(self, model_name, input_dim, sub_hidden=128, num_heads=4, dropout=0.1):
# #         super().__init__()
# #         self.model_name = model_name
# #         self.subnet = SubNetwork(input_dim, hidden_dim=sub_hidden,
# #                                  num_heads=num_heads, dropout=dropout)
# #         self.classifier = nn.Sequential(
# #             nn.Linear(sub_hidden, 64),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(64, 2)
# #         )
# #
# #     def forward(self, features_dict):
# #         # features_dict 中只包含 self.model_name 对应的特征
# #         x = features_dict[self.model_name]   # (batch, seq_len, input_dim)
# #         out = self.subnet(x)                  # (batch, sub_hidden)
# #         logits = self.classifier(out)         # (batch, 2)
# #         return logits