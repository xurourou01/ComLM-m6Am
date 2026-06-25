# data_utils.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

def load_features(model_name, data_type, label_type):
    """
    加载指定模型的特征数据
    model_name: 模型名称（与MODELS中一致）
    data_type: 'train' 或 'test'
    label_type: 'pos' 或 'neg'
    """
    if model_name == 'ERNIE-RNA':
        sub_dir = 'ERNIE-RNA/ernie_token'
        key = 'token_embeddings'
    elif model_name == 'rnafm':
        sub_dir = 'rnafm/RNAFM_token'
        key = 'embeddings'
    elif model_name == 'SpliceBERT':
        sub_dir = 'SpliceBERT/SpliceBERT_token'
        key = 'embeddings'
    elif model_name == 'BiRNA-BERT':
        sub_dir = 'BiRNA-BERT/birnabert_token'
        key = 'embeddings'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    filename = f"{data_type}_{label_type}{data_type}.npz"  # e.g., train_postrain.npz
    filepath = os.path.join(config.FEATURES_DIR, sub_dir, filename)
    data = np.load(filepath, allow_pickle=True)
    return data[key]


def load_all_data():
    """
    加载 config.USE_MODELS 中指定的所有模型的训练和测试数据，返回字典
    """
    train_pos = {}
    train_neg = {}
    test_pos = {}
    test_neg = {}

    for model in config.USE_MODELS:
        train_pos[model] = load_features(model, 'train', 'pos')
        train_neg[model] = load_features(model, 'train', 'neg')
        test_pos[model]  = load_features(model, 'test', 'pos')
        test_neg[model]  = load_features(model, 'test', 'neg')

    return train_pos, train_neg, test_pos, test_neg


class MultiFeatureDataset(Dataset):
    """自定义数据集，返回每个样本的模型特征字典和标签"""
    def __init__(self, pos_features_dict, neg_features_dict, pos_indices, neg_indices):
        """
        pos_features_dict: dict {model_name: np.array} 正样本特征
        neg_features_dict: dict {model_name: np.array} 负样本特征
        pos_indices: list of int 正样本索引
        neg_indices: list of int 负样本索引
        """
        self.samples = []
        # 正样本
        for idx in pos_indices:
            feat = {name: torch.from_numpy(pos_features_dict[name][idx]).float()
                    for name in pos_features_dict}
            self.samples.append((feat, 1))
        # 负样本
        for idx in neg_indices:
            feat = {name: torch.from_numpy(neg_features_dict[name][idx]).float()
                    for name in neg_features_dict}
            self.samples.append((feat, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_downsampled_subsets(neg_count, pos_count, n_subsets=10, seed=None):
    """
    将负样本索引随机分成 n_subsets 份（每份大小等于 pos_count），返回索引列表
    """
    if seed is not None:
        np.random.seed(seed)
    neg_indices = np.arange(neg_count)
    np.random.shuffle(neg_indices)
    split_size = pos_count
    subsets = [neg_indices[i*split_size:(i+1)*split_size].tolist() for i in range(n_subsets)]
    return subsets