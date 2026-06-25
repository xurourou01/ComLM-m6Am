# utils.py

import random
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, f1_score, recall_score)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred, y_prob):
    """
    计算所有评估指标
    y_true: 真实标签 (numpy array)
    y_pred: 预测标签 (numpy array)
    y_prob: 预测为正类的概率 (numpy array)
    返回字典：accuracy, auc, aupr, mcc, f1, sensitivity, specificity
    """
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # 召回率 = 灵敏度
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': acc,
        'auc': auc,
        'aupr': aupr,
        'mcc': mcc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }