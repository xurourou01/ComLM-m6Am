# config.py

import torch
import os

# 路径设置
FEATURES_DIR = 'features'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 模型选择
#单个模型进行消融实验
#USE_MODELS = ['rnafm']
#USE_MODELS = ['SpliceBERT']
#USE_MODELS = ['BiRNA-BERT']
#两个模型组合
#USE_MODELS = ['rnafm', 'SpliceBERT']
#USE_MODELS = ['rnafm', 'BiRNA-BERT']
#USE_MODELS = ['SpliceBERT', 'BiRNA-BERT']
#三个模型
USE_MODELS = ['rnafm', 'SpliceBERT', 'BiRNA-BERT']

# 各模型的特征维度（输入维度）
MODEL_DIMS = {
    'rnafm': 640,
    'SpliceBERT': 512,
    'BiRNA-BERT': 768
}

# 子网络类型：'full'（默认）, 'none', 'cnn', 'attention'
SUB_NETWORK_TYPE = 'full'   # 改为 'none', 'cnn', 'attention' 进行对比实验

#门控融合开关
USE_GATED_FUSION = False  # 设为True使用门控融合，False使用原有多模型融合

# 子网络参数
SUB_HIDDEN = 128          # 子网络输出特征维度
NUM_HEADS = 4              # 多头注意力头数
DROPOUT = 0.2           #原为0.1

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4      #原为1e-4
PATIENCE = 5
RANDOM_SEED = 666      #随机种子0、1、2、3、42、123、3407、2024、256、666



# 学习率调度配置
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'cosine'  # 'step', 'cosine', 'reduce_on_plateau'
LR_STEP_SIZE = 10
LR_GAMMA = 0.5
LR_PATIENCE = 3
LR_FACTOR = 0.5

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加强正则化
# SUB_HIDDEN = 64          # 子网络输出特征维度
# NUM_HEADS = 2              # 多头注意力头数，原为4
# DROPOUT = 0.5        #原为0.1

# # 新增：优化器正则化
# WEIGHT_DECAY = 1e-4          # L2 正则化强度
# LABEL_SMOOTHING = 0.1        # 标签平滑，可减轻过拟合