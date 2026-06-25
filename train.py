#原有多模型组合

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import os

import config
import utils
import data_utils
import model

def train_one_epoch(net, loader, optimizer, criterion, device, is_multi):
    net.train()
    total_loss = 0
    for batch in loader:
        feat_dict, labels = batch
        for name in feat_dict:
            feat_dict[name] = feat_dict[name].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if is_multi:
            logits = net(feat_dict)
        else:
            logits = net(feat_dict)  # SingleModelClassifier 也接收字典
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(net, loader, criterion, device, is_multi):
    net.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            feat_dict, labels = batch
            for name in feat_dict:
                feat_dict[name] = feat_dict[name].to(device)
            labels = labels.to(device)
            if is_multi:
                logits = net(feat_dict)
            else:
                logits = net(feat_dict)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    utils.set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    is_multi = len(config.USE_MODELS) > 1   # 是否为多模型

    # 1. 加载数据（只加载 config.USE_MODELS 指定的模型）
    print("加载数据...")
    train_pos, train_neg, test_pos, test_neg = data_utils.load_all_data()
    P = len(train_pos[config.USE_MODELS[0]])   # 正样本数
    N = len(train_neg[config.USE_MODELS[0]])   # 负样本数
    print(f"训练集: 正样本 {P}, 负样本 {N}")

    # 2. 下采样创建10个子集的负样本索引
    neg_subsets = data_utils.create_downsampled_subsets(N, P, n_subsets=10, seed=config.RANDOM_SEED)

    # 3. 存储所有子集的交叉验证结果
    all_fold_results = []  # 每个元素是该子集10折的指标字典列表

    # 4. 外层循环：每个子集
    for subset_idx, neg_indices in enumerate(neg_subsets):
        print(f"\n=== 处理子训练集 {subset_idx+1}/10 ===")
        pos_indices = list(range(P))
        full_dataset = data_utils.MultiFeatureDataset(train_pos, train_neg,
                                                      pos_indices, neg_indices)
        labels_full = [1]*P + [0]*len(neg_indices)

        # 十折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_SEED)
        fold_metrics = []   # 存储该子集每折的指标

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_full)), labels_full)):
            print(f"    Fold {fold+1}/10")
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

            # 初始化模型
            if is_multi:
                if config.USE_GATED_FUSION:
                    net = model.GatedFusionClassifier(
                        model_dims={name: config.MODEL_DIMS[name] for name in config.USE_MODELS},
                        sub_hidden=config.SUB_HIDDEN,
                        num_heads=config.NUM_HEADS,
                        dropout=config.DROPOUT,
                        sub_type=config.SUB_NETWORK_TYPE  # 新增
                    ).to(device)
                else:
                    net = model.MultiModelClassifier(
                        model_dims={name: config.MODEL_DIMS[name] for name in config.USE_MODELS},
                        sub_hidden=config.SUB_HIDDEN,
                        num_heads=config.NUM_HEADS,
                        dropout=config.DROPOUT,
                        sub_type=config.SUB_NETWORK_TYPE  # 新增
                    ).to(device)
            else:
                model_name = config.USE_MODELS[0]
                net = model.SingleModelClassifier(
                    model_name=model_name,
                    input_dim=config.MODEL_DIMS[model_name],
                    sub_hidden=config.SUB_HIDDEN,
                    num_heads=config.NUM_HEADS,
                    dropout=config.DROPOUT,
                    sub_type=config.SUB_NETWORK_TYPE  # 新增
                ).to(device)

            optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0
            best_metrics = None
            best_model_path = os.path.join(config.CHECKPOINT_DIR,
                                           f"subset{subset_idx+1}_fold{fold+1}_best.pt")

            for epoch in range(config.EPOCHS):
                train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device, is_multi)
                val_loss, val_labels, val_preds, val_probs = validate(net, val_loader, criterion, device, is_multi)
                val_acc = (val_preds == val_labels).mean()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    # 计算该折最佳指标
                    best_metrics = utils.compute_metrics(val_labels, val_preds, val_probs)
                    # 保存模型
                    torch.save(net.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= config.PATIENCE:
                        print(f"        早停于 epoch {epoch+1}")
                        break

                if (epoch+1) % 10 == 0:
                    print(f"        Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

            print(f"    Fold {fold+1} 最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch+1})")
            fold_metrics.append(best_metrics)

        # 计算该子集10折的平均指标
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            avg_metrics[metric] = (np.mean(values), np.std(values))
        print(f"子集 {subset_idx+1} 10折平均指标:")
        for metric, (mean_val, std_val) in avg_metrics.items():
            print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
        all_fold_results.append(fold_metrics)

    # 5. 输出所有子集的总体平均指标
    print("\n=== 所有子集总体平均指标 ===")
    overall_metrics = {}
    for metric in fold_metrics[0].keys():
        all_values = []
        for subset_metrics in all_fold_results:
            for fold_m in subset_metrics:
                all_values.append(fold_m[metric])
        overall_metrics[metric] = (np.mean(all_values), np.std(all_values))
    for metric, (mean_val, std_val) in overall_metrics.items():
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()








# # 集成门控融合与小型MLP
#
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from sklearn.model_selection import StratifiedKFold
# from torch.utils.data import DataLoader, Subset
# import os
#
# import config
# import utils
# import data_utils
# import model
#
# def train_one_epoch(net, loader, optimizer, criterion, device, is_multi):
#     net.train()
#     total_loss = 0
#     for batch in loader:
#         feat_dict, labels = batch
#         for name in feat_dict:
#             feat_dict[name] = feat_dict[name].to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#         logits = net(feat_dict)   # 统一使用字典输入
#         loss = criterion(logits, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) #===新增梯度裁剪=====
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)
#
# def validate(net, loader, criterion, device, is_multi):
#     net.eval()
#     total_loss = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []
#     with torch.no_grad():
#         for batch in loader:
#             feat_dict, labels = batch
#             for name in feat_dict:
#                 feat_dict[name] = feat_dict[name].to(device)
#             labels = labels.to(device)
#             logits = net(feat_dict)
#             loss = criterion(logits, labels)
#             total_loss += loss.item()
#             probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             preds = torch.argmax(logits, dim=1).cpu().numpy()
#             all_probs.extend(probs)
#             all_preds.extend(preds)
#             all_labels.extend(labels.cpu().numpy())
#     return total_loss / len(loader), np.array(all_labels), np.array(all_preds), np.array(all_probs)
#
# def main():
#     utils.set_seed(config.RANDOM_SEED)
#     device = config.DEVICE
#     is_multi = len(config.USE_MODELS) > 1
#
#     # 1. 加载数据
#     print("加载数据...")
#     train_pos, train_neg, test_pos, test_neg = data_utils.load_all_data()
#     P = len(train_pos[config.USE_MODELS[0]])
#     N = len(train_neg[config.USE_MODELS[0]])
#     print(f"训练集: 正样本 {P}, 负样本 {N}")
#
#     # 2. 下采样创建10个子集的负样本索引
#     neg_subsets = data_utils.create_downsampled_subsets(N, P, n_subsets=10, seed=config.RANDOM_SEED)
#
#     # 3. 存储所有子集的交叉验证结果
#     all_fold_results = []
#
#     # 4. 外层循环：每个子集
#     for subset_idx, neg_indices in enumerate(neg_subsets):
#         print(f"\n=== 处理子训练集 {subset_idx+1}/10 ===")
#         pos_indices = list(range(P))
#         full_dataset = data_utils.MultiFeatureDataset(train_pos, train_neg,
#                                                       pos_indices, neg_indices)
#         labels_full = [1]*P + [0]*len(neg_indices)
#
#         # 十折交叉验证
#         skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_SEED)
#         fold_metrics = []
#
#         for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_full)), labels_full)):
#             print(f"    Fold {fold+1}/10")
#             train_subset = Subset(full_dataset, train_idx)
#             val_subset = Subset(full_dataset, val_idx)
#
#             train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
#             val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
#
#             # 初始化模型（根据是否多模型及是否门控选择）
#             if is_multi:
#                 if config.USE_GATED_FUSION:
#                     net = model.GatedFusionClassifier(
#                         model_dims={name: config.MODEL_DIMS[name] for name in config.USE_MODELS},
#                         sub_hidden=config.SUB_HIDDEN,
#                         num_heads=config.NUM_HEADS,
#                         dropout=config.DROPOUT
#                     ).to(device)
#                 else:
#                     net = model.MultiModelClassifier(
#                         model_dims={name: config.MODEL_DIMS[name] for name in config.USE_MODELS},
#                         sub_hidden=config.SUB_HIDDEN,
#                         num_heads=config.NUM_HEADS,
#                         dropout=config.DROPOUT
#                     ).to(device)
#             else:
#                 model_name = config.USE_MODELS[0]
#                 net = model.SingleModelClassifier(
#                     model_name=model_name,
#                     input_dim=config.MODEL_DIMS[model_name],
#                     sub_hidden=config.SUB_HIDDEN,
#                     num_heads=config.NUM_HEADS,
#                     dropout=config.DROPOUT
#                 ).to(device)
#
#             # optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
#             # criterion = nn.CrossEntropyLoss()
#
#             # ===== 修改点：添加权重衰减 weight_decay 和标签平滑 label_smoothing =====
#             optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
#             criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
#
#
#             best_val_acc = 0.0
#             best_epoch = 0
#             patience_counter = 0
#             best_metrics = None
#             # 模型保存路径：根据是否门控添加前缀
#             prefix = "gated_" if (is_multi and config.USE_GATED_FUSION) else ""
#             best_model_path = os.path.join(config.CHECKPOINT_DIR,
#                                            f"{prefix}subset{subset_idx+1}_fold{fold+1}_best.pt")
#
#             for epoch in range(config.EPOCHS):
#                 train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device, is_multi)
#                 val_loss, val_labels, val_preds, val_probs = validate(net, val_loader, criterion, device, is_multi)
#                 val_acc = (val_preds == val_labels).mean()
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_epoch = epoch
#                     patience_counter = 0
#                     best_metrics = utils.compute_metrics(val_labels, val_preds, val_probs)
#                     torch.save(net.state_dict(), best_model_path)
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= config.PATIENCE:
#                         print(f"        早停于 epoch {epoch+1}")
#                         break
#
#                 if (epoch+1) % 10 == 0:
#                     print(f"        Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#             print(f"    Fold {fold+1} 最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch+1})")
#             fold_metrics.append(best_metrics)
#
#         # 计算该子集10折的平均指标
#         avg_metrics = {}
#         for metric in fold_metrics[0].keys():
#             values = [m[metric] for m in fold_metrics]
#             avg_metrics[metric] = (np.mean(values), np.std(values))
#         print(f"子集 {subset_idx+1} 10折平均指标:")
#         for metric, (mean_val, std_val) in avg_metrics.items():
#             print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
#         all_fold_results.append(fold_metrics)
#
#     # 5. 输出所有子集的总体平均指标
#     print("\n=== 所有子集总体平均指标 ===")
#     overall_metrics = {}
#     for metric in fold_metrics[0].keys():
#         all_values = []
#         for subset_metrics in all_fold_results:
#             for fold_m in subset_metrics:
#                 all_values.append(fold_m[metric])
#         overall_metrics[metric] = (np.mean(all_values), np.std(all_values))
#     for metric, (mean_val, std_val) in overall_metrics.items():
#         print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
#
# if __name__ == "__main__":
#     main()
