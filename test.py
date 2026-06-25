# 通过子网络处理特征后，再合并模型（非直接集成）

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import config
import utils
import data_utils
import model
from train import train_one_epoch, validate   # 复用训练函数

def main():
    utils.set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    is_multi = len(config.USE_MODELS) > 1

    # 1. 加载数据
    print("加载数据...")
    train_pos, train_neg, test_pos, test_neg = data_utils.load_all_data()
    P = len(train_pos[config.USE_MODELS[0]])
    N = len(train_neg[config.USE_MODELS[0]])

    # 2. 子采样子集划分（与交叉验证相同）
    neg_subsets = data_utils.create_downsampled_subsets(N, P, n_subsets=10, seed=config.RANDOM_SEED)

    # 3. 为每个子集训练一个最终模型
    final_models = []
    for subset_idx, neg_indices in enumerate(neg_subsets):
        print(f"训练最终模型 {subset_idx+1}/10")
        pos_indices = list(range(P))
        full_dataset = data_utils.MultiFeatureDataset(train_pos, train_neg,
                                                      pos_indices, neg_indices)
        full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        # 初始化模型（根据是否多模型及是否门控选择）
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
                sub_type = config.SUB_NETWORK_TYPE
            ).to(device)

        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # # ===== 加强正则化修改点：添加 weight_decay 和 label_smoothing =====
        # optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)


        best_loss = float('inf')
        patience_counter = 0
        prefix = "gated_" if (is_multi and config.USE_GATED_FUSION) else ""
        best_model_path = os.path.join(config.CHECKPOINT_DIR, f"{prefix}final_subset{subset_idx+1}_best.pt")

        for epoch in range(config.EPOCHS):
            train_loss = train_one_epoch(net, full_loader, optimizer, criterion, device, is_multi)
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                torch.save(net.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"    早停于 epoch {epoch+1}")
                    break

            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{config.EPOCHS}, Loss: {train_loss:.4f}")

        # 加载最佳模型用于后续集成
        net.load_state_dict(torch.load(best_model_path))
        final_models.append(net)

    # 4. 构建测试集
    test_pos_indices = list(range(len(test_pos[config.USE_MODELS[0]])))
    test_neg_indices = list(range(len(test_neg[config.USE_MODELS[0]])))
    test_dataset = data_utils.MultiFeatureDataset(test_pos, test_neg,
                                                  test_pos_indices, test_neg_indices)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 5. 集成预测
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            feat_dict, labels = batch
            for name in feat_dict:
                feat_dict[name] = feat_dict[name].to(device)
            labels = labels.to(device)

            logits_sum = None
            for net in final_models:
                net.eval()
                logits = net(feat_dict)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
            avg_logits = logits_sum / len(final_models)
            probs = torch.softmax(avg_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 6. 计算测试集指标
    metrics = utils.compute_metrics(all_labels, all_preds, all_probs)
    print("\n=== 测试集评估结果 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()












# 原有多模型融合测试
#
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import os
#
# import config
# import utils
# import data_utils
# import model
# from train import train_one_epoch, validate   # 复用函数
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
#
#     # 2. 下采样子集划分（与交叉验证相同）
#     neg_subsets = data_utils.create_downsampled_subsets(N, P, n_subsets=10, seed=config.RANDOM_SEED)
#
#     # 3. 为每个子集训练一个最终模型
#     final_models = []
#     for subset_idx, neg_indices in enumerate(neg_subsets):
#         print(f"训练最终模型 {subset_idx+1}/10")
#         pos_indices = list(range(P))
#         full_dataset = data_utils.MultiFeatureDataset(train_pos, train_neg,
#                                                       pos_indices, neg_indices)
#         full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#
#         # 初始化模型
#         if is_multi:
#             net = model.MultiModelClassifier(
#                 model_dims={name: config.MODEL_DIMS[name] for name in config.USE_MODELS},
#                 sub_hidden=config.SUB_HIDDEN,
#                 num_heads=config.NUM_HEADS,
#                 dropout=config.DROPOUT
#             ).to(device)
#         else:
#             model_name = config.USE_MODELS[0]
#             net = model.SingleModelClassifier(
#                 model_name=model_name,
#                 input_dim=config.MODEL_DIMS[model_name],
#                 sub_hidden=config.SUB_HIDDEN,
#                 num_heads=config.NUM_HEADS,
#                 dropout=config.DROPOUT
#             ).to(device)
#
#         optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
#         criterion = nn.CrossEntropyLoss()
#
#         best_loss = float('inf')
#         patience_counter = 0
#         best_model_path = os.path.join(config.CHECKPOINT_DIR, f"final_subset{subset_idx+1}_best.pt")
#
#         for epoch in range(config.EPOCHS):
#             train_loss = train_one_epoch(net, full_loader, optimizer, criterion, device, is_multi)
#             if train_loss < best_loss:
#                 best_loss = train_loss
#                 patience_counter = 0
#                 torch.save(net.state_dict(), best_model_path)
#             else:
#                 patience_counter += 1
#                 if patience_counter >= config.PATIENCE:
#                     print(f"    早停于 epoch {epoch+1}")
#                     break
#
#             if (epoch+1) % 10 == 0:
#                 print(f"    Epoch {epoch+1}/{config.EPOCHS}, Loss: {train_loss:.4f}")
#
#         # 加载最佳模型用于后续集成
#         net.load_state_dict(torch.load(best_model_path))
#         final_models.append(net)
#
#     # 4. 构建测试集
#     test_pos_indices = list(range(len(test_pos[config.USE_MODELS[0]])))
#     test_neg_indices = list(range(len(test_neg[config.USE_MODELS[0]])))
#     test_dataset = data_utils.MultiFeatureDataset(test_pos, test_neg,
#                                                   test_pos_indices, test_neg_indices)
#     test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
#
#     # 5. 集成预测
#     all_labels = []
#     all_preds = []
#     all_probs = []
#     with torch.no_grad():
#         for batch in test_loader:
#             feat_dict, labels = batch
#             for name in feat_dict:
#                 feat_dict[name] = feat_dict[name].to(device)
#             labels = labels.to(device)
#
#             logits_sum = None
#             for net in final_models:
#                 net.eval()
#                 if is_multi:
#                     logits = net(feat_dict)
#                 else:
#                     logits = net(feat_dict)
#                 if logits_sum is None:
#                     logits_sum = logits
#                 else:
#                     logits_sum += logits
#             avg_logits = logits_sum / len(final_models)
#             probs = torch.softmax(avg_logits, dim=1)[:, 1].cpu().numpy()
#             preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
#             all_probs.extend(probs)
#             all_preds.extend(preds)
#             all_labels.extend(labels.cpu().numpy())
#
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)
#     all_probs = np.array(all_probs)
#
#     # 6. 计算测试集指标
#     metrics = utils.compute_metrics(all_labels, all_preds, all_probs)
#     print("\n=== 测试集评估结果 ===")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")
#
# if __name__ == "__main__":
#     main()