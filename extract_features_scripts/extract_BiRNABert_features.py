#基于本地文件
import os
import torch
import numpy as np
from Bio import SeqIO
import json

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
OUTPUT_DIR = "features/BiRNA-BERT/birnabert_token"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 路径定义
BASE_MODEL_PATH = "RNA_LLM/BiRNA-BERT-main"
TOKENIZER_PATH = os.path.join(BASE_MODEL_PATH, "TOKENIZER")

print(f"模型基础路径: {BASE_MODEL_PATH}")
print(f"Tokenizer路径: {TOKENIZER_PATH}")


def check_model_structure():
    """检查模型文件结构"""
    print("\n检查模型结构:")

    # 检查必要文件
    required_model_files = ["pytorch_model.bin", "config.json"]
    required_tokenizer_files = ["tokenizer_config.json", "vocab.txt", "tokenizer.json"]

    print("1. 模型主目录:")
    for file in required_model_files:
        file_path = os.path.join(BASE_MODEL_PATH, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ 缺失: {file}")

    print("\n2. Tokenizer目录:")
    if os.path.exists(TOKENIZER_PATH):
        for file in required_tokenizer_files:
            file_path = os.path.join(TOKENIZER_PATH, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ 缺失: {file}")
    else:
        print(f"  Tokenizer目录不存在")


def load_model_with_separate_paths():
    """分别加载tokenizer和模型"""
    from transformers import AutoTokenizer, AutoModel

    print("\n加载模型和tokenizer...")

    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    if os.path.exists(TOKENIZER_PATH):
        # 从TOKENIZER子目录加载
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True
        )
        print(f"  ✓ Tokenizer从 {TOKENIZER_PATH} 加载")
    else:
        # 尝试从主目录加载
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        print(f"  ✓ Tokenizer从 {BASE_MODEL_PATH} 加载")

    # 2. 加载模型
    print("2. 加载模型...")
    # 首先修复 config.json
    config_path = os.path.join(BASE_MODEL_PATH, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 确保有 model_type
        if "model_type" not in config:
            print("  修复 config.json: 添加 model_type 字段")
            config["model_type"] = "bert"

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    # 加载模型
    try:
        model = AutoModel.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        print(f"  ✓ 模型从 {BASE_MODEL_PATH} 加载")
    except Exception as e:
        print(f"  ✗ AutoModel失败: {e}")
        # 尝试手动加载
        from transformers import BertModel, BertConfig

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = BertConfig.from_dict(config_dict)
        model = BertModel(config)

        # 加载权重
        model_path = os.path.join(BASE_MODEL_PATH, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ 手动加载BertModel成功")

    model = model.to(DEVICE)
    model.eval()

    return tokenizer, model


def main():
    """主函数"""
    # 1. 检查结构
    check_model_structure()

    # 2. 加载模型
    tokenizer, model = load_model_with_separate_paths()

    if tokenizer is None or model is None:
        print("模型加载失败")
        return

    # 3. 验证
    print("\n验证加载:")
    print(f"  Tokenizer类型: {type(tokenizer).__name__}")
    print(f"  模型类型: {type(model).__name__}")

    if hasattr(model, 'config'):
        print(f"  隐藏层维度: {model.config.hidden_size}")
        print(f"  词汇表大小: {model.config.vocab_size}")

    # 4. 处理数据
    fasta_files = [
        "dataset/train/negtrain.fasta",
        "dataset/train/postrain.fasta",
        "dataset/test/negtest.fasta",
        "dataset/test/postest.fasta"
    ]

    for fasta_path in fasta_files:
        print(f"\n{'=' * 60}")
        print(f"处理: {fasta_path}")

        if not os.path.exists(fasta_path):
            print(f"文件不存在: {fasta_path}")
            continue

        subset_name = os.path.basename(fasta_path).replace(".fasta", "")
        records = list(SeqIO.parse(fasta_path, "fasta"))
        seq_ids = [rec.id for rec in records]

        # BiRNA-BERT 需要空格分隔
        sequences = [' '.join(str(rec.seq).upper()) for rec in records]

        all_embeddings = []

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_sequences = sequences[i:i + BATCH_SIZE]
            print(f"  批次 {i // BATCH_SIZE + 1}: {len(batch_sequences)} 条序列")

            # Tokenization
            inputs = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False
            )

            # 验证 token 数量
            input_shape = inputs["input_ids"].shape
            print(f"    输入形状: {input_shape}")

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # 特征提取
            with torch.no_grad():
                outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.cpu().numpy()
            all_embeddings.append(embeddings)

            print(f"    输出形状: {embeddings.shape}")

        if all_embeddings:
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"\n最终形状: {final_embeddings.shape}")

            # 确保每个序列是 41 个位置
            if final_embeddings.shape[1] > 41:
                final_embeddings = final_embeddings[:, :41, :]
                print(f"截断为: {final_embeddings.shape}")

            # 保存
            output_path = os.path.join(OUTPUT_DIR, f"{subset_name}.npz")
            np.savez(output_path, embeddings=final_embeddings, ids=seq_ids)
            print(f"保存到: {output_path}")

    print(f"\n{'=' * 60}")
    print("BiRNA-BERT特征提取完成!")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()






# import os
# import torch
# import numpy as np
# from Bio import SeqIO
# from transformers import AutoTokenizer, AutoModel
#
# # 配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# batch_size = 32
# output_dir = "features/BiRNA-BERT/birnabert_token"
# os.makedirs(output_dir, exist_ok=True)
#
# # 设置镜像（可选）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#
# model_name = "buetnlpbio/birna-bert"
#
# print("Loading BiRNA-BERT model and tokenizer...")
# # 使用 AutoTokenizer 并禁用 fast 版本
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
# model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
# model.eval()
#
# fasta_files = [
#     "dataset/train/negtrain.fasta",
#     "dataset/train/postrain.fasta",
#     "dataset/test/negtest.fasta",
#     "dataset/test/postest.fasta"
# ]
#
# for fasta_path in fasta_files:
#     print(f"\nProcessing {fasta_path}...")
#     subset_name = fasta_path.replace("dataset/", "").replace("/", "_").replace(".fasta", "")
#
#     records = list(SeqIO.parse(fasta_path, "fasta"))
#     seq_ids = [rec.id for rec in records]
#     # 关键：将序列转换为空格分隔的字符串，例如 "A C G T U ..."
#     sequences = [' '.join(str(rec.seq).upper()) for rec in records]  # 不替换T/U
#     n_seq = len(sequences)
#     print(f"Total sequences: {n_seq}")
#
#     all_embeddings = []
#
#     for i in range(0, n_seq, batch_size):
#         batch_sequences = sequences[i:i+batch_size]
#
#         # 分词：不添加特殊token？但官方示例中没有禁用，输出会包含 [CLS] 和 [SEP]
#         # 为了与原始长度对齐，我们可以在后续切片去掉它们
#         inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True,
#                            truncation=True, max_length=512)  # 用足够大的max_length
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = model(**inputs)
#
#         # 获取token嵌入，形状 (batch, seq_len, hidden_size)
#         token_embeddings = outputs.last_hidden_state.cpu().numpy()
#         all_embeddings.append(token_embeddings)
#
#         print(f"  Batch {i//batch_size + 1}/{(n_seq-1)//batch_size + 1}, shape: {token_embeddings.shape}")
#
#     if all_embeddings:
#         token_embeddings = np.concatenate(all_embeddings, axis=0)
#         print(f"Full shape before trimming: {token_embeddings.shape}")
#
#         # 去掉首尾特殊token（假设模型添加了 [CLS] 和 [SEP]）
#         # 我们需要确认 seq_len 是否等于 41+2=43。如果等于43，则切片；否则不做处理
#         if token_embeddings.shape[1] == 43:
#             token_embeddings = token_embeddings[:, 1:-1, :]
#             print(f"Trimmed to shape: {token_embeddings.shape}")
#         elif token_embeddings.shape[1] == 41:
#             print("No special tokens added, keeping original length.")
#         else:
#             print(f"Unexpected sequence length: {token_embeddings.shape[1]}, may need adjustment.")
#
#         np.savez(os.path.join(output_dir, f"{subset_name}.npz"),
#                  embeddings=token_embeddings,
#                  ids=seq_ids)
#         print(f"Saved {len(seq_ids)} sequences for {subset_name}, final shape {token_embeddings.shape}")
#     else:
#         print(f"No sequences found in {fasta_path}")
#
# print("\nAll BiRNA-BERT token features extracted!")