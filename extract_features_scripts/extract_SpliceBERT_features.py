import os
import torch
import numpy as np
from Bio import SeqIO
#from transformers import BertTokenizer, BertModel
from multimolecule import RnaTokenizer, SpliceBertModel

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 32
output_dir = "features/SpliceBERT/SpliceBERT_token"
os.makedirs(output_dir, exist_ok=True)

# 选择正确的模型名称
# 推荐使用 multimolecule 版本，因为兼容性好
model_name = "multimolecule/splicebert"  # 或 "multimolecule/splicebert.510", "yangheng/SpliceBERT-510nt"
print(f"Loading model: {model_name}")

# 如果遇到网络问题，启用镜像站（取消下面一行的注释）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# print("Loading tokenizer...")
# tokenizer = BertTokenizer.from_pretrained(model_name, trust_remote_code=True)
# print("Loading model...")
# model = BertModel.from_pretrained(model_name, trust_remote_code=True).to(device)
# model.eval()

tokenizer = RnaTokenizer.from_pretrained("multimolecule/splicebert")
model = SpliceBertModel.from_pretrained("multimolecule/splicebert").to(device)
model.eval()

fasta_files = [
    "dataset/train/negtrain.fasta",
    "dataset/train/postrain.fasta",
    "dataset/test/negtest.fasta",
    "dataset/test/postest.fasta"
]

for fasta_path in fasta_files:
    print(f"\nProcessing {fasta_path}...")
    subset_name = fasta_path.replace("dataset/", "").replace("/", "_").replace(".fasta", "")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [rec.id for rec in records]
    sequences = [str(rec.seq).upper() for rec in records]
    n_seq = len(sequences)
    print(f"Total sequences: {n_seq}")

    all_embeddings = []

    for i in range(0, n_seq, batch_size):
        batch_sequences = sequences[i:i+batch_size]

        # 分词：不加特殊 token，保证输出长度 = 41
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True,
                           truncation=True, max_length=41, add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state.cpu().numpy()
        all_embeddings.append(token_embeddings)

        print(f"  Batch {i//batch_size + 1}/{(n_seq-1)//batch_size + 1}")

    if all_embeddings:
        token_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"Hidden dimension: {token_embeddings.shape[2]}")  # 通常为 768

        np.savez(os.path.join(output_dir, f"{subset_name}.npz"),
                 embeddings=token_embeddings,
                 ids=seq_ids)
        print(f"Saved {len(seq_ids)} sequences for {subset_name}, token shape {token_embeddings.shape[1:]}")
    else:
        print(f"No sequences found in {fasta_path}")

print("\nAll SpliceBERT token features extracted!")