import os
import torch
import numpy as np
from Bio import SeqIO
import fm

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 32  # 根据你的GPU显存调整，如果CPU可设为8或更小
output_dir = "features/rnafm/RNAFM_token"
os.makedirs(output_dir, exist_ok=True)

print("Loading RNA-FM RNA_LLM...")
model, alphabet = fm.pretrained.rna_fm_t12()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

fasta_files = [
    "dataset/train/negtrain.fasta",
    "dataset/train/postrain.fasta",
    "dataset/test/negtest.fasta",
    "dataset/test/postest.fasta"
]

for fasta_path in fasta_files:
    print(f"Processing {fasta_path}...")
    subset_name = fasta_path.replace("dataset/", "").replace("/", "_").replace(".fasta", "")

    # 读取所有序列
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [rec.id for rec in records]
    sequences = [str(rec.seq).upper().replace("T", "U") for rec in records]
    n_seq = len(sequences)
    print(f"Total sequences: {n_seq}")

    all_embeddings = []

    # 分批处理
    for i in range(0, n_seq, batch_size):
        batch_seq_ids = seq_ids[i:i+batch_size]
        batch_sequences = sequences[i:i+batch_size]
        batch_data = list(zip(batch_seq_ids, batch_sequences))

        _, _, batch_tokens = batch_converter(batch_data)  # shape: (batch, seq_len)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

        token_embeddings = results["representations"][12].cpu().numpy()  # (batch, 41, 640)
        all_embeddings.append(token_embeddings)

        print(f"  Processed batch {i//batch_size + 1}/{(n_seq-1)//batch_size + 1}")

    if all_embeddings:
        token_embeddings = np.concatenate(all_embeddings, axis=0)
        np.savez(os.path.join(output_dir, f"{subset_name}.npz"),
                 embeddings=token_embeddings,
                 ids=seq_ids)
        print(f"Saved {len(seq_ids)} sequences for {subset_name}, token shape {token_embeddings.shape[1:]}")
    else:
        print(f"No sequences found in {fasta_path}")

print("All RNA-FM token features extracted!")