import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from generator import sample_batch_with_beliefs, BASE_VOCAB
from model import TinyTransformerLM

VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def collect_hidden_and_beliefs(model, num_batches=20, batch_size=256, seq_len=9):
    model.eval()

    H_list = []
    B_list = []
    loss_list = []

    for b in range(num_batches):
        X_np, Y_np, B_np = sample_batch_with_beliefs(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=10000 + b
        )

        x = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
        y = torch.tensor(Y_np, dtype=torch.long, device=DEVICE)

        logits, h = model(x, return_hidden=True)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        H_list.append(h.cpu().numpy().reshape(-1, h.shape[-1]))
        B_list.append(B_np.reshape(-1, B_np.shape[-1]))
        loss_list.append(loss.item())

    H = np.concatenate(H_list, axis=0)
    B = np.concatenate(B_list, axis=0)

    print("hidden shape:", H.shape)
    print("belief shape:", B.shape)
    print("eval CE mean:", np.mean(loss_list))

    return H, B


def pca_explained_variance_ratio(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (Xc.shape[0] - 1)
    ratio = var / var.sum()
    return ratio


def linear_probe_r2(H, B, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(H))

    split = int(0.8 * len(H))
    tr_idx = idx[:split]
    te_idx = idx[split:]

    H_tr = H[tr_idx]
    H_te = H[te_idx]
    B_tr = B[tr_idx]
    B_te = B[te_idx]

    H_mean = H_tr.mean(axis=0, keepdims=True)
    H_std = H_tr.std(axis=0, keepdims=True) + 1e-8
    B_mean = B_tr.mean(axis=0, keepdims=True)
    B_std = B_tr.std(axis=0, keepdims=True) + 1e-8

    H_tr_z = (H_tr - H_mean) / H_std
    H_te_z = (H_te - H_mean) / H_std
    B_tr_z = (B_tr - B_mean) / B_std
    B_te_z = (B_te - B_mean) / B_std

    H_tr_aug = np.concatenate([H_tr_z, np.ones((H_tr_z.shape[0], 1))], axis=1)
    H_te_aug = np.concatenate([H_te_z, np.ones((H_te_z.shape[0], 1))], axis=1)

    W, _, _, _ = np.linalg.lstsq(H_tr_aug, B_tr_z, rcond=None)
    B_pred = H_te_aug @ W

    ss_res = ((B_te_z - B_pred) ** 2).sum(axis=0)
    ss_tot = ((B_te_z - B_te_z.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)

    r2_per_dim = 1.0 - ss_res / ss_tot
    return r2_per_dim


def main():
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=96,
        n_heads=3,
        n_layers=3,
        d_ff=256,
        max_len=SEQ_LEN,
        dropout=0.0,
    ).to(DEVICE)

    model.load_state_dict(torch.load("tiny_transformer_independent.pt", map_location=DEVICE))
    print("model loaded")

    H, B = collect_hidden_and_beliefs(model, num_batches=20, batch_size=256, seq_len=9)

    ratio = pca_explained_variance_ratio(H)
    cum_ratio = np.cumsum(ratio)

    for k in [1, 2, 3, 5, 10, 15, 20, 30]:
        if k <= len(cum_ratio):
            print(f"top {k:2d} PCs explain {cum_ratio[k-1]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(cum_ratio[:40], marker="o")
    plt.xlabel("number of principal components")
    plt.ylabel("cumulative explained variance")
    plt.title("PCA on hidden states")
    plt.grid(True)
    plt.show()

    r2 = linear_probe_r2(H, B)
    print("R^2 per belief dim:")
    print(np.round(r2, 4))
    print("mean R^2 (nanmean):", round(float(np.nanmean(r2)), 4))

    for i in range(5):
        block = r2[i*3:(i+1)*3]
        print(f"factor {i+1} mean R^2 = {np.nanmean(block):.4f} | dims = {np.round(block, 4)}")


if __name__ == "__main__":
    main()

import numpy as np
from generator import sample_batch_with_beliefs

X, Y, B = sample_batch_with_beliefs(batch_size=2000, seq_len=9, seed=0)
B = B.reshape(-1, B.shape[-1])   # [N, 15]

for i in range(5):
    block = B[:, i*3:(i+1)*3]
    Xc = block - block.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    print(f"factor {i+1} singular values:", np.round(S, 6))