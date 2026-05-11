import numpy as np
import torch
import matplotlib.pyplot as plt

from generator import sample_batch_with_beliefs, BASE_VOCAB
from model import TinyTransformerLM

VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path="tiny_transformer_independent.pt"):
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=120,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=SEQ_LEN,
        dropout=0.0,
    ).to(DEVICE)

    obj = torch.load(model_path, map_location=DEVICE)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)

    model.eval()
    return model


@torch.no_grad()
def collect_hidden_and_beliefs(model, num_batches=20, batch_size=256, seq_len=9):
    H_list = []
    B_list = []

    for b in range(num_batches):
        X_np, Y_np, B_np = sample_batch_with_beliefs(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=10000 + b
        )

        x = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
        _, h = model(x, return_hidden=True)

        H_list.append(h.cpu().numpy().reshape(-1, h.shape[-1]))   # [N, d_model]
        B_list.append(B_np.reshape(-1, B_np.shape[-1]))           # [N, 15]

    H = np.concatenate(H_list, axis=0)
    B = np.concatenate(B_list, axis=0)
    return H, B


def fit_linear_probe(H, B):
    H_mean = H.mean(axis=0, keepdims=True)
    H_std = H.std(axis=0, keepdims=True) + 1e-8
    B_mean = B.mean(axis=0, keepdims=True)
    B_std = B.std(axis=0, keepdims=True) + 1e-8

    H_z = (H - H_mean) / H_std
    B_z = (B - B_mean) / B_std

    H_aug = np.concatenate([H_z, np.ones((H_z.shape[0], 1))], axis=1)
    W, _, _, _ = np.linalg.lstsq(H_aug, B_z, rcond=None)
    B_hat_z = H_aug @ W

    return B_hat_z


def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    return Y


def plot_factor_panels(B_all, title, filename):
    colors = ["#312e81", "#6b21a8", "#db2777", "#fb7185", "#fdba74"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for i, ax in enumerate(axes):
        block = B_all[:, i*3:(i+1)*3]     # 每個 factor 的 3 維 block
        xy = pca_2d(block)                # 對這個 block 自己做 PCA 2D

        # 視覺上再標準化
        xy = (xy - xy.mean(axis=0, keepdims=True)) / (xy.std(axis=0, keepdims=True) + 1e-8)

        ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.18, c=colors[i], edgecolors="none")
        ax.set_title(f"Factor {i+1}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved: {filename}")


def main():
    model = load_model("tiny_transformer_independent.pt")
    H, B = collect_hidden_and_beliefs(model, num_batches=20, batch_size=256, seq_len=9)

    max_points = 15000
    if len(H) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(H), size=max_points, replace=False)
        H = H[idx]
        B = B[idx]

    # true factor belief geometry
    plot_factor_panels(B, "True factor belief geometry (per-factor PCA)", "factor_panels_true_pca.png")

    # hidden -> predicted belief, then per-factor PCA
    B_hat = fit_linear_probe(H, B)
    plot_factor_panels(B_hat, "Projections from hidden state (per-factor PCA)", "factor_panels_hidden_pca.png")


if __name__ == "__main__":
    main()