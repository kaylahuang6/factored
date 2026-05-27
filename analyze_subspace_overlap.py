import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from generator import (
    build_independent_factors,
    pack_subtokens,
    BASE_VOCAB,
    BOS,
)

from model import TinyTransformerLM


VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8
OBS_LEN = SEQ_LEN - 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_factor_subtokens(factor, length, rng):
    """
    用 generator.py 裡的 FactorProcess 規則，
    產生單一 factor 的 sub-token sequence。
    """
    belief = factor.init.copy()
    zs = []

    for _ in range(length):
        z = factor.sample_obs(belief, rng)
        zs.append(z)
        belief = factor.update(belief, z)

    return np.array(zs, dtype=np.int64)


def make_input_from_subtokens(Z):
    """
    Z shape = [5, OBS_LEN]

    使用 generator.py 裡的 pack_subtokens，
    確保 token 編碼跟訓練時完全一樣。
    """
    obs_tokens = []

    for t in range(OBS_LEN):
        subtokens_t = Z[:, t]
        token = pack_subtokens(subtokens_t)
        obs_tokens.append(token)

    obs_tokens = np.array(obs_tokens, dtype=np.int64)

    X = np.empty(SEQ_LEN, dtype=np.int64)
    X[0] = BOS
    X[1:] = obs_tokens

    return X


@torch.no_grad()
def collect_vary_one_hidden(
    model,
    factor_id,
    n_fixed=100,
    n_var=64,
    seed=1234,
    position="last",
):
    """
    vary-one-factor dataset:

    固定其他四個 factors，
    只變動其中一個 factor，
    然後收集 hidden states。

    position="last" 只取最後一個 token 位置，比較乾淨。
    """
    model.eval()
    rng = np.random.default_rng(seed + factor_id * 10000)

    factors = build_independent_factors()

    H_all = []

    for fixed_id in range(n_fixed):
        fixed_Z = {}

        # 固定其他四個 factors 的 sub-token sequences
        for f in range(5):
            if f != factor_id:
                fixed_Z[f] = sample_factor_subtokens(
                    factors[f],
                    OBS_LEN,
                    rng,
                )

        X_batch = []

        for _ in range(n_var):
            Z = np.zeros((5, OBS_LEN), dtype=np.int64)

            for f in range(5):
                if f == factor_id:
                    Z[f] = sample_factor_subtokens(
                        factors[f],
                        OBS_LEN,
                        rng,
                    )
                else:
                    Z[f] = fixed_Z[f]

            X = make_input_from_subtokens(Z)
            X_batch.append(X)

        X_batch = np.stack(X_batch, axis=0)
        x = torch.tensor(X_batch, dtype=torch.long, device=DEVICE)

        logits, h = model(x, return_hidden=True)
        h_np = h.detach().cpu().numpy()

        if position == "last":
            # 只取最後一個位置
            h_np = h_np[:, -1, :]      # [n_var, d_model]
            h_centered = h_np - h_np.mean(axis=0, keepdims=True)
            H_all.append(h_centered)

        elif position == "all":
            # 取所有非 BOS 位置
            h_np = h_np[:, 1:, :]      # [n_var, OBS_LEN, d_model]
            h_centered = h_np - h_np.mean(axis=0, keepdims=True)
            H_all.append(h_centered.reshape(-1, h_centered.shape[-1]))

        else:
            raise ValueError("position must be 'last' or 'all'")

    H = np.concatenate(H_all, axis=0)
    print(f"factor {factor_id + 1} hidden shape:", H.shape)

    return H


def pca_basis(X, k=2):
    """
    對 vary-one hidden states 做 PCA，
    取前 k 個 principal components 作為 factor subspace。
    """
    Xc = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    Q = Vt[:k].T

    var = S ** 2
    ratio = var / var.sum()
    cev = np.cumsum(ratio)

    return Q, ratio, cev


def subspace_overlap(QA, QB):
    """
    overlap(A, B) = (1 / d_min) * sum sigma_i^2

    overlap 接近 0：比較正交
    overlap 接近 1：高度重疊
    """
    M = QA.T @ QB
    sigma = np.linalg.svd(M, compute_uv=False)

    d_min = min(QA.shape[1], QB.shape[1])

    return float(np.sum(sigma ** 2) / d_min)


def compute_overlap_matrix(bases):
    n = len(bases)
    overlap = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            overlap[i, j] = subspace_overlap(bases[i], bases[j])

    return overlap


def main():
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=120,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=SEQ_LEN,
        dropout=0.0,
    ).to(DEVICE)

    ckpt_path = "tiny_transformer_independent.pt"
    state = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    print("model loaded:", ckpt_path)

    os.makedirs("figures", exist_ok=True)

    k = 2
    bases = []

    for factor_id in range(5):
        H = collect_vary_one_hidden(
            model=model,
            factor_id=factor_id,
            n_fixed=100,
            n_var=64,
            seed=1234,
            position="last",
        )

        Q, ratio, cev = pca_basis(H, k=k)
        bases.append(Q)

        print(f"\nFactor {factor_id + 1}")
        print("Top 10 explained variance ratio:")
        print(np.round(ratio[:10], 4))
        print("CEV top 2:", round(float(cev[1]), 4))

    overlap = compute_overlap_matrix(bases)

    print("\nSubspace overlap matrix, k = 2")
    print(np.round(overlap, 4))

    off_diag = overlap[~np.eye(overlap.shape[0], dtype=bool)]

    print("\nMean off-diagonal overlap:", round(float(off_diag.mean()), 4))
    print("Max off-diagonal overlap:", round(float(off_diag.max()), 4))
    print("Min off-diagonal overlap:", round(float(off_diag.min()), 4))

    plt.figure(figsize=(6, 5))
    plt.imshow(overlap, vmin=0, vmax=1)
    plt.colorbar(label="subspace overlap")

    plt.xticks(range(5), [f"F{i+1}" for i in range(5)])
    plt.yticks(range(5), [f"F{i+1}" for i in range(5)])
    plt.title("Pairwise Subspace Overlap, k=2")

    for i in range(5):
        for j in range(5):
            plt.text(
                j,
                i,
                f"{overlap[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig("figures/subspace_overlap_k2.png", dpi=300)
    plt.show()

    print("\nsaved: figures/subspace_overlap_k2.png")


if __name__ == "__main__":
    main()