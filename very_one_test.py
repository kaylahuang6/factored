import numpy as np
import torch
import matplotlib.pyplot as plt

from generator import (
    build_independent_factors,
    pack_subtokens,
    BOS,
    BASE_VOCAB,
)
from model import TinyTransformerLM

VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8   # model input length = tokens[:-1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path="tiny_transformer_independent.pt"):
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=96,
        n_heads=3,
        n_layers=3,
        d_ff=256,
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


def sample_single_factor_trajectory(factor, T=8, seed=0):
    """
    對單一 factor 產生長度 T 的 subtoken trajectory
    """
    rng = np.random.default_rng(seed)
    belief = factor.init.copy()

    xs = []
    beliefs = [belief.copy()]

    for _ in range(T):
        x = factor.sample_obs(belief, rng)
        xs.append(int(x))
        belief = factor.update(belief, x)
        beliefs.append(belief.copy())

    return np.array(xs, dtype=np.int64), np.array(beliefs, dtype=np.float32)


def make_base_trajectories(T=8, seed=0):
    """
    為 5 個 factors 各自建立一條固定的 base trajectory
    """
    factors = build_independent_factors()
    trajs = []

    for i, f in enumerate(factors):
        xs, _ = sample_single_factor_trajectory(f, T=T, seed=seed + 1000 * i)
        trajs.append(xs)

    return trajs


def build_input_tokens_from_trajectories(trajs):
    """
    trajs: list of 5 arrays, each shape [T]
    轉成 packed token sequence，再取 model input = tokens[:-1]
    """
    T = len(trajs[0])
    tokens = [BOS]

    for t in range(T):
        subtokens = [trajs[i][t] for i in range(5)]
        tok = pack_subtokens(subtokens)
        tokens.append(tok)

    tokens = np.array(tokens, dtype=np.int64)   # length = T+1 = 9
    x = tokens[:-1]                             # model input length = 8
    return x


@torch.no_grad()
def collect_vary_one_hidden(model, vary_factor, num_samples=3000, T=8, base_seed=123):
    """
    固定其他 4 個 factors，只變動 vary_factor
    收集最後一個 hidden state: [num_samples, d_model]
    """
    factors = build_independent_factors()
    base_trajs = make_base_trajectories(T=T, seed=base_seed)

    X_list = []

    for n in range(num_samples):
        trajs = [arr.copy() for arr in base_trajs]

        # 只重抽第 vary_factor 個 factor
        xs_var, _ = sample_single_factor_trajectory(
            factors[vary_factor],
            T=T,
            seed=base_seed + 100000 + n
        )
        trajs[vary_factor] = xs_var

        x = build_input_tokens_from_trajectories(trajs)
        X_list.append(x)

    X = np.stack(X_list, axis=0)   # [N, 8]
    x = torch.tensor(X, dtype=torch.long, device=DEVICE)

    _, h = model(x, return_hidden=True)   # [N, 8, d_model]
    H = h[:, -1, :].cpu().numpy()         # 只取最後位置
    return H


def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    return Y


def plot_vary_one_panels(H_list):
    colors = ["#312e81", "#6b21a8", "#db2777", "#fb7185", "#fdba74"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for i, ax in enumerate(axes):
        xy = pca_2d(H_list[i])

        # 每個 panel 自己標準化，讓形狀更好看
        xy = (xy - xy.mean(axis=0, keepdims=True)) / (
            xy.std(axis=0, keepdims=True) + 1e-8
        )

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=3,
            alpha=0.20,
            c=colors[i],
            edgecolors="none"
        )
        ax.set_title(f"Factor {i+1}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    fig.suptitle("Vary-one-factor test (final hidden state)", fontsize=18)
    plt.tight_layout()
    plt.savefig("vary_one_panels.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("saved: vary_one_panels.png")


def main():
    model = load_model("tiny_transformer_independent.pt")
    print("model loaded")

    H_list = []
    for i in range(5):
        H = collect_vary_one_hidden(
            model,
            vary_factor=i,
            num_samples=3000,
            T=8,
            base_seed=123
        )
        H_list.append(H)
        print(f"factor {i+1} hidden shape: {H.shape}")

    plot_vary_one_panels(H_list)


if __name__ == "__main__":
    main()