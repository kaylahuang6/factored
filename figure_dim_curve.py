import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from generator import sample_batch, BASE_VOCAB
from model import TinyTransformerLM

VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_from_ckpt(path):
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=120,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_len=SEQ_LEN,
        dropout=0.0,
    ).to(DEVICE)

    obj = torch.load(path, map_location=DEVICE)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        step = obj.get("step", -1)
    else:
        model.load_state_dict(obj)
        step = -1

    model.eval()
    return model, step


@torch.no_grad()
def collect_hidden(model, num_batches=6, batch_size=128, seq_len=9):
    """
    論文 Appendix F 的近似版：
    收集 final hidden activations，並把 batch 與 sequence positions 全部攤平。
    這裡用的是你目前模型能取到的最後 hidden state。
    """
    H_list = []

    for b in range(num_batches):
        X_np, Y_np = sample_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=20000 + b
        )

        x = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
        _, h = model(x, return_hidden=True)   # [B, T, d_model]

        # 攤平成 [B*T, d_model]
        H_list.append(h.cpu().numpy().reshape(-1, h.shape[-1]))

    H = np.concatenate(H_list, axis=0)
    return H


def pca_stats(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    lam = (S ** 2) / (Xc.shape[0] - 1)
    ratio = lam / lam.sum()
    cum = np.cumsum(ratio)
    return ratio, cum


def dims_for_threshold(cum, threshold=0.95):
    return int(np.searchsorted(cum, threshold) + 1)


def main():
    paths = sorted(glob.glob("checkpoints/ckpt_*.pt"))

    if not paths:
        print("No checkpoints found in checkpoints/")
        return

    steps = []
    dims95 = []
    dims98 = []

    print("device:", DEVICE)
    print(f"found {len(paths)} checkpoints")

    for path in paths:
        model, step = load_model_from_ckpt(path)
        H = collect_hidden(model, num_batches=6, batch_size=128, seq_len=9)

        _, cum = pca_stats(H)
        d95 = dims_for_threshold(cum, 0.95)
        d98 = dims_for_threshold(cum, 0.98)

        steps.append(step)
        dims95.append(d95)
        dims98.append(d98)

        print(f"step = {step:5d} | dim95 = {d95:3d} | dim98 = {d98:3d}")

    steps = np.array(steps)
    dims95 = np.array(dims95)
    dims98 = np.array(dims98)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, dims95, marker="o", label="95% variance")
    plt.plot(steps, dims98, marker="s", label="98% variance")
    plt.axhline(10, linestyle="--", label="factored prediction (10)")
    plt.xlabel("training steps")
    plt.ylabel("dimensions")
    plt.title("Dimensions for explained variance over training")
    plt.grid(True)
    plt.legend()
    plt.savefig("dim_curve_95.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("saved: dim_curve_95.png")


if __name__ == "__main__":
    main()