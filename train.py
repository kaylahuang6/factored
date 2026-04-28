import os
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from generator import sample_batch, BASE_VOCAB
from model import TinyTransformerLM

VOCAB_SIZE = BASE_VOCAB + 1
SEQ_LEN = 8
BATCH_SIZE = 256
NUM_STEPS = 3000
PRINT_EVERY = 50
SAVE_EVERY = 100
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_history = []

    os.makedirs("checkpoints", exist_ok=True)

    print("device:", DEVICE)
    print("uniform baseline ~ log(432) =", round(math.log(BASE_VOCAB), 4))

    for step in range(1, NUM_STEPS + 1):
        X_np, Y_np = sample_batch(batch_size=BATCH_SIZE, seq_len=9, seed=step)

        x = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
        y = torch.tensor(Y_np, dtype=torch.long, device=DEVICE)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % SAVE_EVERY == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                },
                f"checkpoints/ckpt_{step:05d}.pt"
            )

        loss_history.append(loss.item())

        if step % PRINT_EVERY == 0 or step == 1:
            avg_loss = sum(loss_history[-50:]) / min(len(loss_history), 50)
            print(f"step {step:4d} | loss = {loss.item():.4f} | avg50 = {avg_loss:.4f}")

    torch.save(model.state_dict(), "tiny_transformer_independent.pt")
    print("training finished")
    print("saved: tiny_transformer_independent.pt")


if __name__ == "__main__":
    main()