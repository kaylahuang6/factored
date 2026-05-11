import os
import numpy as np
import matplotlib.pyplot as plt

LOSS_FILE = "loss_history.npy"

loss_history = np.load(LOSS_FILE)

print("loss_history 筆數:", len(loss_history))
print("first loss:", loss_history[0])
print("last loss:", loss_history[-1])

steps = np.arange(1, len(loss_history) + 1)

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(steps, loss_history)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curve")
plt.grid(True)

plt.savefig("figures/loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: figures/loss_curve.png")