
import numpy as np
import torch

# Configuration
batch_size = 4
block_size = 8

# Load binary datasets
train_data = np.memmap(
    "train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    "val.bin",
    dtype=np.uint16,
    mode="r"
)

def get_batch(split):

    data = train_data if split == "train" else val_data

    # Random starting positions
    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        torch.from_numpy(
            data[i:i + block_size].astype(np.int64)
        )
        for i in ix
    ])

    # Next-token targets
    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + block_size + 1].astype(np.int64)
        )
        for i in ix
    ])

    return x, y


# Test
xb, yb = get_batch("train")

print("Input shape:", xb.shape)
print("Target shape:", yb.shape)

print(xb)
print(yb)
