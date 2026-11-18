
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
batch_size = 32
block_size = 8
learning_rate = 0.001
max_iters = 2000

device = "cpu"

# Load data
import numpy as np

train_data = np.memmap(
    "train.bin", dtype=np.uint16, mode="r"
)

val_data = np.memmap(
    "val.bin", dtype=np.uint16, mode="r"
)

meta = __import__("pickle").load(
    open("meta.pkl", "rb")
)

vocab_size = len(meta["stoi"])


def get_batch(data):
    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        torch.tensor(
            data[i:i+block_size].astype(np.int64)
        )
        for i in ix
    ])

    y = torch.stack([
        torch.tensor(
            data[i+1:i+block_size+1].astype(np.int64)
        )
        for i in ix
    ])

    return x, y


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

for step in range(max_iters):

    xb, yb = get_batch(train_data)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item())
