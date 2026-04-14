import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from parallel_utils import ring_all_reduce

def worker(rank, left_conn, right_conn, world_size, model_fn, dataset):

    model = model_fn()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # split dataset
    size = len(dataset) // world_size
    subset = Subset(dataset, range(rank * size, (rank + 1) * size))
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    # hook for gradient sync
    def make_hook():
        def hook(grad):
            return ring_all_reduce(grad, left_conn, right_conn, world_size)
        return hook

    for param in model.parameters():
        param.register_hook(make_hook())

    # training loop
    for epoch in range(5):
        total_loss = 0

        for X, y in loader:
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Rank {rank}] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    left_conn.close()
    right_conn.close()