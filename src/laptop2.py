import torch
import torch.nn as nn
import time
import socket

hostname = socket.gethostname()

def run_single_machine(world_size=2):
    torch.manual_seed(0)

    total_start = time.perf_counter()

    grads = []

    for rank in range(world_size):
        model = nn.Linear(1, 1, bias=False)

        x = torch.tensor([[rank + 1.0]])
        y = torch.tensor([[2.0]])

        # Forward
        fwd_start = time.perf_counter()
        output = model(x)
        loss = (output - y).pow(2).mean()
        fwd_time = time.perf_counter() - fwd_start

        # Backward
        bwd_start = time.perf_counter()
        loss.backward()
        bwd_time = time.perf_counter() - bwd_start

        grad = model.weight.grad.clone()
        grads.append(grad)

        print(f"[{hostname} | SimRank {rank}] Forward: {fwd_time:.6f}s")
        print(f"[{hostname} | SimRank {rank}] Backward: {bwd_time:.6f}s")
        print(f"[{hostname} | SimRank {rank}] Local grad: {grad}")

    # ---- Averaging (no communication) ----
    avg_start = time.perf_counter()

    total = sum(grads)
    avg_grad = total / world_size

    avg_time = time.perf_counter() - avg_start

    print(f"[{hostname}] Averaged grad: {avg_grad}")
    print(f"[{hostname}] Averaging time (no comm): {avg_time:.6f}s")

    print(f"[{hostname}] Total execution time: {time.perf_counter() - total_start:.6f}s")


if __name__ == "__main__":
    print(f"[{hostname}] Running single-machine baseline...")
    run_single_machine()