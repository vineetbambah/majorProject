import torch
import torch.nn as nn
import socket
import pickle
import sys
import time

def send_tensor(sock, tensor):
    data = pickle.dumps(tensor)
    sock.sendall(len(data).to_bytes(4, 'big') + data)

def recv_tensor(sock):
    length = int.from_bytes(sock.recv(4), 'big')
    data = b''
    while len(data) < length:
        data += sock.recv(4096)
    return pickle.loads(data)

def worker(rank, world_size, master_ip, hostname, port=5000):
    torch.manual_seed(0)

    total_start = time.perf_counter()

    # ---- Setup socket ----
    setup_start = time.perf_counter()

    if rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((master_ip, port))
        server.listen(1)
        print(f"[{hostname} | Rank {rank}] Waiting for connection...")
        conn, _ = server.accept()
        left_sock = conn
        right_sock = conn
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((master_ip, port))
        left_sock = sock
        right_sock = sock

    print(f"[{hostname} | Rank {rank}] Connection established in {time.perf_counter() - setup_start:.4f}s")

    # ---- Model ----
    model = nn.Linear(1, 1, bias=False)

    x = torch.tensor([[rank + 1.0]])
    y = torch.tensor([[2.0]])

    # ---- Forward ----
    fwd_start = time.perf_counter()
    output = model(x)
    loss = (output - y).pow(2).mean()
    print(f"[{hostname} | Rank {rank}] Forward done in {time.perf_counter() - fwd_start:.6f}s")

    # ---- Backward ----
    bwd_start = time.perf_counter()
    loss.backward()
    print(f"[{hostname} | Rank {rank}] Backward done in {time.perf_counter() - bwd_start:.6f}s")

    print(f"[{hostname} | Rank {rank}] Initial weight: {model.weight.data}")
    print(f"[{hostname} | Rank {rank}] Local grad: {model.weight.grad}")

    # ---- Ring All-Reduce ----
    comm_start = time.perf_counter()

    grad = model.weight.grad.clone()
    total = grad.clone()
    send_tensor_data = grad.clone()

    for step in range(world_size - 1):
        print(f"[{hostname} | Rank {rank}] Step {step}: sending {send_tensor_data}")

        send_tensor(right_sock, send_tensor_data)

        recv_tensor_data = recv_tensor(left_sock)
        print(f"[{hostname} | Rank {rank}] Step {step}: received {recv_tensor_data}")

        total += recv_tensor_data
        send_tensor_data = recv_tensor_data.clone()

    avg_grad = total / world_size
    model.weight.grad = avg_grad

    print(f"[{hostname} | Rank {rank}] Averaged grad: {model.weight.grad}")
    print(f"[{hostname} | Rank {rank}] Communication time: {time.perf_counter() - comm_start:.6f}s")

    # ---- Total time ----
    print(f"[{hostname} | Rank {rank}] Total execution time: {time.perf_counter() - total_start:.6f}s")

    left_sock.close()
    right_sock.close()


if __name__ == "__main__":
    hostname = socket.gethostname()

    rank = int(sys.argv[1])
    world_size = 2
    master_ip = sys.argv[2]

    print(f"[{hostname} | Rank {rank}] Starting...")

    worker(rank, world_size, master_ip, hostname)