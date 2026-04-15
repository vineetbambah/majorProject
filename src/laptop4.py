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


def worker(rank, world_size, ip_list, base_port=5000):
    torch.manual_seed(0)
    hostname = socket.gethostname()

    total_start = time.perf_counter()

    # ---- Setup ring sockets ----
    setup_start = time.perf_counter()

    left_rank = (rank - 1 + world_size) % world_size
    right_rank = (rank + 1) % world_size

    my_port = base_port + rank
    right_port = base_port + right_rank

    # Server: receive from left neighbor
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", my_port))
    server.listen(1)

    # Client: send to right neighbor
    right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Allow all servers to start
    time.sleep(2)

    right_ip = ip_list[right_rank]
    right_sock.connect((right_ip, right_port))
    print(f"[{hostname} | Rank {rank}] Connected to right ({right_ip}:{right_port})")

    left_sock, _ = server.accept()
    print(f"[{hostname} | Rank {rank}] Connected from left")

    print(f"[{hostname} | Rank {rank}] Setup time: {time.perf_counter() - setup_start:.4f}s")

    # ---- Model ----
    model = nn.Linear(1, 1, bias=False)

    x = torch.tensor([[rank + 1.0]])
    y = torch.tensor([[2.0]])

    # ---- Forward ----
    fwd_start = time.perf_counter()
    output = model(x)
    loss = (output - y).pow(2).mean()
    print(f"[{hostname} | Rank {rank}] Forward time: {time.perf_counter() - fwd_start:.6f}s")

    # ---- Backward ----
    bwd_start = time.perf_counter()
    loss.backward()
    print(f"[{hostname} | Rank {rank}] Backward time: {time.perf_counter() - bwd_start:.6f}s")

    print(f"[{hostname} | Rank {rank}] Initial weight: {model.weight.data}")
    print(f"[{hostname} | Rank {rank}] Local grad: {model.weight.grad}")

    # ---- Ring All-Reduce ----
    comm_start = time.perf_counter()

    grad = model.weight.grad.clone()
    total = grad.clone()
    send_data = grad.clone()

    for step in range(world_size - 1):
        print(f"[{hostname} | Rank {rank}] Step {step}: sending {send_data}")

        send_tensor(right_sock, send_data)

        recv_data = recv_tensor(left_sock)
        print(f"[{hostname} | Rank {rank}] Step {step}: received {recv_data}")

        total += recv_data
        send_data = recv_data.clone()

    avg_grad = total / world_size
    model.weight.grad = avg_grad

    print(f"[{hostname} | Rank {rank}] Averaged grad: {model.weight.grad}")
    print(f"[{hostname} | Rank {rank}] Communication time: {time.perf_counter() - comm_start:.6f}s")

    print(f"[{hostname} | Rank {rank}] Total time: {time.perf_counter() - total_start:.6f}s")

    left_sock.close()
    right_sock.close()
    server.close()


if __name__ == "__main__":
    hostname = socket.gethostname()

    # Usage:
    # python script.py <rank> <ip0> <ip1> <ip2> <ip3>
    rank = int(sys.argv[1])
    world_size = 4

    ip_list = sys.argv[2:2 + world_size]

    print(f"[{hostname} | Rank {rank}] Starting...")

    worker(rank, world_size, ip_list)