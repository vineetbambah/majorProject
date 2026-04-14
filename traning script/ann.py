import torch
import torch.nn as nn
import torch.optim as optim
import time
import socket
import pickle
import sys

# -------------------------
# Simple ANN Model
# -------------------------
class ANN(nn.Module):
    def _init_(self):
        super(ANN, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------
# Communication Utils
# -------------------------
def send_data(sock, data):
    data_bytes = pickle.dumps(data)
    sock.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)

def recv_data(sock):
    data_len = int.from_bytes(sock.recv(4), 'big')
    data = b''
    while len(data) < data_len:
        packet = sock.recv(4096)
        data += packet
    return pickle.loads(data)

# -------------------------
# Setup Connection
# -------------------------
def setup_connection(rank, master_ip, port=5000):
    if rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((master_ip, port))
        server.listen(1)
        print("[Rank 0] Waiting for connection...")
        conn, addr = server.accept()
        print("[Rank 0] Connected to", addr)
        return conn
    else:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((master_ip, port))
        print("[Rank 1] Connected to server")
        return client

# -------------------------
# Training Step
# -------------------------
def train_step(model, optimizer, criterion, x, y):
    start = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)

    fwd_time = time.time() - start

    start = time.time()
    loss.backward()
    bwd_time = time.time() - start

    # Collect gradients (flatten)
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)

    return grads, fwd_time, bwd_time

# -------------------------
# Main
# -------------------------
def main():
    rank = int(sys.argv[1])   # 0 or 1
    master_ip = sys.argv[2]

    # Model
    model = ANN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy data (each machine has different data)
    torch.manual_seed(rank)
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)

    # Setup connection
    conn = setup_connection(rank, master_ip)

    # Train one step
    grads, fwd_time, bwd_time = train_step(model, optimizer, criterion, x, y)

    print(f"[Rank {rank}] Forward time: {fwd_time:.6f}s")
    print(f"[Rank {rank}] Backward time: {bwd_time:.6f}s")
    print(f"[Rank {rank}] Local grad norm: {grads.norm()}")

    # -------------------------
    # Gradient Synchronization
    # -------------------------
    comm_start = time.time()

    if rank == 0:
        # Receive from rank 1
        other_grads = recv_data(conn)

        # Average
        avg_grads = (grads + other_grads) / 2

        # Send back averaged gradients
        send_data(conn, avg_grads)

    else:
        # Send to rank 0
        send_data(conn, grads)

        # Receive averaged gradients
        avg_grads = recv_data(conn)

    comm_time = time.time() - comm_start

    print(f"[Rank {rank}] Communication time: {comm_time:.6f}s")
    print(f"[Rank {rank}] Averaged grad norm: {avg_grads.norm()}")

    conn.close()

if _name_ == "_main_":
    main()