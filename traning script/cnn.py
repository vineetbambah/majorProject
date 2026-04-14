import torch
import torch.nn as nn
import torch.optim as optim
import time
import socket
import pickle
import sys

# -------------------------
# CNN Model
# -------------------------
class CNN(nn.Module):
    def _init_(self):
        super(CNN, self)._init_()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 26 * 26, 10)
        )

    def forward(self, x):
        return self.net(x)

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
def setup_connection(rank, master_ip, port=5001):
    if rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((master_ip, port))
        server.listen(1)
        print("[Rank 0] Waiting...")
        conn, _ = server.accept()
        return conn
    else:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((master_ip, port))
        return client

# -------------------------
# Train Step
# -------------------------
def train_step(model, optimizer, criterion, x, y):
    t = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)

    fwd = time.time() - t

    t = time.time()
    loss.backward()
    bwd = time.time() - t

    grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return grads, fwd, bwd

# -------------------------
# Main
# -------------------------
def main():
    rank = int(sys.argv[1])
    master_ip = sys.argv[2]

    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(rank)
    x = torch.randn(32, 1, 28, 28)
    y = torch.randn(32, 10)

    conn = setup_connection(rank, master_ip)

    grads, fwd, bwd = train_step(model, optimizer, criterion, x, y)

    print(f"[Rank {rank}] Forward: {fwd:.6f}s")
    print(f"[Rank {rank}] Backward: {bwd:.6f}s")

    comm_t = time.time()

    if rank == 0:
        other = recv_data(conn)
        avg = (grads + other) / 2
        send_data(conn, avg)
    else:
        send_data(conn, grads)
        avg = recv_data(conn)

    comm = time.time() - comm_t

    print(f"[Rank {rank}] Comm: {comm:.6f}s")
    print(f"[Rank {rank}] Grad size: {grads.numel()}")

    conn.close()

if _name_ == "_main_":
    main()