import torch
import torch.nn as nn
import torch.optim as optim
import time
import socket
import pickle
import sys

# -------------------------
# RNN Model
# -------------------------
class RNNModel(nn.Module):
    def _init_(self):
        super(RNNModel, self)._init_()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

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
        data += sock.recv(4096)
    return pickle.loads(data)

# -------------------------
# Setup Connection
# -------------------------
def setup_connection(rank, master_ip, port=5002):
    if rank == 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((master_ip, port))
        s.listen(1)
        conn, _ = s.accept()
        return conn
    else:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect((master_ip, port))
        return c

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

    model = RNNModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(rank)
    x = torch.randn(32, 5, 8)  # (batch, seq_len, features)
    y = torch.randn(32, 1)

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