import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -------------------- CONFIG --------------------
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.1

torch.manual_seed(42)

# -------------------- DATA --------------------
df = pd.read_csv('fmnist_small.csv')

X = df.iloc[:, 1:].values / 255.0
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- DATASET --------------------
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------- MODEL --------------------
class MyNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


model = MyNN(X_train.shape[1])

# -------------------- TRAIN SETUP --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# -------------------- TRAIN --------------------
def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# -------------------- EVALUATE --------------------
def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

    print(f"Accuracy: {correct/total:.4f}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    train()
    evaluate()

def get_model():
    return MyNN(input_size=784)