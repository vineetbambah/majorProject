import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# -------------------- CONFIG --------------------
EPOCHS = 100
LR = 0.001

torch.manual_seed(42)

# -------------------- LOAD DATA --------------------
df = pd.read_csv('100_Unique_QA_Dataset.csv')

# -------------------- TOKENIZER --------------------
def tokenize(text):
    text = text.lower().replace('?', '').replace("'", "")
    return text.split()

# -------------------- BUILD VOCAB --------------------
vocab = {'<UNK>': 0}

def build_vocab(row):
    tokens = tokenize(row['question']) + tokenize(row['answer'])
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

df.apply(build_vocab, axis=1)

# reverse vocab
idx_to_word = {idx: word for word, idx in vocab.items()}

# -------------------- TEXT → INDICES --------------------
def text_to_indices(text):
    return [vocab.get(token, vocab['<UNK>']) for token in tokenize(text)]

# -------------------- DATASET --------------------
class QADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        q = torch.tensor(text_to_indices(self.df.iloc[idx]['question']))
        a = torch.tensor(text_to_indices(self.df.iloc[idx]['answer']))
        return q, a

dataset = QADataset(df)

# IMPORTANT: keep batch_size = 1 (like your original)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# -------------------- MODEL --------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

model = SimpleRNN(len(vocab))

# -------------------- TRAIN SETUP --------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------- TRAIN --------------------
def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for q, a in dataloader:
            optimizer.zero_grad()

            output = model(q)

            # predict first word of answer
            loss = criterion(output, a[0][0].unsqueeze(0))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# -------------------- PREDICT --------------------
def predict(question):
    model.eval()

    q = torch.tensor(text_to_indices(question)).unsqueeze(0)

    with torch.no_grad():
        output = model(q)
        _, idx = torch.max(output, dim=1)

    print("Answer:", idx_to_word[idx.item()])

# -------------------- MAIN --------------------
if __name__ == "__main__":
    train()

    print("\n--- Testing ---")
    predict("What is the capital of France?")
    