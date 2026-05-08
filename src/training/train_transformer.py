import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

# Add root to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from models.transformer import MusicTransformer

class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        seq = self.tokens[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = root / "data" / "processed"
    train_tokens = np.load(data_dir / "tokens_train.npy")
    val_tokens = np.load(data_dir / "tokens_val.npy")

    train_loader = DataLoader(TokenDataset(train_tokens), batch_size=32, shuffle=True)
    val_loader = DataLoader(TokenDataset(val_tokens), batch_size=32, shuffle=False)

    model = MusicTransformer(vocab_size=284).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    for epoch in range(5): # Short training for script demo
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, 284), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, PPL: {math.exp(min(avg_loss, 20)):.2f}")

    # Save
    checkpoint_dir = root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "transformer_script.pth")
    print(f"Model saved to {checkpoint_dir / 'transformer_script.pth'}")

if __name__ == "__main__":
    train()
