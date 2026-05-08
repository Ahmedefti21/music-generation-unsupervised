import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add root to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from models.vae import MusicVAE

class PianoRollDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Data is stored as (88, 128), LSTM expects (128, 88)
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x.permute(1, 0)

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Use small beta (0.05) as found in our experiments to mitigate posterior collapse
    return BCE + 0.05 * KLD

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = root / "data" / "train_test_split"
    train_loader = DataLoader(PianoRollDataset(data_dir / "train.npy"), batch_size=64, shuffle=True)

    model = MusicVAE(input_size=88, latent_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting Task 2 training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save
    checkpoint_dir = root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "vae_script.pth")
    print(f"Model saved to {checkpoint_dir / 'vae_script.pth'}")

if __name__ == "__main__":
    train()
