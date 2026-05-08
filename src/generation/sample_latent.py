import torch
import numpy as np

def sample_vae_latent(model, num_samples=1, latent_size=128, device='cpu'):
    """Samples from the VAE latent space and decodes into piano rolls."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_size).to(device)
        samples = model.decode(z)
        # Apply sigmoid because the decoder output is logits
        samples = torch.sigmoid(samples)
    return samples.cpu().numpy()

def reconstruct_autoencoder(model, data, device='cpu'):
    """Reconstructs given piano rolls using the Autoencoder."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).to(device)
        recon, _ = model(x)
        recon = torch.sigmoid(recon)
    return recon.cpu().numpy()
