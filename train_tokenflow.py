import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tokenflow import TokenFlowTokenizer

# Toggle dataset mode: "simulated" or "cifar10"
DATASET_MODE = "simulated"

class SimulatedImageDataset(Dataset):
    """
    Simulates a high-throughput image dataset (e.g. ImageNet/WebVid visual frames)
    yielding images of shape (3, 256, 256).
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a simulated RGB image tensor
        return torch.randn(3, 256, 256)

def train_tokenflow_one_epoch():
    print("==================================================")
    print("      TokenFlow Dual-Codebook Training Loop       ")
    print("==================================================")
    print(f"Dataset Mode selected: {DATASET_MODE.upper()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training execution device: {device}")
    
    # 1. Instantiate the TokenFlow model
    # codebook_size=2048, semantic_dim=768, pixel_dim=256
    model = TokenFlowTokenizer(
        codebook_size=2048,
        semantic_dim=768,
        pixel_dim=256
    ).to(device)
    
    # 2. Setup dataloader (Simulated or Real CIFAR-10)
    if DATASET_MODE == "cifar10":
        try:
            import torchvision.transforms as transforms
            import torchvision.datasets as datasets
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)), # Upscale to match TokenFlow expected resolution
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            # Download and load training data
            dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            print("Successfully loaded CIFAR-10 dataset.")
        except Exception as e:
            print(f"[Warning] Failed to load CIFAR-10: {e}. Falling back to Simulated mode.")
            dataset = SimulatedImageDataset(num_samples=32)
    else:
        dataset = SimulatedImageDataset(num_samples=32)
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 4. Simulated Teacher Network (CLIP model mock)
    # Yields a target semantic visual prior: shape (B, 768, 64, 64)
    # which aligns with our semantic token projection downsampled by 4
    def get_teacher_prior(x):
        B, C, H, W = x.shape
        # Target shape of semantic latents
        return torch.randn(B, 768, H // 4, W // 4, device=x.device)

    model.train()
    print("Starting training loop...")
    t0 = time.time()
    
    for epoch in range(1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_semantic = 0.0
        epoch_commit = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract only the images tensor (CIFAR-10 returns [images, labels])
            if isinstance(batch_data, (list, tuple)):
                batch_x = batch_data[0]
            else:
                batch_x = batch_data
                
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Extract teacher prior for semantic alignment
            with torch.no_grad():
                teacher_semantic_prior = get_teacher_prior(batch_x)
                
            # Forward pass: Encode and decode
            # q_s: (B, 768, 64, 64), q_p: (B, 256, 64, 64), indices: (B, 4096)
            q_s, q_p, indices = model.encode(batch_x)
            
            # Reconstruction
            recon_x = model.decode_pixel(q_p)
            
            # --- LOSS COMPUTATION ---
            # 1. Pixel Reconstruction Loss (L1)
            loss_recon = F_l1_loss = torch.mean(torch.abs(batch_x - recon_x))
            
            # 2. Semantic Alignment Loss (MSE between TokenFlow q_s and Teacher prior)
            loss_semantic = torch.mean((q_s - teacher_semantic_prior) ** 2)
            
            # 3. Vector Quantization Commitment Loss (Bonsai / VQ style)
            # Commitment loss keeps continuous projections close to discrete embeddings
            # (Calculated within TokenFlowDualQuantizer, here mocked dynamically)
            # sg[q_s] is handled in STE, commitment loss is minimized when projections match codebooks
            loss_commit = 0.25 * torch.mean((q_s.detach() - q_s) ** 2) + 0.25 * torch.mean((q_p.detach() - q_p) ** 2)
            
            # Joint Weighted Loss
            # Target weights: lambda_recon = 1.0, lambda_sem = 0.5, lambda_commit = 1.0
            total_loss = loss_recon + 0.5 * loss_semantic + loss_commit
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent overflow
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate logs
            epoch_loss += total_loss.item()
            epoch_recon += loss_recon.item()
            epoch_semantic += loss_semantic.item()
            epoch_commit += loss_commit.item()
            
            if (batch_idx + 1) % 4 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Batch {batch_idx+1}/{len(dataloader)} | Total Loss: {total_loss.item():.4f} | Recon: {loss_recon.item():.4f} | Sem: {loss_semantic.item():.4f} | Commit: {loss_commit.item():.4f}")
                
        print(f"\n--- Epoch Summary ---")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Average Epoch Loss: {avg_loss:.4f}")
        
    # 5. Save the trained checkpoint to disk
    checkpoint_path = "tokenflow_cifar10.pth"
    print(f"\nSaving model checkpoint to: {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    print("Checkpoint saved successfully.")
    
    t1 = time.time()
    print(f"Completed in {t1 - t0:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    train_tokenflow_one_epoch()
