#!/usr/bin/env python3
"""Train TokenFlow on real CIFAR-10 data from pre-saved tensor."""
import os, sys, math, time
import torch, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenflow import TokenFlowTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# Load pre-processed CIFAR-10
data = torch.load('data/cifar10_64.pt')
print(f"CIFAR-10: {data.shape}")

train_data, val_data = data[:47000], data[47000:]
train_dl = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True, num_workers=0)
val_dl = DataLoader(TensorDataset(val_data), batch_size=16, shuffle=False, num_workers=0)

model = TokenFlowTokenizer(codebook_size=2048, semantic_dim=256, pixel_dim=128).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

best_recon = float('inf')
best_util = 0.0
t0 = time.time()

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for batch_x in train_dl:
        opt.zero_grad()
        x = batch_x[0].to(DEVICE)
        q_s, q_p, indices, q_losses = model.encode_with_losses(x)
        recon = model.decode_pixel(q_p)

        loss_recon = F.l1_loss(recon, x) + F.mse_loss(recon, x)
        loss_cb = q_losses['codebook_s'] + q_losses['codebook_p']
        loss_cmt = 0.25 * (q_losses['commit_s'] + q_losses['commit_p'])
        total = loss_recon + loss_cb + loss_cmt
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        util = indices.unique().numel() / 2048 * 100
        best_util = max(best_util, util)
        best_recon = min(best_recon, loss_recon.item())
        epoch_loss += loss_recon.item()

    # Validation
    model.eval()
    val_mse = 0.0
    val_util = 0.0
    with torch.no_grad():
        for batch_x in val_dl:
            x = batch_x[0].to(DEVICE)
            q_s, q_p, idx = model.encode(x)
            recon = model.decode_pixel(q_p)
            val_mse += F.mse_loss(recon, x).item()
            val_util += idx.unique().numel() / 2048 * 100
    val_mse /= len(val_dl)
    val_util /= len(val_dl)
    val_psnr = 20 * math.log10(1.0/math.sqrt(val_mse)) if val_mse > 1e-10 else 60

    elapsed = time.time() - t0
    print(f"  E{epoch+1} | Train: recon={epoch_loss/len(train_dl):.4f} | "
          f"Val: MSE={val_mse:.5f} PSNR={val_psnr:.1f}dB Util={val_util:.1f}% | "
          f"{elapsed/60:.1f}min")

torch.save(model.state_dict(), f"{SAVE_DIR}/tokenflow_cifar.pth")
print(f"\nSaved! Best util: {best_util:.1f}% | Best val MSE: {val_mse:.5f} | "
      f"Best val PSNR: {val_psnr:.1f}dB")
print(f"Total: {(time.time()-t0)/60:.1f} min")
