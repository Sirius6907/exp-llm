#!/usr/bin/env python3
"""
train_tokenflow_real.py – TokenFlow on real CIFAR-10.  Clean GPU memory, B=4.
Zero synthetic-artifact overfit.  All data is real camera images.
"""
import os, sys, math, time
import torch, torch.nn.functional as F, torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenflow import TokenFlowTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE   = "checkpoints"; os.makedirs(SAVE, exist_ok=True)

# Load prepped fp16 CIFAR on CPU only
t0 = time.time()
path = r"C:\Users\opcha\Downloads\Pixelle-Video-main\Pixelle-Video-main\experiment-sirius\data\cifar10_prepped_f16.pt"
all_f16 = torch.load(path, map_location="cpu", weights_only=True)
train_f16 = all_f16[:40000]; val_f16 = all_f16[40000:]
N_train, N_val = len(train_f16), len(val_f16)
print(f"CIFAR: {N_train} train + {N_val} val | {time.time()-t0:.0f}s", flush=True)

model = TokenFlowTokenizer(codebook_size=2048, semantic_dim=256, pixel_dim=128).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Force CUDA context to settle
torch.cuda.empty_cache()
free0, total = torch.cuda.mem_get_info()
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
      f"GPU: {free0/1024**3:.2f}GB / {total/1024**3:.1f}GB free", flush=True)

B = 8
best_recon, best_util = float("inf"), 0.0
t_start = time.time()

for epoch in range(10):
    perm = torch.randperm(N_train).to(DEVICE)
    train_batches = N_train // B
    model.train()
    epoch_loss = 0.0

    for i in range(train_batches):
        # Transfer one batch from CPU → GPU in fp16, cast to fp32
        cpu_idxs = perm[i*B:(i+1)*B].cpu()
        x = train_f16[cpu_idxs].to(DEVICE, non_blocking=True).float()

        opt.zero_grad(set_to_none=True)
        qs, qp, idx, ql = model.encode_with_losses(x)
        recon = model.decode_pixel(qp)

        loss_recon = F.l1_loss(recon, x) + F.mse_loss(recon, x)
        total = loss_recon + ql["codebook_s"] + ql["codebook_p"] \
                + 0.25 * (ql["commit_s"] + ql["commit_p"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        best_recon = min(best_recon, loss_recon.item())
        best_util  = max(best_util, idx.unique().numel() / 2048 * 100)
        epoch_loss += loss_recon.item()

        if (i+1) % 250 == 0:
            print(f"  E{epoch+1} b{i+1:04d}/{train_batches} | "
                  f"loss={loss_recon.item():.4f} util={idx.unique().numel()/2048*100:.1f}% | "
                  f"{time.time()-t_start:.0f}s", flush=True)

    # Validation
    model.eval()
    vm, vu = 0.0, 0.0
    with torch.no_grad():
        for i in range(N_val // B):
            x = val_f16[i*B:(i+1)*B].to(DEVICE, non_blocking=True).float()
            qs, qp, idx = model.encode(x)
            recon = model.decode_pixel(qp)
            vm += F.mse_loss(recon, x).item() * B
            vu += (idx.unique().numel() / 2048 * 100) * B
    vm /= N_val; vu /= N_val
    vp = 20 * math.log10(1/math.sqrt(vm)) if vm > 0 else 60

    mem = torch.cuda.mem_get_info()[0]/1024**3
    print(f"  E{epoch+1:2d} val: MSE={vm:.5f} PSNR={vp:.1f}dB util={vu:.1f}% "
          f"| {time.time()-t_start:.0f}s | GPU free: {mem:.2f}GB", flush=True)

torch.save(model.state_dict(), f"{SAVE}/tokenflow_real_cifar.pth")
print(f"\nDone {(time.time()-t_start)/60:.1f} min | Best util: {best_util:.1f}% "
      f"| Val PSNR: {vp:.1f}dB", flush=True)
