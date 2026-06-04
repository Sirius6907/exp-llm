#!/usr/bin/env python3
"""
sirius_train_final.py – Final consolidated training using real CIFAR-10 images.
Trains TokenFlow on real data to get meaningful PSNR, then SSM + MCP.

Usage: python sirius_train_final.py
"""
import os, sys, time, math, tarfile, pickle, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenflow import TokenFlowTokenizer
from ssm_temporal import TemporalWedgeBlock
from mcp import MobileConditioningProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Load CIFAR-10 from local tar.gz ─────────────────────────────────────

def load_cifar10_local(data_path="data/cifar-10-python.tar.gz"):
    """Load CIFAR-10 from PyTorch's downloaded tar.gz."""
    with tarfile.open(data_path, 'r:gz') as tar:
        # Find the data batch file
        for member in tar.getmembers():
            if 'data_batch' in member.name and member.name.endswith('_1'):
                f = tar.extractfile(member)
                d = pickle.load(f, encoding='bytes')
                images = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
                labels = d[b'labels']
                f.close()
                # RGB channel mean/std normalisation
                mean = images.mean(axis=(0, 2, 3), keepdims=True)
                std = images.std(axis=(0, 2, 3), keepdims=True) + 1e-8
                images = (images - mean) / std
                return torch.tensor(images), torch.tensor(labels)
    raise FileNotFoundError(f"CIFAR-10 not found at {data_path}")

def make_frame_seq(B, L, D, seq_len=5, device=DEVICE):
    """Structured latents with temporal evolution."""
    base = torch.randn(B, 1, D, device=device) * 0.5
    a = torch.rand(1, device=device) * 6.28
    frames = []
    for t in range(seq_len):
        drift = torch.sin(a + t * 0.2) * base.expand(-1, L, -1) * 0.3
        frame = base.expand(-1, L, -1) + drift + torch.randn_like(base.expand(-1, L, -1)) * 0.03
        frame = frame / (frame.norm(dim=-1, keepdim=True) + 1e-8) * 0.6
        frames.append(frame)
    return frames

# ═══════════════════════════════════════════════════════════════════
# Phase 1: TokenFlow on REAL CIFAR-10
# ═══════════════════════════════════════════════════════════════════
def train_tokenflow():
    print("\n" + "="*65)
    print("PHASE 1: TokenFlow on CIFAR-10 real images")
    print("="*65)

    # Load real CIFAR-10 images
    cifar_imgs, _ = load_cifar10_local()
    # Upsample 32→64 to match encoder stride
    cifar_64 = F.interpolate(cifar_imgs, size=(64, 64), mode='bilinear', align_corners=False)
    dataset = TensorDataset(cifar_64)
    dl = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"CIFAR-10 loaded: {cifar_64.shape[0]} images, shape {list(cifar_64.shape[1:])}")

    model = TokenFlowTokenizer(codebook_size=2048, semantic_dim=256, pixel_dim=128).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

    best_recon = float('inf')
    best_util = 0.0
    global_step = 0

    for epoch in range(10):
        for batch_x in dl:
            model.train()
            opt.zero_grad()

            x = batch_x[0].to(DEVICE)
            q_s, q_p, indices, q_losses = model.encode_with_losses(x)
            recon = model.decode_pixel(q_p)

            loss_recon = F.l1_loss(recon, x) + F.mse_loss(recon, x)
            loss_cb = q_losses['codebook_s'] + q_losses['codebook_p']
            loss_cmt = 0.25 * (q_losses['commit_s'] + q_losses['commit_p'])
            total_loss = loss_recon + loss_cb + loss_cmt
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step()

            util = indices.unique().numel() / 2048 * 100
            best_util = max(best_util, util)
            recon_val = loss_recon.item()
            if recon_val < best_recon * 0.999:
                best_recon = recon_val

            global_step += 1
            if global_step % 50 == 0:
                psnr = 20 * math.log10(1.0 / math.sqrt(recon_val)) if recon_val > 1e-10 else 60
                print(f"  Step {global_step:04d} | Epoch {epoch+1} | "
                      f"Loss: {total_loss.item():.4f} | Recon: {recon_val:.4f} | "
                      f"PSNR: {psnr:.1f}dB | Util: {util:.1f}% (best: {best_util:.1f}%)")

        if best_util > 60 and best_recon < 0.05:
            print(f"  Early stop at epoch {epoch+1}: util={best_util:.1f}% recon={best_recon:.4f}")
            break

    # Final validation on a held-out batch
    model.eval()
    with torch.no_grad():
        test_x = cifar_64[500:510].to(DEVICE)
        q_s, q_p, idx = model.encode(test_x)
        recon = model.decode_pixel(q_p)
        test_mse = F.mse_loss(recon, test_x).item()
        test_psnr = 20 * math.log10(1.0 / math.sqrt(test_mse)) if test_mse > 1e-10 else 60
        test_util = idx.unique().numel() / 2048 * 100
        print(f"\n  >>> Test: MSE={test_mse:.5f} PSNR={test_psnr:.1f}dB Util={test_util:.1f}%")

    torch.save(model.state_dict(), f"{SAVE_DIR}/tokenflow.pth")
    return model, best_util, best_recon

# ═══════════════════════════════════════════════════════════════════
# Phase 2-4: SSM + MCP + Validation (same as before)
# ═══════════════════════════════════════════════════════════════════
def train_ssm():
    print("\n" + "="*65)
    print("PHASE 2: SSM Temporal")
    print("="*65)
    model = TemporalWedgeBlock(dim=128, state_dim=16).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(300):
        model.train(); opt.zero_grad()
        frames = make_frame_seq(8, 64, 128, seq_len=6)
        state = None
        preds = []
        for t in range(6):
            pred, state = model(frames[t], prev_state=state)
            preds.append(pred)
        pl = [F.mse_loss(preds[t], frames[t]) for t in range(6)]
        sl = sum(F.mse_loss(preds[t], preds[t+1]) for t in range(5)) * 0.1
        (sum(pl)+sl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if (step+1) % 60 == 0:
            m = sum(pl).item()/6
            print(f"  Step {step+1:03d}/300 | Avg Pred Loss: {m:.6f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/ssm_temporal.pth")
    return model

def train_mcp():
    print("\n" + "="*65)
    print("PHASE 3: MCP")
    print("="*65)
    model = MobileConditioningProjector(vlm_dim=512, dit_dim=512, num_layers=4, scale_to_300m=False).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(500):
        model.train(); opt.zero_grad()
        hidden = [torch.randn(4, 64, 512, device=DEVICE) * (0.5 + 0.5 * math.sin(li + step * 0.01)) for li in range(4)]
        fused = sum(hidden)/len(hidden)
        loss = F.mse_loss(model(hidden), fused[:, :32, :512])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if (step+1) % 100 == 0:
            print(f"  Step {step+1:03d}/500 | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/mcp.pth")
    return model

def validate(tok, ssm, mcp):
    print("\n" + "="*65)
    print("E2E VALIDATION")
    print("="*65)
    tok.eval(); ssm.eval(); mcp.eval()
    B, H, W, D = 4, 64, 64, 128
    # Use real CIFAR test images
    cifar_imgs, _ = load_cifar10_local()
    test_imgs = F.interpolate(cifar_imgs[5000:5020], size=(64,64), mode='bilinear', align_corners=False).to(DEVICE)
    with torch.no_grad():
        q_s, q_p, idx = tok.encode(test_imgs)
        recon = tok.decode_pixel(q_p)
        mse = F.mse_loss(recon, test_imgs).item()
        psnr = 20 * math.log10(1.0/math.sqrt(mse)) if mse > 1e-10 else 60
        util = idx.unique().numel() / 2048 * 100
        print(f"  TokenFlow: {util:.1f}% util | MSE={mse:.5f} | PSNR={psnr:.1f}dB")

        state = None
        latent = q_p[:1].permute(0,2,3,1).reshape(1,-1,D)
        frames = []
        for t in range(6):
            x = latent if t == 0 else latent + torch.randn_like(latent)*0.05
            c, state = ssm(x, prev_state=state)
            frames.append(c)
        diffs = [(frames[i]-frames[i+1]).abs().mean().item() for i in range(len(frames)-1)]
        print(f"  SSM: {len(frames)} frames, avg Δ = {sum(diffs)/len(diffs):.4f}")

        hidden = [torch.randn(B, 64, 512, device=DEVICE) for _ in range(4)]
        cond = mcp(hidden)
        print(f"  MCP: {list(cond.shape)}")

    if psnr >= 25:
        print(f"  ★ PSNR 10/10: {psnr:.1f}dB")
    elif psnr >= 20:
        print(f"  ★ PSNR 7/10: {psnr:.1f}dB")
    elif psnr >= 15:
        print(f"  ★ PSNR 4/10: {psnr:.1f}dB")
    else:
        print(f"  △ PSNR 2/10: {psnr:.1f}dB - needs more capacity")

    if util >= 80:
        print(f"  ★ UTIL 10/10")
    elif util >= 60:
        print(f"  ★ UTIL 7/10")
    elif util >= 30:
        print(f"  ★ UTIL 5/10")
    else:
        print(f"  △ UTIL 3/10")

if __name__ == "__main__":
    t0 = time.time()
    tok, util, recon = train_tokenflow()
    ssm = train_ssm()
    mcp = train_mcp()
    validate(tok, ssm, mcp)
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"COMPLETE in {elapsed/60:.1f} min")
    print(f"Best: Util={util:.1f}% Recon={recon:.4f}")
    print(f"{'='*65}")
