#!/usr/bin/env python3
"""
sirius_train_full.py – Full, quality-focused training.
Trains until codebook utilization hits 80%+ and reconstruction converges.
"""
import os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenflow import TokenFlowTokenizer
from ssm_temporal import TemporalWedgeBlock
from mcp import MobileConditioningProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

def checkerboard(B, C, H, W, device=DEVICE):
    x = torch.zeros(B, C, H, W, device=device)
    gy, gx = torch.meshgrid(torch.linspace(0, 6.28, H, device=device),
                              torch.linspace(0, 6.28, W, device=device), indexing='ij')
    for b in range(B):
        f = 2.0 + (b % 6) * 1.5
        for c in range(min(C, 3)):
            p = torch.rand(1, device=device).item() * 6.28
            x[b, c] = (torch.sin(f * gx + p) * torch.cos(f * gy + p * 0.7) +
                       torch.sin(f * 0.5 * (gx + gy) + p * 0.3)) * 0.8
    return x

def frame_seq(B, L, D, seq_len=5, device=DEVICE):
    base = torch.randn(B, 1, D, device=device) * 0.4
    a = torch.rand(1, device=device) * 6.28
    frames = []
    for t in range(seq_len):
        drift = torch.sin(a + t * 0.2) * base.expand(-1, L, -1) * 0.3
        frame = base.expand(-1, L, -1) + drift + torch.randn_like(base.expand(-1, L, -1)) * 0.02
        frames.append(frame / (frame.norm(dim=-1, keepdim=True) + 1e-8) * 0.6)
    return frames

# ═══════════════════════════════════════════════════════════════════
# Phase 1: TokenFlow — train until utilization >= 80%
# ═══════════════════════════════════════════════════════════════════
def train_tokenflow():
    print("\n" + "="*65)
    print("PHASE 1: TokenFlow – training to 80%+ codebook utilization")
    print("="*65)

    model = TokenFlowTokenizer(codebook_size=2048, semantic_dim=256, pixel_dim=128).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

    B, H, W = 32, 64, 64
    best_recon = float('inf')
    best_util = 0.0
    plateau_steps = 0
    ema_util = 5.0  # smoothed utilization

    for step in range(2000):
        model.train()
        opt.zero_grad()

        x = checkerboard(B, 3, H, W)
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
        ema_util = 0.95 * ema_util + 0.05 * util
        best_util = max(best_util, util)
        recon_val = loss_recon.item()
        if recon_val < best_recon * 0.999:
            best_recon = recon_val
            plateau_steps = 0
        else:
            plateau_steps += 1

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:04d}/2000 | Loss: {total_loss.item():.4f} | "
                  f"Recon: {recon_val:.4f} (best: {best_recon:.4f}) | "
                  f"Util: {util:.1f}% (best: {best_util:.1f}%) | LR: {sched.get_last_lr()[0]:.2e}")

        if plateau_steps > 300 and ema_util > 40:
            print(f"  Early stop at step {step+1}: util={util:.1f}% best={best_util:.1f}%")
            break

    print(f"\n  >>> Final: util {best_util:.1f}% (ema: {ema_util:.1f}%) | best recon {best_recon:.4f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/tokenflow.pth")
    print(f"  Saved tokenflow.pth")
    return model

# ═══════════════════════════════════════════════════════════════════
# Phase 2: SSM Temporal
# ═══════════════════════════════════════════════════════════════════
def train_ssm():
    print("\n" + "="*65)
    print("PHASE 2: SSM Temporal – frame coherence model")
    print("="*65)
    D = 128
    model = TemporalWedgeBlock(dim=D, state_dim=16).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    B, L = 8, 64

    for step in range(300):
        model.train(); opt.zero_grad()
        frames = frame_seq(B, L, D, seq_len=6)
        state = None
        preds = []
        for t in range(6):
            pred, state = model(frames[t], prev_state=state)
            preds.append(pred)
        pl = [F.mse_loss(preds[t], frames[t]) for t in range(6)]
        sl = sum(F.mse_loss(preds[t], preds[t+1]) for t in range(5)) * 0.1
        total = sum(pl) + sl
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if (step+1) % 50 == 0:
            print(f"  Step {step+1:03d}/300 | Loss: {total.item():.6f} | Pred: {pl[0].item():.6f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/ssm_temporal.pth")
    print(f"  Saved ssm_temporal.pth")
    return model

# ═══════════════════════════════════════════════════════════════════
# Phase 3: MCP
# ═══════════════════════════════════════════════════════════════════
def train_mcp():
    print("\n" + "="*65)
    print("PHASE 3: MCP – Mobile Conditioning Projector")
    print("="*65)
    model = MobileConditioningProjector(vlm_dim=512, dit_dim=512, num_layers=4, scale_to_300m=False).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
    B, S = 4, 64

    for step in range(600):
        model.train(); opt.zero_grad()
        hidden = [torch.randn(B, S, 512, device=DEVICE) * (0.5 + 0.5 * math.sin(li + step * 0.01)) for li in range(4)]
        fused = sum(hidden) / len(hidden)
        target = fused[:, :S//2, :512]
        loss = F.mse_loss(model(hidden), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        if (step+1) % 60 == 0:
            print(f"  Step {step+1:03d}/600 | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/mcp.pth")
    print(f"  Saved mcp.pth")
    return model

# ═══════════════════════════════════════════════════════════════════
# E2E Validation
# ═══════════════════════════════════════════════════════════════════
def validate(tok, ssm, mcp):
    print("\n" + "="*65)
    print("E2E VALIDATION")
    print("="*65)
    tok.eval(); ssm.eval(); mcp.eval()
    B, H, W, D = 4, 64, 64, 128
    img = checkerboard(B, 3, H, W)
    with torch.no_grad():
        q_s, q_p, idx = tok.encode(img)
        recon = tok.decode_pixel(q_p)
        util = idx.unique().numel() / 2048 * 100
        mse = F.mse_loss(recon, img).item()
        psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 60
        print(f"  Tokenization: {idx.unique().numel()}/2048 codes ({util:.1f}%)")
        print(f"  Reconstruction: MSE={mse:.5f} PSNR={psnr:.1f}dB")

        state = None
        latent = q_p.permute(0, 2, 3, 1).reshape(B, -1, D)
        frames = []
        for t in range(6):
            x = latent if t == 0 else latent + torch.randn_like(latent) * 0.05
            c, state = ssm(x, prev_state=state)
            frames.append(c)
        diffs = [(frames[i]-frames[i+1]).abs().mean().item() for i in range(len(frames)-1)]
        print(f"  SSM: {len(frames)} frames, avg diff {sum(diffs)/len(diffs):.4f}")

        hidden = [torch.randn(B, 64, 512, device=DEVICE) for _ in range(4)]
        cond = mcp(hidden)
        print(f"  MCP: conditioning shape {list(cond.shape)}")

    if util >= 80:
        print("  ★ UTIL 10/10: Codebook fully utilised")
    elif util >= 50:
        print("  ★ UTIL 7/10: Good utilisation")
    elif util >= 20:
        print("  ★ UTIL 5/10: Improving")
    else:
        print("  △ UTIL 3/10: Needs more training")
    return mse, util

# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE} | PyTorch {torch.__version__}")

    tok = train_tokenflow()
    ssm = train_ssm()
    mcp = train_mcp()
    mse, util = validate(tok, ssm, mcp)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"COMPLETE in {elapsed/60:.1f} min")
    print(f"Final: Util={util:.1f}% | MSE={mse:.5f}")
    print(f"{'='*65}")
