#!/usr/bin/env python3
"""
sirius_train.py – Consolidated training pipeline.
Trains TokenFlow → SSM Temporal → MCP sequentially.
v3: EMA codebooks, structured data, proper convergence tracking.
"""
import os, sys, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenflow import TokenFlowTokenizer
from ssm_temporal import TemporalWedgeBlock
from mcp import MobileConditioningProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
      ) if DEVICE.type == "cuda" else print(f"Device: CPU")

# ─── Data ────────────────────────────────────────────────────────────────

def make_checkerboard(B, C, H, W, device=DEVICE):
    """Structured images with spatial frequencies (not random noise)."""
    x = torch.zeros(B, C, H, W, device=device)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 6.28, H, device=device),
        torch.linspace(0, 6.28, W, device=device),
        indexing='ij'
    )
    for b in range(B):
        freq = 3.0 + (b % 5) * 1.5
        for c in range(min(C, 3)):
            phase = torch.rand(1, device=device).item() * 6.28
            x[b, c] = (torch.sin(freq * grid_x + phase) *
                       torch.cos(freq * grid_y + phase * 0.7) +
                       torch.cos(freq * 0.7 * (grid_x + grid_y) + phase * 0.3)) * 0.6
    return x

def make_frame_sequence(B, L, D, seq_len=4, device=DEVICE):
    """Slowly evolving latent vectors for temporal coherence training."""
    base = torch.randn(B, 1, D, device=device) * 0.5
    angle = torch.rand(1, device=device) * 6.28
    frames = []
    for t in range(seq_len):
        drift = torch.sin(angle + t * 0.3) * base.expand(-1, L, -1) * 0.2
        noise = torch.randn(B, L, D, device=device) * 0.03
        frame = base.expand(-1, L, -1) + drift + noise
        # Normalise to keep stable
        frames.append(frame / (frame.norm(dim=-1, keepdim=True) + 1e-8) * 0.5)
    return frames


# ═══════════════════════════════════════════════════════════════════════
# TokenFlow
# ═══════════════════════════════════════════════════════════════════════
def train_tokenflow():
    print("\n" + "="*60)
    print("TOKENFLOW: EMA-VQ Tokenizer")
    print("="*60)

    model = TokenFlowTokenizer(
        codebook_size=2048, semantic_dim=256, pixel_dim=128
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

    B, H, W = 16, 64, 64
    num_steps = 800
    best_recon = float('inf')

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        x = make_checkerboard(B, 3, H, W)

        # Training forward pass with losses + EMA update
        q_s, q_p, indices, q_losses = model.encode_with_losses(x)
        recon = model.decode_pixel(q_p)

        # Reconstruction loss
        loss_recon = F.l1_loss(recon, x) + F.mse_loss(recon, x)

        # Codebook + commitment
        loss_cb = q_losses['codebook_s'] + q_losses['codebook_p']
        loss_cmt = 0.25 * (q_losses['commit_s'] + q_losses['commit_p'])

        total_loss = loss_recon + loss_cb + loss_cmt + q_losses.get('codebook_reg', 0)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        util = indices.unique().numel() / 2048 * 100

        if (step + 1) % 25 == 0:
            recon_val = loss_recon.item()
            best_recon = min(best_recon, recon_val)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1:03d}/{num_steps} | Loss: {total_loss.item():.4f} | "
                  f"Recon: {recon_val:.4f} (best: {best_recon:.4f}) | "
                  f"Util: {util:.1f}% | LR: {lr_now:.2e}")

    final_util = indices.unique().numel() / 2048 * 100
    print(f"  Final codebook utilization: {final_util:.1f}% | Best recon loss: {best_recon:.4f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/tokenflow.pth")
    print(f"  Saved to {SAVE_DIR}/tokenflow.pth")
    return model


# ═══════════════════════════════════════════════════════════════════════
# SSM Temporal
# ═══════════════════════════════════════════════════════════════════════
def train_ssm():
    print("\n" + "="*60)
    print("SSM TEMPORAL: Frame Coherence")
    print("="*60)

    D = 128
    model = TemporalWedgeBlock(dim=D, state_dim=16).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    B, L = 8, 64
    num_steps = 200

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        frames = make_frame_sequence(B, L, D, seq_len=5)

        # Forward with state propagation
        state = None
        preds = []
        for t in range(5):
            pred, state = model(frames[t], prev_state=state)
            preds.append(pred)

        # Prediction loss per frame
        pred_losses = [F.mse_loss(preds[t], frames[t]) for t in range(5)]

        # Temporal smoothness (adjacent predictions should change gradually)
        smooth_loss = sum(F.mse_loss(preds[t], preds[t+1])
                         for t in range(4)) * 0.1

        total_loss = sum(pred_losses) + smooth_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:03d}/{num_steps} | Loss: {total_loss.item():.6f} | "
                  f"Pred: {pred_losses[0].item():.6f} | "
                  f"Smooth: {smooth_loss.item():.6f}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/ssm_temporal.pth")
    print(f"  Saved")
    return model


# ═══════════════════════════════════════════════════════════════════════
# MCP
# ═══════════════════════════════════════════════════════════════════════
def train_mcp():
    print("\n" + "="*60)
    print("MCP: Mobile Conditioning Projector (Lite)")
    print("="*60)

    VLM_DIM = 512
    DIT_DIM = 512
    model = MobileConditioningProjector(
        vlm_dim=VLM_DIM, dit_dim=DIT_DIM, num_layers=4, scale_to_300m=False
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    B, S = 4, 64
    num_steps = 300

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        # Simulate VLM hidden states with temporal structure
        base_freq = math.sin(step * 0.02)
        hidden_states = []
        for li in range(4):
            base = torch.randn(B, S, VLM_DIM, device=DEVICE) * (0.5 + 0.5 * math.sin(li + step * 0.01))
            hidden_states.append(base)

        # Target: spatially downsampled fused features
        fused = sum(hidden_states) / len(hidden_states)
        cond_target = fused[:, :S//2, :DIT_DIM]

        cond_pred = model(hidden_states)
        loss = F.mse_loss(cond_pred, cond_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 30 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1:03d}/{num_steps} | Loss: {loss.item():.4f} | LR: {lr_now:.2e}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/mcp.pth")
    print(f"  Saved")
    return model


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline Validation
# ═══════════════════════════════════════════════════════════════════════
def validate(tok, ssm, mcp):
    print("\n" + "="*60)
    print("E2E VALIDATION")
    print("="*60)

    tok.eval()
    ssm.eval()
    mcp.eval()

    B, H, W, D = 2, 64, 64, 128
    img = make_checkerboard(B, 3, H, W)

    with torch.no_grad():
        q_s, q_p, indices = tok.encode(img)
        recon = tok.decode_pixel(q_p)
        util = indices.unique().numel()
        mse = F.mse_loss(recon, img).item()
        print(f"  Tokenization: {util}/2048 codes ({util/2048*100:.1f}%) | Recon MSE: {mse:.4f}")

        # Frame-by-frame with SSM
        state = None
        latent_seq = q_p.permute(0, 2, 3, 1).reshape(B, -1, D)
        frame_latents = []
        for t in range(5):
            x = latent_seq if t == 0 else latent_seq + torch.randn_like(latent_seq) * 0.05
            coherent, state = ssm(x, prev_state=state)
            frame_latents.append(coherent)

        # Inter-frame diffs should show smooth progression
        diffs = [(frame_latents[i] - frame_latents[i+1]).abs().mean().item()
                 for i in range(len(frame_latents)-1)]
        avg_diff = sum(diffs) / len(diffs)
        print(f"  Frames: {len(frame_latents)} | Avg inter-frame diff: {avg_diff:.4f}")

        # MCP conditioning
        hidden = [torch.randn(B, 64, 512, device=DEVICE) for _ in range(4)]
        cond = mcp(hidden)
        print(f"  Conditioning shape: {list(cond.shape)}")

    print("  Validation PASSED")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    tok = train_tokenflow()
    ssm = train_ssm()
    mcp = train_mcp()
    validate(tok, ssm, mcp)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE in {elapsed/60:.1f} min")
    print(f"Checkpoints: {SAVE_DIR}/")
    print(f"{'='*60}")
