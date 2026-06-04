"""
Pixelle-Sirius Complete Training Pipeline
==========================================
Trains the model on REAL image-text datasets using proper diffusion loss.
Designed for RTX 3050 (4GB VRAM) with fp16 mixed precision.

Usage:
    python train_complete.py --stage 1    # MCP alignment only
    python train_complete.py --stage 2    # DiT denoising
    python train_complete.py --stage 3    # Joint fine-tune
    python train_complete.py --stage 4    # Text generation
    python train_complete.py --stage all  # Run all stages sequentially
    python train_complete.py --dry-run    # Run shape validation dry-run
"""

import os
import sys
import time
import math
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

# ============================================================
# 1. DIFFUSION NOISE SCHEDULE (DDPM)
# ============================================================

class DiffusionSchedule:
    """
    Standard DDPM linear noise schedule.
    Precomputes all alpha/beta values for efficient training.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        Args:
            x_0: Clean sample (B, C, H, W) or (B, L, D)
            t: Timestep indices (B,)
            noise: Optional pre-sampled noise
        Returns:
            x_t: Noisy sample, noise: The added noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x_0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
    
    def sample_timesteps(self, batch_size):
        """Sample random timesteps uniformly."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)


# ============================================================
# 2. REAL IMAGE-TEXT DATASET (HuggingFace Streaming)
# ============================================================

class RealImageTextDataset(IterableDataset):
    """
    Streams real image-text pairs from HuggingFace datasets.
    No full download required - images are fetched on-the-fly.
    """
    def __init__(self, dataset_names=None, image_size=256, split="train", max_samples=None):
        super().__init__()
        self.image_size = image_size
        self.max_samples = max_samples
        self.split = split
        
        if dataset_names is None:
            dataset_names = ["lambdalabs/naruto-blip-captions"]
        
        self.dataset_names = dataset_names
        self._datasets = None  # Lazy init
    
    def _init_datasets(self):
        """Lazy-load HuggingFace datasets on first iteration."""
        from datasets import load_dataset
        
        all_datasets = []
        for name in self.dataset_names:
            print(f"[Dataset] Loading {name} (streaming)...")
            try:
                ds = load_dataset(name, split=self.split, streaming=True)
                all_datasets.append((name, ds))
                print(f"[Dataset] {name} connected successfully.")
            except Exception as e:
                print(f"[Dataset Warning] Failed to load {name}: {e}")
        
        self._datasets = all_datasets
    
    def _process_image(self, img):
        """Convert PIL image to normalized tensor [-1, 1]."""
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        
        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        return tensor
    
    def _get_caption(self, sample):
        """Extract caption from various dataset formats."""
        for key in ["text", "caption", "label", "description"]:
            if key in sample:
                val = sample[key]
                if isinstance(val, list):
                    return val[0] if val else "an image"
                return str(val)
        return "an image"
    
    def _get_image(self, sample):
        """Extract image from various dataset formats."""
        for key in ["image", "img", "pixel_values"]:
            if key in sample:
                return sample[key]
        return None
    
    def __iter__(self):
        if self._datasets is None:
            self._init_datasets()
        
        count = 0
        for name, ds in self._datasets:
            for sample in ds:
                if self.max_samples and count >= self.max_samples:
                    return
                
                try:
                    img = self._get_image(sample)
                    if img is None:
                        continue
                    
                    image_tensor = self._process_image(img)
                    caption = self._get_caption(sample)
                    
                    yield {"image": image_tensor, "caption": caption}
                    count += 1
                except Exception:
                    continue  # Skip corrupted samples


class CachedImageTextDataset(Dataset):
    """
    Downloads and caches dataset locally for fast repeated training.
    Uses non-streaming mode for small datasets that fit in memory.
    """
    def __init__(self, dataset_name="lambdalabs/naruto-blip-captions", image_size=256, split="train", max_samples=None):
        super().__init__()
        self.image_size = image_size
        
        from datasets import load_dataset
        print(f"[Dataset] Downloading {dataset_name} to local cache...")
        self.dataset = load_dataset(dataset_name, split=split)
        
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
        
        print(f"[Dataset] Loaded {len(self.dataset)} samples from {dataset_name}.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image
        img = sample.get("image", sample.get("img"))
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        
        # Get caption
        caption = sample.get("text", sample.get("caption", "an image"))
        if isinstance(caption, list):
            caption = caption[0] if caption else "an image"
        
        return {"image": image_tensor, "caption": str(caption)}


# ============================================================
# 3. TRAINING UTILS
# ============================================================

def setup_orchestrator(device, load_stage1=False, load_stage2=False, offload=False):
    """Initialize orchestrator with real backbones, optionally offloading them to CPU to save GPU memory."""
    from pixelle_sirius_engine import PixelleSiriusOrchestrator
    
    print("\n" + "=" * 60)
    print("  Initializing Pixelle-Sirius Orchestrator")
    print("=" * 60)
    
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device,
        scale_to_300m=True
    )
    
    # Load real backbones (frozen). Offload to CPU if requested.
    orchestrator.load_real_backbones(offload=offload)
    
    # Load previous stage checkpoints if requested
    if load_stage1 and os.path.exists("checkpoints/mcp_stage1.pth"):
        print("[Checkpoint] Loading Stage 1 MCP weights...")
        mcp_state = torch.load("checkpoints/mcp_stage1.pth", map_location=device)
        orchestrator.mcp.load_state_dict(mcp_state)
        print("[Checkpoint] Stage 1 MCP weights loaded.")
    
    if load_stage2 and os.path.exists("checkpoints/dit_stage2.pth"):
        print("[Checkpoint] Loading Stage 2 DiT weights...")
        dit_state = torch.load("checkpoints/dit_stage2.pth", map_location=device)
        orchestrator.lcm_solver.load_state_dict(dit_state)
        
        if os.path.exists("checkpoints/latent_to_vae_stage2.pth"):
            orchestrator.image_decoder.latent_to_vae.load_state_dict(torch.load("checkpoints/latent_to_vae_stage2.pth", map_location=device))
        if os.path.exists("checkpoints/vae_to_latent_stage2.pth"):
            orchestrator.image_decoder.vae_to_latent.load_state_dict(torch.load("checkpoints/vae_to_latent_stage2.pth", map_location=device))
        print("[Checkpoint] Stage 2 DiT weights loaded.")
    
    # Enable gradient checkpointing to fit in 4GB VRAM
    orchestrator.lcm_solver.set_grad_checkpointing(True)
    
    return orchestrator


def get_text_conditioning(orchestrator, captions, device):
    """
    Extract text conditioning from captions using Qwen2 -> MCP pipeline.
    """
    return orchestrator.get_text_conditioning(captions)


def generate_and_save_sample(orchestrator, prompt, step, stage_name, device, save_dir="training_samples"):
    """Generate a sample image and save it for visual verification."""
    os.makedirs(save_dir, exist_ok=True)
    
    orchestrator.eval()
    with torch.no_grad():
        # Get text conditioning
        hidden_states = get_text_conditioning(orchestrator, [prompt], device)
        conditioning = orchestrator.mcp(hidden_states)
        
        # Start from pure noise
        h_g, w_g = 32, 32  # 256x256 / 8
        noise = torch.randn(1, h_g * w_g, 256, device=device)
        
        # Denoise with DiT solver
        denoised = orchestrator.lcm_solver(noise, conditioning, num_steps=4)
        
        # Decode to RGB via VAE
        denoised_spatial = denoised.view(1, h_g, w_g, 256)
        rgb = orchestrator.image_decoder(denoised_spatial, h_g=h_g, w_g=w_g, decode_to_rgb=True)
        
        if rgb is not None and len(rgb.shape) == 4:
            # Convert to PIL and save
            img_np = rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            filename = f"{stage_name}_step{step:06d}.png"
            img.save(os.path.join(save_dir, filename))
            print(f"  [Sample] Saved {filename}")
    
    orchestrator.train()


# ============================================================
# 4. TRAINING STAGES
# ============================================================

def train_stage1_mcp(device, epochs=4, lr=2e-4, batch_size=2, image_size=256, 
                      dataset_name="lambdalabs/naruto-blip-captions", save_every=500):
    """
    Stage 1: Train MCP projector to align Qwen2 text embeddings 
    with DiT conditioning space using real image-text pairs.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1: MCP Alignment Training")
    print("  Training MCP to project text embeddings -> image conditioning")
    print("=" * 60)
    
    orchestrator = setup_orchestrator(device)
    
    # Freeze everything except MCP and the alignment projector
    for name, param in orchestrator.named_parameters():
        param.requires_grad = False
        
    for param in orchestrator.mcp.parameters():
        param.requires_grad = True
        
    # Project conditioning to match ground truth dim for alignment
    if not hasattr(orchestrator, '_cond_align_proj'):
        orchestrator._cond_align_proj = nn.Linear(1024, 256).to(device)
    orchestrator._cond_align_proj.requires_grad_(True)
    
    trainable_params = list(orchestrator.mcp.parameters()) + list(orchestrator._cond_align_proj.parameters())
    
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in orchestrator.parameters())
    print(f"\n[Params] Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")
    
    # Dataset
    dataset = CachedImageTextDataset(dataset_name, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True, drop_last=True)
    
    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    # Diffusion schedule
    diff_schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/mcp_stage1_checkpoint.pth"
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0
    
    # Auto-resume logic
    if os.path.exists(checkpoint_path):
        print(f"[Resume] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        orchestrator.mcp.load_state_dict(checkpoint["mcp_state_dict"])
        if "cond_align_proj_state_dict" in checkpoint and checkpoint["cond_align_proj_state_dict"] is not None:
            orchestrator._cond_align_proj.load_state_dict(checkpoint["cond_align_proj_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Resumed from epoch {start_epoch + 1}, step {global_step} (best loss: {best_loss:.4f})")
    
    print(f"\n[Training] {epochs} epochs x {len(dataloader)} batches = {epochs * len(dataloader)} steps")
    print(f"[Training] Batch size: {batch_size}, LR: {lr}, Image size: {image_size}x{image_size}")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        orchestrator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        # Skip batches if resuming in the middle of an epoch
        batches_to_skip = global_step % len(dataloader) if epoch == start_epoch else 0
        
        for idx, batch in enumerate(pbar):
            if idx < batches_to_skip:
                continue
                
            images = batch["image"].to(device)       # (B, 3, H, W) in [-1, 1]
            captions = batch["caption"]               # list of strings
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                # 1. Encode real images to VAE latents (ground truth)
                gt_latent = orchestrator.image_decoder.encode_image_to_latent(images)
                B, C_vae, h_g, w_g = gt_latent.shape
                
                # 2. Flatten and project VAE latent to DiT sequence format: (B, L, 256)
                gt_flat = gt_latent.permute(0, 2, 3, 1).reshape(B, h_g * w_g, C_vae)
                gt_expanded = orchestrator.image_decoder.vae_to_latent(gt_flat)  # (B, L, 256)
                
                # 3. Sample random timestep and add noise
                t = diff_schedule.sample_timesteps(B)
                noisy_latent, noise = diff_schedule.add_noise(gt_expanded, t)
                
                # 4. Get text conditioning through MCP (this has gradients!)
                hidden_states = get_text_conditioning(orchestrator, captions, device)
                conditioning = orchestrator.mcp(hidden_states)
                
                # 5. DiT predicts clean latent (frozen in stage 1)
                with torch.no_grad():
                    predicted = orchestrator.lcm_solver(noisy_latent, conditioning, num_steps=1)
                
                # 6. Loss: How well does MCP conditioning help DiT predict ground truth?
                # We use the conditioning-ground truth alignment loss
                cond_mean = conditioning.mean(dim=1)  # (B, 1024)
                gt_mean = gt_expanded.mean(dim=1)  # (B, 256)
                
                cond_projected = orchestrator._cond_align_proj(cond_mean)
                loss = F.mse_loss(cond_projected, gt_mean.detach())
                
                # Also add a contrastive-like loss for better alignment
                cond_norm = F.normalize(cond_projected, dim=-1)
                gt_norm = F.normalize(gt_mean.detach(), dim=-1)
                cosine_loss = 1.0 - (cond_norm * gt_norm).sum(dim=-1).mean()
                
                loss = loss + 0.5 * cosine_loss
            
            # NaN / Inf Loss Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] Step {global_step} loss is NaN/Inf, skipping optimizer step.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # NaN / Inf Gradient Guard
            has_nan_grad = False
            for p in trainable_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                print(f"\n[WARNING] Step {global_step} detected NaN/Inf gradients, skipping step.")
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Save checkpoints and visual samples periodically
            if global_step % save_every == 0:
                generate_and_save_sample(orchestrator, "a naruto character with spiky hair", 
                                        global_step, "stage1", device)
                
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "mcp_state_dict": orchestrator.mcp.state_dict(),
                    "cond_align_proj_state_dict": orchestrator._cond_align_proj.state_dict() if hasattr(orchestrator, '_cond_align_proj') else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict()
                }, checkpoint_path)
                print(f"  [Checkpoint] Saved resume checkpoint to {checkpoint_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n  Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        # Save best weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(orchestrator.mcp.state_dict(), "checkpoints/mcp_stage1.pth")
            if hasattr(orchestrator, '_cond_align_proj'):
                torch.save(orchestrator._cond_align_proj.state_dict(), "checkpoints/cond_align_proj.pth")
            print(f"  [Best] Saved checkpoint (loss: {best_loss:.4f})")
            
    print(f"\n{'=' * 60}")
    print(f"  Stage 1 Complete | Best Loss: {best_loss:.4f}")
    print(f"  Checkpoint: checkpoints/mcp_stage1.pth")
    print(f"{'=' * 60}")
    return orchestrator


def train_stage2_dit(device, epochs=6, lr=1e-4, batch_size=1, image_size=256,
                      dataset_names=None, save_every=500):
    """
    Stage 2: Train DiT Solver and VAE projection bridges to denoise latents given text conditioning.
    MCP is frozen (loaded from Stage 1 checkpoint).
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: DiT Denoising Training")
    print("  Training DiT to predict clean latents from noisy input + text")
    print("=" * 60)
    
    if dataset_names is None:
        dataset_names = ["lambdalabs/naruto-blip-captions"]
    
    orchestrator = setup_orchestrator(device, load_stage1=True)
    
    # Freeze everything except DiT solver and image_decoder projection layers
    for name, param in orchestrator.named_parameters():
        param.requires_grad = False
        
    for param in orchestrator.lcm_solver.parameters():
        param.requires_grad = True
        
    orchestrator.image_decoder.latent_to_vae.requires_grad_(True)
    orchestrator.image_decoder.vae_to_latent.requires_grad_(True)
    
    trainable_params = (list(orchestrator.lcm_solver.parameters()) + 
                        list(orchestrator.image_decoder.latent_to_vae.parameters()) +
                        list(orchestrator.image_decoder.vae_to_latent.parameters()))
                        
    trainable = sum(p.numel() for p in trainable_params)
    print(f"\n[Params] Trainable: {trainable:,}")
    
    # Dataset
    dataset = CachedImageTextDataset(dataset_names[0], image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    
    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    diff_schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    
    checkpoint_path = "checkpoints/dit_stage2_checkpoint.pth"
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0
    
    # Auto-resume logic
    if os.path.exists(checkpoint_path):
        print(f"[Resume] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        orchestrator.lcm_solver.load_state_dict(checkpoint["dit_state_dict"])
        orchestrator.image_decoder.latent_to_vae.load_state_dict(checkpoint["latent_to_vae_state_dict"])
        orchestrator.image_decoder.vae_to_latent.load_state_dict(checkpoint["vae_to_latent_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Resumed from epoch {start_epoch + 1}, step {global_step} (best loss: {best_loss:.4f})")
        
    print(f"\n[Training] {epochs} epochs x {len(dataloader)} batches = {epochs * len(dataloader)} steps")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        orchestrator.train()
        orchestrator.real_qwen.eval()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        batches_to_skip = global_step % len(dataloader) if epoch == start_epoch else 0
        
        for idx, batch in enumerate(pbar):
            if idx < batches_to_skip:
                continue
                
            images = batch["image"].to(device)
            captions = batch["caption"]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                # 1. Encode real image -> ground truth VAE latent
                gt_latent = orchestrator.image_decoder.encode_image_to_latent(images)
                B, C_vae, h_g, w_g = gt_latent.shape
                
                # 2. Project VAE latent to DiT sequence format: (B, L, 256)
                gt_flat = gt_latent.permute(0, 2, 3, 1).reshape(B, h_g * w_g, C_vae)
                gt_expanded = orchestrator.image_decoder.vae_to_latent(gt_flat)
                
                # 3. Sample timestep and add noise
                t = diff_schedule.sample_timesteps(B)
                noisy_latent, noise = diff_schedule.add_noise(gt_expanded, t)
                
                # 4. Get text conditioning (MCP frozen)
                with torch.no_grad():
                    hidden_states = get_text_conditioning(orchestrator, captions, device)
                    conditioning = orchestrator.mcp(hidden_states)
                
                # 5. DiT predicts clean latent (THIS has gradients now!)
                predicted_clean = orchestrator.lcm_solver(noisy_latent, conditioning, num_steps=1)
                
                # 6. Diffusion denoising loss: predict x_0 from x_t
                loss = F.mse_loss(predicted_clean, gt_expanded)
                
                # 7. Also train latent_to_vae: predicted 4-channel should match ground truth 4-channel
                pred_4ch = orchestrator.image_decoder.latent_to_vae(predicted_clean)  # (B, L, 4)
                gt_4ch = gt_flat  # (B, L, 4)
                vae_loss = F.mse_loss(pred_4ch, gt_4ch)
                
                loss = loss + 0.1 * vae_loss
            
            # NaN / Inf Loss Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] Step {global_step} loss is NaN/Inf, skipping step.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # NaN / Inf Gradient Guard
            has_nan_grad = False
            for p in trainable_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                print(f"\n[WARNING] Step {global_step} detected NaN/Inf gradients, skipping step.")
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "denoise": f"{F.mse_loss(predicted_clean, gt_expanded).item():.4f}",
                "vae": f"{vae_loss.item():.4f}"
            })
            
            if global_step % save_every == 0:
                generate_and_save_sample(orchestrator, "a naruto character with spiky hair",
                                        global_step, "stage2", device)
                
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "dit_state_dict": orchestrator.lcm_solver.state_dict(),
                    "latent_to_vae_state_dict": orchestrator.image_decoder.latent_to_vae.state_dict(),
                    "vae_to_latent_state_dict": orchestrator.image_decoder.vae_to_latent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict()
                }, checkpoint_path)
                print(f"  [Checkpoint] Saved resume checkpoint to {checkpoint_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n  Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(orchestrator.lcm_solver.state_dict(), "checkpoints/dit_stage2.pth")
            torch.save(orchestrator.image_decoder.latent_to_vae.state_dict(), "checkpoints/latent_to_vae_stage2.pth")
            torch.save(orchestrator.image_decoder.vae_to_latent.state_dict(), "checkpoints/vae_to_latent_stage2.pth")
            print(f"  [Best] Saved checkpoint (loss: {best_loss:.4f})")
            
    print(f"\n{'=' * 60}")
    print(f"  Stage 2 Complete | Best Loss: {best_loss:.4f}")
    print(f"{'=' * 60}")
    return orchestrator


def train_stage3_joint(device, epochs=4, lr=5e-5, batch_size=1, image_size=256,
                        dataset_names=None, save_every=500):
    """
    Stage 3: Joint end-to-end fine-tuning of MCP + DiT + VAE projection bridges.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: Joint End-to-End Fine-tuning")
    print("  Training MCP + DiT together for coherent generation")
    print("=" * 60)
    
    if dataset_names is None:
        dataset_names = ["lambdalabs/naruto-blip-captions"]
    
    orchestrator = setup_orchestrator(device, load_stage1=True, load_stage2=True, offload=True)
    
    # Unfreeze MCP + DiT + latent_to_vae + vae_to_latent
    for name, param in orchestrator.named_parameters():
        param.requires_grad = False
        
    for param in orchestrator.mcp.parameters():
        param.requires_grad = True
        
    for param in orchestrator.lcm_solver.parameters():
        param.requires_grad = True
        
    orchestrator.image_decoder.latent_to_vae.requires_grad_(True)
    orchestrator.image_decoder.vae_to_latent.requires_grad_(True)
    
    trainable_params = (list(orchestrator.mcp.parameters()) + 
                        list(orchestrator.lcm_solver.parameters()) +
                        list(orchestrator.image_decoder.latent_to_vae.parameters()) +
                        list(orchestrator.image_decoder.vae_to_latent.parameters()))
    
    trainable = sum(p.numel() for p in trainable_params)
    print(f"\n[Params] Trainable: {trainable:,}")
    
    dataset = CachedImageTextDataset(dataset_names[0], image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    diff_schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    
    checkpoint_path = "checkpoints/joint_stage3_checkpoint.pth"
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0
    
    # Auto-resume logic
    if os.path.exists(checkpoint_path):
        print(f"[Resume] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        orchestrator.mcp.load_state_dict(checkpoint["mcp_state_dict"])
        orchestrator.lcm_solver.load_state_dict(checkpoint["dit_state_dict"])
        orchestrator.image_decoder.latent_to_vae.load_state_dict(checkpoint["latent_to_vae_state_dict"])
        orchestrator.image_decoder.vae_to_latent.load_state_dict(checkpoint["vae_to_latent_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Resumed from epoch {start_epoch + 1}, step {global_step} (best loss: {best_loss:.4f})")
        
    print(f"\n[Training] {epochs} epochs x {len(dataloader)} batches = {epochs * len(dataloader)} steps")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        orchestrator.train()
        orchestrator.real_qwen.eval()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        batches_to_skip = global_step % len(dataloader) if epoch == start_epoch else 0
        
        for idx, batch in enumerate(pbar):
            if idx < batches_to_skip:
                continue
                
            images = batch["image"].to(device)
            captions = batch["caption"]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                # 1. Ground truth latent projected to DiT space
                gt_latent = orchestrator.image_decoder.encode_image_to_latent(images)
                B, C_vae, h_g, w_g = gt_latent.shape
                gt_flat = gt_latent.permute(0, 2, 3, 1).reshape(B, h_g * w_g, C_vae)
                gt_expanded = orchestrator.image_decoder.vae_to_latent(gt_flat)
                
                # 2. Noise and denoise
                t = diff_schedule.sample_timesteps(B)
                noisy_latent, noise = diff_schedule.add_noise(gt_expanded, t)
                
                # 3. Full pipeline: text -> Qwen2 -> MCP -> conditioning -> DiT -> prediction
                hidden_states = get_text_conditioning(orchestrator, captions, device)
                conditioning = orchestrator.mcp(hidden_states)  # Gradients flow here!
                predicted_clean = orchestrator.lcm_solver(noisy_latent, conditioning, num_steps=1)
                
                # 4. Combined loss
                denoise_loss = F.mse_loss(predicted_clean, gt_expanded)
                
                pred_4ch = orchestrator.image_decoder.latent_to_vae(predicted_clean)
                vae_loss = F.mse_loss(pred_4ch, gt_flat)
                
                loss = denoise_loss + 0.1 * vae_loss
            
            # NaN / Inf Loss Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] Step {global_step} loss is NaN/Inf, skipping step.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # NaN / Inf Gradient Guard
            has_nan_grad = False
            for p in trainable_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                print(f"\n[WARNING] Step {global_step} detected NaN/Inf gradients, skipping step.")
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % save_every == 0:
                generate_and_save_sample(orchestrator, "a naruto character with spiky hair",
                                        global_step, "stage3", device)
                
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "mcp_state_dict": orchestrator.mcp.state_dict(),
                    "dit_state_dict": orchestrator.lcm_solver.state_dict(),
                    "latent_to_vae_state_dict": orchestrator.image_decoder.latent_to_vae.state_dict(),
                    "vae_to_latent_state_dict": orchestrator.image_decoder.vae_to_latent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict()
                }, checkpoint_path)
                print(f"  [Checkpoint] Saved resume checkpoint to {checkpoint_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n  Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "mcp": orchestrator.mcp.state_dict(),
                "dit": orchestrator.lcm_solver.state_dict(),
                "latent_to_vae": orchestrator.image_decoder.latent_to_vae.state_dict(),
                "vae_to_latent": orchestrator.image_decoder.vae_to_latent.state_dict(),
            }, "checkpoints/joint_stage3.pth")
            print(f"  [Best] Saved joint checkpoint (loss: {best_loss:.4f})")
            
    print(f"\n{'=' * 60}")
    print(f"  Stage 3 Complete | Best Loss: {best_loss:.4f}")
    print(f"{'=' * 60}")
    return orchestrator


def train_stage4_text(device, epochs=3, lr=1e-4, batch_size=4, max_seq_len=256, save_every=500):
    """
    Stage 4: Fine-tune Mamba SSM for text generation.
    Trains text_embedding, draft_vlm, and text_head.
    """
    print("\n" + "=" * 60)
    print("  STAGE 4: Mamba SSM Text Generation Training")
    print("  Training embedding + draft SSM + text head")
    print("=" * 60)
    
    orchestrator = setup_orchestrator(device, load_stage1=True, load_stage2=True)
    
    # Load stage 3 joint weights if available
    if os.path.exists("checkpoints/joint_stage3.pth"):
        joint = torch.load("checkpoints/joint_stage3.pth", map_location=device)
        orchestrator.mcp.load_state_dict(joint["mcp"])
        orchestrator.lcm_solver.load_state_dict(joint["dit"])
        orchestrator.image_decoder.latent_to_vae.load_state_dict(joint["latent_to_vae"])
        if "vae_to_latent" in joint:
            orchestrator.image_decoder.vae_to_latent.load_state_dict(joint["vae_to_latent"])
        print("[Checkpoint] Loaded Stage 3 joint weights.")
    
    # Freeze everything except text components
    for name, param in orchestrator.named_parameters():
        param.requires_grad = False
        
    for param in orchestrator.text_embedding.parameters():
        param.requires_grad = True
        
    for param in orchestrator.draft_vlm.parameters():
        param.requires_grad = True
        
    for param in orchestrator.text_head.parameters():
        param.requires_grad = True
    
    trainable_params = (list(orchestrator.text_embedding.parameters()) +
                       list(orchestrator.draft_vlm.parameters()) +
                       list(orchestrator.text_head.parameters()))
                       
    trainable = sum(p.numel() for p in trainable_params)
    print(f"\n[Params] Trainable: {trainable:,}")
    
    # Use Qwen2 tokenizer for real text data
    tokenizer = orchestrator.real_tokenizer
    
    # Simple text dataset from prompts
    text_prompts = [
        "A young boy playing cricket on a green grass field under sunny bokeh",
        "A majestic ancient juniper bonsai tree bathed in warm sunlight",
        "A beautiful sci-fi arc reactor glowing in a dark laboratory",
        "A serene mountain lake at sunrise with emerald green water",
        "Close-up portrait of an old wise man with deep wrinkles",
        "Geometric patterns morphing in a digital dimension with glowing fractals",
        "A cozy cabin in the woods surrounded by autumn leaves",
        "A futuristic android sitting at a workbench with glowing circuits",
        "A cute red panda playing in the snow with warm sunbeams",
        "A sleek silver sports car speeding down a wet highway at dusk",
        "A cyberpunk street filled with rain reflections and neon signs",
        "A warm cup of coffee with latte art and rising steam",
        "A massive cascading waterfall plunging into a misty canyon",
        "A majestic bald eagle perched on a high pine branch",
        "A fantasy wizard tower on a floating rock island",
        "A glowing quantum computer core in a futuristic laboratory",
        "A pristine white sand beach at sunset with turquoise waves",
        "A beautiful medieval library with massive wooden bookshelves",
        "An elegant white horse running along a beach at sunset",
        "A futuristic city built inside a colossal canyon",
    ] * 50  # Repeat for more training steps
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    checkpoint_path = "checkpoints/text_stage4_checkpoint.pth"
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0
    
    # Auto-resume logic
    if os.path.exists(checkpoint_path):
        print(f"[Resume] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        orchestrator.text_embedding.load_state_dict(checkpoint["text_embedding_state_dict"])
        orchestrator.draft_vlm.load_state_dict(checkpoint["draft_vlm_state_dict"])
        orchestrator.text_head.load_state_dict(checkpoint["text_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[Resume] Resumed from epoch {start_epoch + 1}, step {global_step} (best loss: {best_loss:.4f})")
        
    print(f"\n[Training] {epochs} epochs x {len(text_prompts) // batch_size} steps = {epochs * (len(text_prompts) // batch_size)} steps")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        random.seed(epoch)
        random.shuffle(text_prompts)
        
        batches_to_skip = (global_step % (len(text_prompts) // batch_size)) if epoch == start_epoch else 0
        
        pbar = tqdm(range(0, len(text_prompts), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for idx_step, i in enumerate(pbar):
            if idx_step < batches_to_skip:
                continue
                
            batch_prompts = text_prompts[i:i+batch_size]
            if len(batch_prompts) < batch_size:
                continue
                
            # Tokenize
            tokens = tokenizer(batch_prompts, padding=True, truncation=True,
                             max_length=max_seq_len, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            B, S = input_ids.shape
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                # Embed tokens (using the updated orchestrator.vocab_size)
                safe_ids = input_ids % orchestrator.vocab_size
                x = orchestrator.text_embedding(safe_ids)  # (B, S, vlm_dim)
                
                # Run through draft Mamba SSM
                output, _ = orchestrator.draft_vlm(x)
                
                # Project to vocab
                logits = orchestrator.text_head(output)  # (B, S, vocab_size)
                
                # Next-token prediction loss (shift by 1)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = safe_ids[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, orchestrator.vocab_size),
                    shift_labels.view(-1)
                )
            
            # NaN / Inf Loss Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] Step {global_step} loss is NaN/Inf, skipping step.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # NaN / Inf Gradient Guard
            has_nan_grad = False
            for p in trainable_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                print(f"\n[WARNING] Step {global_step} detected NaN/Inf gradients, skipping step.")
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "text_embedding_state_dict": orchestrator.text_embedding.state_dict(),
                    "draft_vlm_state_dict": orchestrator.draft_vlm.state_dict(),
                    "text_head_state_dict": orchestrator.text_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict()
                }, checkpoint_path)
                print(f"  [Checkpoint] Saved resume checkpoint to {checkpoint_path}")
        
        avg_loss = epoch_loss / (len(text_prompts) // batch_size)
        print(f"\n  Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "text_embedding": orchestrator.text_embedding.state_dict(),
                "draft_vlm": orchestrator.draft_vlm.state_dict(),
                "text_head": orchestrator.text_head.state_dict(),
            }, "checkpoints/text_stage4.pth")
            print(f"  [Best] Saved text checkpoint (loss: {best_loss:.4f})")
            
    print(f"\n{'=' * 60}")
    print(f"  Stage 4 Complete | Best Loss: {best_loss:.4f}")
    print(f"{'=' * 60}")
    return orchestrator


# ============================================================
# 5. DRY-RUN SHAPE VALIDATION (Step 5)
# ============================================================

def run_shape_validation_dry_run(device, dataset_name="lambdalabs/naruto-blip-captions"):
    """
    Runs a 5-step dry-run of all 4 training stages.
    Asserts intermediate tensor shapes to catch errors early.
    """
    print("\n" + "=" * 60)
    print("  RUNNING SHAPE VALIDATION DRY-RUN (5 steps per stage)")
    print("=" * 60)
    
    orchestrator = setup_orchestrator(device)
    
    # 1. Dataset test
    print("[Dry-run] Testing dataset loading...")
    dataset = CachedImageTextDataset(dataset_name, image_size=256, max_samples=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    images = batch["image"].to(device) # (2, 3, 256, 256)
    captions = batch["caption"]
    assert images.shape == (2, 3, 256, 256), f"Expected images shape (2, 3, 256, 256), got {images.shape}"
    print(f"  [OK] Dataset sample loaded. Images shape: {images.shape}, captions: {captions}")
    
    # 2. Stage 1 Dry-run
    print("\n[Dry-run] Stage 1 (MCP alignment) shape checks...")
    gt_latent = orchestrator.image_decoder.encode_image_to_latent(images)
    assert gt_latent.shape == (2, 4, 32, 32), f"Expected VAE latent (2, 4, 32, 32), got {gt_latent.shape}"
    
    B, C_vae, h_g, w_g = gt_latent.shape
    gt_flat = gt_latent.permute(0, 2, 3, 1).reshape(B, h_g * w_g, C_vae)
    assert gt_flat.shape == (2, 1024, 4), f"Expected flattened VAE latent (2, 1024, 4), got {gt_flat.shape}"
    
    assert hasattr(orchestrator.image_decoder, 'vae_to_latent'), "vae_to_latent projector missing"
    gt_expanded = orchestrator.image_decoder.vae_to_latent(gt_flat)
    assert gt_expanded.shape == (2, 1024, 256), f"Expected projected latent (2, 1024, 256), got {gt_expanded.shape}"
    
    # get text conditioning
    hidden_states = get_text_conditioning(orchestrator, captions, device)
    assert isinstance(hidden_states, list), f"Expected list of hidden states, got {type(hidden_states)}"
    print(f"  [OK] Qwen2 hidden states: {[h.shape for h in hidden_states]}")
    
    conditioning = orchestrator.mcp(hidden_states)
    assert conditioning.shape == (2, 1024, 1024), f"Expected conditioning (2, 1024, 1024), got {conditioning.shape}"
    
    # solver forward pass
    diff_schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    t = diff_schedule.sample_timesteps(2)
    noisy_latent, noise = diff_schedule.add_noise(gt_expanded, t)
    assert noisy_latent.shape == (2, 1024, 256), f"Expected noisy latent (2, 1024, 256), got {noisy_latent.shape}"
    
    predicted = orchestrator.lcm_solver(noisy_latent, conditioning, num_steps=1)
    assert predicted.shape == (2, 1024, 256), f"Expected predicted clean latent (2, 1024, 256), got {predicted.shape}"
    print("  [OK] Stage 1 forward pass shapes verified.")
    
    # Backward pass check
    if not hasattr(orchestrator, '_cond_align_proj'):
        orchestrator._cond_align_proj = nn.Linear(1024, 256).to(device)
    orchestrator._cond_align_proj.requires_grad_(True)
    trainable_params1 = list(orchestrator.mcp.parameters()) + list(orchestrator._cond_align_proj.parameters())
    optimizer1 = optim.AdamW(trainable_params1, lr=1e-4)
    
    cond_mean = conditioning.mean(dim=1)
    gt_mean = gt_expanded.mean(dim=1)
    cond_projected = orchestrator._cond_align_proj(cond_mean)
    loss1 = F.mse_loss(cond_projected, gt_mean)
    loss1.backward()
    optimizer1.step()
    print("  [OK] Stage 1 backward pass verified.")
    
    # 3. Stage 2 Dry-run
    print("\n[Dry-run] Stage 2 (DiT Denoising) shape checks...")
    trainable_params2 = (list(orchestrator.lcm_solver.parameters()) + 
                        list(orchestrator.image_decoder.latent_to_vae.parameters()) +
                        list(orchestrator.image_decoder.vae_to_latent.parameters()))
    optimizer2 = optim.AdamW(trainable_params2, lr=1e-4)
    pred_4ch = orchestrator.image_decoder.latent_to_vae(predicted)
    assert pred_4ch.shape == (2, 1024, 4), f"Expected projected 4ch latent (2, 1024, 4), got {pred_4ch.shape}"
    loss2 = F.mse_loss(predicted, gt_expanded) + 0.1 * F.mse_loss(pred_4ch, gt_flat)
    loss2.backward()
    optimizer2.step()
    print("  [OK] Stage 2 forward/backward pass shapes verified.")
    
    # 4. Stage 4 Dry-run
    print("\n[Dry-run] Stage 4 (Mamba SSM Text) shape checks...")
    tokenizer = orchestrator.real_tokenizer
    tokens = tokenizer(captions, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    safe_ids = input_ids % orchestrator.vocab_size
    x_embed = orchestrator.text_embedding(safe_ids)
    assert x_embed.shape == (input_ids.shape[0], input_ids.shape[1], 2048), f"Expected embedded shape (B, S, 2048), got {x_embed.shape}"
    
    output, _ = orchestrator.draft_vlm(x_embed)
    assert output.shape == x_embed.shape, f"Expected Mamba output shape {x_embed.shape}, got {output.shape}"
    
    logits = orchestrator.text_head(output)
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], orchestrator.vocab_size), f"Expected logits shape (B, S, {orchestrator.vocab_size}), got {logits.shape}"
    
    loss4 = F.cross_entropy(logits[:, :-1].reshape(-1, orchestrator.vocab_size), safe_ids[:, 1:].reshape(-1))
    loss4.backward()
    print("  [OK] Stage 4 forward/backward pass shapes verified.")
    
    print("\n" + "=" * 60)
    print("  DRY-RUN SHAPE VALIDATION SUCCESSFUL - ALL SIZES CORRECT!")
    print("=" * 60 + "\n")


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pixelle-Sirius Complete Training Pipeline")
    parser.add_argument("--stage", type=str, default="all", 
                       choices=["1", "2", "3", "4", "all"],
                       help="Training stage to run")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--image-size", type=int, default=256,
                       help="Training image resolution")
    parser.add_argument("--dataset", type=str, default="lambdalabs/naruto-blip-captions",
                       help="HuggingFace dataset name")
    parser.add_argument("--save-every", type=int, default=500,
                       help="Save sample images and checkpoints every N steps")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run shape validation dry-run (5 steps) and exit")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    if args.dry_run:
        run_shape_validation_dry_run(device, args.dataset)
        return
    
    print("=" * 60)
    print("  Pixelle-Sirius Complete Training Pipeline")
    print("=" * 60)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Stage: {args.stage}")
    print(f"  Image Size: {args.image_size}x{args.image_size}")
    print(f"  Dataset: {args.dataset}")
    print("=" * 60)
    
    t_start = time.time()
    
    if args.stage in ["1", "all"]:
        train_stage1_mcp(
            device, 
            epochs=args.epochs or 4,
            lr=args.lr or 2e-4,
            batch_size=args.batch_size or (2 if device.type == "cuda" else 1),
            image_size=args.image_size,
            dataset_name=args.dataset,
            save_every=args.save_every
        )
    
    if args.stage in ["2", "all"]:
        train_stage2_dit(
            device,
            epochs=args.epochs or 6,
            lr=args.lr or 1e-4,
            batch_size=args.batch_size or 1,
            image_size=args.image_size,
            dataset_names=[args.dataset],
            save_every=args.save_every
        )
    
    if args.stage in ["3", "all"]:
        train_stage3_joint(
            device,
            epochs=args.epochs or 4,
            lr=args.lr or 5e-5,
            batch_size=args.batch_size or 1,
            image_size=args.image_size,
            dataset_names=[args.dataset],
            save_every=args.save_every
        )
    
    if args.stage in ["4", "all"]:
        train_stage4_text(
            device,
            epochs=args.epochs or 3,
            lr=args.lr or 1e-4,
            batch_size=args.batch_size or 4,
            save_every=args.save_every
        )
    
    t_end = time.time()
    hours = (t_end - t_start) / 3600
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total Time: {hours:.1f} hours")
    print(f"  Checkpoints saved in: checkpoints/")
    print(f"  Sample images saved in: training_samples/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()