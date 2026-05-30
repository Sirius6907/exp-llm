import os
os.environ["USE_LIBUV"] = "0"
import sys
import math
import random
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from any_to_any import AnyToAnyOrchestrator

class LargeScaleMultimodalDataset(Dataset):
    """
    Production-grade Dataset loader designed for distributed high-throughput DDP streams.
    Loads real dataset paths or falls back to synthetic tensors.
    """
    def __init__(self, num_samples=64, text_dir=None, image_dir=None, video_dir=None, audio_dir=None):
        self.num_samples = num_samples
        
        # Real directory paths
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        
        # Scan directories
        self.text_files = self._list_files(text_dir, ['.txt'])
        self.image_files = self._list_files(image_dir, ['.png', '.jpg', '.jpeg'])
        self.video_files = self._list_files(video_dir, ['.mp4', '.avi'])
        self.audio_files = self._list_files(audio_dir, ['.wav', '.mp3'])

    def _list_files(self, directory, extensions):
        if not directory or not os.path.exists(directory):
            return []
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, f))
        return files

    def __len__(self):
        return self.num_samples

    def _get_text(self, idx):
        if self.text_files:
            try:
                path = self.text_files[idx % len(self.text_files)]
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                tokens = [ord(c) % 32000 for c in text[:512]]
                if len(tokens) < 512:
                    tokens += [0] * (512 - len(tokens))
                return torch.tensor(tokens, dtype=torch.long)
            except Exception:
                pass
        return torch.randint(0, 32000, (512,), dtype=torch.long)

    def _get_image(self, idx):
        if self.image_files:
            try:
                path = self.image_files[idx % len(self.image_files)]
                img = Image.open(path).convert('RGB').resize((64, 64))
                return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            except Exception:
                pass
        return torch.randn(3, 64, 64)

    def _get_video(self, idx):
        return torch.randn(4, 3, 64, 64)

    def _get_audio(self, idx):
        if self.audio_files:
            try:
                path = self.audio_files[idx % len(self.audio_files)]
                from scipy.io import wavfile
                sample_rate, data = wavfile.read(path)
                if len(data.shape) > 1:
                    data = data[:, 0]
                data = data.astype(np.float32) / 32768.0
                if len(data) < 16000:
                    data = np.pad(data, (0, 16000 - len(data)))
                else:
                    data = data[:16000]
                return torch.from_numpy(data).unsqueeze(0)
            except Exception:
                pass
        return torch.randn(1, 16000)

    def __getitem__(self, idx):
        return {
            "text": self._get_text(idx),
            "image": self._get_image(idx),
            "video": self._get_video(idx),
            "audio": self._get_audio(idx)
        }

def setup_ddp():
    """Initializes distributed process group across multiple GPUs."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running under torchrun launcher
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        # Bypassed / Local single process execution
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_large_scale():
    # 1. Initialize Distributed Environment
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)
    
    if is_master:
        print("==================================================")
        print("      Distributed Large-Scale DDP Pre-Trainer     ")
        print("==================================================")
        print(f"Total Rank Processes (World Size): {world_size}")
        
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    if is_master:
        print(f"Rank Master executing on device: {device}")
        
    # 2. Instantiate Base Model
    orchestrator = AnyToAnyOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    
    # 3. Wrap Model in DistributedDataParallel
    if world_size > 1:
        orchestrator = nn.parallel.DistributedDataParallel(
            orchestrator,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True
        )
        
    # 4. Setup Distributed Dataset & Sampler
    dataset = LargeScaleMultimodalDataset(
        num_samples=32,
        text_dir=None,
        image_dir=None,
        video_dir=None,
        audio_dir=None
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, pin_memory=True, num_workers=4, persistent_workers=True)
    
    # 5. Optimizer & Mixed Precision Scaler
    optimizer = optim.AdamW(orchestrator.parameters(), lr=2e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Define pathways and pre-training steps
    modalities = ["text", "image", "video", "audio"]
    routes = [f"{src}_to_{tgt}" for src in modalities for tgt in modalities]
    
    # Load checkpoint if resuming training
    checkpoint_path = "large_scale_checkpoint.pth"
    start_epoch = 0
    global_step = 0
    
    if os.path.exists(checkpoint_path):
        if is_master:
            print(f"Loading existing checkpoint to resume: {checkpoint_path}...")
        # Load maps tensors to corresponding GPU ranks safely to prevent memory leak
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if world_size > 1:
            orchestrator.module.load_state_dict(checkpoint["model_state"])
        else:
            orchestrator.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        
    orchestrator.train()
    
    # Hyperparameters for large-scale training
    epochs = 1
    gradient_accumulation_steps = 4
    warmup_steps = 10
    total_steps = len(dataloader) * epochs
    
    if is_master:
        print(f"Starting large-scale distributed loops. Warmup steps: {warmup_steps} | Accumulation: {gradient_accumulation_steps}")
        
    t0 = time.time()
    
    for epoch in range(start_epoch, epochs):
        # Set epoch on DistributedSampler to ensure correct random seeds across epochs
        sampler.set_epoch(epoch)
        
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Send batch data to current local device
            text_batch = batch_data["text"].to(device, non_blocking=True)
            image_batch = batch_data["image"].to(device, non_blocking=True)
            video_batch = batch_data["video"].to(device, non_blocking=True)
            audio_batch = batch_data["audio"].to(device, non_blocking=True)
            
            # Select route randomly
            active_route = random.choice(routes)
            src = active_route.split("_to_")[0]
            tgt = active_route.split("_to_")[1]
            
            # Dynamic Learning Rate warmup + cosine decay
            global_step += 1
            if global_step < warmup_steps:
                lr_mult = float(global_step) / float(max(1, warmup_steps))
            else:
                progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-4 * lr_mult
                
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = orchestrator.module.route(
                    mode=active_route,
                    text_input=text_batch,
                    image_input=image_batch,
                    video_input=video_batch,
                    audio_input=audio_batch,
                    num_frames=4,
                    audio_len=16000
                ) if world_size > 1 else orchestrator.route(
                    mode=active_route,
                    text_input=text_batch,
                    image_input=image_batch,
                    video_input=video_batch,
                    audio_input=audio_batch,
                    num_frames=4,
                    audio_len=16000
                )
                
                # --- targeted DDP Loss ---
                if tgt == "text":
                    vocab_size = 32000
                    min_S = min(output.shape[1], text_batch.shape[1])
                    logits = output[:, :min_S].contiguous()
                    targets = text_batch[:, :min_S].contiguous()
                    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                elif tgt == "image":
                    loss = F.l1_loss(output, image_batch)
                elif tgt == "video":
                    loss = F.l1_loss(output, video_batch)
                elif tgt == "audio":
                    loss = F.l1_loss(output, audio_batch)
                
                # Adjust loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                # Clip gradients to enforce weight stability
                if world_size > 1:
                    torch.nn.utils.clip_grad_norm_(orchestrator.module.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(orchestrator.parameters(), max_norm=1.0)
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            if is_master and ((batch_idx + 1) % 2 == 0 or batch_idx == len(dataloader) - 1):
                print(f"[Batch {batch_idx+1}/{len(dataloader)}] Pathway: {active_route.upper()} | Loss: {loss.item()*gradient_accumulation_steps:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
        # Save checkpoints periodically on rank 0 only to prevent file locks
        if is_master:
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\n--- Epoch {epoch+1} Completed ---")
            print(f"Average Large-Scale Pipeline Loss: {avg_epoch_loss:.4f}")
            
            # Pack save weights
            save_model_state = orchestrator.module.state_dict() if world_size > 1 else orchestrator.state_dict()
            checkpoint = {
                "model_state": save_model_state,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Rank Master saved checkpoint: {checkpoint_path}\n")
            
    t1 = time.time()
    if is_master:
        print(f"Completed DDP Pre-Training in {t1 - t0:.2f} seconds.")
        print("==================================================")
        
    cleanup_ddp()

if __name__ == "__main__":
    train_large_scale()
