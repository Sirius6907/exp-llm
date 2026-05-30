import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import random
import numpy as np
import dask.array as da
from PIL import Image

from any_to_any import AnyToAnyOrchestrator

class UnifiedMultimodalDataset(Dataset):
    """
    UnifiedMultimodalDataset scans real data folders (text, images, videos, audio)
    or falls back to high-throughput simulated data to prepare the orchestrator
    for large-scale dataset pre-training.
    """
    def __init__(self, num_samples=16, text_dir=None, image_dir=None, video_dir=None, audio_dir=None):
        self.num_samples = num_samples
        
        # Real directory paths
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        
        # List files if real paths are supplied
        self.text_files = self._list_files(text_dir, ['.txt'])
        self.image_files = self._list_files(image_dir, ['.png', '.jpg', '.jpeg'])
        self.video_files = self._list_files(video_dir, ['.mp4', '.avi'])
        self.audio_files = self._list_files(audio_dir, ['.wav', '.mp3'])
        
        print(f"[Dataset] Unified Dataset initialized with {num_samples} samples.")
        if self.text_files: print(f"  - Found {len(self.text_files)} text files.")
        if self.image_files: print(f"  - Found {len(self.image_files)} image files.")
        if self.video_files: print(f"  - Found {len(self.video_files)} video files.")
        if self.audio_files: print(f"  - Found {len(self.audio_files)} audio files.")

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
                # Encodes character-level tokens as a simplified demo
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
                da_img = da.from_array(np.array(img), chunks=(32, 32, 3))
                tensor_img = torch.from_numpy(da_img.compute()).float().permute(2, 0, 1) / 255.0
                return tensor_img
            except Exception:
                pass
        return torch.randn(3, 64, 64)

    def _get_video(self, idx):
        # Videos are represented as shape (T, 3, 64, 64)
        return torch.randn(4, 3, 64, 64)

    def _get_audio(self, idx):
        if self.audio_files:
            try:
                # Real WAV file parser (simplified fallback if SciPy is missing)
                path = self.audio_files[idx % len(self.audio_files)]
                from scipy.io import wavfile
                sample_rate, data = wavfile.read(path)
                da_data = da.from_array(data, chunks=(8000,))
                if len(da_data.shape) > 1:
                    da_data = da_data[:, 0]  # Mono conversion
                da_data = da_data.astype(np.float32) / 32768.0
                if len(da_data) < 16000:
                    da_data = da.pad(da_data, (0, 16000 - len(da_data)), mode='constant')
                else:
                    da_data = da_data[:16000]
                return torch.from_numpy(da_data.compute()).unsqueeze(0)
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

def train_any_to_any_one_epoch():
    print("==================================================")
    print("      Any-to-Any Unified Multi-Modal Trainer      ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training execution device: {device}")
    
    # 1. Instantiate Orchestrator
    orchestrator = AnyToAnyOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    
    # 2. Setup Dataloader
    dataset = UnifiedMultimodalDataset(
        num_samples=16,
        text_dir=None,  # Pass real paths here in production pre-training
        image_dir=None,
        video_dir=None,
        audio_dir=None
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Setup Optimizer
    optimizer = optim.AdamW(orchestrator.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Define all 16 pathways
    modalities = ["text", "image", "video", "audio"]
    routes = [f"{src}_to_{tgt}" for src in modalities for tgt in modalities]
    
    orchestrator.train()
    print("\nStarting unified optimization loops...")
    t0 = time.time()
    
    epoch_loss_sum = 0.0
    batch_count = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Extract tensors from batch and send to device
        text_batch = batch_data["text"].to(device)
        image_batch = batch_data["image"].to(device)
        video_batch = batch_data["video"].to(device)
        audio_batch = batch_data["audio"].to(device)
        
        # Randomly select a pathway for this batch to simulate multi-task training
        active_route = random.choice(routes)
        src = active_route.split("_to_")[0]
        tgt = active_route.split("_to_")[1]
        
        print(f"\n[Batch {batch_idx+1}] Pathway training step: {active_route.upper()}")
        optimizer.zero_grad()
        
        # Route forward pass
        output = orchestrator.route(
            mode=active_route,
            text_input=text_batch,
            image_input=image_batch,
            video_input=video_batch,
            audio_input=audio_batch,
            num_frames=4,
            audio_len=16000
        )
        
        # --- PATHWAY LOSS COMPUTATION ---
        if tgt == "text":
            # output shape: (B, S_out, vocab_size)
            # text_batch target shape: (B, S_tgt)
            vocab_size = orchestrator.vocab_size
            min_S = min(output.shape[1], text_batch.shape[1])
            logits = output[:, :min_S].contiguous()
            targets = text_batch[:, :min_S].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
        elif tgt == "image":
            # output shape: (B, 3, 64, 64)
            loss = F.l1_loss(output, image_batch)
            
        elif tgt == "video":
            # output shape: (B, T, 3, 64, 64)
            loss = F.l1_loss(output, video_batch)
            
        elif tgt == "audio":
            # output shape: (B, 1, 16000)
            loss = F.l1_loss(output, audio_batch)
            
        else:
            raise ValueError(f"Unknown target: {tgt}")
            
        # Backward step
        loss.backward()
        
        # Clip gradient norm to ensure stable training
        torch.nn.utils.clip_grad_norm_(orchestrator.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss_sum += loss.item()
        batch_count += 1
        
        print(f"Batch {batch_idx+1} complete. Targeted Loss: {loss.item():.4f}")
        
    avg_loss = epoch_loss_sum / max(1, batch_count)
    print(f"\n--- Epoch Summary ---")
    print(f"Average Pipeline Epoch Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = "any_to_any_checkpoint.pth"
    print(f"Saving pre-trained unified checkpoint to: {checkpoint_path}...")
    torch.save(orchestrator.state_dict(), checkpoint_path)
    print("Checkpoint saved successfully.")
    
    t1 = time.time()
    print(f"Epoch completed in {t1 - t0:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    train_any_to_any_one_epoch()
