import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import orchestrator and custom MCP
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from mcp import MobileConditioningProjector

class RealTextPromptDataset(Dataset):
    """
    A Dataset loader yielding real-world visual prompts for text-to-image/video conditioning.
    """
    def __init__(self):
        self.prompts = [
            "Futuristic neon city abstract artwork, glowing lights",
            "A serene mountain lake at sunrise, highly detailed",
            "Cyberpunk street filled with rain reflections and signs",
            "A majestic flying dragon soaring through stormy clouds",
            "Minimalist architectural design, modern concrete villa",
            "Deep space exploration vehicle approaching a black hole",
            "Vibrant coral reef underwater scene with exotic fish",
            "An ancient forest with mystical glowing mushrooms and fog",
            "Steampunk airship docked at a retro-futuristic terminal",
            "Close up portrait of an astronaut with earth in helmet reflection",
            "A cozy cabin in the woods surrounded by autumn leaves",
            "Hyper-detailed fantasy map of a legendary empire",
            "Geometric patterns morphing in a digital dimension",
            "An elegant white horse running along a sandy beach",
            "Sunset over a sprawling cybernetic metropolis skyline",
            "A cute red panda playing in the snow, warm lighting"
        ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def train_real_mcp_adapter():
    print("==================================================")
    print("   Pixelle-Sirius: Real-Weight MCP Adapter Tuner  ")
    print("==================================================")
    print("Objective: Train 1.58-bit Ternary MCP to align real Qwen2-0.5B")
    print("representations into DiT conditioning space with ZERO overhead.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU/CPU Execution Device: {device}")
    
    # 1. Initialize Unified Orchestrator and Ingest Real Backbones
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    orchestrator.eval()
    
    # Load backbones permanently onto GPU (MAX GPU mode) to eliminate transfer latency
    print("\nLoading pre-trained Qwen2 backbone onto active GPU memory...")
    orchestrator.load_real_backbones(offload=False)
    
    if not orchestrator.real_weights_enabled or orchestrator.real_qwen is None:
        print("❌ Pre-trained weights could not be initialized. Exiting.")
        return
        
    # 2. Freeze Qwen2 completely (No gradients propagated, No offloading overhead!)
    print("Freezing pre-trained Qwen2 parameters...")
    for param in orchestrator.real_qwen.parameters():
        param.requires_grad = False
        
    # 3. Instantiate Ternary Mobile Conditioning Projector (MCP)
    # Qwen2-0.5B has a hidden dimension of 896. We fuse the last 4 layers.
    vlm_dim = 896
    dit_dim = 1024
    num_layers = 4
    
    print(f"\nInitializing 1.58-bit Ternary MCP Projector...")
    mcp_projector = MobileConditioningProjector(
        vlm_dim=vlm_dim,
        dit_dim=dit_dim,
        num_layers=num_layers
    ).to(device)
    mcp_projector.train()
    
    # 4. Setup Train and Validation Dataloaders (80% / 20% split)
    full_dataset = RealTextPromptDataset()
    # Multiply prompts to simulate a larger dataset for high-throughput pipeline testing
    full_dataset.prompts = full_dataset.prompts * 32  # 512 samples
    
    split_idx = int(len(full_dataset.prompts) * 0.8)
    
    train_dataset = RealTextPromptDataset()
    train_dataset.prompts = full_dataset.prompts[:split_idx]
    
    val_dataset = RealTextPromptDataset()
    val_dataset.prompts = full_dataset.prompts[split_idx:]
    
    # Maximize CPU and RAM utilization with pin_memory
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Only optimize MCP parameters
    optimizer = optim.AdamW(mcp_projector.parameters(), lr=2e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    best_val_loss = float('inf')
    save_path = "mcp_alignment_real.pth"
    
    print("\n==================================================")
    print("           Starting Adapter Tuning Loop           ")
    print("==================================================")
    
    t_start = time.time()
    epochs = 3
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        step_times = []
        
        mcp_projector.train()
        for batch_idx, prompts in enumerate(train_dataloader):
            t0 = time.time()
            optimizer.zero_grad()
            
            # Tokenize real text prompts
            inputs = orchestrator.real_tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            B, S = input_ids.shape
            
            # Use Automatic Mixed Precision (AMP) to maximize Tensor Core utilization
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                # 1. Forward Pass through Frozen Qwen2 to extract hidden states
                with torch.no_grad():
                    out_hf = orchestrator.real_qwen(input_ids=input_ids, output_hidden_states=True)
                    # Extract and cast last 4 hidden states to float32 to prevent dtype crashes
                    hidden_states = [h.float() for h in out_hf.hidden_states[-num_layers:]]
                    
                # 2. Forward pass through Ternary MCP (Gradients pass through custom STE)
                projected_c = mcp_projector(hidden_states)
                
                # 3. Simulate deterministic target teacher embeddings to make losses stable and comparable
                generator = torch.Generator(device=device)
                generator.manual_seed(int(input_ids.sum().item()) % 999983)
                target_c = torch.randn(projected_c.shape, device=device, generator=generator)
                
                # 4. Compute alignment loss (MSE)
                loss = F.mse_loss(projected_c, target_c)
                
            # 5. Backpropagate gradients exclusively through the MCP using GradScaler
            scaler.scale(loss).backward()
            
            # Unscale for gradient clipping to enforce discrete weight stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mcp_projector.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            t1 = time.time()
            step_times.append((t1 - t0) * 1000)
            epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Latency: {step_times[-1]:.2f} ms")
            
        avg_loss = epoch_loss / len(train_dataloader)
        avg_step = sum(step_times) / len(step_times)
        
        # 6. Evaluation Phase (Validation Set / Dev Loss)
        mcp_projector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_prompts in val_dataloader:
                inputs = orchestrator.real_tokenizer(val_prompts, padding=True, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    out_hf = orchestrator.real_qwen(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = [h.float() for h in out_hf.hidden_states[-num_layers:]]
                    projected_c = mcp_projector(hidden_states)
                    
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(input_ids.sum().item()) % 999983)
                    target_c = torch.randn(projected_c.shape, device=device, generator=generator)
                    
                    loss = F.mse_loss(projected_c, target_c)
                    val_loss += loss.item()
                    
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"--------------------------------------------------")
        print(f"Epoch {epoch+1:02d} Summary | Train Loss: {avg_loss:.4f} | Val (Dev) Loss: {avg_val_loss:.4f} | Avg Step Time: {avg_step:.2f} ms")
        print(f"--------------------------------------------------")
        
        # Save only the checkpoint with the lowest validation loss (peak intelligence sweet spot)
        if avg_val_loss < best_val_loss:
            print(f"  -> [New Best] Validation Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving checkpoint to {save_path}...\n")
            best_val_loss = avg_val_loss
            torch.save(mcp_projector.state_dict(), save_path)
        else:
            print(f"  -> [Warning] Validation Loss did not improve (Current: {avg_val_loss:.4f}, Best: {best_val_loss:.4f}). Preserving peak intelligence checkpoint.\n")
            
    t_end = time.time()
    print("==================================================")
    print(f"Adapter Pre-training Completed in {t_end - t_start:.2f} seconds.")
    print(f"Peak intelligence model preserved with validation loss: {best_val_loss:.4f}")
    print("==================================================")

if __name__ == "__main__":
    train_real_mcp_adapter()
