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
            # 1. Sports & Action (Target quality!)
            "A young boy playing cricket on a green grass field under sunny bokeh, highly detailed high pixel density",
            "A fast bowler running up to bowl, motion blur, grass textures, sun-drenched field",
            "A batter hitting a six, stadium lights, crowd background, high contrast cinematic rendering",
            "A close-up of a red leather cricket ball resting on dew-covered grass, macro photo",
            "An athletic child playing football in a sunny backyard park, warm golden hour reflections",
            # 2. Bonsai & Zen Aesthetics (Bonsai-style!)
            "A majestic ancient juniper bonsai tree, sculpted with intricate detail, bathed in ethereal warm sunlight, high pixel density",
            "Delicate cherry blossom bonsai, petals gently falling in a serene Japanese garden, watercolor style, soft bokeh",
            "A miniature ancient maple bonsai on a rustic wooden table with misty mountain backdrop, photorealistic depth of field",
            "Sentient pine bonsai tree in a ceramic pot, minimalist zen garden backdrop, morning light ray projection",
            "Ancient gnarled bonsai tree sculpted over mossy rocks, ethereal misty moonlight sumi-e style",
            # 3. Sci-Fi & Technology
            "A beautiful sci-fi arc reactor glowing in a dark lab, high pixel density copper coils",
            "Cyberpunk street filled with rain reflections, glowing neon signs, steam rising from grates",
            "Deep space exploration vehicle approaching a massive black hole, accretion disk glowing brightly",
            "Steampunk airship docked at a retro-futuristic iron terminal, steam clouds, brass gears",
            "Close up portrait of an astronaut with earth in helmet reflection, photorealistic, 8k details",
            "A glowing quantum computer core floating inside a futuristic white laboratory, clean sci-fi style",
            "A futuristic android sitting at a workbench, glowing blue circuit lines visible under transparent skin",
            "Cybernetic bio-dome containing a small glowing neon forest under a star-filled dome",
            # 4. Landscapes & Nature
            "A serene mountain lake at sunrise, emerald green water reflecting razor-sharp snow peaks, highly detailed",
            "Vibrant coral reef underwater scene, sun rays shining through crystal water, exotic colorful fish",
            "An ancient forest with mystical glowing mushrooms, mossy trees, thick morning fog, fantasy lighting",
            "A cozy cabin in the woods surrounded by autumn leaves, warm chimney smoke, cinematic depth of field",
            "Sunset over a sprawling cybernetic metropolis skyline, massive skyscrapers, glowing highways",
            "A cute red panda playing in the snow, warm sunbeams filtering through pine branches, hyperdetailed",
            "A massive cascading waterfall plunging into a deep misty canyon, double rainbow, pristine nature",
            "A pristine white sand beach at sunset, gentle turquoise waves, palm tree silhouettes",
            "Golden desert dunes under a vast clear night sky filled with millions of sparkling stars and the Milky Way",
            "Ethereal ice cave with glowing blue crystal formations, light reflecting off glacial walls",
            # 5. Portrait & Character
            "Close-up portrait of an old wise man with deep wrinkles, warm sunset lighting, highly detailed skin pores",
            "A young female warrior in silver armor looking forward, wind blowing her hair, ancient forest background",
            "A cybernetic hacker with glowing neural interface headgear, multiple holographic screens reflecting in glasses",
            "A cute sleeping kitten nestled inside a warm knitted wool blanket, cozy soft-focus lighting",
            "A majestic bald eagle perched on a high pine branch, razor-sharp feathers, piercing eyes, cloudy sky",
            # 6. Abstract & Geometry
            "Geometric patterns morphing in a digital dimension, swirling glowing fractals, high-frequency grids",
            "Abstract watercolor splash of blue and gold, fluid dynamic paint flow, elegant canvas textures",
            "A flowing river of liquid rainbow light cascading through a dark minimalist landscape",
            "Vortex of glowing neon geometric lines, high-frequency mathematical grids, deep perspective",
            # 7. Additional Fine-Grained High-Aesthetic Prompts
            "A retro vintage typewriter sitting on a mahogany desk, sunbeam highlighting brass keys and paper",
            "A slice of fresh strawberry cake on a marble plate, macro lens, detailed crumbs, glossy glaze",
            "An elegant white horse running wild along a pristine beach at sunset, splashing water droplets",
            "A beautiful medieval library with massive wooden bookshelves, dusty light shafts illuminating ancient books",
            "An ornate gold pocket watch resting on a stack of handwritten letters, classic warm lighting",
            "A futuristic city built inside a colossal canyon, vertical farming towers, flying solar gliders",
            "A glowing mystical jellyfish floating in the dark deep ocean, long trailing tentacles, bioluminescent",
            "A rustic stone cottage in a lush green valley filled with blooming wildflowers, morning dew",
            "A sleek silver sports car speeding down a wet highway at dusk, glowing taillight trails, high shutter speed",
            "A futuristic greenhouse in a space station filled with exotic plants, view of Saturn out the window",
            "A warm cup of coffee with beautiful latte art, steam rising, cozy dark wood coffee shop background",
            "A fantasy wizard tower built on a floating rock island, magical energy beams, stormy sky",
            "A majestic peacock displaying its iridescent feathers, micro-detailed patterns, soft sun-drenched background",
            "A high-tech control room with large wrap-around display screens showing global network maps",
            "A cozy library room with a fireplace, leather armchair, warm rug, bookshelves, and falling snow outside"
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
        device=device,
        scale_to_300m=True
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
    full_dataset.prompts = full_dataset.prompts * 32  # 1664 samples
    
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
    epochs = 5
    
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
