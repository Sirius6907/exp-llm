import os
import sys
import torch
import time
import numpy as np
from PIL import Image

# Set console encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import local modules
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from comfy_nodes import PixelleSiriusImageGen

def run_sunset_gen():
    print("==================================================")
    print("      Pixelle-Sirius Custom Sunset Generation     ")
    print("==================================================")
    
    device = torch.device("cpu")
    print(f"[INFO] Target Execution Device: {device}")
    
    # 1. Initialize Orchestrator
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device="cpu"
    )
    orchestrator.eval()
    
    print("\n[INFO] Loading pre-trained backbones...")
    orchestrator.load_real_backbones(offload=False)
    
    # Load our newly aligned RL preference weights if present
    rl_weights = "mcp_rl_aligned.pth"
    if os.path.exists(rl_weights):
        print(f"  -> Loading RL DPO aligned weights: {rl_weights}")
        if hasattr(orchestrator, "mcp"):
            orchestrator.mcp.load_state_dict(torch.load(rl_weights, map_location=device), strict=False)
            
    print("\n[OK] Engine successfully loaded!")
    print("==================================================")
    
    # 2. Run Image Generation
    prompt = "GENERATE A SUNSET VIEW FROM A WINDOW FROM HOUSE"
    print(f"\nFeeding prompt to model: '{prompt}'...")
    
    image_gen_node = PixelleSiriusImageGen()
    t0 = time.time()
    
    # Generate 16:9 widescreen sunset image
    final_img, latent_dict = image_gen_node.generate_image(
        model=orchestrator,
        prompt=prompt,
        aspect_ratio="16:9",
        custom_aspect_ratio="16:9",
        steps=2
    )
    t1 = time.time()
    
    # Convert RGB tensor (0.0 - 1.0) to PIL image and save
    img_np = (final_img[0].numpy() * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_filename = "custom_sunset_output.png"
    img_pil.save(img_filename)
    
    print("\n------------------ GENERATION OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Visual Tensor Shape (BHWC): {final_img.shape}")
    print(f"  Latent Shape (BCHW): {latent_dict['samples'].shape}")
    print(f"  Resolution: {img_pil.width}x{img_pil.height} (Super High-Resolution 1024px)")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Image: {os.path.abspath(img_filename)}")
    print("-------------------------------------------------------")
    print("\n==================================================")

if __name__ == "__main__":
    run_sunset_gen()
