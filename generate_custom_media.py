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
from comfy_nodes import PixelleSiriusImageGen, PixelleSiriusVideoGen

def run_custom_generation():
    print("==================================================")
    print("   Pixelle-Sirius: Scaled 300M Media Generator   ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Target Execution Device: {device}")
    
    # 1. Initialize Orchestrator with scale_to_300m=True
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=896,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device,
        scale_to_300m=True
    )
    orchestrator.eval()
    
    print("\n[INFO] Loading pre-trained backbones...")
    orchestrator.load_real_backbones(offload=False)
    
    # Load our newly aligned RL preference weights if present
    rl_weights = "mcp_rl_aligned.pth"
    if os.path.exists(rl_weights):
        print(f"  -> Loading DPO aligned 300M weights: {rl_weights}")
        # Load weights into custom 300M structures
        checkpoint = torch.load(rl_weights, map_location=device)
        mcp_state = {}
        lcm_state = {}
        for k, v in checkpoint.items():
            if k.startswith("mcp") or hasattr(orchestrator, "mcp") and k in orchestrator.mcp.state_dict():
                mcp_state[k] = v
            else:
                lcm_state[k] = v
                
        if mcp_state:
            orchestrator.mcp.load_state_dict(mcp_state, strict=False)
        if lcm_state:
            orchestrator.lcm_solver.load_state_dict(lcm_state, strict=False)
            
    print("\n[OK] Engine successfully loaded!")
    print("==================================================")
    
    # 2. Generate Custom Image
    image_prompt = "A breathtaking custom landscape sunset seen from a cozy mountain cabin window, 4k resolution"
    print(f"\n>>> Generating custom image (Prompt: '{image_prompt}')...")
    
    image_gen_node = PixelleSiriusImageGen()
    t0 = time.time()
    final_img, latent_dict = image_gen_node.generate_image(
        model=orchestrator,
        prompt=image_prompt,
        aspect_ratio="16:9",
        custom_aspect_ratio="16:9",
        steps=2
    )
    t1 = time.time()
    
    # Convert and save
    img_np = (final_img[0].numpy() * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_filename = "cloud_sunset.png"
    img_pil.save(img_filename)
    
    print("------------------ IMAGE OUTPUT ------------------")
    print(f"  Visual Tensor Shape: {final_img.shape}")
    print(f"  Resolution: {img_pil.width}x{img_pil.height}")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Image: {os.path.abspath(img_filename)}")
    print("--------------------------------------------------")
    
    # 3. Generate Custom Video
    video_prompt = "A cinematic video of a flying spaceship approaching a glowing futuristic space terminal"
    print(f"\n>>> Generating custom video (Prompt: '{video_prompt}')...")
    
    video_gen_node = PixelleSiriusVideoGen()
    t0 = time.time()
    final_video_imgs, video_latent_dict = video_gen_node.generate_video(
        model=orchestrator,
        prompt=video_prompt,
        aspect_ratio="16:9",
        custom_aspect_ratio="16:9",
        steps=4
    )
    t1 = time.time()
    
    # Compile frames to GIF
    frames_pil = []
    for f in range(4):
        frame_np = (final_video_imgs[f].numpy() * 255.0).astype(np.uint8)
        frames_pil.append(Image.fromarray(frame_np))
        
    vid_filename = "cloud_space_terminal.gif"
    frames_pil[0].save(
        vid_filename, 
        save_all=True, 
        append_images=frames_pil[1:], 
        duration=200, 
        loop=0
    )
    
    print("------------------ VIDEO OUTPUT ------------------")
    print(f"  Visual Frames Shape: {final_video_imgs.shape}")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Video: {os.path.abspath(vid_filename)}")
    print("--------------------------------------------------")
    print("\n==================================================")

if __name__ == "__main__":
    run_custom_generation()
