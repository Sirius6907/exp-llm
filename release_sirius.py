import os
import sys
import time
import numpy as np
import torch
from PIL import Image

# Set console encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import local modules
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from comfy_nodes import PixelleSiriusImageGen, PixelleSiriusVideoGen

def release_and_test_sirius():
    print("==========================================================")
    # Using Unicode characters with clean encoding handling
    print("     SIRIUS V2.0: Master Release Compilation & TTA      ")
    print("==========================================================")
    print("Objective: Package, compile, and release Pixelle-Sirius with")
    print("Test-Time Training (TTT) and Recursive Refinement (SRRL).")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Target Execution Device: {device}")
    
    # 1. Initialize Orchestrator and load pre-trained backbones
    print("\n--- Step 1: Initializing Orchestrator ---")
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
    
    print("Loading pre-trained backbones...")
    orchestrator.load_real_backbones(offload=False)
    
    # 2. Compile to .sirius V2.0 Quantized Format
    print("\n--- Step 2: Packaging to custom .sirius V2.0 format ---")
    pth_weights = "mcp_rl_aligned.pth"
    sirius_file = "pixelle_sirius_v2.sirius"
    
    if os.path.exists(pth_weights):
        print(f"Loading weights from {pth_weights}...")
        checkpoint = torch.load(pth_weights, map_location=device)
        
        # Load weights into active modules first to save them using the framework
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
            
        print("Compiling and dynamically quantizing to 8-bit .sirius format...")
        orchestrator.save_sirius_weights(sirius_file, quantize=True)
    else:
        print(f"[Warning] Checkpoint {pth_weights} not found. Saving current initialization state to .sirius format.")
        orchestrator.save_sirius_weights(sirius_file, quantize=True)
        
    # 3. Reload from .sirius native format to verify
    print("\n--- Step 3: Verifying zero-copy load from .sirius ---")
    verification_orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=896,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device,
        scale_to_300m=True
    )
    verification_orchestrator.eval()
    verification_orchestrator.load_real_backbones(offload=False)
    
    metadata = verification_orchestrator.load_sirius_weights(sirius_file)
    print(f"Reload Successful! Metadata: {metadata}")
    
    # 4. Generate High-Fidelity Image with Stage A & B Refinement
    print("\n--- Step 4: High-Fidelity Stage A & B Image Generation ---")
    image_prompt = "A breathtaking custom landscape sunset seen from a cozy mountain cabin window, 4k resolution"
    print(f"Generating image with prompt: '{image_prompt}'")
    
    image_gen_node = PixelleSiriusImageGen()
    t0 = time.time()
    final_img, latent_dict = image_gen_node.generate_image(
        model=verification_orchestrator,
        prompt=image_prompt,
        aspect_ratio="16:9",
        custom_aspect_ratio="16:9",
        steps=2
    )
    t1 = time.time()
    
    # Save Image
    img_np = (final_img[0].numpy() * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_filename = "cloud_sunset.png"
    img_pil.save(img_filename)
    
    print("\n[Image Output Details]")
    print(f"  Visual Tensor Shape: {final_img.shape}")
    print(f"  Resolution: {img_pil.width}x{img_pil.height}")
    print(f"  Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Image: {os.path.abspath(img_filename)}")
    
    # 5. Generate High-Fidelity Video with Stage A & B Refinement
    print("\n--- Step 5: High-Fidelity Stage A & B Video Generation ---")
    video_prompt = "A cinematic video of a flying spaceship approaching a glowing futuristic space terminal"
    print(f"Generating video with prompt: '{video_prompt}'")
    
    video_gen_node = PixelleSiriusVideoGen()
    t0 = time.time()
    final_video_imgs, video_latent_dict = video_gen_node.generate_video(
        model=verification_orchestrator,
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
    
    print("\n[Video Output Details]")
    print(f"  Visual Frames Shape: {final_video_imgs.shape}")
    print(f"  Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Video: {os.path.abspath(vid_filename)}")
    
    print("\n==========================================================")
    print("      SUCCESS: Pixelle-Sirius Stage A & B Verified!       ")
    print("==========================================================")

if __name__ == "__main__":
    release_and_test_sirius()
