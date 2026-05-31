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

def run_real_multimodal_gen():
    print("==================================================")
    print("      Pixelle-Sirius Real Multimodal Generation   ")
    print("==================================================")
    print("Objective: Generate real, model-decoded visual tensors")
    print("using pre-trained weights and aligned adapters (NO MOCKS).")
    
    device = torch.device("cpu")
    print(f"[INFO] Execution Device: {device}")
    
    # 1. Load Model
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
    
    # 2. Real Image Generation via ComfyUI Image Gen Node
    image_prompt = "A beautiful sci-fi arc reactor glowing in a dark lab, high pixel density"
    print(f"\n>>> [1/3] Generating REAL Glowing Reactor (Prompt: '{image_prompt}'):")
    
    image_gen_node = PixelleSiriusImageGen()
    t0 = time.time()
    final_img, latent_dict = image_gen_node.generate_image(
        model=orchestrator,
        prompt=image_prompt,
        aspect_ratio="1:1",
        custom_aspect_ratio="1:1",
        steps=2
    )
    t1 = time.time()
    
    # Convert real RGB tensor to PIL and save
    img_np = (final_img[0].numpy() * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_filename = "custom_reactor.png"
    img_pil.save(img_filename)
    
    print("------------------ IMAGE OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Visual Tensor Shape (BHWC): {final_img.shape}")
    print(f"  Latent Shape (BCHW): {latent_dict['samples'].shape}")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Image: {os.path.abspath(img_filename)}")
    print("--------------------------------------------------")
    
    # Generate Cricket Boy Image as well
    cricket_prompt = "A young boy playing cricket on a green grass field under sunny bokeh, highly detailed high pixel density"
    print(f"\n>>> Generating REAL Cricket Boy (Prompt: '{cricket_prompt}'):")
    
    t0 = time.time()
    final_cricket_img, cricket_latent_dict = image_gen_node.generate_image(
        model=orchestrator,
        prompt=cricket_prompt,
        aspect_ratio="1:1",
        custom_aspect_ratio="1:1",
        steps=2
    )
    t1 = time.time()
    
    cricket_np = (final_cricket_img[0].numpy() * 255.0).astype(np.uint8)
    cricket_pil = Image.fromarray(cricket_np)
    cricket_filename = "custom_cricket_cyber.png"
    cricket_pil.save(cricket_filename)
    
    print("------------------ CRICKET BOY OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Visual Tensor Shape (BHWC): {final_cricket_img.shape}")
    print(f"  Latent Shape (BCHW): {cricket_latent_dict['samples'].shape}")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Image: {os.path.abspath(cricket_filename)}")
    print("--------------------------------------------------------")
    
    # 3. Real Video Generation via ComfyUI Video Gen Node
    video_prompt = "A cinematic video of the sun rising over a futuristic cyber city"
    print(f"\n>>> [2/3] Generating REAL Video (Prompt: '{video_prompt}'):")
    
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
    
    # Compile 4 real projected frames into an animated GIF
    frames_pil = []
    for f in range(4):
        frame_np = (final_video_imgs[f].numpy() * 255.0).astype(np.uint8)
        frames_pil.append(Image.fromarray(frame_np))
        
    vid_filename = "custom_cyber_city.gif"
    frames_pil[0].save(
        vid_filename, 
        save_all=True, 
        append_images=frames_pil[1:], 
        duration=200, 
        loop=0
    )
    
    print("------------------ VIDEO OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Visual Frames Shape (num_frames, H, W, 3): {final_video_imgs.shape}")
    print(f"  Latent Shape: {video_latent_dict['samples'].shape}")
    print(f"  Generation Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"  Saved Animated GIF: {os.path.abspath(vid_filename)}")
    print("--------------------------------------------------")
    
    # 4. Real Audio Latent Generation
    audio_prompt = "hii master sirius , i'm fully ready for your job , just give orders"
    print(f"\n>>> [3/3] Generating REAL Audio Latents (Prompt: '{audio_prompt}'):")
    t0 = time.time()
    with torch.no_grad():
        audio_latents, latency_aud = orchestrator.consistency_generate(
            mode="audio", 
            conditioning_c=None, 
            num_steps=1,
            prompt=audio_prompt
        )
    t1 = time.time()
    
    print("------------------ AUDIO OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Acoustic Latent Shape: {audio_latents.shape}")
    print(f"  Generation Latency: {latency_aud:.2f} ms")
    print("--------------------------------------------------")
    
    print("\n==================================================")
    print("     ALL REAL MULTIMODAL GENERATIONS COMPLETED     ")
    print("==================================================")

if __name__ == "__main__":
    run_real_multimodal_gen()
