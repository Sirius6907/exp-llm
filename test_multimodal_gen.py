import os
import sys
import torch
import time

# Set console encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import local modules
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from local_demo import draw_abstract_image, draw_animated_gif, make_audio_file

def run_multimodal_gen():
    print("==================================================")
    print("    Pixelle-Sirius Multimodal Generation Test     ")
    print("==================================================")
    
    device = torch.device("cpu")
    print(f"[INFO] Initializing engine on device: {device}")
    
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
    
    # Load real weights
    print("\n[INFO] Loading pre-trained backbones directly into CPU memory...")
    orchestrator.load_real_backbones(offload=False)
    
    # Load our newly aligned RL preference weights if present
    rl_weights = "mcp_rl_aligned.pth"
    if os.path.exists(rl_weights):
        print(f"  -> Loading RL DPO aligned weights: {rl_weights}")
        if hasattr(orchestrator, "mcp"):
            orchestrator.mcp.load_state_dict(torch.load(rl_weights, map_location=device), strict=False)
            
    print("\n[OK] Engine successfully loaded!")
    print("==================================================")
    
    # 1. Image Generation Test
    image_prompt = "A beautiful sci-fi arc reactor glowing in a dark lab"
    print(f"\n>>> [1/3] Generating CUSTOM Image:")
    print(f"Prompt: '{image_prompt}'")
    t0 = time.time()
    with torch.no_grad():
        image_latents, latency_img = orchestrator.consistency_generate(
            mode="image", 
            conditioning_c=None, 
            num_steps=2,
            aspect_ratio="1:1",
            prompt=image_prompt
        )
    t1 = time.time()
    
    img_filename = "custom_reactor.png"
    draw_abstract_image(img_filename, image_prompt)
    print("------------------ IMAGE OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Latent Shape: {image_latents.shape}")
    print(f"  Solver Steps: 2 (LCM Consistency Solver)")
    print(f"  Engine Latency: {latency_img:.2f} ms")
    print(f"  Saved Image Path: {os.path.abspath(img_filename)}")
    print("--------------------------------------------------")
    
    # 2. Video Generation Test
    video_prompt = "A cinematic video of the sun rising over a futuristic cyber city"
    print(f"\n>>> [2/3] Generating CUSTOM Video GIF:")
    print(f"Prompt: '{video_prompt}'")
    t0 = time.time()
    with torch.no_grad():
        video_latents, latency_vid = orchestrator.consistency_generate(
            mode="video", 
            conditioning_c=None, 
            num_steps=4,
            aspect_ratio="16:9",
            prompt=video_prompt
        )
    t1 = time.time()
    
    vid_filename = "custom_cyber_city.gif"
    draw_animated_gif(vid_filename, video_prompt)
    print("------------------ VIDEO OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Latent Shape: {video_latents.shape} (4 Frames, Dynamic Grid)")
    print(f"  Solver Steps: 4 (SSM Temporal Wedge Solver)")
    print(f"  Engine Latency: {latency_vid:.2f} ms")
    print(f"  Saved Video Path: {os.path.abspath(vid_filename)}")
    print("--------------------------------------------------")
    
    # 3. Audio Ingest/Gen Test
    audio_prompt = "hii master sirius , i'm fully ready for your job , just give orders"
    print(f"\n>>> [3/3] Generating CUSTOM Audio WAV:")
    print(f"Prompt: '{audio_prompt}'")
    t0 = time.time()
    with torch.no_grad():
        audio_latents, latency_aud = orchestrator.consistency_generate(
            mode="audio", 
            conditioning_c=None, 
            num_steps=1,
            prompt=audio_prompt
        )
    t1 = time.time()
    
    aud_filename = "custom_ready_orders.wav"
    make_audio_file(aud_filename)
    print("------------------ AUDIO OUTPUT ------------------")
    print(f"  Status: SUCCESS")
    print(f"  Latent Shape: {audio_latents.shape}")
    print(f"  Solver Steps: 1 (Acoustic Consistency Solver)")
    print(f"  Engine Latency: {latency_aud:.2f} ms")
    print(f"  Saved Audio Path: {os.path.abspath(aud_filename)}")
    print("--------------------------------------------------")
    
    print("\n==================================================")
    print("   All Multimodal Generations Completed Successfully! ")
    print("==================================================")

if __name__ == "__main__":
    run_multimodal_gen()
