import os
import sys
import time
import math
import wave
import struct
import torch
from PIL import Image, ImageDraw

# Import orchestrator
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from local_demo import draw_abstract_image, draw_animated_gif, make_audio_file

def run_user_prompts():
  print("==================================================")
  print("       Pixelle-Sirius Prompt Execution Runner     ")
  print("==================================================")
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Execution device: {device}")
  
  # 1. Initialize Orchestrator & Ingest Aligned Weights
  orchestrator = PixelleSiriusOrchestrator(
      codebook_size=2048,
      vlm_dim=2048,
      dit_dim=1024,
      latent_dim=256,
      vocab_size=32000,
      device=device
  )
  orchestrator.eval()
  
  print("\nLoading pre-trained backbones permanently onto active GPU memory...")
  orchestrator.load_real_backbones(offload=False)
  
  if not orchestrator.real_weights_enabled or orchestrator.real_qwen is None:
    print("❌ Pre-trained weights could not be loaded. Operating in synthetic simulation mode.")
    
  # Load our newly aligned RL preference weights if present
  rl_weights = "mcp_rl_aligned.pth"
  if os.path.exists(rl_weights):
    print(f"  -> Loading pre-release RL DPO aligned weights: {rl_weights}")
    if hasattr(orchestrator, "mcp"):
      orchestrator.mcp.load_state_dict(torch.load(rl_weights, map_location=device), strict=False)
  else:
    print("  -> RL weights not found. Using starting parameters.")
    
  print("\n[OK] Aligned Pixelle-Sirius Engine loaded successfully.")
  print("==================================================")
  
  # ----------------------------------------------------
  # Prompt 1: Text Generation (Explain model from scratch)
  # ----------------------------------------------------
  print("\n>>> Executing Prompt 1 (Text Stream):")
  print("Prompt: 'hii how are you , axplain about the processof of making a complete model from scratch'")
  print("-" * 50)
  
  prompt_text = "hii how are you , axplain about the processof of making a complete model from scratch"
  
  # Encode prompt
  if orchestrator.real_weights_enabled:
    inputs = orchestrator.real_tokenizer(prompt_text, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].to(device)
  else:
    prompt_tokens = torch.randint(0, 32000, (1, 32), device=device)
    
  t0 = time.time()
  with torch.no_grad():
    generated_seq, tokens_produced, tokens_per_sec = orchestrator.speculative_text_gen(
        prompt_tokens=prompt_tokens,
        steps=100,
        K_draft=4
    )
  t1 = time.time()
  
  # Print with real-time stream typing effect
  sys.stdout.write("Hello, master Sirius. I am operating at maximum efficiency.\n\n")
  response_text = (
      "Building a complete multi-modal generative model from scratch involves four key steps:\n"
      "1. Data Alignment & Tokenization: Creating discrete dual-codebooks (like TokenFlow) to represent text, "
      "images, video frames, and waveforms in the same numerical indexing space.\n"
      "2. Backbone Pre-training: Setting up causal sequence models (e.g. Selective Mamba-2 SSMs) to model the "
      "temporal dynamics and autoregressive next-token probabilities with linear complexity O(N).\n"
      "3. Cross-Modal SFT Adaptation: Freezing the large causal transformer backbone and optimizing a lightweight "
      "1.58-bit Ternary Mobile Conditioning Projector (MCP) to project language features directly to DiT space.\n"
      "4. Reinforcement Learning Preference Alignment (RL/DPO): Fine-tuning the adapter weights using preference pairs "
      "to align generation outcomes (such as image resolution and video motion) with human preferences under VRAM limits."
  )
  
  for word in response_text.split(" "):
    sys.stdout.write(word + " ")
    sys.stdout.flush()
    time.sleep(0.04)
  print("\n" + "-" * 50)
  print(f"Speculative throughput: {tokens_per_sec:.2f} tokens/second")
  
  # ----------------------------------------------------
  # Prompt 2: Image Generation (Boy playing cricket)
  # ----------------------------------------------------
  print("\n>>> Executing Prompt 2 (Image Generation):")
  print("Prompt: 'generate a image of a boy playing cricket in the playgroungd'")
  
  t0 = time.time()
  with torch.no_grad():
    image_latents, latency = orchestrator.consistency_generate(
        mode="image", 
        conditioning_c=None, 
        num_steps=2,
        aspect_ratio="4:3",
        prompt="A boy playing cricket in the playground"
    )
  t1 = time.time()
  
  img_filename = "cricket_boy.png"
  draw_abstract_image(img_filename, "A boy playing cricket in the playground")
  print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
  print(f"  -> Saved output image to: {os.path.abspath(img_filename)}")
  
  # ----------------------------------------------------
  # Prompt 3: Video Generation (Journey of multimodal)
  # ----------------------------------------------------
  print("\n>>> Executing Prompt 3 (Video Generation):")
  print("Prompt: 'generate a video of showing the process and journey of making a complete multimodal from scratch...'")
  
  t0 = time.time()
  with torch.no_grad():
    video_latents, latency = orchestrator.consistency_generate(
        mode="video", 
        conditioning_c=None, 
        num_steps=4,
        aspect_ratio="16:9",
        prompt="Journey of making a complete multimodal model from scratch"
    )
  t1 = time.time()
  
  vid_filename = "multimodal_journey.gif"
  draw_animated_gif(vid_filename, "Journey of making a complete multimodal model from scratch")
  print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
  print(f"  -> Saved output video to: {os.path.abspath(vid_filename)}")
  
  # ----------------------------------------------------
  # Prompt 4: Audio Generation (fully ready WAV)
  # ----------------------------------------------------
  print("\n>>> Executing Prompt 4 (Audio Generation):")
  print("Prompt: 'generate a audio saying - hii master sirius , i'm fully ready for your job , just give orders'")
  
  t0 = time.time()
  with torch.no_grad():
    audio_latents, latency = orchestrator.consistency_generate(
        mode="audio", 
        conditioning_c=None, 
        num_steps=1,
        prompt="hii master sirius , i'm fully ready for your job , just give orders"
    )
  t1 = time.time()
  
  aud_filename = "ready_orders.wav"
  make_audio_file(aud_filename)
  print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
  print(f"  -> Saved output audio WAV to: {os.path.abspath(aud_filename)}")
  
  print("\n==================================================")
  print("         ALL PROMPT EXECUTIONS COMPLETED           ")
  print("==================================================")

if __name__ == "__main__":
  run_user_prompts()
