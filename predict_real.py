import torch
import time
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def run_real_prediction_pipeline():
    print("==================================================")
    print("   Pixelle-Sirius: Real-Weight Inference Pipeline ")
    print("==================================================")
    print("Zero-Overhead Active Modal Routing Path (Inference)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU/CPU Execution Device: {device}")
    
    # 1. Initialize the Unified Orchestrator
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    orchestrator.eval()
    
    # 2. Ingest Pre-trained Backbones (Phase 4 Real Weights)
    print("\nLoading pre-trained Qwen2, SigLIP, and Whisper backbones directly to GPU (MAX GPU Mode)...")
    orchestrator.load_real_backbones(offload=False)
    
    if not orchestrator.real_weights_enabled:
        print("❌ Could not initialize pre-trained backbones. Exiting.")
        return
        
    print("\n==================================================")
    print("   Running Zero-Overhead Any-to-Any Inference     ")
    print("==================================================")
    
    # Pathway 1: Text-to-Image Generation (Real Text -> MCP -> LCM Solver)
    print("\n--- Route 1: TEXT-TO-IMAGE ---")
    prompt = "Futuristic neon city abstract artwork"
    print(f"Input Prompt: '{prompt}'")
    
    t0 = time.time()
    with torch.no_grad():
        # Encode using real Qwen2 text backbone
        inputs = orchestrator.real_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Get real text features
        out_hf = orchestrator.real_qwen(input_ids=input_ids)
        text_features = out_hf.last_hidden_state # (1, S, 896)
        
        # Ingest text features into consistency solver to generate image latents
        # Denoises real text representation to a 256-latent image in only 2 steps!
        image_latents, latency = orchestrator.consistency_generate(
            mode="image", 
            conditioning_c=text_features, 
            num_steps=2
        )
        
    t1 = time.time()
    print(f"  -> SUCCESS | Generation Time: {(t1 - t0)*1000:.2f} ms (Inference)")
    print(f"  -> Generated Image Latent Shape: {image_latents.shape}")
    
    # Pathway 2: Image-to-Video Coherence (Real Image -> SSM Wedge -> Video Stack)
    print("\n--- Route 2: IMAGE-TO-VIDEO ---")
    print("Feeding visual features into SSM Temporal Wedge recurrence block...")
    
    t0 = time.time()
    with torch.no_grad():
        # Flatten spatial aspect ratio dimensions if image_latents is 4D (B, H, W, D)
        if len(image_latents.shape) == 4:
            B_img, H_img, W_img, D_img = image_latents.shape
            flat_latents = image_latents.view(B_img, H_img * W_img, D_img)
        else:
            flat_latents = image_latents
        projected_latents = orchestrator.image_encoder(flat_latents)
        visual_inputs = projected_latents[:, :16, :]
        
        # Project through MCP
        conditioning_c = orchestrator.mcp(visual_inputs)
        
        # Generate 4 coherent video frames sequentially using the SSM Temporal Wedge
        video_latents, latency = orchestrator.consistency_generate(
            mode="video", 
            conditioning_c=conditioning_c, 
            num_steps=4
        )
        
    t1 = time.time()
    print(f"  -> SUCCESS | Generation Time: {(t1 - t0)*1000:.2f} ms (Inference)")
    print(f"  -> Generated 4-Frame Video Latent Shape: {video_latents.shape}")
    
    # Pathway 3: Real Speculative Text Generation (Fast Autoregressive Decoding)
    print("\n--- Route 3: REAL SPECULATIVE TEXT GEN ---")
    prompt_text = "Mamba-2 state space model possesses linear"
    print(f"Input Prompt: '{prompt_text}'")
    prompt_tokens = orchestrator.real_tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    
    t0 = time.time()
    with torch.no_grad():
        generated_seq, tokens_produced, tokens_per_sec = orchestrator.speculative_text_gen(
            prompt_tokens=prompt_tokens,
            steps=30,
            K_draft=4
        )
    t1 = time.time()
    
    decoded_text = orchestrator.real_tokenizer.decode(generated_seq[0], skip_special_tokens=True)
    print(f"  -> SUCCESS | Decoding Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  -> Generated Text Stream: '{decoded_text}'")
    print(f"  -> Speculative Decoding Throughput: {tokens_per_sec:.2f} tokens/sec")
    
    print("\n==================================================")
    print("   Pipeline Success: Zero-Overhead Mode Active    ")
    print("==================================================")

if __name__ == "__main__":
    run_real_prediction_pipeline()
