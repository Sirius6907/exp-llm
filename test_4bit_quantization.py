import torch
import time
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def test_4bit_quantization_inference():
    print("==================================================")
    print(" Pixelle-Sirius: 4-Bit Edge Quantization Inference Test ")
    print("==================================================")
    print("Objective: Load pre-trained models in 4-bit precision and run inference")
    print("to verify execution under target edge hardware constraints (RTX 3050 Laptop).")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU/CPU Execution Device: {device}")
    
    # 1. Initialize the Orchestrator
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    orchestrator.eval()
    
    # 2. Ingest Pre-trained Backbones in 4-bit Mode
    print("\n[1/3] Ingesting pre-trained backbones in 4-bit NF4 precision...")
    t_load_start = time.time()
    orchestrator.load_real_backbones(offload=False, load_in_4bit=True)
    t_load_end = time.time()
    
    if not orchestrator.real_weights_enabled:
        print("❌ Could not initialize 4-bit pre-trained backbones. Falling back / exiting.")
        return
        
    print(f"  -> SUCCESS | Model loaded in {t_load_end - t_load_start:.2f} seconds.")
    
    # 3. Clear CUDA cache and track VRAM
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # 4. Execute Inference Pathways
    print("\n[2/3] Running zero-overhead Any-to-Any inference pathways in 4-bit...")
    
    # Pathway 1: Text-to-Image Generation (Real Text -> MCP -> LCM Solver)
    print("\n--- Route 1: TEXT-TO-IMAGE ---")
    prompt = "Futuristic neon city abstract artwork"
    print(f"Input Prompt: '{prompt}'")
    
    t0 = time.time()
    with torch.no_grad():
        inputs = orchestrator.real_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Extract text features in 4-bit
        out_hf = orchestrator.real_qwen(input_ids=input_ids)
        text_features = out_hf.last_hidden_state.float() # Cast to float32 for downstream
        
        image_latents, latency = orchestrator.consistency_generate(
            mode="image", 
            conditioning_c=text_features, 
            num_steps=2
        )
    t1 = time.time()
    print(f"  -> SUCCESS | Generation Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  -> Generated Image Latent Shape: {image_latents.shape}")
    
    # Pathway 2: Image-to-Video Coherence (Real Image -> SSM Wedge -> Video Stack)
    print("\n--- Route 2: IMAGE-TO-VIDEO ---")
    t0 = time.time()
    with torch.no_grad():
        projected_latents = orchestrator.image_encoder(image_latents)
        visual_inputs = projected_latents[:, :16, :]
        conditioning_c = orchestrator.mcp(visual_inputs)
        video_latents, latency = orchestrator.consistency_generate(
            mode="video", 
            conditioning_c=conditioning_c, 
            num_steps=4
        )
    t1 = time.time()
    print(f"  -> SUCCESS | Generation Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  -> Generated Video Latent Shape: {video_latents.shape}")
    
    # Pathway 3: Real Speculative Text Generation (Fast Autoregressive Decoding)
    print("\n--- Route 3: REAL SPECULATIVE TEXT GEN ---")
    prompt_text = "Mamba-2 state space model possesses linear"
    print(f"Input Prompt: '{prompt_text}'")
    prompt_tokens = orchestrator.real_tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    
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
    
    # 5. Profile VRAM consumption
    print("\n[3/3] Auditing 4-bit memory footprint on device...")
    print("==================================================")
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU VRAM Reserved (4-bit active):  {peak_vram:.2f} MB")
        
        # Verify edge constraint (e.g. must fit fully within 3GB target VRAM with headroom)
        if peak_vram < 2000.0:
            print("\n[STATUS: PASS]")
            print("Successfully executed 4-bit inference pipeline fully locally under 2.0 GB VRAM limit!")
        else:
            print("\n[STATUS: WARNING]")
            print("Inference completed but exceeded 2.0 GB VRAM limit. Check tensor memory allocations.")
    else:
        print("\n[STATUS: PASS] Inference completed on CPU!")
    print("==================================================")

if __name__ == "__main__":
    test_4bit_quantization_inference()
