import os
import sys
import torch
import time

# Set console encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import local modules
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def run_specific_prompt():
    print("==================================================")
    print("      Pixelle-Sirius Custom Prompt Execution      ")
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
        device="cpu",
        scale_to_300m=True
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
    
    prompt_text = "solve this logical prompt: explain why the selective SSM Mamba-2 architecture is extremely efficient on consumer edge laptops"
    print(f"\nPrompt: '{prompt_text}'")
    
    # Encode prompt using Qwen2 tokenizer
    if orchestrator.real_weights_enabled and orchestrator.real_tokenizer is not None:
        inputs = orchestrator.real_tokenizer(prompt_text, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].to(device)
        
        # Test 1: Explicit Thinking Mode (thinking_mode=True)
        print("\n--- TEST 1: EXPLICIT THINKING MODE (thinking_mode=True) ---")
        print("Generating response with explicit thoughts visible...")
        with torch.no_grad():
            generated_seq, tokens_produced, tokens_per_sec = orchestrator.speculative_text_gen(
                prompt_tokens=prompt_tokens,
                steps=50,
                K_draft=4,
                thinking_mode=True,
                max_thinking_tokens=100
            )
        
        output_ids = generated_seq[0].cpu().tolist()
        full_text = orchestrator.real_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if full_text.startswith(prompt_text):
            response = full_text[len(prompt_text):].strip()
        else:
            response = full_text.strip()
            
        print("\n------------------ GENERATED RESPONSE (EXPLICIT) ------------------")
        print(response)
        print("-------------------------------------------------------------------")
        print(f"Generation metrics: {tokens_produced} tokens | {tokens_per_sec:.2f} tok/sec")
        
        # Test 2: Silent Thinking Mode (thinking_mode=False)
        print("\n--- TEST 2: SILENT THINKING MODE (thinking_mode=False) ---")
        print("Generating response with thinking performed internally but stripped from output...")
        with torch.no_grad():
            generated_seq_silent, tokens_produced_silent, tokens_per_sec_silent = orchestrator.speculative_text_gen(
                prompt_tokens=prompt_tokens,
                steps=50,
                K_draft=4,
                thinking_mode=False,
                max_thinking_tokens=100
            )
        
        output_ids_silent = generated_seq_silent[0].cpu().tolist()
        full_text_silent = orchestrator.real_tokenizer.decode(output_ids_silent, skip_special_tokens=True)
        
        if full_text_silent.startswith(prompt_text):
            response_silent = full_text_silent[len(prompt_text):].strip()
        else:
            response_silent = full_text_silent.strip()
            
        print("\n------------------ GENERATED RESPONSE (SILENT) ------------------")
        print(response_silent)
        print("-----------------------------------------------------------------")
        print(f"Generation metrics: {tokens_produced_silent} tokens | {tokens_per_sec_silent:.2f} tok/sec")
    else:
        print("❌ Real pre-trained backbones or tokenizer is not available.")
        
    print("\n==================================================")

if __name__ == "__main__":
    run_specific_prompt()
