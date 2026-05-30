import torch
import time
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def benchmark_pixelle_sirius():
    print("==================================================")
    print("      Pixelle-Sirius SSM-Diffusion Profiler       ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing engine profiling on: {device}")
    
    # 1. Initialize Pixelle-Sirius Orchestrator
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    
    orchestrator.eval()
    
    # Reset peak memory stats if on CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    print("\n--------------------------------------------------")
    print("  Benchmark 1: Speculative Text/Code Generation")
    print("--------------------------------------------------")
    
    # Mock prompt tokens (batch_size = 1, seq_len = 64)
    prompt_tokens = torch.randint(0, 32000, (1, 64), device=device)
    
    # Run Speculative Drafting
    with torch.no_grad():
        generated_seq, tokens_produced, tokens_per_sec = orchestrator.speculative_text_gen(
            prompt_tokens=prompt_tokens,
            steps=100,
            K_draft=4
        )
        
    print(f"  -> Generated Sequence Length: {generated_seq.shape[1]} tokens")
    print(f"  -> Speculative Tokens Produced: {tokens_produced}")
    print(f"  -> Generation Throughput: {tokens_per_sec:.2f} tokens/sec")
    assert tokens_per_sec >= 100.0 or device.type == "cpu", \
        f"Speed benchmark failed! Got {tokens_per_sec:.2f} tokens/sec, expected >= 100.0"
        
    print("\n--------------------------------------------------")
    print("  Benchmark 2: Multi-Modal Consistency Generation")
    print("--------------------------------------------------")
    
    modes = ["image", "video", "audio"]
    steps_list = [1, 2, 4]
    
    with torch.no_grad():
        for mode in modes:
            print(f"\n[Modality: {mode.upper()}]")
            for steps in steps_list:
                output, latency = orchestrator.consistency_generate(
                    mode=mode,
                    conditioning_c=None,
                    num_steps=steps
                )
                print(f"  -> {steps}-Step Consistency: Latency = {latency:.2f} ms | Output Shape = {list(output.shape)}")
                
    print("\n--------------------------------------------------")
    print("  Benchmark 3: Active GPU VRAM Allocation Audit")
    print("--------------------------------------------------")
    
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"  -> Peak VRAM Allocated: {peak_allocated:.2f} MB")
        print(f"  -> Peak VRAM Reserved:  {peak_reserved:.2f} MB")
        print(f"  -> VRAM Target Status:  {'PASS (Strictly under 3GB ceiling)' if peak_reserved < 3072.0 else 'FAIL (Exceeded 3GB ceiling)'}")
        assert peak_reserved < 3072.0, "VRAM target constraint violated!"
    else:
        print("  -> Execution completed on CPU. GPU VRAM audit skipped.")
        
    print("\n==================================================")
    print("        Pixelle-Sirius Verification Passed        ")
    print("==================================================")

if __name__ == "__main__":
    benchmark_pixelle_sirius()
