import torch
import time
from any_to_any import AnyToAnyOrchestrator

def test_any_to_any():
    print("==================================================")
    print("      Any-to-Any 16-Pathway System Profiler       ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing orchestration on: {device}")
    
    # 1. Initialize Any-to-Any Orchestrator
    orchestrator = AnyToAnyOrchestrator(
        codebook_size=4096,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    
    # Define all 16 pathways
    modalities = ["text", "image", "video", "audio"]
    routes = [f"{src}_to_{tgt}" for src in modalities for tgt in modalities]
    
    # Mock inputs
    mock_text = torch.randint(0, 32000, (1, 512), device=device)
    mock_image = torch.randn(1, 3, 64, 64, device=device)
    mock_video = torch.randn(1, 4, 3, 64, 64, device=device)
    mock_audio = torch.randn(1, 1, 16000, device=device)
    
    print(f"\nCreated mock inputs:")
    print(f"  - Text tensor:  {mock_text.shape}")
    print(f"  - Image tensor: {mock_image.shape}")
    print(f"  - Video tensor: {mock_video.shape}")
    print(f"  - Audio tensor: {mock_audio.shape}")
    
    print("\nStarting 16-Pathway Execution Loop...\n")
    
    orchestrator.eval()
    
    success_count = 0
    t_start_total = time.time()
    
    with torch.no_grad():
        for route_idx, route in enumerate(routes, 1):
            print(f"[{route_idx}/16] Routing: {route.upper()}")
            t0 = time.time()
            
            # Reset memory statistics for individual pathways
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
            try:
                output = orchestrator.route(
                    mode=route,
                    text_input=mock_text,
                    image_input=mock_image,
                    video_input=mock_video,
                    audio_input=mock_audio,
                    num_frames=4,
                    audio_len=16000
                )
                t1 = time.time()
                elapsed = (t1 - t0) * 1000
                
                # Verify output type and shape
                assert output is not None, "Output is None!"
                
                tgt = route.split("_to_")[1]
                if tgt == "text":
                    # Text logits shape: (B, S, vocab_size)
                    assert len(output.shape) == 3 and output.shape[0] == 1 and output.shape[2] == 32000, \
                        f"Unexpected text output shape: {output.shape}"
                    shape_str = f"shape: {list(output.shape)}"
                elif tgt == "image":
                    # Image pixels shape: (B, 3, H, W)
                    assert len(output.shape) == 4 and output.shape[0] == 1 and output.shape[1] == 3, \
                        f"Unexpected image output shape: {output.shape}"
                    shape_str = f"shape: {list(output.shape)}"
                elif tgt == "video":
                    # Video frames shape: (B, T, 3, H, W)
                    assert len(output.shape) == 5 and output.shape[0] == 1 and output.shape[1] == 4 and output.shape[2] == 3, \
                        f"Unexpected video output shape: {output.shape}"
                    shape_str = f"shape: {list(output.shape)}"
                elif tgt == "audio":
                    # Audio waveform shape: (B, 1, S)
                    assert len(output.shape) == 3 and output.shape[0] == 1 and output.shape[1] == 1 and output.shape[2] == 16000, \
                        f"Unexpected audio output shape: {output.shape}"
                    shape_str = f"shape: {list(output.shape)}"
                    
                print(f"  -> SUCCESS | Latency: {elapsed:.2f} ms | Output {shape_str}")
                
                if device.type == "cuda":
                    allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                    print(f"  -> Peak VRAM Allocated: {allocated:.2f} MB | Peak VRAM Reserved: {reserved:.2f} MB")
                    
                success_count += 1
                
            except Exception as e:
                print(f"  -> FAILURE | Reason: {str(e)}")
                raise e
                
            print("-" * 50)
        
    t_end_total = time.time()
    total_time = (t_end_total - t_start_total) * 1000
    
    print("\n==================================================")
    print(f"                  Summary Metrics                 ")
    print("==================================================")
    print(f"Successful Pathways:  {success_count} / 16")
    print(f"Total Execution Time: {total_time:.2f} ms")
    
    total_params = sum(p.numel() for p in orchestrator.parameters())
    print(f"Total Engine parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")
    
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"Overall Peak VRAM Allocated: {peak_allocated:.2f} MB")
        print(f"Overall Peak VRAM Reserved:  {peak_reserved:.2f} MB")
        print(f"3GB Target Constraint Status:  {'PASS (Strictly under 3GB VRAM ceiling)' if peak_reserved < 3072.0 else 'FAIL (Exceeded 3GB VRAM ceiling)'}")
        assert peak_reserved < 3072.0, "VRAM ceiling constraint violated!"
    else:
        print("Note: Run complete on CPU; GPU VRAM profiling skipped.")
        
    print("==================================================")
    print("            Any-to-Any Verification Passed        ")
    print("==================================================")

if __name__ == "__main__":
    test_any_to_any()
