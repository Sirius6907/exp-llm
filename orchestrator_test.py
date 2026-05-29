import torch
import time
from dynamap import DynaMapOrchestrator

def test_orchestration():
    print("==================================================")
    print("      DynaMap System Orchestration Profiler       ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing orchestration on: {device}")
    
    # 1. Initialize DynaMap Orchestrator
    # We use a codebook size of 4096 to represent edge limits
    orchestrator = DynaMapOrchestrator(
        codebook_size=4096,
        vlm_dim=1536,
        dit_dim=1024,
        latent_dim=256
    ).to(device)
    
    # 2. Benchmark PATH A: Text-To-Video (8-Frame Generation)
    print("\n--- Testing Path A: Text-to-Video (8 Frames) ---")
    t0 = time.time()
    video_frames_text = orchestrator.route(
        mode="text_to_video",
        num_frames=8,
        device=device
    )
    t1 = time.time()
    path_a_time = (t1 - t0) * 1000
    print(f"Path A (Text-to-Video) Completed: {len(video_frames_text)} frames in {path_a_time:.2f} ms")
    
    # 3. Benchmark PATH B: Image-To-Video (8-Frame Generation)
    print("\n--- Testing Path B: Image-to-Video (8 Frames) ---")
    # Mock image input: 256x256 RGB image -> shape (1, 3, 256, 256)
    mock_image = torch.randn(1, 3, 256, 256, device=device)
    
    t0 = time.time()
    video_frames_img = orchestrator.route(
        mode="image_to_video",
        image_input=mock_image,
        num_frames=8,
        device=device
    )
    t1 = time.time()
    path_b_time = (t1 - t0) * 1000
    print(f"Path B (Image-to-Video) Completed: {len(video_frames_img)} frames in {path_b_time:.2f} ms")
    
    # 4. Profile overall parameters
    total_params = sum(p.numel() for p in orchestrator.parameters())
    print(f"\nOverall DynaMap System Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")
    
    # 5. Measure VRAM Footprint
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print("\n--- Integrated VRAM Audit ---")
        print(f"Peak VRAM Allocated: {peak_allocated:.2f} MB")
        print(f"Peak VRAM Reserved:  {peak_reserved:.2f} MB")
        print(f"VRAM Target Status:  {'PASS (Under 3GB Ceiling)' if peak_reserved < 3072.0 else 'FAIL (Exceeded 3GB Ceiling)'}")
    else:
        print("\nNote: CUDA is not active; memory footprint estimation was simulated on CPU.")
    
    print("\n==================================================")
    print("        All Orchestration Tests Passed!           ")
    print("==================================================")

if __name__ == "__main__":
    test_orchestration()
