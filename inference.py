import torch
import time
from tokenflow import TokenFlowTokenizer

def run_loaded_inference():
    print("==================================================")
    print("       Trained TokenFlow Model Inference          ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference execution device: {device}")
    
    # 1. Initialize the tokenizer architecture
    model = TokenFlowTokenizer(
        codebook_size=2048,
        semantic_dim=768,
        pixel_dim=256
    ).to(device)
    
    # 2. Load the trained CIFAR-10 checkpoint weights
    checkpoint_path = "tokenflow_cifar10.pth"
    print(f"Loading checkpoint from: {checkpoint_path}...")
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Trained model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Checkpoint file '{checkpoint_path}' not found!")
        print("Please run train_tokenflow.py first to generate the checkpoint file.")
        return
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
        
    model.eval()
    
    # 3. Create simulated normalized input image: shape (1, 3, 256, 256)
    test_image = torch.randn(1, 3, 256, 256, device=device)
    print(f"\nProcessing test input image: shape={test_image.shape}")
    
    # 4. Execute inference
    t0 = time.time()
    with torch.no_grad():
        # Encode image into quantized semantic, pixel, and shared index tensors
        q_s, q_p, indices = model.encode(test_image)
        
        # Decode the pixel latents back to a reconstructed image
        reconstructed_image = model.decode_pixel(q_p)
    t1 = time.time()
    
    print("\n--- Inference Output Tensors ---")
    print(f"Quantized Semantic Latents shape: {q_s.shape}")
    print(f"Quantized Pixel Latents shape:    {q_p.shape}")
    print(f"Shared Codebook Indices shape:     {indices.shape}")
    print(f"Reconstructed Image shape:         {reconstructed_image.shape}")
    print(f"Inference latency: {(t1 - t0) * 1000:.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    run_loaded_inference()
