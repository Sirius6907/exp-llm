import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from offloader import LayerWiseGPUOffloader

def load_and_wrap_real_model():
    print("==================================================")
    print("       Real-Model Hugging Face Wrapper Test       ")
    print("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target execution device: {device}")
    
    # 1. Select a small real transformer backbone for verification
    # We use Qwen2-0.5B (approx 490M parameters) as a representative test backbone
    model_id = "Qwen/Qwen2-0.5B"
    print(f"\nDownloading and loading pre-trained weights for: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load model entirely on CPU RAM first
        raw_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        print("Model weights loaded successfully on CPU RAM.")
    except Exception as e:
        print(f"❌ Failed to load model from Hugging Face: {e}")
        print("Please check your internet connection or install transformers: pip install transformers")
        return

    # 2. Extract transformer block layers
    # In Qwen2, the transformer layers are located under model.layers (nn.ModuleList)
    if hasattr(raw_model, "layers"):
        layers_list = list(raw_model.layers)
        print(f"Detected {len(layers_list)} transformer block layers.")
    elif hasattr(raw_model, "h"):
        layers_list = list(raw_model.h)
        print(f"Detected {len(layers_list)} transformer block layers.")
    else:
        print("❌ Could not dynamically extract transformer layers. Model structure unknown.")
        return

    # 3. Instantiate the VRAM Offloader wrapper
    print("\nWrapping layers with LayerWiseGPUOffloader...")
    offloaded_backbone = LayerWiseGPUOffloader(layers_list, execution_device=device)
    
    # Replace model's sequential layers with our offloaded layers
    if hasattr(raw_model, "layers"):
        raw_model.layers = offloaded_backbone.layers
    elif hasattr(raw_model, "h"):
        raw_model.h = offloaded_backbone.layers

    # 4. Run inference step
    text_prompt = "Ternary quantization enables multimodal video generation under a 3GB VRAM limit."
    print(f"\nEncoding test prompt: '{text_prompt}'")
    
    inputs = tokenizer(text_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"] # (B, S)
    
    # Note: When using LayerWiseGPUOffloader on Hugging Face models,
    # the forward pass of the base model automatically invokes our layer-wise forward logic.
    print("\nExecuting forward pass through real wrapped layers...")
    with torch.no_grad():
        # Keep inputs on CPU for embedding lookup since raw_model is on CPU RAM
        t0 = time.time()
        
        # 1. Retrieve embeddings on CPU
        hidden_states_cpu = raw_model.embed_tokens(input_ids.to("cpu")) # (B, S, D)
        # 2. Transfer the embedded latent tensor to the target execution device (GPU)
        hidden_states = hidden_states_cpu.to(device)
        
        # Pass sequentially through offloaded layers
        fused_hidden_states = offloaded_backbone(hidden_states)
        
        # 3. Swap the final normalization layer to the GPU device
        raw_model.norm.to(device)
        out = raw_model.norm(fused_hidden_states)
        # 4. Offload the normalization layer back to CPU RAM to conserve VRAM
        raw_model.norm.to("cpu")
        t1 = time.time()
        
    print("\n--- Execution Stats ---")
    print(f"Output shape: {out.shape}")
    print(f"Execution time: {(t1 - t0) * 1000:.2f} ms")
    
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak VRAM used: {peak_vram:.2f} MB")
        print("Success: Real model weights dynamically loaded, swapped, and executed.")
    else:
        print("Success: Real model weights dynamically swapped and executed on CPU.")
        
    print("==================================================")

if __name__ == "__main__":
    import time
    load_and_wrap_real_model()
