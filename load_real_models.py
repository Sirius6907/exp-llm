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

    # 3. Wrap each layer in our custom OffloadedLayerWrapper
    print("\nWrapping layers with OffloadedLayerWrapper...")
    from offloader import OffloadedLayerWrapper
    wrapped_layers = [OffloadedLayerWrapper(layer, execution_device=device) for layer in layers_list]
    wrapped_layers_module = nn.ModuleList(wrapped_layers)
    
    # Replace model's sequential layers with our offloaded wrapped layers
    if hasattr(raw_model, "layers"):
        raw_model.layers = wrapped_layers_module
    elif hasattr(raw_model, "h"):
        raw_model.h = wrapped_layers_module
        
    # Move lightweight normalization and rotary embedding modules to the GPU permanently
    if hasattr(raw_model, "norm") and raw_model.norm is not None:
        raw_model.norm.to(device)
    if hasattr(raw_model, "rotary_emb") and raw_model.rotary_emb is not None:
        raw_model.rotary_emb.to(device)

    # 4. Run inference step
    text_prompt = "Ternary quantization enables multimodal video generation under a 3GB VRAM limit."
    print(f"\nEncoding test prompt: '{text_prompt}'")
    
    inputs = tokenizer(text_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"] # (B, S)
    
    # Note: When using OffloadedLayerWrapper, the native Hugging Face model call
    # automatically computes causal masks and RoPE embeddings on CPU/GPU seamlessly.
    print("\nExecuting forward pass through native Hugging Face model (with offloaded layers)...")
    with torch.no_grad():
        # Keep input_ids on CPU for embedding lookup
        t0 = time.time()
        
        # Run standard Hugging Face forward pass!
        out_hf = raw_model(input_ids=input_ids.to("cpu"))
        out = out_hf.last_hidden_state
        
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
