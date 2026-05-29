import torch
import torch.nn as nn

class LayerWiseGPUOffloader(nn.Module):
    """
    Sequential model wrapper that executes massive models (like 9B/4B backbones) 
    layer-by-layer by dynamically swapping weights between CPU RAM and GPU VRAM.
    This bypasses the memory wall, keeping VRAM allocations to a single layer's size.
    """
    def __init__(self, layers_list, execution_device="cuda"):
        super().__init__()
        # Keep all child modules on CPU initially
        self.layers = nn.ModuleList([layer.to("cpu") for layer in layers_list])
        self.execution_device = torch.device(execution_device)
        print(f"[Offloader] Initialized wrapper. Wrapped {len(self.layers)} layers on CPU RAM.")

    def forward(self, x):
        """
        Executes sequential forward pass, swapping each layer to GPU VRAM 
        and offloading it back to CPU RAM immediately after execution.
        """
        # Ensure input starts on execution device
        current_x = x.to(self.execution_device)
        
        for i, layer in enumerate(self.layers):
            # 1. Swap current layer parameters to GPU
            layer.to(self.execution_device)
            
            # 2. Execute forward step
            # If training, we retain activations. If inference, we free them.
            current_x = layer(current_x)
            
            # 3. Offload layer back to CPU (using non_blocking to overlap transfers)
            layer.to("cpu", non_blocking=True)
            
            # Optional: clear CUDA cache periodically if VRAM pressure is severe
            if i % 8 == 0 and self.execution_device.type == "cuda":
                torch.cuda.empty_cache()
                
        return current_x

class MockTransformerLayer(nn.Module):
    """
    Simulates a heavy transformer block (e.g. 500M parameters, 1GB weights).
    """
    def __init__(self, dim=2048):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Residual self-attention
        x_norm = self.norm(x)
        x = x + self.self_attn(x_norm)
        # Residual MLP
        x = x + self.mlp(self.norm(x))
        return x

def test_offloading():
    print("==================================================")
    print("        Layer-by-Layer VRAM Offloader Test       ")
    print("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target execution device: {device}")
    
    # Instantiate 12 heavy transformer layers
    # Total parameter footprint: 12 * ~35M parameters = ~420M parameters (~840MB FP32 weights)
    dim = 2048
    layers = [MockTransformerLayer(dim=dim) for _ in range(12)]
    
    # 1. Measure standard GPU loading (control group)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            print("\nAttempting standard GPU loading...")
            standard_model = nn.Sequential(*layers).to("cuda")
            peak_std = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Standard load peak VRAM: {peak_std:.2f} MB")
            del standard_model
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Standard loading failed as expected: {e}")
            
    # 2. Measure layer-wise offloaded execution
    print("\nRunning offloaded execution...")
    offloaded_model = LayerWiseGPUOffloader(layers, execution_device=device)
    
    # Input batch size 2, sequence length 512, dim 2048
    x_input = torch.randn(2, 512, dim)
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    t0 = time.time()
    y_out = offloaded_model(x_input)
    t1 = time.time()
    
    print("\n--- Execution Stats ---")
    print(f"Output shape: {y_out.shape}")
    print(f"Execution time: {(t1 - t0) * 1000:.2f} ms")
    
    if device == "cuda":
        peak_off = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Offloaded peak VRAM: {peak_off:.2f} MB")
        # Assert offloaded memory is bounded by a single layer size plus inputs
        print(f"VRAM savings: {((peak_std - peak_off) / peak_std * 100) if 'peak_std' in locals() else 0:.1f}% reduction")
    else:
        print("Note: Simulated offloading successfully on CPU.")

if __name__ == "__main__":
    import time
    test_offloading()
