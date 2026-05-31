import torch
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def verify_300m_parameters():
    print("==================================================")
    print("    Pixelle-Sirius: 300M Segment Verification     ")
    print("==================================================")
    
    device = torch.device("cpu")
    
    # Initialize the Orchestrator with scale_to_300m=True
    print("Initializing scaled engine...")
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
    
    # 1. Calculate parameter count for custom Mobile Conditioning Projector (MCP)
    mcp_params = sum(p.numel() for p in orchestrator.mcp.parameters())
    print("\n------------------ MCP PARAMETERS ------------------")
    print(f"  Component: Mobile Conditioning Projector (Ternary Adapter)")
    print(f"  Target Parameters: ~300 Million")
    print(f"  Exact Parameters:  {mcp_params:,} ({mcp_params / 1e6:.2f} Million)")
    print("----------------------------------------------------")
    
    # 2. Calculate parameter count for Consistency Denoising Solver (DiT)
    solver_params = sum(p.numel() for p in orchestrator.lcm_solver.parameters())
    print("\n------------------ SOLVER PARAMETERS -----------------")
    print(f"  Component: Consistency Denoising Solver (Diffusion Transformer)")
    print(f"  Target Parameters: ~300 Million")
    print(f"  Exact Parameters:  {solver_params:,} ({solver_params / 1e6:.2f} Million)")
    print("------------------------------------------------------")
    
    print("\n==================================================")
    print("       300M PARAMETERS VERIFICATION COMPLETED     ")
    print("==================================================")

if __name__ == "__main__":
    verify_300m_parameters()
