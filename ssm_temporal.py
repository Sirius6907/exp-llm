import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Lightweight, memory-efficient Selective State Space Model (SSM) temporal block.
    Designed specifically to pass compressed hidden states from Frame N to Frame N+1
    with linear time-complexity and minimal VRAM footprints.
    """
    def __init__(self, dim=256, state_dim=16, dt_rank=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # S4 parameter A (log scale initialization to enforce stability)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).view(1, 1, -1)))
        
        # Parameter B and C projections (dynamic selective parameters)
        self.x_proj = nn.Linear(dim, dt_rank + state_dim * 2, bias=False)
        nn.init.normal_(self.x_proj.weight, std=0.005) # Initialize with small weights to prevent representation drift
        
        # Step size (dt) projections and bias initialization
        self.dt_proj = nn.Linear(dt_rank, dim, bias=True)
        nn.init.constant_(self.dt_proj.weight, 0.0) # Zero-initialize weights so dt starts as stable constant bias
        # Initialize dt_proj bias to ensure step size starts in a stable regime
        nn.init.constant_(self.dt_proj.bias, 0.1)

    def forward(self, x, prev_state=None):
        """
        Args:
            x (Tensor): Input features of frame N of shape (B, L, D), where
                        B = batch size, L = spatial token sequence length (e.g., 256 for 256x256),
                        D = latent channel dimension.
            prev_state (Tensor, optional): Recurrent hidden state of shape (B, L, D, N_ssm)
                                          representing the compressed state of Frame N-1.
                                          
        Returns:
            y (Tensor): Coherent output features of shape (B, L, D).
            next_state (Tensor): The updated recurrent state of shape (B, L, D, N_ssm).
        """
        B, L, D = x.shape
        device = x.device
        
        # 1. Retrieve the stable SSM state matrix A: shape (1, D, N_ssm)
        # We expand it over the channel dimension to allow channel-wise selective dynamics.
        A = -torch.exp(self.A_log.repeat(1, D, 1)) # (1, D, N_ssm)
        
        # 2. Project input to selective inputs (Delta, B, C)
        # x_proj_out shape: (B, L, dt_rank + state_dim * 2)
        proj_out = self.x_proj(x)
        dt_rank_out, B_out, C_out = torch.split(proj_out, [self.x_proj.in_features if hasattr(self, 'dt_rank') else 16, self.state_dim, self.state_dim], dim=-1)
        
        # 3. Compute step size delta and project to channel dimension: shape (B, L, D)
        dt = F.softplus(self.dt_proj(dt_rank_out)) 
        
        # 4. Format B and C for tensor multiplication
        # B_out: (B, L, N_ssm), C_out: (B, L, N_ssm)
        # We expand/reshape them to shape (B, L, 1, N_ssm)
        B_val = B_out.unsqueeze(-2)
        C_val = C_out.unsqueeze(-2)
        
        # 5. Initialize recurrent state if none is provided
        # h shape: (B, L, D, N_ssm)
        if prev_state is None:
            h = torch.zeros(B, L, D, self.state_dim, device=device)
        else:
            h = prev_state
            
        # 6. Discretization using Euler method (First-order hold approximation)
        # dt_expanded: shape (B, L, D, 1)
        dt_expanded = dt.unsqueeze(-1)
        
        # A_bar = exp(dt * A)
        # shape: (B, L, D, N_ssm)
        A_bar = torch.exp(dt_expanded * A.unsqueeze(0))
        
        # B_bar = dt * B
        # shape: (B, L, D, N_ssm)
        B_bar = dt_expanded * B_val
        
        # x_expanded: shape (B, L, D, 1)
        x_expanded = x.unsqueeze(-1)
        
        # 7. Recurrent update step: h_t = A_bar * h_{t-1} + B_bar * x_t
        next_state = A_bar * h + B_bar * x_expanded
        
        # 8. Output computation: y_t = C_bar * h_t
        # C_val: shape (B, L, 1, N_ssm)
        # We sum over the state_dim (N_ssm) to get (B, L, D)
        y = torch.sum(next_state * C_val, dim=-1)
        
        return y, next_state

class TemporalWedgeBlock(nn.Module):
    """
    Temporal Wedge block that wraps the SelectiveSSM to act as a frame-to-frame conditioning module,
    taking spatial latents of Frame N and blending them recurrently with prior frame states.
    """
    def __init__(self, dim=256, state_dim=16):
        super().__init__()
        self.ssm = SelectiveSSM(dim=dim, state_dim=state_dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        # Zero-initialize the final projection layer to enforce near-zero representation loss on step 0
        nn.init.constant_(self.ffn[2].weight, 0.0)
        nn.init.constant_(self.ffn[2].bias, 0.0)

    def forward(self, x, prev_state=None):
        # 1. Apply Selective SSM recurrent step
        y, next_state = self.ssm(self.layer_norm(x), prev_state)
        
        # 2. Residual connection & FFN
        out = x + y
        out = out + self.ffn(self.layer_norm(out))
        
        return out, next_state
