import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenFlowDualQuantizer(nn.Module):
    """
    TokenFlow Dual-Codebook Quantizer with Shared Index Mapping.
    Decouples semantic representation (for reasoning) and pixel detail (for reconstruction)
    while maintaining absolute alignment via a single shared index codebook.
    """
    def __init__(self, codebook_size=16384, semantic_dim=768, pixel_dim=256, lambda_s=1.0, lambda_p=1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.semantic_dim = semantic_dim
        self.pixel_dim = pixel_dim
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        
        # 1. Initialize Semantic Codebook: shape (M, d_s)
        self.semantic_codebook = nn.Parameter(torch.randn(codebook_size, semantic_dim))
        # 2. Initialize Pixel/Detail Codebook: shape (M, d_p)
        self.pixel_codebook = nn.Parameter(torch.randn(codebook_size, pixel_dim))
        
        # Normalize codebooks for stable cosine/Euclidean distance matching
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x_s, x_p):
        """
        Args:
            x_s (Tensor): Continuous semantic latent feature, shape (B, L, d_s)
            x_p (Tensor): Continuous pixel latent feature, shape (B, L, d_p)
            
        Returns:
            quantized_s (Tensor): Quantized semantic latents, shape (B, L, d_s)
            quantized_p (Tensor): Quantized pixel latents, shape (B, L, d_p)
            encoding_indices (Tensor): Shared index tokens, shape (B, L)
        """
        B, L, _ = x_s.shape
        device = x_s.device
        
        # Flatten inputs for distance computation
        flat_xs = x_s.reshape(-1, self.semantic_dim) # (B*L, d_s)
        flat_xp = x_p.reshape(-1, self.pixel_dim)    # (B*L, d_p)
        
        # 1. Compute pairwise squared L2 distances to all codebook entries
        # Distances to Semantic Codebook: shape (B*L, M)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a.b
        dist_s = (
            torch.sum(flat_xs**2, dim=1, keepdim=True)
            + torch.sum(self.semantic_codebook**2, dim=1)
            - 2 * torch.matmul(flat_xs, self.semantic_codebook.t())
        )
        
        # Distances to Pixel Codebook: shape (B*L, M)
        dist_p = (
            torch.sum(flat_xp**2, dim=1, keepdim=True)
            + torch.sum(self.pixel_codebook**2, dim=1)
            - 2 * torch.matmul(flat_xp, self.pixel_codebook.t())
        )
        
        # 2. Shared Index Minimization
        # Find the single index for each patch that minimizes the joint weighted distance
        joint_dist = self.lambda_s * dist_s + self.lambda_p * dist_p # (B*L, M)
        encoding_indices = torch.argmin(joint_dist, dim=1) # (B*L,)
        
        # 3. Retrieve Quantized Vectors
        # Quantized semantic vectors: (B*L, d_s)
        quantized_flat_s = F.embedding(encoding_indices, self.semantic_codebook)
        # Quantized pixel vectors: (B*L, d_p)
        quantized_flat_p = F.embedding(encoding_indices, self.pixel_codebook)
        
        # Reshape back to original dimensions
        quantized_s = quantized_flat_s.view(B, L, self.semantic_dim)
        quantized_p = quantized_flat_p.view(B, L, self.pixel_dim)
        encoding_indices = encoding_indices.view(B, L)
        
        # 4. Straight-Through Estimator (STE)
        # Bypasses the discrete argmin step during the backward pass to enable end-to-end training
        quantized_s = x_s + (quantized_s - x_s).detach()
        quantized_p = x_p + (quantized_p - x_p).detach()
        
        return quantized_s, quantized_p, encoding_indices

class TokenFlowTokenizer(nn.Module):
    """
    TokenFlow Tokenizer module wrapping specialized encoders and our custom
    dual-codebook quantizer to bridge visual understanding and generation.
    """
    def __init__(self, codebook_size=16384, semantic_dim=768, pixel_dim=256):
        super().__init__()
        # Simple high-speed downsampling convolutional encoders for continuous latents
        self.encoder_sem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, semantic_dim, kernel_size=1)
        )
        
        self.encoder_pix = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, pixel_dim, kernel_size=1)
        )
        
        self.quantizer = TokenFlowDualQuantizer(
            codebook_size=codebook_size,
            semantic_dim=semantic_dim,
            pixel_dim=pixel_dim
        )
        
        # Simple transposed convolution pixel decoder for reconstruction
        self.decoder_pix = nn.Sequential(
            nn.ConvTranspose2d(pixel_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def encode(self, x):
        # x shape: (B, 3, H, W)
        B, C, H, W = x.shape
        
        # 1. Project to continuous latent spaces
        xs_cont = self.encoder_sem(x) # (B, d_s, H//4, W//4)
        xp_cont = self.encoder_pix(x) # (B, d_p, H//4, W//4)
        
        # 2. Reshape to sequence vectors: (B, L, D)
        L = (H // 4) * (W // 4)
        xs_seq = xs_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        xp_seq = xp_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        
        # 3. Perform dual quantization
        q_s, q_p, indices = self.quantizer(xs_seq, xp_seq)
        
        # Reshape quantized sequences back to spatial layouts for decoders/VLMs
        q_s_spatial = q_s.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        q_p_spatial = q_p.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        
        return q_s_spatial, q_p_spatial, indices

    def decode_pixel(self, q_p_spatial):
        return self.decoder_pix(q_p_spatial)
