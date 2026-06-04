import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFlowDualQuantizer(nn.Module):
    """
    TokenFlow Dual-Codebook Quantizer with EMA + dead code replacement.
    Periodically replaces unused codebook entries with actual encoder outputs.
    """
    def __init__(self, codebook_size=16384, semantic_dim=768, pixel_dim=256,
                 lambda_s=1.0, lambda_p=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.semantic_dim = semantic_dim
        self.pixel_dim = pixel_dim
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        limit = 1 / (semantic_dim ** 0.5)
        self.semantic_codebook = nn.Parameter(torch.empty(codebook_size, semantic_dim).uniform_(-limit, limit))
        limit = 1 / (pixel_dim ** 0.5)
        self.pixel_codebook = nn.Parameter(torch.empty(codebook_size, pixel_dim).uniform_(-limit, limit))

        # EMA buffers
        self.register_buffer('ema_semantic', torch.zeros(codebook_size, semantic_dim))
        self.register_buffer('ema_pixel', torch.zeros(codebook_size, pixel_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('initialized', torch.tensor(False))

        # Usage tracking for dead code replacement
        self.register_buffer('usage_count', torch.zeros(codebook_size, dtype=torch.long))
        self._step_counter = 0

    def forward(self, x_s, x_p, return_losses=False):
        B, L, _ = x_s.shape
        device = x_s.device

        flat_xs = x_s.reshape(-1, self.semantic_dim)
        flat_xp = x_p.reshape(-1, self.pixel_dim)

        # 1. Pairwise squared L2 distances
        dist_s = (
            torch.sum(flat_xs**2, dim=1, keepdim=True)
            + torch.sum(self.semantic_codebook**2, dim=1)
            - 2 * torch.matmul(flat_xs, self.semantic_codebook.t())
        )
        dist_p = (
            torch.sum(flat_xp**2, dim=1, keepdim=True)
            + torch.sum(self.pixel_codebook**2, dim=1)
            - 2 * torch.matmul(flat_xp, self.pixel_codebook.t())
        )

        # Normalise distances per codebook so neither dominates
        dist_s = dist_s / (dist_s.mean(dim=1, keepdim=True) + 1e-8)
        dist_p = dist_p / (dist_p.mean(dim=1, keepdim=True) + 1e-8)

        # 2. Shared index
        joint_dist = self.lambda_s * dist_s + self.lambda_p * dist_p
        encoding_indices = torch.argmin(joint_dist, dim=1).long()

        # 3. Retrieve quantized vectors
        quantized_flat_s = F.embedding(encoding_indices, self.semantic_codebook)
        quantized_flat_p = F.embedding(encoding_indices, self.pixel_codebook)

        quantized_s = quantized_flat_s.view(B, L, self.semantic_dim)
        quantized_p = quantized_flat_p.view(B, L, self.pixel_dim)
        encoding_indices_out = encoding_indices.view(B, L)

        # 4. Losses
        losses = {}
        if return_losses:
            losses['codebook_s'] = F.mse_loss(quantized_flat_s.detach(), flat_xs)
            losses['codebook_p'] = F.mse_loss(quantized_flat_p.detach(), flat_xp)
            losses['commit_s'] = F.mse_loss(quantized_flat_s, flat_xs.detach())
            losses['commit_p'] = F.mse_loss(quantized_flat_p, flat_xp.detach())
            # Also track total codebook as simple l2 regularisation
            losses['codebook_reg'] = self.semantic_codebook.norm(2).mean() * 0.01 + self.pixel_codebook.norm(2).mean() * 0.01

        # 5. Straight-Through
        quantized_s = x_s + (quantized_s - x_s).detach()
        quantized_p = x_p + (quantized_p - x_p).detach()

        # 6. EMA update + dead code replacement (training only)
        if self.training and return_losses:
            with torch.no_grad():
                flat_indices = encoding_indices  # (B*L,)
                enc_one_hot = F.one_hot(flat_indices, self.codebook_size).float()
                cluster_size = enc_one_hot.sum(0)  # (M,)

                # Track usage (across steps)
                self.usage_count += cluster_size.long()

                if not self.initialized:
                    scale = cluster_size.clamp_min(1).unsqueeze(1)
                    self.ema_semantic.data = (enc_one_hot.T @ flat_xs) / scale
                    self.ema_pixel.data = (enc_one_hot.T @ flat_xp) / scale
                    self.semantic_codebook.data.copy_(self.ema_semantic)
                    self.pixel_codebook.data.copy_(self.ema_pixel)
                    self.ema_cluster_size.data = cluster_size
                    self.initialized = torch.tensor(True).to(self.initialized.device)
                else:
                    self.ema_cluster_size.data = (
                        self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
                    )
                    self.ema_semantic.data = (
                        self.decay * self.ema_semantic + (1 - self.decay) * (enc_one_hot.T @ flat_xs)
                    )
                    self.ema_pixel.data = (
                        self.decay * self.ema_pixel + (1 - self.decay) * (enc_one_hot.T @ flat_xp)
                    )

                    n = self.ema_cluster_size.sum()
                    smoothed = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                    self.semantic_codebook.data = self.ema_semantic / smoothed.clamp_min(1).unsqueeze(1)
                    self.pixel_codebook.data = self.ema_pixel / smoothed.clamp_min(1).unsqueeze(1)

                # Dead code replacement: only replace codes with tiny EMA mass over long interval
                self._step_counter += 1
                if self._step_counter % 75 == 0:
                    # Dead codes = those with EMA count below threshold and never used
                    dead = (self.ema_cluster_size < 0.5).nonzero(as_tuple=True)[0]
                    n_dead = dead.numel()
                    if n_dead > 0:
                        n_replace = min(n_dead, flat_xs.size(0) // 3)
                        if n_replace > 0:
                            # Sample random encoder outputs to replace dead codes
                            perm = torch.randperm(flat_xs.size(0))[:n_replace]
                            self.semantic_codebook.data[dead[:n_replace]] = flat_xs[perm]
                            self.pixel_codebook.data[dead[:n_replace]] = flat_xp[perm]
                            # Re-init EMA for replaced slots
                            self.ema_semantic.data[dead[:n_replace]] = flat_xs[perm]
                            self.ema_pixel.data[dead[:n_replace]] = flat_xp[perm]
                            self.ema_cluster_size.data[dead[:n_replace]] = 1.0

        if return_losses:
            return quantized_s, quantized_p, encoding_indices_out, losses
        return quantized_s, quantized_p, encoding_indices_out


class TokenFlowTokenizer(nn.Module):
    """
    TokenFlow Tokenizer module wrapping encoders + dual-codebook quantizer.
    """
    def __init__(self, codebook_size=16384, semantic_dim=768, pixel_dim=256):
        super().__init__()
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

        self.decoder_pix = nn.Sequential(
            nn.ConvTranspose2d(pixel_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def encode(self, x):
        B, C, H, W = x.shape
        xs_cont = self.encoder_sem(x)
        xp_cont = self.encoder_pix(x)
        L = (H // 4) * (W // 4)
        xs_seq = xs_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        xp_seq = xp_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        q_s, q_p, indices = self.quantizer(xs_seq, xp_seq)
        q_s_spatial = q_s.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        q_p_spatial = q_p.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        return q_s_spatial, q_p_spatial, indices

    def encode_with_losses(self, x):
        """Training forward with full loss + EMA/dead-code replacement."""
        B, C, H, W = x.shape
        xs_cont = self.encoder_sem(x)
        xp_cont = self.encoder_pix(x)
        L = (H // 4) * (W // 4)
        xs_seq = xs_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        xp_seq = xp_cont.permute(0, 2, 3, 1).reshape(B, L, -1)
        q_s, q_p, indices, losses = self.quantizer(xs_seq, xp_seq, return_losses=True)
        q_s_spatial = q_s.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        q_p_spatial = q_p.permute(0, 2, 1).reshape(B, -1, H // 4, W // 4)
        return q_s_spatial, q_p_spatial, indices, losses

    def decode_pixel(self, q_p_spatial):
        return self.decoder_pix(q_p_spatial)
