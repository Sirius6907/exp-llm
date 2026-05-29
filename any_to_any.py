import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from tokenflow import TokenFlowDualQuantizer, TokenFlowTokenizer
from mcp import MobileConditioningProjector
from ssm_temporal import TemporalWedgeBlock
from dynamap import MiniCPMSALAMock, BonsaiDiffusionMock
from offloader import OffloadedLayerWrapper

class AudioTokenFlowTokenizer(nn.Module):
    """
    AudioTokenFlowTokenizer wraps 1D convolutional encoders and a custom 1D
    dual-codebook quantizer to bridge acoustic understanding and generation.
    """
    def __init__(self, codebook_size=16384, semantic_dim=768, pixel_dim=256):
        super().__init__()
        # 1D downsampling convolutional encoders for continuous latents (factor of 8 downsampling)
        self.encoder_sem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(256, semantic_dim, kernel_size=1)
        )
        
        self.encoder_pix = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(256, pixel_dim, kernel_size=1)
        )
        
        self.quantizer = TokenFlowDualQuantizer(
            codebook_size=codebook_size,
            semantic_dim=semantic_dim,
            pixel_dim=pixel_dim
        )
        
        # 1D transposed convolution pixel decoder for reconstruction
        self.decoder_pix = nn.Sequential(
            nn.ConvTranspose1d(pixel_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        # x shape: (B, 1, S_audio)
        B, C, S_audio = x.shape
        
        # 1. Project to continuous latent spaces
        xs_cont = self.encoder_sem(x) # (B, semantic_dim, S_audio // 8)
        xp_cont = self.encoder_pix(x) # (B, pixel_dim, S_audio // 8)
        
        # 2. Reshape to sequence vectors: (B, L, D)
        L = S_audio // 8
        xs_seq = xs_cont.permute(0, 2, 1).reshape(B, L, -1)
        xp_seq = xp_cont.permute(0, 2, 1).reshape(B, L, -1)
        
        # 3. Perform dual quantization
        q_s, q_p, indices = self.quantizer(xs_seq, xp_seq)
        
        # Reshape quantized sequences back to layouts for decoders/VLMs
        q_s_spatial = q_s.permute(0, 2, 1).reshape(B, -1, S_audio // 8)
        q_p_spatial = q_p.permute(0, 2, 1).reshape(B, -1, S_audio // 8)
        
        return q_s_spatial, q_p_spatial, indices

    def decode_pixel(self, q_p_spatial):
        return self.decoder_pix(q_p_spatial)

class VLMTextDecoderHead(nn.Module):
    """
    Causal autoregressive projection head mapping VLM hidden states to text token logits.
    """
    def __init__(self, vlm_dim=1536, vocab_size=32000):
        super().__init__()
        self.proj = nn.Linear(vlm_dim, vocab_size)

    def forward(self, hidden_states):
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        return self.proj(hidden_states)

class BonsaiAudioMock(nn.Module):
    """
    Mock audio diffusion model that takes raw audio spatial tokens and modulates them
    using cross-modal conditioning from the Mobile Conditioning Projector (MCP).
    """
    def __init__(self, dit_dim=1024, latent_dim=256):
        super().__init__()
        self.dit_dim = dit_dim
        self.latent_dim = latent_dim
        
        self.proj_cond = nn.Linear(dit_dim, latent_dim)
        self.audio_transformer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x_latent, conditioning_c):
        """
        Args:
            x_latent: Audio spatial tokens of shape (B, L_audio, latent_dim)
            conditioning_c: Cross-modal MCP context of shape (B, L_cond, dit_dim)
        """
        cond_projected = self.proj_cond(conditioning_c) # (B, L_cond, latent_dim)
        pooled_cond = cond_projected.mean(dim=1, keepdim=True) # (B, 1, latent_dim)
        
        out = x_latent * torch.sigmoid(pooled_cond)
        out = self.audio_transformer(out)
        return out

class AnyToAnyOrchestrator(nn.Module):
    """
    Any-to-Any Multimodal Generative Orchestrator supporting all 16 routing pathways
    between Text, Image, Video, and Audio under a strict 3GB VRAM ceiling.
    """
    def __init__(self, codebook_size=4096, vlm_dim=1536, dit_dim=1024, latent_dim=256, vocab_size=32000, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.vlm_dim = vlm_dim
        
        # 1. Tokenizers
        self.image_tokenizer = TokenFlowTokenizer(
            codebook_size=codebook_size,
            semantic_dim=768,
            pixel_dim=latent_dim
        ).to(self.device)
        
        self.audio_tokenizer = AudioTokenFlowTokenizer(
            codebook_size=codebook_size,
            semantic_dim=768,
            pixel_dim=latent_dim
        ).to(self.device)
        
        # 2. Text Decoder Head
        self.text_head = VLMTextDecoderHead(
            vlm_dim=vlm_dim,
            vocab_size=vocab_size
        ).to(self.device)
        
        # 3. State Space Model (SSM) Temporal Wedge Block for frame-to-frame coherence
        self.temporal_wedge = TemporalWedgeBlock(
            dim=latent_dim,
            state_dim=16
        ).to(self.device)
        
        # 4. Heavy models wrapped in OffloadedLayerWrapper to target the 3GB VRAM limit
        self.vlm = OffloadedLayerWrapper(
            MiniCPMSALAMock(vlm_dim=vlm_dim, num_layers=4),
            execution_device=self.device
        )
        
        self.mcp = OffloadedLayerWrapper(
            MobileConditioningProjector(vlm_dim=vlm_dim, dit_dim=dit_dim, num_layers=4),
            execution_device=self.device
        )
        
        self.diffusion_image = OffloadedLayerWrapper(
            BonsaiDiffusionMock(dit_dim=dit_dim, latent_dim=latent_dim),
            execution_device=self.device
        )
        
        self.diffusion_audio = OffloadedLayerWrapper(
            BonsaiAudioMock(dit_dim=dit_dim, latent_dim=latent_dim),
            execution_device=self.device
        )
        
        print(f"[AnyToAnyOrchestrator] Any-to-Any routing engine initialized on device {self.device}.")

    def to(self, device):
        device = torch.device(device)
        self.device = device
        super().to(device)
        # Update wrappers' execution devices
        self.vlm.execution_device = device
        self.mcp.execution_device = device
        self.diffusion_image.execution_device = device
        self.diffusion_audio.execution_device = device
        return self

    def route(self, mode="text_to_text", text_input=None, image_input=None, video_input=None, audio_input=None, num_frames=4, audio_len=16000):
        """
        Dynamically analyzes input formats and executes the cheapest performance path.
        Supports all 16 cross-modal paths with dynamic batch sizes.
        """
        t_start = time.time()
        print(f"[AnyToAnyOrchestrator] Routing pathway: '{mode.upper()}' on device: {self.device}")
        
        # Determine Batch Size dynamically
        B = 1
        if text_input is not None:
            B = text_input.shape[0]
        elif image_input is not None:
            B = image_input.shape[0]
        elif video_input is not None:
            B = video_input.shape[0]
        elif audio_input is not None:
            B = audio_input.shape[0]
            
        # Determine Source Modality
        src = mode.split("_to_")[0]
        tgt = mode.split("_to_")[1]
        
        # --- SOURCE PROCESSING ---
        if src == "text":
            text_tokens = text_input if isinstance(text_input, torch.Tensor) else torch.randint(0, self.vocab_size, (B, 512), device=self.device)
            vlm_hidden_states = self.vlm(text_tokens=text_tokens)
            conditioning_c = self.mcp(vlm_hidden_states)
            
        elif src == "image":
            img = image_input if isinstance(image_input, torch.Tensor) else torch.randn(B, 3, 64, 64, device=self.device)
            q_s, q_p, indices = self.image_tokenizer.encode(img)
            vlm_hidden_states = self.vlm(visual_latents=q_s)
            conditioning_c = self.mcp(vlm_hidden_states)
            
        elif src == "video":
            vid = video_input if isinstance(video_input, torch.Tensor) else torch.randn(B, num_frames, 3, 64, 64, device=self.device)
            q_s_list = []
            q_p_list = []
            for t in range(vid.shape[1]):
                qs_t, qp_t, idx_t = self.image_tokenizer.encode(vid[:, t])
                q_s_list.append(qs_t)
                q_p_list.append(qp_t)
            q_s_mean = torch.stack(q_s_list, dim=1).mean(dim=1)
            vlm_hidden_states = self.vlm(visual_latents=q_s_mean)
            conditioning_c = self.mcp(vlm_hidden_states)
            
        elif src == "audio":
            aud = audio_input if isinstance(audio_input, torch.Tensor) else torch.randn(B, 1, audio_len, device=self.device)
            q_s, q_p, indices = self.audio_tokenizer.encode(aud)
            # Reshape 1D audio semantic feature to 4D to be fully compatible with VLM mock visual_latents parameter
            q_s_vlm = q_s.unsqueeze(-1)
            vlm_hidden_states = self.vlm(visual_latents=q_s_vlm)
            conditioning_c = self.mcp(vlm_hidden_states)
            
        else:
            raise ValueError(f"Unknown source modality: {src}")
            
        # --- TARGET DECODING & GENERATION ---
        if tgt == "text":
            output = self.text_head(vlm_hidden_states)
            
        elif tgt == "image":
            noise = torch.randn(B, 256, self.latent_dim, device=self.device)
            diffused_latent = self.diffusion_image(noise, conditioning_c)
            diffused_latent_spatial = diffused_latent.permute(0, 2, 1).reshape(B, self.latent_dim, 16, 16)
            output = self.image_tokenizer.decode_pixel(diffused_latent_spatial)
            
        elif tgt == "video":
            state = None
            frames = []
            for t in range(num_frames):
                if mode == "image_to_video" and t == 0:
                    # In image-to-video, first frame utilizes encoded pixel latent
                    # Reshape q_p from (B, C, H, W) to (B, L, C)
                    current_latent = q_p.permute(0, 2, 3, 1).reshape(B, -1, self.latent_dim)
                else:
                    noise = torch.randn(B, 256, self.latent_dim, device=self.device)
                    current_latent = self.diffusion_image(noise, conditioning_c)
                    
                coherent_latent, state = self.temporal_wedge(current_latent, state)
                coherent_latent_spatial = coherent_latent.permute(0, 2, 1).reshape(B, self.latent_dim, 16, 16)
                frame = self.image_tokenizer.decode_pixel(coherent_latent_spatial)
                frames.append(frame)
            output = torch.stack(frames, dim=1)
            
        elif tgt == "audio":
            L_audio_latent = audio_len // 8
            noise = torch.randn(B, L_audio_latent, self.latent_dim, device=self.device)
            diffused_latent = self.diffusion_audio(noise, conditioning_c)
            diffused_latent_spatial = diffused_latent.permute(0, 2, 1) # (B, latent_dim, L_audio_latent)
            output = self.audio_tokenizer.decode_pixel(diffused_latent_spatial)
            
        else:
            raise ValueError(f"Unknown target modality: {tgt}")
            
        t_end = time.time()
        latency = (t_end - t_start) * 1000
        print(f"[AnyToAnyOrchestrator] Completed routing in {latency:.2f} ms.")
        return output
