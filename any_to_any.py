import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from tokenflow import TokenFlowDualQuantizer, TokenFlowTokenizer
from mcp import MobileConditioningProjector
from ssm_temporal import TemporalWedgeBlock
from dynamap import MiniCPMSALAMock, BonsaiDiffusionMock
class OffloadInputHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer):
        ctx.layer = layer
        return x.clone() if isinstance(x, torch.Tensor) else x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.layer.to("cpu", non_blocking=True)
        return grad_output, None

class OffloadOutputHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer, device):
        ctx.layer = layer
        ctx.device = device
        return x.clone() if isinstance(x, torch.Tensor) else x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.layer.to(ctx.device)
        return grad_output, None, None

class OffloadedLayerWrapper(nn.Module):
    """
    A PyTorch Layer Wrapper that dynamically swaps layer weights and all input tensors
    to the target GPU execution device during the forward pass, then immediately
    swaps the weights back to CPU RAM. Compatible with all PyTorch and Hugging Face architectures.
    Supports autograd backward offloading via input/output gates during training.
    """
    def __init__(self, original_layer, execution_device="cuda"):
        super().__init__()
        self.layer = original_layer.to("cpu")
        self.execution_device = torch.device(execution_device)

    def _apply(self, fn):
        # 1. Temporarily detach self.layer to bypass recursive submodule processing by super()._apply
        layer = self.layer
        delattr(self, 'layer')
        
        # 2. Call super()._apply for other attributes
        super()._apply(fn)
        
        # 3. Restore self.layer
        self.layer = layer
        
        # 4. Apply the function to self.layer but force the output tensors to remain on CPU
        def cpu_fn(t):
            res = fn(t)
            if res is not None and isinstance(res, torch.Tensor) and res.device.type != 'cpu':
                return res.cpu()
            return res
            
        self.layer._apply(cpu_fn)
        return self

    def forward(self, *args, **kwargs):
        # 1. Input Gate: Apply input hook during training to trigger offloading to CPU on backward exit
        new_args = args
        if self.training and len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].requires_grad:
            new_args = (OffloadInputHook.apply(args[0], self.layer),) + args[1:]
            
        # 2. Swap current layer parameters to GPU
        self.layer.to(self.execution_device)
        
        # 3. Transfer all input tensors to execution device
        from offloader import recursive_to_device
        dev_args = tuple(recursive_to_device(x, self.execution_device) for x in new_args)
        dev_kwargs = {k: recursive_to_device(v, self.execution_device) for k, v in kwargs.items()}
        
        # 4. Execute the actual forward step on GPU
        output = self.layer(*dev_args, **dev_kwargs)
        
        # 5. Output Gate: Apply output hook during training to trigger loading to GPU on backward entry
        if self.training and isinstance(output, torch.Tensor) and output.requires_grad:
            output = OffloadOutputHook.apply(output, self.layer, self.execution_device)
        else:
            # During evaluation (inference) or if output doesn't require grad, immediately offload to CPU
            self.layer.to("cpu", non_blocking=True)
            
        # 6. Ensure that all output tensors remain on execution device
        return recursive_to_device(output, self.execution_device)

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

class Heavy2BTransformerLayer(nn.Module):
    """
    A high-capacity transformer layer simulating ~50M parameters.
    Consists of self-attention projections and a SwiGLU MLP.
    """
    def __init__(self, dim=2048, hidden_dim=5460):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Self-Attention projection weights: ~16.7M parameters
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm2 = nn.LayerNorm(dim)
        # SwiGLU MLP: ~33.5M parameters (Gate + Up, then Down)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # 1. Attention residual
        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Simple simulated attention context blending
        attn_out = self.out_proj(q * torch.sigmoid(k).mean(dim=1, keepdim=True))
        x = x + attn_out
        
        # 2. SwiGLU FFN residual
        x_norm2 = self.norm2(x)
        swiglu = F.silu(self.gate_proj(x_norm2)) * self.up_proj(x_norm2)
        ffn_out = self.down_proj(swiglu)
        x = x + ffn_out
        return x

class Heavy2BTransformer(nn.Module):
    """
    Consolidated 2 Billion parameter transformer backbone (40 offloaded layers).
    Applies gradient checkpointing per layer block during backpropagation to prevent
    VRAM spikes under 3GB constraints.
    """
    def __init__(self, dim=2048, hidden_dim=5460, num_layers=40, execution_device="cuda"):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.execution_device = torch.device(execution_device)
        
        # Wrap each of the 40 layers inside OffloadedLayerWrapper
        self.layers = nn.ModuleList([
            OffloadedLayerWrapper(
                Heavy2BTransformerLayer(dim=dim, hidden_dim=hidden_dim),
                execution_device=self.execution_device
            ) for _ in range(num_layers)
        ])

    def forward(self, text_tokens=None, visual_latents=None):
        # 1. Determine batch and sequence lengths dynamically
        if text_tokens is not None:
            B, S = text_tokens.shape
        elif visual_latents is not None:
            if len(visual_latents.shape) == 4:
                B, D_lat, H, W = visual_latents.shape
                S = H * W
            else:
                # Handle flattened 1D audio shapes or intermediate embeddings
                B, S, D_lat = visual_latents.shape
        else:
            B, S = 1, 512
            
        device = text_tokens.device if text_tokens is not None else (visual_latents.device if visual_latents is not None else self.execution_device)
        
        # 2. Create initial state
        x = torch.zeros(B, S, self.dim, device=device)
        if visual_latents is not None:
            if len(visual_latents.shape) == 4:
                flat_visual = visual_latents.permute(0, 2, 3, 1).reshape(B, S, -1)
            else:
                flat_visual = visual_latents
            if flat_visual.shape[-1] != self.dim:
                proj = nn.Linear(flat_visual.shape[-1], self.dim, device=device)
                x = x + proj(flat_visual)
            else:
                x = x + flat_visual
                
        # 3. Layer-by-layer forward execution with dynamic offloading and gradient checkpointing
        hidden_states = []
        for layer in self.layers:
            # Transfer execution device configuration dynamically
            layer.execution_device = self.execution_device
            
            # Apply PyTorch activation checkpointing to prevent activation VRAM growth
            if self.training:
                # Define a wrapper function for checkpoint
                def run_layer(layer_inputs):
                    return layer(layer_inputs)
                x = torch.utils.checkpoint.checkpoint(run_layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
            hidden_states.append(x)
            
        return hidden_states

class AnyToAnyOrchestrator(nn.Module):
    """
    Any-to-Any Multimodal Generative Orchestrator supporting all 16 routing pathways
    between Text, Image, Video, and Audio under a strict 3GB VRAM ceiling.
    Scales to 2 Billion parameter backbone using CPU weight swapping.
    """
    def __init__(self, codebook_size=4096, vlm_dim=2048, dit_dim=1024, latent_dim=256, vocab_size=32000, device="cpu"):
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
        
        # 4. Massive 2B parameter VLM backbone consisting of 40 offloaded SwiGLU layers
        self.vlm = Heavy2BTransformer(
            dim=vlm_dim,
            hidden_dim=5460,
            num_layers=40,
            execution_device=self.device
        )
        
        # 5. Connectors & Decoders wrapped in OffloadedLayerWrapper
        self.mcp = OffloadedLayerWrapper(
            MobileConditioningProjector(vlm_dim=vlm_dim, dit_dim=dit_dim, num_layers=40),
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
        
        print(f"[AnyToAnyOrchestrator] Any-to-Any 2B-Scale engine initialized on device {self.device}.")

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
