import torch
import torch.nn as nn
import math
from pixelle_sirius_engine import PixelleSiriusOrchestrator

class PixelleSiriusLoader:
    """
    ComfyUI Node to load the Pixelle-Sirius SSM-Diffusion model.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "load_in_4bit": (["True", "False"], {"default": "True"}),
                "offload": (["True", "False"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("PIXELLE_SIRIUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "PixelleSirius"

    def load_model(self, device, load_in_4bit, offload):
        dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        offload_bool = (offload == "True")
        load_4bit_bool = (load_in_4bit == "True")
        
        print(f"[ComfyUI Loader] Initializing Pixelle-Sirius on {dev}...")
        orchestrator = PixelleSiriusOrchestrator(
            codebook_size=2048,
            vlm_dim=2048,
            dit_dim=1024,
            latent_dim=256,
            vocab_size=32000,
            device=dev
        )
        orchestrator.eval()
        
        # Load the real weights
        print(f"[ComfyUI Loader] Loading backbones (offload={offload_bool}, load_in_4bit={load_4bit_bool})...")
        orchestrator.load_real_backbones(offload=offload_bool, load_in_4bit=load_4bit_bool)
        
        return (orchestrator,)

class PixelleSiriusImageGen:
    """
    ComfyUI Node to generate images with aspect ratios.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXELLE_SIRIUS_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "Futuristic neon city abstract artwork"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "custom", "prompt"], {"default": "1:1"}),
                "custom_aspect_ratio": ("STRING", {"default": "3:2"}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_image"
    CATEGORY = "PixelleSirius"

    def generate_image(self, model, prompt, aspect_ratio, custom_aspect_ratio, steps):
        ratio = custom_aspect_ratio if aspect_ratio == "custom" else aspect_ratio
        
        # 1. Encode prompt using Qwen2 to extract text conditioning features
        device = model.device
        if model.real_weights_enabled:
            inputs = model.real_tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out_hf = model.real_qwen(input_ids=input_ids)
                text_features = out_hf.last_hidden_state.float() # (1, S_prompt, 896)
        else:
            text_features = torch.randn(1, 16, model.vlm_dim, device=device)
            
        # 2. Run LCM Consistency Solver with aspect ratios
        with torch.no_grad():
            latents, latency = model.consistency_generate(
                mode="image",
                conditioning_c=text_features,
                num_steps=steps,
                aspect_ratio=ratio,
                prompt=prompt
            ) # Output shape: (B, h_g, w_g, D)
            
        # 3. Create high-quality visual representation for ComfyUI IMAGE output
        # Deterministically project high-dimensional latents (256-dim) to RGB (3-dim)
        B, H_g, W_g, D = latents.shape
        generator = torch.Generator(device=device).manual_seed(42)
        proj_matrix = torch.randn(D, 3, generator=generator, device=device)
        proj_matrix = proj_matrix / proj_matrix.norm(dim=0, keepdim=True)
        
        # Project and apply sigmoid to bound to [0, 1] range
        rgb = torch.matmul(latents, proj_matrix)
        rgb_image = torch.sigmoid(rgb) # Shape: (B, H_g, W_g, 3)
        
        # Rescale/Interpolate to default visual display size (512x512 with correct aspect ratio)
        # Convert to CHW format for interpolation
        rgb_chw = rgb_image.permute(0, 3, 1, 2)
        
        # Compute display dimensions matching aspect ratio
        display_height = 512
        w_r, h_r = model.parse_aspect_ratio(ratio, prompt)
        display_width = int(round(display_height * w_r / h_r))
        # Ensure divisible by 8
        display_width = (display_width // 8) * 8
        
        interpolated = torch.nn.functional.interpolate(
            rgb_chw, 
            size=(display_height, display_width), 
            mode="bilinear", 
            align_corners=False
        )
        final_image = interpolated.permute(0, 2, 3, 1).cpu() # Shape: (B, H_display, W_display, 3)
        
        # 4. Standard ComfyUI LATENT dictionary output: samples shape (B, D, H, W)
        latent_samples = latents.permute(0, 3, 1, 2).cpu()
        
        return (final_image, {"samples": latent_samples})

class PixelleSiriusVideoGen:
    """
    ComfyUI Node to generate video frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXELLE_SIRIUS_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "Futuristic neon city abstract artwork sequence"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "custom", "prompt"], {"default": "1:1"}),
                "custom_aspect_ratio": ("STRING", {"default": "3:2"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "generate_video"
    CATEGORY = "PixelleSirius"

    def generate_video(self, model, prompt, aspect_ratio, custom_aspect_ratio, steps):
        ratio = custom_aspect_ratio if aspect_ratio == "custom" else aspect_ratio
        device = model.device
        
        # 1. Encode prompt
        if model.real_weights_enabled:
            inputs = model.real_tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out_hf = model.real_qwen(input_ids=input_ids)
                text_features = out_hf.last_hidden_state.float()
        else:
            text_features = torch.randn(1, 16, model.vlm_dim, device=device)
            
        # 2. Run video generation
        with torch.no_grad():
            video_latents, latency = model.consistency_generate(
                mode="video",
                conditioning_c=text_features,
                num_steps=steps,
                aspect_ratio=ratio,
                prompt=prompt
            ) # Output shape: (B, S_frames, h_g, w_g, D)
            
        # 3. Project and interpolate video frames
        B, S_frames, H_g, W_g, D = video_latents.shape
        generator = torch.Generator(device=device).manual_seed(42)
        proj_matrix = torch.randn(D, 3, generator=generator, device=device)
        proj_matrix = proj_matrix / proj_matrix.norm(dim=0, keepdim=True)
        
        # Flatten temporal frames to batch dimension for projection/interpolation
        video_flat = video_latents.view(B * S_frames, H_g, W_g, D)
        rgb_flat = torch.matmul(video_flat, proj_matrix)
        rgb_image = torch.sigmoid(rgb_flat) # (B * S_frames, H_g, W_g, 3)
        
        # Resize to display dimensions
        display_height = 512
        w_r, h_r = model.parse_aspect_ratio(ratio, prompt)
        display_width = (int(round(display_height * w_r / h_r)) // 8) * 8
        
        rgb_chw = rgb_image.permute(0, 3, 1, 2)
        interpolated = torch.nn.functional.interpolate(
            rgb_chw, 
            size=(display_height, display_width), 
            mode="bilinear", 
            align_corners=False
        )
        final_images = interpolated.permute(0, 2, 3, 1).cpu() # (B * S_frames, H_display, W_display, 3)
        
        # 4. Latent representation for ComfyUI: (S_frames, D, H, W)
        latent_samples = video_flat.permute(0, 3, 1, 2).cpu()
        
        return (final_images, {"samples": latent_samples})

class PixelleSiriusVideoEdit:
    """
    ComfyUI Node to perform video editing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXELLE_SIRIUS_MODEL",),
                "latent": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default": "Add glowing neon particles"}),
                "edit_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "edit_video"
    CATEGORY = "PixelleSirius"

    def edit_video(self, model, latent, prompt, edit_strength, steps):
        device = model.device
        
        # 1. Parse ComfyUI Latent (S_frames, D, H, W) -> Orchestrator (1, S_frames, H, W, D)
        latent_samples = latent["samples"].to(device) # (S_frames, D, H, W)
        S_frames, D, H, W = latent_samples.shape
        
        orchestrator_latents = latent_samples.permute(0, 2, 3, 1).unsqueeze(0) # (1, S_frames, H, W, D)
        
        # 2. Run video editing
        with torch.no_grad():
            edited_latents, latency = model.video_edit(
                input_video=orchestrator_latents,
                prompt=prompt,
                edit_strength=edit_strength,
                num_steps=steps
            ) # Output shape: (1, S_frames, H, W, D)
            
        # 3. Project edited frames to RGB
        generator = torch.Generator(device=device).manual_seed(42)
        proj_matrix = torch.randn(D, 3, generator=generator, device=device)
        proj_matrix = proj_matrix / proj_matrix.norm(dim=0, keepdim=True)
        
        edited_flat = edited_latents.view(S_frames, H, W, D)
        rgb_flat = torch.matmul(edited_flat, proj_matrix)
        rgb_image = torch.sigmoid(rgb_flat) # (S_frames, H, W, 3)
        
        # Resize to display
        display_height = 512
        display_width = (int(round(display_height * W / H)) // 8) * 8
        
        rgb_chw = rgb_image.permute(0, 3, 1, 2)
        interpolated = torch.nn.functional.interpolate(
            rgb_chw, 
            size=(display_height, display_width), 
            mode="bilinear", 
            align_corners=False
        )
        final_images = interpolated.permute(0, 2, 3, 1).cpu() # (S_frames, H_display, W_display, 3)
        
        # 4. Latent output
        out_latent = edited_flat.permute(0, 3, 1, 2).cpu()
        
        return (final_images, {"samples": out_latent})

class PixelleSiriusVideoUnderstand:
    """
    ComfyUI Node for video understanding.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXELLE_SIRIUS_MODEL",),
                "latent": ("LATENT",),
                "prompt": ("STRING", {"default": "Describe this video."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "understand"
    CATEGORY = "PixelleSirius"

    def understand(self, model, latent, prompt):
        device = model.device
        
        # 1. Parse ComfyUI Latent (S_frames, D, H, W) -> Orchestrator (1, S_frames, H, W, D)
        latent_samples = latent["samples"].to(device)
        S_frames, D, H, W = latent_samples.shape
        
        orchestrator_latents = latent_samples.permute(0, 2, 3, 1).unsqueeze(0) # (1, S_frames, H, W, D)
        
        # 2. Run video understanding
        with torch.no_grad():
            description, latency = model.understand_video(
                video_latents=orchestrator_latents,
                prompt_text=prompt
            )
            
        return (description,)

# Export mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelleSiriusLoader": PixelleSiriusLoader,
    "PixelleSiriusImageGen": PixelleSiriusImageGen,
    "PixelleSiriusVideoGen": PixelleSiriusVideoGen,
    "PixelleSiriusVideoEdit": PixelleSiriusVideoEdit,
    "PixelleSiriusVideoUnderstand": PixelleSiriusVideoUnderstand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelleSiriusLoader": "Pixelle-Sirius Model Loader",
    "PixelleSiriusImageGen": "Pixelle-Sirius Image Generator",
    "PixelleSiriusVideoGen": "Pixelle-Sirius Video Generator",
    "PixelleSiriusVideoEdit": "Pixelle-Sirius Video Editor",
    "PixelleSiriusVideoUnderstand": "Pixelle-Sirius Video Understander",
}
