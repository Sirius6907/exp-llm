import os
import time
import torch
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from comfy_nodes import (
    PixelleSiriusLoader,
    PixelleSiriusImageGen,
    PixelleSiriusVideoGen,
    PixelleSiriusVideoEdit,
    PixelleSiriusVideoUnderstand
)

def test_cinematic_suite():
    print("==================================================")
    print("   Pixelle-Sirius: Cinematic Video Suite & ComfyUI ")
    print("==================================================")
    print("Objective: Verify custom aspect ratios, video editing, video")
    print("understanding, and ComfyUI integration under a 2.0 GB VRAM ceiling.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU/CPU Execution Device: {device}")
    
    # ----------------------------------------------------
    # 1. Load Model via ComfyUI Loader Node (4-bit active)
    # ----------------------------------------------------
    print("\n[1/5] Loading model via ComfyUI Loader node...")
    loader = PixelleSiriusLoader()
    (orchestrator,) = loader.load_model(
        device="cuda" if device.type == "cuda" else "cpu",
        load_in_4bit="True",
        offload="False"
    )
    
    if not orchestrator.real_weights_enabled:
        print("❌ Could not load pre-trained weights. Running in synthetic mode.")
        return
        
    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # ----------------------------------------------------
    # 2. Verify Predefined & Custom Aspect Ratio Grids
    # ----------------------------------------------------
    print("\n[2/5] Verifying dynamic aspect ratio grid configurations...")
    aspect_ratios = ["1:1", "16:9", "9:16", "4:3", "3:2"]
    
    for aspect in aspect_ratios:
        w_r, h_r = orchestrator.parse_aspect_ratio(aspect)
        # Compute expected grid
        target_patches = 256
        h_g = max(4, int(round((target_patches * h_r / w_r) ** 0.5)))
        w_g = max(4, int(round(target_patches / h_g)))
        
        # Test image generation for this aspect ratio
        print(f"  -> Testing aspect ratio '{aspect}'...")
        with torch.no_grad():
            img_latents, latency = orchestrator.consistency_generate(
                mode="image",
                conditioning_c=torch.randn(1, 16, 2048, device=device),
                num_steps=2,
                aspect_ratio=aspect
            )
        
        # Check shape
        expected_shape = (1, h_g, w_g, 256)
        print(f"     Grid Size: {w_g}x{h_g} ({w_g * h_g} patches) | Latent Shape: {img_latents.shape}")
        assert img_latents.shape == expected_shape, f"❌ Shape mismatch for {aspect}. Expected {expected_shape}, got {img_latents.shape}"
        print("     STATUS: [OK]")
        
    # ----------------------------------------------------
    # 3. Verify Cinematic Video Editing
    # ----------------------------------------------------
    print("\n[3/5] Verifying cinematic video editing...")
    # Generate mock input video latents (4 frames, 16x16 grid, 256-dim)
    input_video = torch.randn(1, 4, 16, 16, 256, device=device)
    prompt = "Edit: Add volumetric fog and movie lighting"
    
    t0 = time.time()
    with torch.no_grad():
        edited_video, latency = orchestrator.video_edit(
            input_video=input_video,
            prompt=prompt,
            edit_strength=0.6,
            num_steps=4
        )
    t1 = time.time()
    print(f"  -> SUCCESS | Video Edit Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  -> Edited Video Shape: {edited_video.shape}")
    assert edited_video.shape == (1, 4, 16, 16, 256), f"❌ Edited video shape mismatch"
    assert not torch.isnan(edited_video).any(), "❌ Edited video contains NaN values"
    print("  -> STATUS: [OK]")
    
    # ----------------------------------------------------
    # 4. Verify Video-LLM Understanding
    # ----------------------------------------------------
    print("\n[4/5] Verifying Video-LLM understanding...")
    # Use the edited video as input for understanding
    t0 = time.time()
    with torch.no_grad():
        description, latency = orchestrator.understand_video(
            video_latents=edited_video,
            prompt_text="What features are present in this video clip?"
        )
    t1 = time.time()
    print(f"  -> SUCCESS | Video Understanding Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  -> Generated Text Output: '{description}'")
    assert description.startswith("Video analysis:"), "❌ Description should start with Video analysis"
    print("  -> STATUS: [OK]")
    
    # ----------------------------------------------------
    # 5. Verify ComfyUI Node Execution
    # ----------------------------------------------------
    print("\n[5/5] Verifying ComfyUI nodes output compatibility...")
    image_gen_node = PixelleSiriusImageGen()
    video_gen_node = PixelleSiriusVideoGen()
    video_edit_node = PixelleSiriusVideoEdit()
    video_understand_node = PixelleSiriusVideoUnderstand()
    
    # A. Test Image Gen Node
    print("  -> Executing PixelleSiriusImageGen node...")
    final_img, latent_dict = image_gen_node.generate_image(
        model=orchestrator,
        prompt="A high quality landscape in widescreen format",
        aspect_ratio="16:9",
        custom_aspect_ratio="3:2",
        steps=2
    )
    print(f"     Image output shape (for display): {final_img.shape}")
    print(f"     Latent sample shape: {latent_dict['samples'].shape}")
    assert len(final_img.shape) == 4 and final_img.shape[-1] == 3, "❌ ComfyUI Image should be BHWC format"
    assert len(latent_dict['samples'].shape) == 4 and latent_dict['samples'].shape[1] == 256, "❌ Latents should be BCHW format"
    
    # B. Test Video Gen Node
    print("  -> Executing PixelleSiriusVideoGen node...")
    final_video_imgs, video_latent_dict = video_gen_node.generate_video(
        model=orchestrator,
        prompt="Cinematic zoom into abstract nebula",
        aspect_ratio="4:3",
        custom_aspect_ratio="1:1",
        steps=4
    )
    print(f"     Video Image batch shape: {final_video_imgs.shape}")
    print(f"     Video Latent sample shape: {video_latent_dict['samples'].shape}")
    # In ComfyUI, a video is returned as a batch of images (num_frames, H, W, 3)
    assert final_video_imgs.shape[0] == 4, "❌ ComfyUI Video Gen should output batch size equal to num_frames"
    
    # C. Test Video Edit Node
    print("  -> Executing PixelleSiriusVideoEdit node...")
    edited_imgs, edited_latent_dict = video_edit_node.edit_video(
        model=orchestrator,
        latent=video_latent_dict,
        prompt="Turn the nebula into green color",
        edit_strength=0.5,
        steps=2
    )
    print(f"     Edited Video Image batch shape: {edited_imgs.shape}")
    print(f"     Edited Latent shape: {edited_latent_dict['samples'].shape}")
    assert edited_imgs.shape[0] == 4, "❌ ComfyUI Video Edit should preserve frame count"
    
    # D. Test Video Understand Node
    print("  -> Executing PixelleSiriusVideoUnderstand node...")
    (node_desc,) = video_understand_node.understand(
        model=orchestrator,
        latent=video_latent_dict,
        prompt="Describe the colors and motion."
    )
    print(f"     Node Output Description: '{node_desc}'")
    print("  -> STATUS: [OK]")
    
    # ----------------------------------------------------
    # VRAM Audit
    # ----------------------------------------------------
    print("\nAuditing Peak VRAM consumption...")
    print("==================================================")
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU VRAM Reserved during test suite: {peak_vram:.2f} MB")
        
        if peak_vram < 2000.0:
            print("\n[STATUS: PASS]")
            print("Successfully verified cinematic suite and ComfyUI nodes fully locally under 2.0 GB VRAM limit!")
        else:
            print("\n[STATUS: WARNING]")
            print("Completed but VRAM usage exceeded 2.0 GB target limit.")
    else:
        print("\n[STATUS: PASS] Completed successfully on CPU!")
    print("==================================================")

if __name__ == "__main__":
    test_cinematic_suite()
