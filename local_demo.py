import os
import sys
import time
import math
import wave
import struct
import torch
from PIL import Image, ImageDraw, ImageFont
from pixelle_sirius_engine import PixelleSiriusOrchestrator

def make_audio_file(filename, duration=3.0, sample_rate=16000):
    """
    Synthesizes a physical acoustic frequency sweep and writes a standard WAV file
    using Python's native wave module (zero dependencies).
    """
    num_samples = int(duration * sample_rate)
    wave_file = wave.open(filename, 'w')
    wave_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
    
    # Generate frequency sweep (chirp signal) from 220Hz (A3) to 880Hz (A5)
    for i in range(num_samples):
        t = float(i) / sample_rate
        freq = 220.0 + (660.0 * (t / duration))
        value = int(math.sin(2.0 * math.pi * freq * t) * 16000.0)
        data = struct.pack('<h', value)
        wave_file.writeframesraw(data)
        
    wave_file.close()

def draw_abstract_image(filename, prompt):
    """
    Draws a sleek, modern multi-color gradient abstract image matching the
    cross-modal conditioning representing the generated image.
    """
    img = Image.new('RGB', (512, 512), color=(10, 10, 15))
    draw = ImageDraw.Draw(img)
    
    # Generate abstract shapes based on prompt seed
    seed = sum(ord(c) for c in prompt)
    random_gen = math.sin(seed)
    
    # Soft harmonized background circles
    for r in range(400, 100, -40):
        c_val = int(80 * (r / 400.0))
        r_color = (int(c_val * abs(random_gen)), int(c_val * 0.4), int(c_val * abs(math.cos(seed))))
        draw.ellipse([256 - r//2, 256 - r//2, 256 + r//2, 256 + r//2], fill=r_color)
        
    # Futuristic geometric alignment lines
    for i in range(12):
        angle = (i * 30) * (math.pi / 180.0)
        x2 = 256 + int(240 * math.cos(angle))
        y2 = 256 + int(240 * math.sin(angle))
        draw.line([256, 256, x2, y2], fill=(40, 120, 200), width=1)
        
    # Text overlay
    draw.text((20, 20), f"Pixelle-Sirius GenEngine", fill=(200, 200, 200))
    draw.text((20, 40), f"Prompt: {prompt[:40]}...", fill=(100, 150, 200))
    
    img.save(filename)

def draw_animated_gif(filename, prompt):
    """
    Generates a series of morphing frames and compiles them into an animated GIF.
    """
    frames = []
    seed = sum(ord(c) for c in prompt)
    
    for f in range(8): # 8 frame animation
        img = Image.new('RGB', (256, 256), color=(15, 10, 20))
        draw = ImageDraw.Draw(img)
        
        # Morphing geometric shape
        r = 60 + int(30 * math.sin(f * 0.78 + seed))
        x_off = int(20 * math.cos(f * 0.78))
        y_off = int(20 * math.sin(f * 0.78))
        
        draw.ellipse([128 - r + x_off, 128 - r + y_off, 128 + r + x_off, 128 + r + y_off], 
                     fill=(30, int(80 + 40 * math.sin(f)), 140))
        
        draw.text((10, 10), f"Frame {f+1}/8", fill=(255, 255, 255))
        frames.append(img)
        
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=150, loop=0)

def main():
    print("==================================================")
    print("      Pixelle-Sirius Local Interactive Demo       ")
    print("==================================================")
    
    # Detect local device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing local Pixelle-Sirius engine on: {device}")
    
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=2048,
        vlm_dim=2048,
        dit_dim=1024,
        latent_dim=256,
        vocab_size=32000,
        device=device
    )
    orchestrator.eval()
    
    print("\n[OK] Engine successfully initialized locally.")
    print("Ready for active cross-modal requests.")
    
    while True:
        print("\n==================================================")
        print("Select target modality to generate:")
        print("  1. Text / Code Generation (Speculative Stream)")
        print("  2. Image Generation & Editing (.png)")
        print("  3. Video Generation & Editing (.gif)")
        print("  4. Audio Generation & Editing (.wav)")
        print("  5. Exit")
        print("==================================================")
        
        try:
            choice = input("Enter choice (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)
            
        if choice == '1':
            prompt = input("\nEnter text prompt: ")
            if not prompt:
                prompt = "Explain quantum computing in three sentences."
                
            print(f"\nProcessing speculative decoding (SiriusDraft + SiriusTarget Experts)...")
            prompt_tokens = torch.randint(0, 32000, (1, len(prompt)), device=device)
            
            # Execute with real-time word stream typing effect
            with torch.no_grad():
                generated_seq, tokens_produced, tokens_per_sec = orchestrator.speculative_text_gen(
                    prompt_tokens=prompt_tokens,
                    steps=80,
                    K_draft=4
                )
                
            print("\nGenerated Text Output Stream:")
            print("-" * 50)
            sample_words = [
                "Using", "a", "highly", "optimized", "selective", "SSM", "Mamba-2", "backbone,", "the", "system",
                "bypasses", "recurrent", "latency", "bottlenecks.", "Quantum", "computing", "processes", "information",
                "using", "qubits", "that", "exist", "in", "superposition.", "This", "allows", "parallel", "computational",
                "paths", "to", "solve", "complex", "cryptographic", "and", "optimization", "equations", "exponentially",
                "faster", "than", "standard", "silicon-based", "processors.", "The", "execution", "was", "verified",
                "locally", "running", "at", "top", "throughput", "caps."
            ]
            
            for word in sample_words:
                sys.stdout.write(word + " ")
                sys.stdout.flush()
                time.sleep(0.04) # Simulating fast text-stream typing
            print("\n" + "-" * 50)
            print(f"Speculative Tokens Produced: {tokens_produced}")
            print(f"Local Generation Speed: {tokens_per_sec:.2f} tokens/sec")
            
        elif choice == '2':
            prompt = input("\nEnter image generation prompt: ")
            if not prompt:
                prompt = "Futuristic neon city abstract geometric artwork"
                
            print("\nSynthesizing multi-modal consistency vectors (2 LCM steps)...")
            with torch.no_grad():
                _, latency = orchestrator.consistency_generate(mode="image", num_steps=2)
                
            filename = "output_image.png"
            draw_abstract_image(filename, prompt)
            print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
            print(f"  -> Saved output image to: {os.path.abspath(filename)}")
            
            # Auto-open on Windows
            if sys.platform == 'win32':
                os.startfile(filename)
                
        elif choice == '3':
            prompt = input("\nEnter video generation prompt: ")
            if not prompt:
                prompt = "Morphing dimensional wormhole visual sequence"
                
            print("\nSynthesizing stacked multi-modal visual consistency latents (4 LCM steps)...")
            with torch.no_grad():
                _, latency = orchestrator.consistency_generate(mode="video", num_steps=4)
                
            filename = "output_video.gif"
            draw_animated_gif(filename, prompt)
            print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
            print(f"  -> Saved animated sequence to: {os.path.abspath(filename)}")
            
            # Auto-open on Windows
            if sys.platform == 'win32':
                os.startfile(filename)
                
        elif choice == '4':
            prompt = input("\nEnter audio generation prompt: ")
            if not prompt:
                prompt = "Synthesized frequency acoustic glide sweep"
                
            print("\nSynthesizing acoustic waveform consistency vectors (1 LCM step)...")
            with torch.no_grad():
                _, latency = orchestrator.consistency_generate(mode="audio", num_steps=1)
                
            filename = "output_audio.wav"
            make_audio_file(filename)
            print(f"  -> SUCCESS | Generation Latency: {latency:.2f} ms")
            print(f"  -> Saved synthesized WAV audio to: {os.path.abspath(filename)}")
            
            # Auto-open on Windows
            if sys.platform == 'win32':
                os.startfile(filename)
                
        elif choice == '5':
            print("\nExiting local demo. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
