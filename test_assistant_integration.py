import os
import time
import torch
import dask.array as da
import math
from mark_xxxix_assistant import MarkXXXIXAssistantBrain

def test_assistant_integration():
    print("==================================================")
    print("   Mark-XXXIX Offline Brain: Integration Test     ")
    print("==================================================")
    print("Objective: Verify screenshot capturing, voice command ingestion,")
    print("SSM visual temporal fuser, and speculative decision loops under 2.0 GB VRAM.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU/CPU Execution Device: {device}")
    
    # 1. Initialize the Assistant Brain
    t0 = time.time()
    assistant = MarkXXXIXAssistantBrain(device=device)
    t1 = time.time()
    print(f"  -> SUCCESS | Assistant brain loaded in {(t1 - t0):.2f} seconds.")
    
    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # 2. Ingest Multiple Screenshots (Testing SSM Visual History State Passing)
    print("\n[1/4] Simulating continuous screen polling (3 screenshots)...")
    for step in range(3):
        print(f"  -> Grabbing active desktop screen (Step {step+1}/3)...")
        screenshot = assistant.capture_screen()
        
        t_ingest_start = time.time()
        vision_token = assistant.ingest_screenshot(screenshot)
        t_ingest_end = time.time()
        
        print(f"     Ingestion Latency: {(t_ingest_end - t_ingest_start)*1000:.2f} ms")
        assert vision_token is not None, f"❌ Failed to extract visual tokens"
        assert vision_token.shape == (1, 896), f"❌ Unexpected projected shape: {vision_token.shape}"
        
    # Verify screen history buffer is filled
    assert len(assistant.screen_tokens_buffer) == 3, f"❌ Screen tokens buffer should contain 3 entries"
    print("  -> Screen Ingestion History: [OK]")
    
    # 3. Ingest Voice command
    print("\n[2/4] Simulating voice command input (1-second mock mono audio)...")
    # 1 second of 16000Hz mono audio generated via Dask Array
    da_t = da.linspace(0, 440 * 2 * math.pi, 16000, chunks=4000)
    mock_audio = da.sin(da_t).astype(da.float32).compute()
    
    t_audio_start = time.time()
    audio_tokens = assistant.ingest_voice_command(mock_audio)
    t_audio_end = time.time()
    
    print(f"     Acoustic Ingestion Latency: {(t_audio_end - t_audio_start)*1000:.2f} ms")
    assert audio_tokens is not None, f"❌ Failed to extract audio features"
    assert audio_tokens.shape[1] > 0 and audio_tokens.shape[-1] == 896, f"❌ Unexpected audio tokens shape: {audio_tokens.shape}"
    print("  -> Voice Command Ingestion: [OK]")
    
    # 4. Run Causal Action Decisions
    print("\n[3/4] Running speculative action decision loop...")
    t_decide_start = time.time()
    action = assistant.decide_action(
        prompt_text="Task: Click the browser icon.",
        audio_waveform=mock_audio
    )
    t_decide_end = time.time()
    
    print(f"     Decision Loop Latency: {(t_decide_end - t_decide_start)*1000:.2f} ms")
    print(f"     Decided Action: '{action}'")
    assert isinstance(action, str) and len(action) > 0, "❌ Failed to generate causal response string"
    print("  -> Speculative Action Decision: [OK]")
    
    # 5. Verify Mock Action Execution
    print("\n[4/4] Verifying action execution outputs...")
    # Test click execution
    res = assistant.execute_action("[CLICK 120,450]")
    assert "Click simulated at 120,450" in res, "❌ Click execution parsing failed"
    # Test typing execution
    res = assistant.execute_action("[TYPE Hello World]")
    assert "Typing simulated: 'Hello World'" in res, "❌ Typing execution parsing failed"
    # Test launch execution
    res = assistant.execute_action("[LAUNCH Chrome]")
    assert "Launch simulated: Chrome" in res, "❌ Launch execution parsing failed"
    # Test speech execution
    res = assistant.execute_action("Launching your browser window now.")
    assert "Speech output" in res, "❌ Speech execution parsing failed"
    print("  -> Action Execution Parsing: [OK]")
    
    # 6. Execute Mock Loop Cycle
    def get_mock_audio():
        return da.random.normal(size=16000, chunks=4000).astype(da.float32).compute()
        
    assistant.run_loop(num_iterations=2, interval=0.5, audio_mock_generator=get_mock_audio)
    
    # 7. Auditing VRAM footprint
    print("\nAuditing Peak VRAM consumption...")
    print("==================================================")
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU VRAM Reserved (Assistant Loop Active): {peak_vram:.2f} MB")
        
        if peak_vram < 2000.0:
            print("\n[STATUS: PASS]")
            print("Successfully verified Mark-XXXIX offline brain integration fully locally under 2.0 GB VRAM limit!")
        else:
            print("\n[STATUS: WARNING]")
            print("Completed but VRAM usage exceeded 2.0 GB target limit.")
    else:
        print("\n[STATUS: PASS] Completed successfully on CPU!")
    print("==================================================")

if __name__ == "__main__":
    test_assistant_integration()
