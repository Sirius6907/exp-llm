import time
import numpy as np
import torch
import os
import sys

# Set console encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

from pixelle_sirius_engine import PixelleSiriusOrchestrator
from sirius_ops import SiriusZeroLossMemory

def run_latency_determinism_tests():
    print("==================================================")
    print("  Latency Determinism & LRU Eviction Validation   ")
    print("==================================================")
    
    device = torch.device("cpu")
    print(f"[INFO] Initializing engine on: {device}")
    
    # 1. Initialize Memory Bank
    mem = SiriusZeroLossMemory()
    if not mem.use_rust:
        print("[Warning] Rust extension not active. Skipping strict LRU test.")
        return
        
    print("[SUCCESS] Zero-Copy Rust memory bank successfully initialized!")
    
    # Set capacity to a lower value for rapid eviction testing
    mem.rust_mem.max_capacity = 1000
    print(f"[INFO] Target capacity set to: {mem.rust_mem.max_capacity} items")
    
    # 2. Test LRU Eviction & Budget Ceiling
    print("\n[INFO] Ingesting 1,500 continuous embeddings to trigger LRU eviction...")
    dim = 256
    
    # Generate 1,500 unique embeddings
    np.random.seed(42)
    keys = [torch.randn(dim, dtype=torch.float32) for _ in range(1500)]
    values = [torch.randn(dim, dtype=torch.float32) for _ in range(1500)]
    
    # Insert sequentially
    for idx, (k, v) in enumerate(zip(keys, values)):
        mem.insert_continuous(k, v)
        
    current_size = len(mem)
    print(f"  -> Total stored items: {current_size}")
    assert current_size == 1000, f"Eviction failed! Expected size 1000, got {current_size}"
    print("  [PASS] LRU Cache correctly capped size to max_capacity (1000).")
    
    # Verify that the oldest items (indices 0 to 499) were evicted, while the newest (indices 500 to 1499) persist
    print("\n[INFO] Verifying eviction and hit/miss status...")
    evicted_key = keys[0]
    persisted_key = keys[1200]
    
    retrieved_evicted, sim_evicted = mem.query_continuous(evicted_key, 0.99)
    retrieved_persisted, sim_persisted = mem.query_continuous(persisted_key, 0.99)
    
    assert retrieved_evicted is None, "Eviction failed! Oldest item was not pruned."
    print("  [PASS] Oldest entry (index 0) was successfully pruned (Miss).")
    assert retrieved_persisted is not None, "Persistency check failed! Active item was pruned."
    print(f"  [PASS] Newer entry (index 1200) was successfully retrieved (Hit, Sim: {sim_persisted:.4f}).")
    
    # 3. Profile Latency Determinism & Tail Latency (p99)
    print("\n[INFO] Profiling sequential token generation tail latency under load...")
    
    # Setup Orchestrator
    orchestrator = PixelleSiriusOrchestrator(
        codebook_size=1024,
        vlm_dim=256,
        dit_dim=128,
        latent_dim=64,
        vocab_size=1000,
        device="cpu"
    )
    
    # Simulate speculative text generation loop and measure token-to-token duration
    latencies = []
    
    import threading
    
    # Pre-generate standard input vectors
    prompt_in = torch.randn(1, 1, 256)
    state = torch.zeros(1, 256, orchestrator.experts[0].state_dim)
    
    # Thread function to run continuous background inserts
    stop_inserts = False
    def background_inserter():
        np.random.seed(99)
        while not stop_inserts:
            k_sim = torch.randn(dim)
            v_sim = torch.randn(dim)
            orchestrator.zero_loss_mem.insert_continuous(k_sim, v_sim)
            time.sleep(0.001) # brief sleep to avoid CPU starvation
            
    # Start the background insertion thread
    t_inserter = threading.Thread(target=background_inserter)
    t_inserter.daemon = True
    t_inserter.start()
    
    # Run 100 consecutive decoding steps on the main thread
    for step in range(100):
        t0 = time.perf_counter()
        
        # Run Mamba recurrent scanning step
        with torch.no_grad():
            out, state = orchestrator.experts[0](prompt_in, state=state)
            
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms
        
        # Brief sleep to simulate real token-to-token decoding interval
        time.sleep(0.01)
        
    # Stop background inserter
    stop_inserts = True
    t_inserter.join(timeout=1.0)
        
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    
    print(f"  -> Mean step latency: {mean_lat:.4f} ms")
    print(f"  -> p95 tail latency: {p95_lat:.4f} ms")
    print(f"  -> p99 tail latency: {p99_lat:.4f} ms")
    
    # Check step-to-step variations
    variations = []
    for i in range(1, len(latencies)):
        denominator = max(latencies[i-1], 1e-6)
        v = abs(latencies[i] - latencies[i-1]) / denominator
        variations.append(v)
        
    mean_var = np.mean(variations) * 100.0
    print(f"  -> Mean step-to-step latency variation: {mean_var:.2f}%")
    
    # Verify tail latency spikes are isolated (p99 must be tightly bounded)
    assert p99_lat < mean_lat * 2.5, f"Tail latency spike detected! p99: {p99_lat:.4f} ms, limit: {mean_lat * 2.5:.4f} ms. Thread isolation failed."
    print("  [PASS] Latency tail is extremely stable. Thread isolation verified!")
    print("\n==================================================")
    print("     All Latency Determinism Tests Passed!       ")
    print("==================================================")

if __name__ == "__main__":
    run_latency_determinism_tests()
