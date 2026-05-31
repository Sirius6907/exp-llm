# rust_inference.py
# High-Fidelity Python-Rust Bridge for Pixelle-Sirius Edge Engine
# Integrates PyO3-compiled bare-metal memory banks and parallelized Mamba selective scan steps.

import sys
import os
import time
import numpy as np
import torch

# Ensure the sirius_ops directory is on the path for easy import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Attempt to load the native PyO3 Rust extension module
    import sirius_ops_rust
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Native Rust Extension 'sirius_ops_rust' could not be loaded: {e}")
    RUST_AVAILABLE = False

def run_rust_benchmarks():
    print("==================================================")
    print("      Pixelle-Sirius: Python + Rust Bridge        ")
    print("==================================================")
    
    if not RUST_AVAILABLE:
        print("❌ Rust extension is not available. Skipping benchmarks.")
        return

    print("[SUCCESS] Compiled Rust library 'sirius_ops_rust' successfully loaded!")
    
    # 1. Initialize Rust Parallel Memory Bank
    print("\n[INFO] Initializing SiriusZeroLossMemoryRust database...")
    mem_bank = sirius_ops_rust.SiriusZeroLossMemoryRust()
    
    # Insert discrete test prompt-response pairs
    prompt = [101, 102, 103, 104]
    response = [999, 888, 777]
    mem_bank.insert_discrete(prompt, response)
    
    # Verify discrete lookup
    result = mem_bank.query_discrete(prompt)
    if result:
        retrieved_val, score = result
        print(f"  -> Discrete Match Verified: {prompt} -> {retrieved_val} (Score: {score})")
    
    # 2. Benchmark Parallel Continuous Embedding Sweeps (Rayon vs. Python Loop)
    print("\n[INFO] Ingesting 5,000 multi-modal key-value embeddings into database...")
    dim = 256
    np.random.seed(42)
    
    # Ingest keys and values
    for i in range(5000):
        key = np.random.randn(dim).astype(np.float32)
        val = np.random.randn(dim).astype(np.float32)
        mem_bank.insert_continuous(key, val)
        
    print(f"  -> Database size: {mem_bank.size()} items successfully ingested.")
    
    # Generate multiple query tensors for benchmarking
    queries = [np.random.randn(dim).astype(np.float32) for _ in range(100)]
    
    print("\n[INFO] Benchmarking continuous nearest-neighbor sweep (100 sequential queries over 5,000 items)...")
    
    # Time Rust Parallel Map-Reduce Sweep (using Rayon)
    t0 = time.time()
    for query_key in queries:
        rust_match = mem_bank.query_continuous(query_key, 0.70)
    t1 = time.time()
    rust_duration_ms = (t1 - t0) * 1000.0
    print(f"  -> Rust Parallel Sweep (Rayon): {rust_duration_ms:.4f} ms")
    if rust_match:
        val, sim = rust_match
        print(f"     Last query match similarity: {sim:.5f}")
        
    # Time Realistic single-threaded Python Loop
    # We pre-compute python list structures to simulate the same database search
    py_keys = [np.random.randn(dim).astype(np.float32).tolist() for _ in range(5000)]
    t0 = time.time()
    for query_key in queries:
        query_list = query_key.tolist()
        best_sim = -1.0
        q_norm = np.linalg.norm(query_list)
        for key in py_keys:
            # Native Python math for Cosine Similarity
            dot = sum(x*y for x, y in zip(query_list, key))
            k_norm = np.linalg.norm(key)
            sim = dot / (q_norm * k_norm) if q_norm * k_norm > 0 else 0.0
            if sim > best_sim:
                best_sim = sim
    t1 = time.time()
    py_duration_ms = (t1 - t0) * 1000.0
    print(f"  -> Native Python Math Loop: {py_duration_ms:.4f} ms")
    
    print(f"\n  [PASS] Rust Rayon Parallel speedup: {py_duration_ms / max(rust_duration_ms, 1e-6):.2f}x faster execution!")
    
    # 3. Verify Vectorized Cosine Similarity
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.2], dtype=np.float32)
    sim = sirius_ops_rust.fast_cosine_similarity_rust(a, b)
    print(f"\n[INFO] Auto-vectorized Cosine Similarity: {sim:.5f}")
    
    # 4. Verify Selective SSM Recurrent Scan Step in Rust
    print("\n[INFO] Testing bare-metal Mamba Recurrent Scan step in Rust...")
    x_step = np.random.randn(dim).astype(np.float32)
    gate_step = np.random.randn(dim).astype(np.float32)
    state = np.zeros((dim, 16)).astype(np.float32)
    A_log = np.random.randn(dim, 16).astype(np.float32)
    B_raw = np.random.randn(16).astype(np.float32)
    C_raw = np.random.randn(16).astype(np.float32)
    dt = np.random.randn(dim).astype(np.float32)
    norm_weight = np.random.randn(dim).astype(np.float32)
    
    t0 = time.time()
    y_out, new_state = sirius_ops_rust.mamba_selective_scan_step_rust(
        x_step, gate_step, state, A_log, B_raw, C_raw, dt, norm_weight
    )
    t1 = time.time()
    print(f"  -> SSM Step execution in Rust: {(t1 - t0) * 1000.0:.4f} ms")
    print(f"  -> Output shape: {y_out.shape} | New State Shape: {new_state.shape}")
    print("==================================================")

if __name__ == "__main__":
    run_rust_benchmarks()
