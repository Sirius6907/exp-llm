import torch
import torch.nn as nn
import numpy as np
import sys
import os
from .ops import fast_cosine_similarity

# Ensure directory is on the path to import local PyO3 extension
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sirius_ops_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class SiriusZeroLossMemory(nn.Module):
    """
    Episodic Non-Parametric Zero-Loss Memory Bank.
    Stores and retrieves multi-modal samples in O(1) time.
    Supports both exact discrete lookups (for text token IDs) and 
    approximate continuous lookups (for visual/acoustic embeddings).
    Delegates to a high-speed, thread-safe PyO3 Rust backend with zero-copy NumPy pointer sharing when available.
    """
    def __init__(self):
        super().__init__()
        # Discrete memory bank (Prompt Token Tuple -> Response Token List)
        self.discrete_store = {}
        
        # Continuous memory bank (Context Tensor -> Output Tensor)
        self.continuous_keys = []
        self.continuous_values = []
        
        self.use_rust = RUST_AVAILABLE
        if self.use_rust:
            self.rust_mem = sirius_ops_rust.SiriusZeroLossMemoryRust()
            # Explicitly initialize the Rayon thread pool to avoid asymmetric core stalls
            try:
                # Target primary worker threads (e.g. 4 performance threads)
                sirius_ops_rust.init_rayon_thread_pool(4)
            except Exception:
                pass
        else:
            self.rust_mem = None

    def insert_discrete(self, key_tokens, value_tokens):
        """
        Inserts a discrete key-value pair (e.g. text prompt token tuple to target response tokens).
        """
        if isinstance(key_tokens, torch.Tensor):
            key_tuple = tuple(key_tokens.cpu().flatten().tolist())
        else:
            key_tuple = tuple(key_tokens)
            
        if isinstance(value_tokens, torch.Tensor):
            value_list = value_tokens.cpu().flatten().tolist()
        else:
            value_list = list(value_tokens)
            
        self.discrete_store[key_tuple] = value_list
        
        if self.use_rust:
            self.rust_mem.insert_discrete(list(key_tuple), value_list)

    def insert_continuous(self, key_tensor, value_tensor):
        """
        Inserts a continuous key-value pair (e.g. multimodal embedding to target representation).
        """
        # Ensure key is a flat 1D float32 tensor
        k_flat = key_tensor.detach().float().cpu().view(-1)
        v_tensor = value_tensor.detach().cpu()
        
        # Sync with Python storage for strict compatibility
        match_idx = -1
        for idx, existing_k in enumerate(self.continuous_keys):
            sim = fast_cosine_similarity(k_flat, existing_k)
            if sim >= 0.999:
                match_idx = idx
                break
                
        if match_idx >= 0:
            self.continuous_values[match_idx] = v_tensor
        else:
            self.continuous_keys.append(k_flat)
            self.continuous_values.append(v_tensor)
            
        if self.use_rust:
            # Direct shared pointer view passing - absolute zero-copy!
            k_np = k_flat.numpy().astype(np.float32)
            v_np = v_tensor.numpy().astype(np.float32)
            self.rust_mem.insert_continuous(k_np, v_np)

    def insert(self, key, value):
        """
        Generic entry-point for dynamic O(1) episodic memorization.
        """
        # Decide if key is discrete or continuous
        if isinstance(key, (list, tuple)) or (isinstance(key, torch.Tensor) and key.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8)):
            self.insert_discrete(key, value)
        else:
            self.insert_continuous(key, value)

    def query_discrete(self, query_tokens):
        """
        Look up discrete exact prompt matches.
        """
        if self.use_rust:
            if isinstance(query_tokens, torch.Tensor):
                q_list = query_tokens.cpu().flatten().tolist()
            else:
                q_list = list(query_tokens)
            result = self.rust_mem.query_discrete(q_list)
            if result is not None:
                retrieved_val, score = result
                return retrieved_val, score
            return None, 0.0
            
        # Fallback to Python exact/prefix matching
        if isinstance(query_tokens, torch.Tensor):
            query_tuple = tuple(query_tokens.cpu().flatten().tolist())
        else:
            query_tuple = tuple(query_tokens)
            
        # Check for exact matches
        if query_tuple in self.discrete_store:
            return self.discrete_store[query_tuple], 1.0
            
        # Check for subsequence/prefix matches
        for stored_key, stored_val in self.discrete_store.items():
            if len(query_tuple) >= len(stored_key) and query_tuple[:len(stored_key)] == stored_key:
                return stored_val, 1.0
                
        return None, 0.0

    def query_continuous(self, query_tensor, similarity_threshold=0.98):
        """
        Look up continuous key matches using cosine similarity.
        """
        if self.use_rust:
            q_flat = query_tensor.detach().float().cpu().view(-1)
            q_np = q_flat.numpy().astype(np.float32)
            result = self.rust_mem.query_continuous(q_np, similarity_threshold)
            if result is not None:
                retrieved_val, score = result
                return torch.tensor(retrieved_val, dtype=torch.float32), float(score)
            return None, 0.0

        # Fallback to Python math loops
        if len(self.continuous_keys) == 0:
            return None, 0.0
            
        q_flat = query_tensor.detach().float().cpu().view(-1)
        
        best_sim = -1.0
        best_idx = -1
        
        for idx, key in enumerate(self.continuous_keys):
            sim = fast_cosine_similarity(q_flat, key)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
                
        if best_sim >= similarity_threshold and best_idx >= 0:
            return self.continuous_values[best_idx], float(best_sim)
            
        return None, 0.0

    def query(self, query_key, similarity_threshold=0.98):
        """
        Queries the memory bank. 
        Returns (retrieved_value, score). If no match is found, returns (None, 0.0).
        """
        if isinstance(query_key, (list, tuple)) or (isinstance(query_key, torch.Tensor) and query_key.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8)):
            return self.query_discrete(query_key)
        else:
            return self.query_continuous(query_key, similarity_threshold=similarity_threshold)

    def clear(self):
        """
        Wipes the memory store completely.
        """
        self.discrete_store.clear()
        self.continuous_keys.clear()
        self.continuous_values.clear()
        if self.use_rust:
            self.rust_mem.clear()
        
    def __len__(self):
        if self.use_rust:
            return self.rust_mem.size()
        return len(self.discrete_store) + len(self.continuous_keys)
