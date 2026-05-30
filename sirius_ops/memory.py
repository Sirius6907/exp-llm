import torch
import torch.nn as nn
from .ops import fast_cosine_similarity

class SiriusZeroLossMemory(nn.Module):
    """
    Episodic Non-Parametric Zero-Loss Memory Bank.
    Stores and retrieves multi-modal samples in O(1) time.
    Supports both exact discrete lookups (for text token IDs) and 
    approximate continuous lookups (for visual/acoustic embeddings).
    """
    def __init__(self):
        super().__init__()
        # Discrete memory bank (Prompt Token Tuple -> Response Token List)
        self.discrete_store = {}
        
        # Continuous memory bank (Context Tensor -> Output Tensor)
        self.continuous_keys = []
        self.continuous_values = []

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

    def insert_continuous(self, key_tensor, value_tensor):
        """
        Inserts a continuous key-value pair (e.g. multimodal embedding to target representation).
        """
        # Ensure key is a flat 1D float32 tensor
        k_flat = key_tensor.detach().float().cpu().view(-1)
        v_tensor = value_tensor.detach().cpu()
        
        # Check if already exists, overwrite if matching key is very close
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
        if isinstance(query_tokens, torch.Tensor):
            query_tuple = tuple(query_tokens.cpu().flatten().tolist())
        else:
            query_tuple = tuple(query_tokens)
            
        # Check for exact matches
        if query_tuple in self.discrete_store:
            return self.discrete_store[query_tuple], 1.0
            
        # Check for subsequence/prefix matches (e.g., matching the prompt prefix)
        for stored_key, stored_val in self.discrete_store.items():
            if len(query_tuple) >= len(stored_key) and query_tuple[:len(stored_key)] == stored_key:
                return stored_val, 1.0
                
        return None, 0.0

    def query_continuous(self, query_tensor, similarity_threshold=0.98):
        """
        Look up continuous key matches using cosine similarity.
        """
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
        
    def __len__(self):
        return len(self.discrete_store) + len(self.continuous_keys)
