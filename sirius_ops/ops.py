import torch

def fast_cosine_similarity(a, b):
    """
    Computes the cosine similarity between two 1D PyTorch tensors.
    Truncates to the shorter vector's length to avoid silent dimension mismatch.
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)
        
    a = a.view(-1)
    b = b.view(-1)
    
    # Truncate to common length to avoid zero-padding distortion
    if a.shape[0] != b.shape[0]:
        min_len = min(a.shape[0], b.shape[0])
        a = a[:min_len]
        b = b[:min_len]
        
    dot_product = torch.dot(a, b)
    norm_a = torch.linalg.norm(a)
    norm_b = torch.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

def fast_quantized_lookup(embeddings, codebook):
    """
    Performs rapid, low-compute nearest-neighbor search to map embeddings to codebook vectors.
    Computes distances in a vectorized sweep to reduce loop overhead.
    """
    # embeddings: (B, S, D) or (N, D)
    # codebook: (Codebook_Size, D)
    orig_shape = embeddings.shape
    flat_emb = embeddings.view(-1, orig_shape[-1]) # (N_tokens, D)
    
    # Vectorized L2 squared distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x . y^T
    x_squared = torch.sum(flat_emb ** 2, dim=1, keepdim=True) # (N_tokens, 1)
    y_squared = torch.sum(codebook ** 2, dim=1, keepdim=True).t() # (1, Codebook_Size)
    xy_dot = torch.matmul(flat_emb, codebook.t()) # (N_tokens, Codebook_Size)
    
    distances = x_squared + y_squared - 2 * xy_dot
    indices = torch.argmin(distances, dim=-1) # (N_tokens,)
    
    return indices.view(orig_shape[:-1])
