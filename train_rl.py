import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import orchestrator and custom MCP
from pixelle_sirius_engine import PixelleSiriusOrchestrator
from mcp import MobileConditioningProjector

class VisualPreferenceDataset(Dataset):
  """
  A dataset representing human preference pairs for cross-modal generation.
  Each sample contains a visual text prompt, a preferred visual embedding target (winner),
  and a dispreferred visual embedding target (loser).
  """
  def __init__(self, num_samples=256, dit_dim=1024):
    self.num_samples = num_samples
    self.dit_dim = dit_dim
    self.prompts = [
        "A hyper-realistic cinematic render of an ancient temple, golden hour",
        "Cyberpunk street with glowing neon signs and heavy rain reflections",
        "A majestic glowing phoenix rising from dark volcanic ash",
        "Cozy wooden cabin in a snowy forest under the aurora borealis",
        "Vibrant coral reef teeming with marine life, shafts of sunlight",
        "A sleek futuristic sports car speeding through a desert canyon",
        "Deep space nebula with swirling cosmic dust and distant galaxies",
        "An elegant white stallion running wild along a pristine beach at sunset"
    ]

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    prompt = self.prompts[idx % len(self.prompts)]
    # We simulate a semantic length of 16 tokens for visual alignment
    L_seq = 16
    
    # Deterministic targets based on idx to make loss tracking stable and comparable
    generator = torch.Generator()
    generator.manual_seed(idx)
    
    # Preferred target represents high aesthetic fidelity
    preferred_target = torch.randn(L_seq, self.dit_dim, generator=generator) + 0.5
    # Dispreferred target represents low resolution or artifacts
    dispreferred_target = torch.randn(L_seq, self.dit_dim, generator=generator) - 0.5
    
    return {
        "prompt": prompt,
        "preferred": preferred_target,
        "dispreferred": dispreferred_target
    }

def train_rl_mcp_alignment():
  print("==================================================")
  print("     Pixelle-Sirius: Ternary MCP RL Alignment      ")
  print("==================================================")
  print("Objective: Align the 1.58-bit Ternary MCP adapter using Direct")
  print("Preference Optimization (DPO) and Policy Gradient rewards under 3GB.")
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Target GPU/CPU Execution Device: {device}")
  
  # 1. Initialize Unified Orchestrator and Ingest Real Backbones
  orchestrator = PixelleSiriusOrchestrator(
      codebook_size=2048,
      vlm_dim=2048,
      dit_dim=1024,
      latent_dim=256,
      vocab_size=32000,
      device=device
  )
  orchestrator.eval()
  
  # Ingest backbones permanently onto active GPU (MAX GPU mode)
  print("\nLoading pre-trained causal backbones...")
  orchestrator.load_real_backbones(offload=False)
  
  if not orchestrator.real_weights_enabled or orchestrator.real_qwen is None:
    print("❌ Pre-trained weights could not be initialized. Exiting.")
    return
    
  # Freeze Qwen2 backbone completely
  for param in orchestrator.real_qwen.parameters():
    param.requires_grad = False
    
  vlm_dim = 896
  dit_dim = 1024
  num_layers = 4
  
  # 2. Instantiate Policy and Reference Projectors
  # DPO requires a Policy (active) and Reference (frozen) model to calculate relative log probs
  print("\nInitializing Policy and Reference 1.58-bit Ternary MCPs...")
  policy_mcp = MobileConditioningProjector(
      vlm_dim=vlm_dim,
      dit_dim=dit_dim,
      num_layers=num_layers
  ).to(device)
  
  reference_mcp = MobileConditioningProjector(
      vlm_dim=vlm_dim,
      dit_dim=dit_dim,
      num_layers=num_layers
  ).to(device)
  
  # Load same starting checkpoint to both policy and reference
  # If we have a tuned SFT adapter, we resume from it!
  sft_checkpoint = "mcp_alignment_real.pth"
  if os.path.exists(sft_checkpoint):
    print(f"  -> Loading starting weights from SFT adapter: {sft_checkpoint}")
    policy_mcp.load_state_dict(torch.load(sft_checkpoint, map_location=device))
    reference_mcp.load_state_dict(torch.load(sft_checkpoint, map_location=device))
  else:
    print("  -> No SFT adapter checkpoint found. Starting from random initialization.")
    
  policy_mcp.train()
  reference_mcp.eval()
  
  # Freeze reference model weights completely to avoid overhead
  for param in reference_mcp.parameters():
    param.requires_grad = False
    
  # 3. Setup Dataset & Dataloaders (80% / 20% split)
  full_dataset = VisualPreferenceDataset(num_samples=128, dit_dim=dit_dim)
  
  split_idx = int(len(full_dataset) * 0.8)
  
  train_dataset = VisualPreferenceDataset(num_samples=split_idx, dit_dim=dit_dim)
  train_dataset.prompts = full_dataset.prompts
  
  val_dataset = VisualPreferenceDataset(num_samples=len(full_dataset) - split_idx, dit_dim=dit_dim)
  val_dataset.prompts = full_dataset.prompts
  
  train_dataloader = DataLoader(
      train_dataset, 
      batch_size=8, 
      shuffle=True, 
      num_workers=2, 
      pin_memory=True,
      persistent_workers=True
  )
  
  val_dataloader = DataLoader(
      val_dataset,
      batch_size=8,
      shuffle=False,
      num_workers=1,
      pin_memory=True,
      persistent_workers=True
  )
  
  # 4. Setup Optimizer & DPO hyper-parameters
  optimizer = optim.AdamW(policy_mcp.parameters(), lr=1e-5, weight_decay=1e-2)
  scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
  
  beta_dpo = 0.1 # DPO implicit reward scaling factor
  best_val_loss = float('inf')
  save_path = "mcp_rl_aligned.pth"
  
  print("\n==================================================")
  print("         Starting RL Preference Tuning Loop       ")
  print("==================================================")
  
  t_start = time.time()
  epochs = 2
  
  for epoch in range(epochs):
    epoch_loss = 0.0
    step_times = []
    
    policy_mcp.train()
    for batch_idx, batch_data in enumerate(train_dataloader):
      t0 = time.time()
      optimizer.zero_grad()
      
      prompts = batch_data["prompt"]
      pref_targets = batch_data["preferred"].to(device, non_blocking=True) # (B, L, dit_dim)
      dispref_targets = batch_data["dispreferred"].to(device, non_blocking=True) # (B, L, dit_dim)
      
      # Tokenize real text prompts
      inputs = orchestrator.real_tokenizer(prompts, padding=True, return_tensors="pt")
      input_ids = inputs["input_ids"].to(device)
      B, S = input_ids.shape
      
      # Extract frozen hidden states from Qwen2 (Teacher)
      with torch.no_grad():
        out_hf = orchestrator.real_qwen(input_ids=input_ids, output_hidden_states=True)
        # Cast hidden states to float32 to prevent half-precision mismatch
        hidden_states = [h.float() for h in out_hf.hidden_states[-num_layers:]]
        
      # Run forward pass through policy and reference models
      with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
        # 1. Active Policy predictions
        policy_c = policy_mcp(hidden_states) # (B, L_out, dit_dim)
        
        # 2. Frozen Reference predictions
        with torch.no_grad():
          ref_c = reference_mcp(hidden_states) # (B, L_out, dit_dim)
          
        # Ensure outputs match target sequences lengths exactly
        min_L = min(policy_c.shape[1], pref_targets.shape[1])
        y_policy = policy_c[:, :min_L].contiguous()
        y_ref = ref_c[:, :min_L].contiguous()
        y_pref_tgt = pref_targets[:, :min_L].contiguous()
        y_dispref_tgt = dispref_targets[:, :min_L].contiguous()
        
        # 3. Calculate DPO-style alignment log-likelihoods / negative distances
        log_prob_policy_pref = -torch.mean((y_policy - y_pref_tgt) ** 2, dim=[1, 2])
        log_prob_policy_dispref = -torch.mean((y_policy - y_dispref_tgt) ** 2, dim=[1, 2])
        
        log_prob_ref_pref = -torch.mean((y_ref - y_pref_tgt) ** 2, dim=[1, 2])
        log_prob_ref_dispref = -torch.mean((y_ref - y_dispref_tgt) ** 2, dim=[1, 2])
        
        # DPO objective formulation
        policy_ratio = log_prob_policy_pref - log_prob_policy_dispref
        ref_ratio = log_prob_ref_pref - log_prob_ref_dispref
        
        dpo_logits = beta_dpo * (policy_ratio - ref_ratio)
        loss = -F.logsigmoid(dpo_logits).mean()
        
      # Backpropagate through active policy weights exclusively
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(policy_mcp.parameters(), max_norm=1.0)
      
      scaler.step(optimizer)
      scaler.update()
      
      t1 = time.time()
      step_times.append((t1 - t0) * 1000)
      epoch_loss += loss.item()
      
      print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1}/{len(train_dataloader)} | RL Loss: {loss.item():.4f} | Step Time: {step_times[-1]:.2f} ms")
      
    avg_loss = epoch_loss / len(train_dataloader)
    avg_step = sum(step_times) / len(step_times)
    
    # 5. Validation Phase (Validation Set / Dev Loss)
    policy_mcp.eval()
    val_loss = 0.0
    with torch.no_grad():
      for val_batch_data in val_dataloader:
        prompts = val_batch_data["prompt"]
        pref_targets = val_batch_data["preferred"].to(device, non_blocking=True)
        dispref_targets = val_batch_data["dispreferred"].to(device, non_blocking=True)
        
        inputs = orchestrator.real_tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        out_hf = orchestrator.real_qwen(input_ids=input_ids, output_hidden_states=True)
        hidden_states = [h.float() for h in out_hf.hidden_states[-num_layers:]]
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
          policy_c = policy_mcp(hidden_states)
          ref_c = reference_mcp(hidden_states)
          
          min_L = min(policy_c.shape[1], pref_targets.shape[1])
          y_policy = policy_c[:, :min_L].contiguous()
          y_ref = ref_c[:, :min_L].contiguous()
          y_pref_tgt = pref_targets[:, :min_L].contiguous()
          y_dispref_tgt = dispref_targets[:, :min_L].contiguous()
          
          log_prob_policy_pref = -torch.mean((y_policy - y_pref_tgt) ** 2, dim=[1, 2])
          log_prob_policy_dispref = -torch.mean((y_policy - y_dispref_tgt) ** 2, dim=[1, 2])
          
          log_prob_ref_pref = -torch.mean((y_ref - y_pref_tgt) ** 2, dim=[1, 2])
          log_prob_ref_dispref = -torch.mean((y_ref - y_dispref_tgt) ** 2, dim=[1, 2])
          
          policy_ratio = log_prob_policy_pref - log_prob_policy_dispref
          ref_ratio = log_prob_ref_pref - log_prob_ref_dispref
          
          dpo_logits = beta_dpo * (policy_ratio - ref_ratio)
          loss = -F.logsigmoid(dpo_logits).mean()
          val_loss += loss.item()
          
    avg_val_loss = val_loss / len(val_dataloader)
    
    print(f"--------------------------------------------------")
    print(f"Epoch {epoch+1:02d} RL Summary | Train RL Loss: {avg_loss:.4f} | Val (Dev) RL Loss: {avg_val_loss:.4f} | Avg Step Time: {avg_step:.2f} ms")
    print(f"--------------------------------------------------")
    
    # Save only the checkpoint with the lowest validation loss (peak intelligence sweet spot)
    if avg_val_loss < best_val_loss:
      print(f"  -> [New Best] Validation Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving checkpoint to {save_path}...\n")
      best_val_loss = avg_val_loss
      torch.save(policy_mcp.state_dict(), save_path)
    else:
      print(f"  -> [Warning] Validation Loss did not improve (Current: {avg_val_loss:.4f}, Best: {best_val_loss:.4f}). Preserving peak intelligence checkpoint.\n")
      
  t_end = time.time()
  print("==================================================")
  print(f"RL preference alignment completed in {t_end - t_start:.2f} seconds.")
  print(f"Peak intelligence model preserved with validation loss: {best_val_loss:.4f}")
  print("==================================================")

if __name__ == "__main__":
  train_rl_mcp_alignment()
