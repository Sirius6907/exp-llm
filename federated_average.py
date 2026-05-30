import os
import sys
import torch
import torch.nn as nn

def federated_average_checkpoints(checkpoint_paths, output_path="mcp_aligned_federated.pth"):
  """
  Performs Federated Averaging (FedAvg) over multiple state dictionaries
  trained independently on separate Lightning.ai free accounts to consolidate
  learned weights without loss of numerical stability.
  """
  if not checkpoint_paths:
    print("[Error] No checkpoint paths provided for averaging.")
    return False
    
  print(f"\n[FedAvg] Initiating parameter averaging over {len(checkpoint_paths)} checkpoints...")
  
  # Load first checkpoint to initialize state
  first_path = checkpoint_paths[0]
  if not os.path.exists(first_path):
    print(f"[Error] Checkpoint not found: {first_path}")
    return False
    
  base_state = torch.load(first_path, map_location="cpu")
  
  # Check if it's a standard dictionary or wrapped orchestrator state
  is_dict_wrapped = "model_state" in base_state
  state_to_average = base_state["model_state"] if is_dict_wrapped else base_state
  
  # Initialize accumulator for float tensors
  accumulated_state = {k: v.clone().float() for k, v in state_to_average.items() if isinstance(v, torch.Tensor)}
  
  # Accumulate from remaining checkpoints
  for path in checkpoint_paths[1:]:
    if not os.path.exists(path):
      print(f"[Warning] Skipping missing checkpoint: {path}")
      continue
    print(f"  -> Merging gradients and parameters from: {path}")
    current_checkpoint = torch.load(path, map_location="cpu")
    current_state = current_checkpoint["model_state"] if is_dict_wrapped else current_checkpoint
    
    for k, v in current_state.items():
      if k in accumulated_state and isinstance(v, torch.Tensor):
        accumulated_state[k] += v.float()
        
  # Divide by total checkpoints to get the mathematical average
  num_checkpoints = len(checkpoint_paths)
  for k in accumulated_state.keys():
    accumulated_state[k] /= num_checkpoints
    # Cast back to half-precision float16 if original weights were in half precision
    original_dtype = state_to_average[k].dtype
    accumulated_state[k] = accumulated_state[k].to(original_dtype)
    
  # Save the consolidated federated checkpoint
  if is_dict_wrapped:
    base_state["model_state"] = accumulated_state
    torch.save(base_state, output_path)
  else:
    torch.save(accumulated_state, output_path)
    
  print(f"[FedAvg SUCCESS] Consolidated federated weights saved to: {output_path}")
  return True

def assemble_modular_experts(text_path=None, vision_path=None, audio_path=None, mcp_path=None, output_path="pixelle_sirius_assembled.pth"):
  """
  Assembles modular sub-expert weights (Text, Vision, Audio, and Ternary MCP)
  trained independently on separate Lightning.ai free accounts into a single,
  unified high-capacity orchestrator.
  """
  print("\n[Modular Assembly] Assembling expert checkpoints into unified engine...")
  assembled_state = {}
  
  expert_mappings = [
      ("Text Expert / Speculative Causal Model", text_path, ["draft_vlm", "experts.0", "text_head"]),
      ("Vision Expert / LCM Image-Video Solver", vision_path, ["lcm_solver", "image_encoder", "image_decoder", "temporal_fuser"]),
      ("Audio Expert / Whisper Projection Solver", audio_path, ["audio_encoder", "audio_decoder"]),
      ("Ternary MCP Alignment Adapter", mcp_path, ["mcp", "real_mcp"])
  ]
  
  for description, path, prefix_list in expert_mappings:
    if not path or not os.path.exists(path):
      print(f"  -> [Skipped] {description} weights not supplied or missing. Using initial defaults.")
      continue
      
    print(f"  -> [Injecting] {description} from: {path}")
    state = torch.load(path, map_location="cpu")
    actual_state = state["model_state"] if "model_state" in state else state
    
    # Filter state parameters matching target sub-components
    for k, v in actual_state.items():
      if any(k.startswith(prefix) for prefix in prefix_list):
        assembled_state[k] = v
        
  if not assembled_state:
    print("[Error] No expert parameters were assembled. Exiting.")
    return False
    
  torch.save(assembled_state, output_path)
  print(f"[Assembly SUCCESS] Assembled unified model weights saved to: {output_path}")
  return True

def main():
  print("==================================================")
  print("    Pixelle-Sirius: Decentralized Training Fuser  ")
  print("==================================================")
  
  if len(sys.argv) < 2:
    print("Usage Options:")
    print("  1. Federated Averaging (FedAvg) over multiple checkpoints:")
    print("     python federated_average.py average checkpoint1.pth checkpoint2.pth ...")
    print("  2. Assemble Modular Experts trained on separate accounts:")
    print("     python federated_average.py assemble --text text.pth --vision vision.pth --mcp mcp.pth")
    return
    
  action = sys.argv[1].lower()
  
  if action == "average":
    checkpoint_paths = sys.argv[2:]
    federated_average_checkpoints(checkpoint_paths)
    
  elif action == "assemble":
    text_path = None
    vision_path = None
    audio_path = None
    mcp_path = None
    
    # Simple argument parser
    args = sys.argv[2:]
    for i in range(len(args)):
      if args[i] == "--text" and i+1 < len(args):
        text_path = args[i+1]
      elif args[i] == "--vision" and i+1 < len(args):
        vision_path = args[i+1]
      elif args[i] == "--audio" and i+1 < len(args):
        audio_path = args[i+1]
      elif args[i] == "--mcp" and i+1 < len(args):
        mcp_path = args[i+1]
        
    assemble_modular_experts(text_path, vision_path, audio_path, mcp_path)
  else:
    print(f"[Error] Unknown action: {action}")

if __name__ == "__main__":
  main()
