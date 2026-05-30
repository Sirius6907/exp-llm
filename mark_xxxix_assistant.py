import os
import time
import torch
import torch.nn as nn
from PIL import Image, ImageGrab
from pixelle_sirius_engine import PixelleSiriusOrchestrator

class MarkXXXIXAssistantBrain(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Mark-XXXIX Assistant] Initializing offline brain on {self.device}...")
        
        # Initialize the Pixelle-Sirius Orchestrator
        self.orchestrator = PixelleSiriusOrchestrator(
            codebook_size=2048,
            vlm_dim=2048,
            dit_dim=1024,
            latent_dim=256,
            vocab_size=32000,
            device=self.device
        )
        self.orchestrator.eval()
        
        # Load the pre-trained models in 4-bit NF4 quantized mode (MAX GPU)
        print("[Mark-XXXIX Assistant] Ingesting backbones in 4-bit NF4 precision...")
        self.orchestrator.load_real_backbones(offload=False, load_in_4bit=True)
        
        # Screen visual state memory
        self.ssm_state = None
        self.screen_tokens_buffer = []
        self.max_screen_history = 5
        
        # Cross-modal projection layers to map pre-trained visual/acoustic features to Qwen2 space
        # Qwen2-0.5B hidden size is 896
        self.qwen_dim = 896
        self.vision_projector = nn.Linear(768, self.qwen_dim).to(self.device)
        self.audio_projector = nn.Linear(384, self.qwen_dim).to(self.device)
        
    def capture_screen(self):
        """
        Grabs the active monitor screen using PIL ImageGrab.
        Provides a synthetic fallback image if running headlessly or in a test environment without GUI.
        """
        try:
            # Attempt to grab active desktop screen
            screenshot = ImageGrab.grab()
        except Exception as e:
            # Fallback to generating a synthetic screen state for testing robustness
            print(f"[Mark-XXXIX Assistant] Screen grab fallback active: {e}")
            screenshot = Image.new('RGB', (1024, 768), color=(20, 25, 30))
            
        # Resize to SigLIP vision encoder standard size
        return screenshot.resize((224, 224))
        
    def ingest_screenshot(self, image):
        """
        Processes a screenshot image through the SigLIP vision encoder, projects the 
        embeddings, and passes them through the Mamba visual temporal fuser.
        """
        if not self.orchestrator.real_weights_enabled or self.orchestrator.real_siglip is None:
            return None
            
        # Process image using SigLIP
        inputs = self.orchestrator.real_siglip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if pixel_values.dtype != torch.float16 and self.device.type == "cuda":
            pixel_values = pixel_values.to(torch.float16)
            
        with torch.no_grad():
            outputs = self.orchestrator.real_siglip.vision_model(pixel_values=pixel_values)
            # Extracted pooled representation of shape (1, 768)
            frame_features = outputs.pooler_output.float()
            
            # Map vision tokens to orchestrator VLM dimension (2048) for SSM fusion
            if not hasattr(self, "mcp_vision_fuser"):
                self.mcp_vision_fuser = nn.Linear(frame_features.shape[-1], self.orchestrator.vlm_dim).to(self.device)
            fuser_input = self.mcp_vision_fuser(frame_features).unsqueeze(1) # (1, 1, vlm_dim)
            
            # Run Mamba SSM temporal fuser to compress visual sequence memory
            fused_token, self.ssm_state = self.orchestrator.draft_vlm(fuser_input, state=self.ssm_state)
            
            # Project fused token to Qwen2 hidden dimension (896)
            qwen_vision_token = self.vision_projector(fused_token.squeeze(1)) # (1, 896)
            
        # Append token to rolling screen state history
        self.screen_tokens_buffer.append(qwen_vision_token)
        if len(self.screen_tokens_buffer) > self.max_screen_history:
            self.screen_tokens_buffer.pop(0)
            
        return qwen_vision_token
        
    def ingest_voice_command(self, audio_waveform):
        """
        Ingests user voice inputs from the microphone, runs them through the Whisper
        encoder, and projects them directly to the Qwen2 semantic space.
        """
        if not self.orchestrator.real_weights_enabled or self.orchestrator.real_whisper is None:
            return None
            
        # audio_waveform should be a 1D numpy array or tensor (16kHz mono)
        inputs = self.orchestrator.real_whisper_processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        if input_features.dtype != torch.float16 and self.device.type == "cuda":
            input_features = input_features.to(torch.float16)
            
        with torch.no_grad():
            outputs = self.orchestrator.real_whisper.encoder(input_features=input_features)
            # Map Whisper acoustic encoder features (B, S_audio, D_whisper) to Qwen2 space
            whisper_features = outputs.last_hidden_state.float() # (1, S_audio, 384)
            qwen_audio_tokens = self.audio_projector(whisper_features) # (1, S_audio, 896)
            
        return qwen_audio_tokens
        
    def decide_action(self, prompt_text="Perform system actions based on current screen states.", audio_waveform=None):
        """
        Consolidates the rolling screen history tokens, voice inputs, and text prompts,
        and decodes them via Qwen2 using Speculative Causal Decoding to output natural actions.
        """
        if not self.orchestrator.real_weights_enabled or self.orchestrator.real_qwen is None:
            return "Execution Error: Real weights not loaded."
            
        # 1. Gather all visual history embeddings
        embeddings_list = []
        if self.screen_tokens_buffer:
            # Concatenate all screen tokens in history: (1, num_frames, 896)
            screen_history = torch.stack(self.screen_tokens_buffer, dim=1).squeeze(0) # (num_frames, 896)
            embeddings_list.append(screen_history.unsqueeze(0))
            
        # 2. Gather voice embedding if provided
        if audio_waveform is not None:
            audio_tokens = self.ingest_voice_command(audio_waveform)
            if audio_tokens is not None:
                embeddings_list.append(audio_tokens)
                
        # 3. Embed the prompt text
        inputs = self.orchestrator.real_tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        text_embeddings = self.orchestrator.real_qwen.embed_tokens(input_ids) # (1, S_text, 896)
        embeddings_list.append(text_embeddings)
        
        # Combine all cross-modal context embeddings into a single sequence
        combined_embeddings = torch.cat(embeddings_list, dim=1) # (1, total_tokens, 896)
        
        # 4. Generate Causal Text Output via the Qwen2 speculative pipeline
        # We project the hidden state from Qwen2 outputs through our target text head
        with torch.no_grad():
            out_hf = self.orchestrator.real_qwen(inputs_embeds=combined_embeddings.half() if self.device.type == "cuda" else combined_embeddings)
            logits = out_hf.last_hidden_state[:, -1, :].float()
            
            if not hasattr(self, "real_text_head") or self.real_text_head.in_features != logits.shape[-1]:
                self.real_text_head = nn.Linear(logits.shape[-1], self.orchestrator.vocab_size).to(self.device)
                
            # Perform draft-decoding verification (autoregressive prediction loop for 16 steps)
            output_tokens = []
            current_embeds = combined_embeddings
            
            for _ in range(16):
                outputs = self.orchestrator.real_qwen(inputs_embeds=current_embeds.half() if self.device.type == "cuda" else current_embeds)
                hidden_state = outputs.last_hidden_state[:, -1, :]
                step_logits = self.real_text_head(hidden_state)
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
                output_tokens.append(next_token.item())
                
                # Append predicted token embedding for autoregressive loop
                next_embed = self.orchestrator.real_qwen.embed_tokens(next_token)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                
            decoded_response = self.orchestrator.real_tokenizer.decode(output_tokens, skip_special_tokens=True)
            
        return decoded_response
        
    def execute_action(self, action_str):
        """
        Parses and simulates the execution of generated desktop automation commands.
        """
        action_str = action_str.strip()
        print(f"[Mark-XXXIX Action Executor] Raw Command Received: '{action_str}'")
        
        if "[CLICK" in action_str:
            coords = action_str.split("[CLICK")[1].split("]")[0].strip()
            print(f"  -> EXECUTING: Simulating mouse click at screen coordinate: ({coords})")
            return f"Click simulated at {coords}."
        elif "[TYPE" in action_str:
            text = action_str.split("[TYPE")[1].split("]")[0].strip()
            print(f"  -> EXECUTING: Simulating keyboard input typing: '{text}'")
            return f"Typing simulated: '{text}'."
        elif "[LAUNCH" in action_str:
            app = action_str.split("[LAUNCH")[1].split("]")[0].strip()
            print(f"  -> EXECUTING: Simulating application launch command: '{app}'")
            return f"Launch simulated: {app}."
        else:
            print(f"  -> EXECUTING: Standard system assistant speech response: \"{action_str}\"")
            return f"Speech output: \"{action_str}\""

    def run_loop(self, num_iterations=3, interval=1.0, audio_mock_generator=None):
        """
        Simulates the background execution loop of the assistant.
        Captures active screenshots, ingests them, polls voice inputs, and runs decision paths.
        """
        print(f"\n--- Starting Mark-XXXIX Offline Brain Loop ({num_iterations} iterations, {interval}s interval) ---")
        for i in range(num_iterations):
            print(f"\n[Iteration {i+1}/{num_iterations}] polling desktop input...")
            t0 = time.time()
            
            # A. Ingest Screen State
            screenshot = self.capture_screen()
            vision_token = self.ingest_screenshot(screenshot)
            
            # B. Ingest Voice command (if generator is available)
            waveform = None
            if audio_mock_generator is not None:
                waveform = audio_mock_generator()
                
            # C. Decide Causal Actions
            prompt = "Determine the next optimal desktop automation action."
            action_decided = self.decide_action(prompt_text=prompt, audio_waveform=waveform)
            
            # D. Execute Desktop Actions
            execution_log = self.execute_action(action_decided)
            
            t1 = time.time()
            print(f"[Iteration {i+1} Done] Loop Cycle Latency: {(t1 - t0)*1000:.2f} ms")
            time.sleep(interval)
            
        print("\n--- Mark-XXXIX Loop Simulation Completed ---")
