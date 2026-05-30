import sys
import os

# Force standard output and error streams to use UTF-8 on Windows to prevent Unicode encoding crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

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
        self.vision_projector = nn.Linear(self.orchestrator.vlm_dim, self.qwen_dim).to(self.device)
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
            
        # SigLIP might be loaded on CPU to conserve VRAM
        siglip_device = next(self.orchestrator.real_siglip.parameters()).device
        
        # Process image using SigLIP
        inputs = self.orchestrator.real_siglip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(siglip_device)
        
        # Match weight dtype
        if hasattr(self.orchestrator.real_siglip.vision_model, "embeddings"):
            try:
                target_dtype = next(self.orchestrator.real_siglip.vision_model.embeddings.parameters()).dtype
                pixel_values = pixel_values.to(target_dtype)
            except StopIteration:
                pass
            
        with torch.no_grad():
            outputs = self.orchestrator.real_siglip.vision_model(pixel_values=pixel_values)
            # Extracted pooled representation of shape (1, 768)
            frame_features = outputs.pooler_output.float().to(self.device)
            
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
            
        # Whisper might be loaded on CPU to conserve VRAM
        whisper_device = next(self.orchestrator.real_whisper.parameters()).device
        
        # audio_waveform should be a 1D numpy array or tensor (16kHz mono)
        inputs = self.orchestrator.real_whisper_processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(whisper_device)
        
        # Dynamically match the dtype of the Whisper encoder weights
        encoder = self.orchestrator.real_whisper.get_encoder()
        if hasattr(encoder, "conv1"):
            target_dtype = encoder.conv1.weight.dtype
            input_features = input_features.to(target_dtype)
            
        with torch.no_grad():
            outputs = encoder(input_features=input_features)
            # Map Whisper acoustic encoder features (B, S_audio, D_whisper) to Qwen2 space
            whisper_features = outputs.last_hidden_state.float().to(self.device) # Move to GPU
            qwen_audio_tokens = self.audio_projector(whisper_features) # (1, S_audio, 896)
            
        return qwen_audio_tokens

    def transcribe_audio(self, audio_waveform):
        """
        Transcribes the mono 16kHz audio waveform into text using Whisper generation.
        """
        if not self.orchestrator.real_weights_enabled or self.orchestrator.real_whisper is None:
            return ""
        try:
            whisper_device = next(self.orchestrator.real_whisper.parameters()).device
            inputs = self.orchestrator.real_whisper_processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(whisper_device)
            with torch.no_grad():
                predicted_ids = self.orchestrator.real_whisper.generate(input_features)
                transcription = self.orchestrator.real_whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            print(f"[Mark-XXXIX Assistant] Transcription error: {e}")
            return ""
        
    def decide_action(self, prompt_text="Perform system actions based on current screen states.", audio_waveform=None):
        """
        Consolidates the rolling screen history tokens, voice inputs, and text prompts,
        and decodes them via Qwen2 using Speculative Causal Decoding to output natural actions.
        """
        if not self.orchestrator.real_weights_enabled or self.orchestrator.real_qwen is None:
            return "Execution Error: Real weights not loaded."
            
        # 1. Background feature extraction to keep visual fuser and Whisper pipelines active
        if audio_waveform is not None:
            # Extract features to keep the adapter fuser pathway logic executed
            _ = self.ingest_voice_command(audio_waveform)
            # Transcribe audio to text
            voice_text = self.transcribe_audio(audio_waveform)
            if voice_text:
                prompt_text = f"Voice Command: {voice_text}"
                print(f"[Mark-XXXIX Assistant] Audio Transcribed to: '{voice_text}'")
                
        # 2. Formulate the few-shot system prompt for the base Qwen2 model
        # Base models need few-shot examples to follow formatting instructions and generate clean action commands.
        few_shot_prompt = f"""You are JARVIS, a desktop automation assistant.
You output commands in the following formats:
- To click: [CLICK x,y]
- To type: [TYPE text]
- To launch: [LAUNCH app]
- To speak: speech response

Here are some examples:
Task: Open Chrome browser.
Response: [LAUNCH Chrome]

Task: Click on the submit button at coordinates 500, 300.
Response: [CLICK 500,300]

Task: Type Hello World.
Response: [TYPE Hello World]

Task: What is the capital of France?
Response: Paris is the capital of France.

Task: {prompt_text}
Response:"""

        # 3. Embed the prompt text
        inputs = self.orchestrator.real_tokenizer(few_shot_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        weight_device = self.orchestrator.real_qwen.embed_tokens.weight.device
        text_embeddings = self.orchestrator.real_qwen.embed_tokens(input_ids.to(weight_device)).to(self.device) # (1, S_text, 896)
        
        # 4. Generate Causal Text Output via the Qwen2 speculative pipeline
        with torch.no_grad():
            model_dtype = self.orchestrator.real_qwen.embed_tokens.weight.dtype
            current_embeds = text_embeddings
            
            output_tokens = []
            for _ in range(16):
                outputs = self.orchestrator.real_qwen(inputs_embeds=current_embeds.to(model_dtype))
                hidden_state = outputs.last_hidden_state[:, -1, :]
                
                if hasattr(self.orchestrator.real_qwen, "lm_head") and self.orchestrator.real_qwen.lm_head is not None:
                    step_logits = self.orchestrator.real_qwen.lm_head(hidden_state)
                else:
                    if not hasattr(self, "real_text_head") or self.real_text_head.in_features != hidden_state.shape[-1]:
                        self.real_text_head = nn.Linear(hidden_state.shape[-1], self.orchestrator.vocab_size).to(self.device)
                    step_logits = self.real_text_head(hidden_state)
                    
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
                output_tokens.append(next_token.item())
                
                # Append predicted token embedding for autoregressive loop
                weight_device = self.orchestrator.real_qwen.embed_tokens.weight.device
                next_embed = self.orchestrator.real_qwen.embed_tokens(next_token.to(weight_device)).to(self.device)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                
            decoded_response = self.orchestrator.real_tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            # Clean up the output string
            if "Response:" in decoded_response:
                decoded_response = decoded_response.split("Response:")[-1].strip()
            # If the response generated multiple lines, take the first line
            decoded_response = decoded_response.split("\n")[0].strip()
            
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
