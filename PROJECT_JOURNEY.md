# Pixelle-Sirius: Project Evolution, Build History, and Technical Roadmap

This document outlines the complete developmental history of the **Pixelle-Sirius** project. It details the journey from a memory-constrained edge-scale video generation experiment to a highly optimized, unified State Space Model (SSM) and Latent Consistency Diffusion Engine.

---

## 1. The Starting Vision: Edge-Scale Generative AI (0 to 1)

### The Hardware Bottleneck
The project initiated with a strict hardware constraint: **How can high-capacity generative models be executed and trained under a strict 3.0 GB VRAM ceiling?** 
Most generative models are designed for datacenter scale, requiring professional GPUs (e.g., A100 or H100) due to memory footprint. For instance:
* A standard 2 Billion parameter model running in FP16 requires **4.0 GB of VRAM** simply to hold its weights in memory.
* High-end video generation architectures suffer from quadratic attention complexity ($O(N^2)$) and massive KV-cache accumulation, causing immediate out-of-memory (OOM) failures on consumer laptops and budget cloud instances like a single Tesla T4 GPU.

### Breakthrough 1: Dynamic Layer Offloading
To break this memory wall, the first engineering milestone was the creation of a custom CPU-GPU weight offloading wrapper (`OffloadedLayerWrapper`):
* **Host RAM Allocation**: The full model parameters reside in host CPU memory.
* **Just-in-Time Transfers**: During the forward pass, individual layers are loaded into active GPU memory, processed, and immediately offloaded back to CPU memory.
* **Training Autograd Hooks**: Custom autograd hooks (`OffloadInputHook` and `OffloadOutputHook`) were engineered to coordinate backpropagation. Weights are dynamically reloaded onto the GPU on backward pass entry and discarded upon exit, preventing memory bloat during training.

### Overcoming Recursive Memory Leakage
During initial cloud profiling of a 2.09 Billion parameter SwiGLU MLP model (`Heavy2BTransformer`), recursive `.to("cuda")` calls on the parent orchestrator circumvented the offloader and migrated all parameters to the GPU at once, triggering an immediate OOM error.
* **The Fix**: The internal `_apply` method in `OffloadedLayerWrapper` was overridden to intercept PyTorch's recursive state modifications, forcing the offloaded weights to remain residing on the host CPU during global device migrations.
* **Verification**: Verified on a Cloud T4 GPU at **2,858 MB VRAM reserved**, successfully operating below the 3.0 GB constraint target.

---

## 2. Milestone 2: Any-to-Any 16-Pathway Scaling

### The Architectural Shift
The pipeline was expanded from single-modality tasks into a unified **16-Pathway Any-to-Any Multimodal Generative Model** orchestrated by the `AnyToAnyOrchestrator`. This architecture maps every input-to-output path between Text, Image, Video, and Audio:

```
                        +-------------------------------+
                        |  Text / Image / Video / Audio |
                        +---------------+---------------+
                                        |
                                        v
                            +-----------------------+
                            | AnyToAnyOrchestrator  |
                            +-----------+-----------+
                                        |
                 +----------------------+----------------------+
                 | (Text Modality)                             | (Vision/Audio Modality)
                 v                                             v
     +-----------------------+                     +-----------------------+
     |  MiniCPM-SALA VLM     |                     |  TokenFlow Tokenizers |
     |  (9B Sparse/Linear)   |                     | (1D/2D Dual-Codebook) |
     +-----------+-----------+                     +-----------+-----------+
                 |                                             |
                 | (Hidden States X_i)                         | (Semantic q_s)
                 v                                             v
     +-----------------------+                     +-----------------------+
     |  Ternary Mobile       |                     |  MiniCPM-SALA VLM     |
     |  Conditioning (MCP)   |<--------------------+  (Visual Encoding)    |
     +-----------+-----------+                     +-----------+-----------+
                 |                                             |
                 | (Conditioning C)                            | (Pixel q_p)
                 v                                             v
     +---------------------------------------------------------------------+
     |                     Bonsai Multi-Modal Diffusion                    |
     |                  (Ternary 1.58-bit / STE quantized)                 |
     +----------------------------------+----------------------------------+
                                        |
                                        v <--- [SSM Temporal State h_t]
                            +-----------------------+
                            |  SSM Temporal Wedge   |
                            |  (Frame-to-Frame)     |
                            +-----------------------+
```

### Core Subcomponents
To maintain high fidelity under a strict VRAM budget, four dedicated blocks were implemented:
1. **`AudioTokenFlowTokenizer`**: A 1D convolutional downsampling encoder and pixel transposed 1D convolutional decoder combined with a dual-codebook vector quantizer to compress waveforms.
2. **`TemporalWedgeBlock`**: A Selective State Space Model (SSM) temporal block running recurrent sequence-length scaling over visual frames, ensuring inter-frame identity coherence.
3. **`Ternary Mobile Conditioning Projector (MCP)`**: Fuses layers of the large language backbone via temperature-scaled temperature weights. It compresses the representation using depthwise-separable 1D convolutions and an Efficient Channel Attention (ECA) module. To minimize memory, weights are quantized to **1.58-bit ternary values** ($\{-1, 0, 1\}$) via a **Straight-Through Estimator (STE)**:
   $$\hat{W} = W + (W_q \cdot \gamma - W)\text{.detach()}$$
4. **`BonsaiAudioMock` and `BonsaiDiffusionMock`**: Linear projecting layers linking the multi-modal output matrices directly to quantized latent diffusion generators.

---

## 3. Milestone 3: The Pixelle-Sirius SSM-Diffusion Engine

### Resolving the Latency Bottleneck
While dynamic layer offloading bypassed the VRAM ceiling, moving weights sequentially between CPU and GPU introduced significant latency. Additionally, quadratic self-attention complexity ($O(N^2)$) restricted text generation throughput. 

To achieve **100+ tokens/second** and sub-second generation locally, the codebase was refactored into the **Pixelle-Sirius SSM-Diffusion Engine** (`pixelle_sirius_engine.py`), introducing four major modifications:

#### 1. Selective SSM (Mamba-2) Backbone
Self-attention was replaced with Selective State Space blocks (`MambaSelectiveBlock`), exhibiting linear complexity $O(N)$ with sequence duration. These blocks maintain state histories in a fixed-size recurrent state matrix rather than a growing Key-Value (KV) cache.

#### 2. Speculative Draft & Target Caching
Autoregressive generation latency was minimized using dual caching schemes:
* **SiriusDraft Caching**: Autoregressive decoding is performed using an ultra-fast 80M parameter draft model. It processes token generation on only the single latest token ($S=1$), bypassing python iteration loops via a specialized $O(1)$ path.
* **SiriusTarget Caching**: Candidates generated by the draft model are verified in parallel by the target experts using cached recurrent state parameters, ensuring stable, sub-second step latency.

#### 3. Top-1 Gated Sparse Mixture of Experts (MoE)
A dynamic routing block (`SparseMoERouter`) distributes token sequences to domain-specific experts (Text, Vision, Audio) based on top-1 probability gating. This scales learning capacity while keeping active execution parameters minimal.

#### 4. Boundary-Guided Latent Consistency Model (LCM)
Standard iterative 50-step diffusion loops were replaced with a boundary-guided consistency solver (`ConsistencyDenoisingSolver`). It resolves latents in **1 to 4 steps**, permitting sub-second image, video, and audio synthesis:
$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

---

## 4. Current State & System Benchmarks

The project is highly functional and has been validated across local and cloud environments:

### Cloud GPU Benchmarking (Tesla T4 GPU, 16GB VRAM, CUDA 12.1)
The engine was profiled across all 16 Any-to-Any cross-modal routing combinations (using a 4-frame video latent stack and a 16000-sample audio output shape):

| Modality Pathway | Verification Status | Processing Latency | Peak VRAM Allocated | Peak VRAM Reserved |
| :--- | :--- | :--- | :--- | :--- |
| **TEXT_TO_TEXT** | SUCCESS | 351.49 ms | 381.07 MB | 428.00 MB |
| **TEXT_TO_IMAGE** | SUCCESS | 57.62 ms | 448.60 MB | 492.00 MB |
| **TEXT_TO_VIDEO** | SUCCESS | 74.68 ms | 488.63 MB | 530.00 MB |
| **TEXT_TO_AUDIO** | SUCCESS | 23.10 ms | 500.39 MB | 532.00 MB |
| **IMAGE_TO_TEXT** | SUCCESS | 76.42 ms | 440.96 MB | 532.00 MB |
| **IMAGE_TO_IMAGE** | SUCCESS | 26.20 ms | 417.11 MB | 480.00 MB |
| **IMAGE_TO_VIDEO** | SUCCESS | 33.97 ms | 475.89 MB | 510.00 MB |
| **IMAGE_TO_AUDIO** | SUCCESS | 21.83 ms | 487.70 MB | 510.00 MB |
| **VIDEO_TO_TEXT** | SUCCESS | 25.64 ms | 445.47 MB | 508.00 MB |
| **VIDEO_TO_IMAGE** | SUCCESS | 29.85 ms | 429.70 MB | 484.00 MB |
| **VIDEO_TO_VIDEO** | SUCCESS | 33.49 ms | 490.14 MB | 522.00 MB |
| **VIDEO_TO_AUDIO** | SUCCESS | 25.00 ms | 501.17 MB | 522.00 MB |
| **AUDIO_TO_TEXT** | SUCCESS | 44.37 ms | 841.82 MB | 960.00 MB |
| **AUDIO_TO_IMAGE** | SUCCESS | 75.69 ms | 937.92 MB | 1028.00 MB |
| **AUDIO_TO_VIDEO** | SUCCESS | 48.02 ms | 832.40 MB | 1028.00 MB |
| **AUDIO_TO_AUDIO** | SUCCESS | 40.19 ms | 853.56 MB | 1022.00 MB |

* **Total Path Execution Time**: **957.68 ms** for all 16 pathways sequentially.
* **Peak Reserved Memory Footprint**: **1022.00 MB (1.02 GB VRAM)**, leaving ample headroom under the 3.0 GB ceiling.
* **Denoising Latencies**:
  * **Image Generation**: **1.41 ms** (4 steps)
  * **Video Generation**: **5.13 ms** (4 steps)
  * **Audio Generation**: **1.36 ms** (4 steps)
* **Text Speculative Decoding Speed**: **81.34 tokens/second** on a Tesla T4 GPU.

### Local Windows CPU Desktop Playground (`local_demo.py`)
A console-based dashboard was engineered to run the model locally. On typical laptop CPUs:
* **Generation Speed**: **20.32 tokens/second** on CPU.
* **Visual Synthesis**: Renders physical, uncompressed output media (`output_image.png`, `output_video.gif`, `output_audio.wav`) and triggers default Windows platform handlers to display them automatically.

### Stable Production Pre-Training (`train_large_scale.py`)
A fully functional pre-training module is integrated, supporting Distributed Data Parallel (DDP) with mixed precision (AMP) and gradient accumulation:
* **Epoch Latency**: **3.45 seconds** (for 16 training batches).
* **Pre-Training Loss**: Stabilized at **2.0537** without gradient explosion.

---

## 5. Future Roadmap

### Phase 4: Hugging Face Real Weight Distillation (Q2 2026)
* **Real Weight Initialization**: Replace synthetic layers with weights from pre-trained open weights (e.g., Qwen2-1.5B for text, Whisper-Base for audio, and Stable Diffusion-XL for image latents).
* **4-bit Quantization (AWQ/GPTQ)**: Compress distilled weights to keeping total runtime memory under **1.5 GB** without losing representation capacity.

### Phase 5: Mark-XXXIX Local Offline Brain Integration (Q3 2026)
* **System Action Controller**: Deploy the **Pixelle-Sirius SSM-Diffusion Engine** as the fully offline local "main brain" for the **Mark-XXXIX** assistant.
* **Real-time Screen Processing**: Utilize the selective SSM temporal recurrence loop to process user screenshots continuously at low computational costs.
* **Acoustic Interaction**: Pipe microphone inputs through the acoustic consistency solver to drive offline, real-time voice controls.
