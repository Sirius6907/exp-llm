# Edge-Scale Video Generation (3GB target)

This repository contains the official, fully reproducible implementation of the edge-scale video generation pipeline designed to operate within a strict **3GB VRAM limit** (e.g., local RTX 3050 GPUs). The project integrates a dual-codebook **TokenFlow** tokenizer, **MiniCPM-SALA** hybrid attention backbone, **Mobile Conditioning Projector (MCP)** quantized to ternary states, and a selective **SSM temporal block** routed via the **DynaMap** runtime orchestrator.

---

## 🛠️ System Architecture Overview

```
                        +----------------------+
                        |   Text / Image / Frame|
                        +-----------+----------+
                                    |
                                    v
                          +------------------+
                          |  DynaMap Router  |
                          +---+----------+---+
                              |          |
            +-----------------+          +-----------------+
            | (Text Modality)                              | (Image Modality)
            v                                              v
+-----------------------+                      +-----------------------+
|  MiniCPM-SALA VLM     |                      |  TokenFlow Tokenizer  |
|  (9B Sparse/Linear)   |                      |  (Dual-Codebook VQ)   |
+-----------+-----------+                      +-----------+-----------+
            |                                              |
            | (Hidden States X_i)                          | (Semantic q_s)
            v                                              v
+-----------------------+                      +-----------------------+
|  Ternary Mobile       |                      |  MiniCPM-SALA VLM     |
|  Conditioning (MCP)   |<---------------------+  (Visual Encoding)    |
+-----------+-----------+                      +-----------+-----------+
            |                                              |
            | (Conditioning C)                             | (Pixel q_p)
            v                                              v
+----------------------------------------------------------------------+
|                       Bonsai Diffusion Generator                     |
|                   (Ternary 1.58-bit / STE quantized)                 |
+----------------------------------+-----------------------------------+
                                   |
                                   v <--- [SSM Temporal State h_t]
                       +-----------------------+
                       |  SSM Temporal Wedge   |
                       |  (Frame-to-Frame)     |
                       +-----------------------+
```

---

## 📂 Repository Structure

```bash
experiment-sirius/
├── mcp.py               # Mobile Conditioning Projector (MCP) implementation
├── ssm_temporal.py      # Selective SSM Temporal Wedge recurrence block
├── tokenflow.py         # TokenFlow dual-codebook vector quantizer & tokenizer
├── dynamap.py           # DynaMap system modality router and orchestrator
├── mock_test.py         # MCP shape validation and Autograd STE test script
├── ssm_test.py          # SSM temporal loop and gradient propagation check
├── orchestrator_test.py # Integrated DynaMap end-to-end routing check
├── trc_application.md   # Prepared Google TPU Research Cloud application
└── README.md            # System developer guide (this document)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ and PyTorch 2.0+ installed.

### 2. Quick Setup
Clone the repository and install dependencies:
```bash
pip install torch torchvision
```

### 3. Run Validation Tests
Verify each subsystem of the pipeline under strict memory constraints:

*   **Test the Mobile Conditioning Projector (MCP):**
    ```bash
    python mock_test.py
    ```
    *Validates sequence downsampling, ternary Conv1d weights range, and autograd backpropagation.*

*   **Test the Selective SSM Recurrence Loop:**
    ```bash
    python ssm_test.py
    ```
    *Validates 8-frame recurrence loop speed, state sizes, and gradient flow across time steps.*

*   **Test the Fully Fused DynaMap Orchestrator:**
    ```bash
    python orchestrator_test.py
    ```
    *Runs both Text-to-Video and Image-to-Video paths, verifying that parameters remain highly compact (~16.95M parameters).*

---

## 💻 Developer API Usage

The central entry point is the **DynaMap Orchestrator**. You can easily route text prompts or visual frames:

```python
import torch
from experiment_sirius.dynamap import DynaMapOrchestrator

# 1. Initialize the orchestrator
device = "cuda" if torch.cuda.is_available() else "cpu"
orchestrator = DynaMapOrchestrator(
    codebook_size=4096,
    vlm_dim=1536,
    dit_dim=1024,
    latent_dim=256
).to(device)

# 2. Path A: Text-to-Video (generate 8 coherent frames)
text_frames = orchestrator.route(
    mode="text_to_video",
    num_frames=8,
    device=device
)

# 3. Path B: Image-to-Video (encode a source image and animate it)
mock_image = torch.randn(1, 3, 256, 256, device=device) # 256x256 RGB image
image_frames = orchestrator.route(
    mode="image_to_video",
    image_input=mock_image,
    num_frames=8,
    device=device
)
```
