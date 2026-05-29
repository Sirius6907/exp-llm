# Google TPU Research Cloud (TRC) Application Draft

This document contains a prepared application draft for the **Google TPU Research Cloud (TRC)** to obtain free access to Cloud TPUs for training and evaluating the 3GB target video-generation pipeline.

---

## 1. Project Title
**Ternary-MCP & SSM: Pushing the Pareto Frontier of Edge-Scale Multimodal Video Generation under a 3GB VRAM Ceiling**

---

## 2. Project Abstract
We propose a novel, edge-optimized multimodal video-generation pipeline that targets real-time local execution under a strict **3GB VRAM limit** (e.g., consumer-grade RTX 3050 GPUs). The architecture integrates a dual-codebook TokenFlow image tokenizer (decoupling semantic and pixel features via shared index mapping), a MiniCPM-SALA hybrid (Sparse/Linear Attention) text-and-vision backbone, a Mobile Conditioning Projector (MCP) compressed using extreme ternary (1.58-bit) quantization, and a recurrent selective State Space Model (SSM) temporal block to maintain inter-frame coherence. 

While inference is strictly designed for local, consumer-grade hardware, scaling the training of our unified vision-language-diffusion pipeline requires significant compute. We apply for the Google TPU Research Cloud (TRC) to:
1. Conduct extensive pre-training and alignment on large-scale video-text datasets (e.g., WebVid-10M, Panda-70M).
2. Perform high-throughput search and ablation studies on the joint indexing mapping of the TokenFlow dual-codebook tokenizer.
3. Train our custom quantized ternary projection layers using Straight-Through Estimators (STE).

Our findings and fully reproducible code will be targeted for submission to the **ICLR 2027 Workshop on Efficient ML** and open-sourced to accelerate on-device generative AI research.

---

## 3. Detailed Description & Research Agenda

### Technical Architecture & Optimization
Our project red-teams VRAM utilization in modern generative diffusion and autoregressive architectures. By leveraging the following components, we target a sub-5% degradation in generation quality with an 8x reduction in projector memory:
- **Ternary Quantization (1.58-bit):** Restricting the weights of our Mobile Conditioning Projector (MCP) to $\{-1, 0, 1\}$ using a custom Straight-Through Estimator (STE) in PyTorch.
- **Selective SSM Temporal Blocks:** Designing a lightweight recurrence mechanism that handles frame-to-frame hidden state propagation, preventing the quadratic memory scaling of temporal self-attention.
- **Hybrid Attention (SALA):** Mixing 25% Sparse Attention (InfLLM-V2) and 75% Linear Attention (Lightning Attention) to process context sizes of up to 1M tokens with linear complexity.

### Proposed TPU Experiments & Workload
We plan to utilize Cloud TPUs to accelerate the following training phases:
1. **Phase A (Dual-Codebook Alignment):** Joint pre-training of the TokenFlow semantic and pixel codebooks using v4-8 or v5p-8 TPU nodes to optimize codebook utilization (target >95%).
2. **Phase B (Ternary Projector Tuning):** Training the 1.58-bit MCP connector. Due to the discrete nature of ternary weights, we will employ high-throughput optimization with large batch sizes, which maps perfectly to TPU pod architectures.
3. **Phase C (Temporal Recurrence Training):** Fine-tuning the selective SSM temporal block on short 4-to-8 frame video clips to establish temporal consistency.

### Expected Impact
Our research directly addresses the "compute and memory walls" of edge AI. Successful realization of this architecture will prove that high-quality, temporally consistent multimodal video understanding and generation can be democratized and executed on consumer edge devices without relying on massive, centralized cloud services.

---

## 4. Requested Resources
*   **Preferred TPU Type:** Cloud TPU v4 or v5e nodes (e.g., `v4-8` or `v5e-8` slices).
*   **Estimated Duration:** 3–6 months of part-time access.
*   **Software Stack:** PyTorch, JAX/XLA, Hugging Face Diffusers, and PyTorch Lightning.
