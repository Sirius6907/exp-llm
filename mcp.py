import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Custom autograd function for ternary quantization (-1, 0, 1) 
    using the Straight-Through Estimator (STE) to avoid memory leaks.
    """
    @staticmethod
    def forward(ctx, weight):
        # Calculate scaling factor gamma (mean absolute value of weights)
        gamma = weight.abs().mean()
        # Scale weights and clip to [-1.0, 1.0]
        scaled_w = weight / (gamma + 1e-5)
        # Round to nearest integer: {-1.0, 0.0, 1.0}
        quantized_w = torch.clamp(torch.round(scaled_w), -1.0, 1.0)
        # Save variables for backward pass
        ctx.save_for_backward(weight)
        ctx.gamma = gamma
        return quantized_w * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # Identity gradient passing for the Straight-Through Estimator (STE)
        return grad_output

class TernaryConv1d(nn.Module):
    """
    1D Convolution layer utilizing ternary weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Learnable weight parameter
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming Uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Dynamically quantize weights during the forward pass
        q_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.conv1d(x, q_weight, self.bias, self.stride, self.padding, groups=self.groups)

class TernaryDepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise-separable 1D Convolution with ternary weights.
    Extremely parameter-efficient and optimized for edge devices.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # Depthwise layer filters each channel independently
        self.depthwise = TernaryConv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # Pointwise layer mixes channels
        self.pointwise = TernaryConv1d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Performs fast, local channel attention without dimensionality reduction.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # x shape: (B, C, L)
        y = self.avg_pool(x)  # (B, C, 1)
        # Conv1D expects shape (B, channels, length) or equivalent.
        # Transpose to perform convolution across the channel dimension.
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = torch.sigmoid(y)
        return x * y

class LayerwiseFeatureFusion(nn.Module):
    """
    Fuses hidden states from multiple model layers using temperature-scaled,
    learnable weights to capture rich multi-level semantic features.
    """
    def __init__(self, num_layers=4, temp=0.07):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.temp = temp

    def forward(self, layer_hidden_states):
        # layer_hidden_states can be a list of K tensors, each of shape (B, S, D)
        if isinstance(layer_hidden_states, list):
            layer_hidden_states = torch.stack(layer_hidden_states, dim=0) # (K, B, S, D)
            
        # Compute normalized softmax weights along the layer dimension with temperature scaling
        norm_weights = F.softmax(self.weights / self.temp, dim=0) # (K,)
        
        # Reshape for broadcasting multiplication
        norm_weights = norm_weights.view(-1, 1, 1, 1)
        
        # Perform weighted sum
        fused = torch.sum(layer_hidden_states * norm_weights, dim=0)
        return fused

class MobileConditioningProjector(nn.Module):
    """
    Mobile Conditioning Projector (MCP) based on Mobile-O.
    Fuses layer-wise hidden states and projects them into the diffusion conditioning space
    using depthwise-separable ternary convolutions.
    """
    def __init__(self, vlm_dim=1536, dit_dim=1024, num_layers=4, temp=0.07):
        super().__init__()
        self.fusion = LayerwiseFeatureFusion(num_layers=num_layers, temp=temp)
        
        # Downsample sequence length by stride=2 to conserve diffusion KV-cache memory
        self.proj = TernaryDepthwiseSeparableConv1d(
            in_channels=vlm_dim,
            out_channels=dit_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.eca = EfficientChannelAttention(channels=dit_dim, kernel_size=3)
        self.layer_norm = nn.LayerNorm(dit_dim)

    def forward(self, layer_hidden_states):
        # 1. Layer-wise fusion: (B, S, vlm_dim)
        fused = self.fusion(layer_hidden_states)
        
        # 2. Reshape for 1D convolution: (B, vlm_dim, S)
        x = fused.transpose(-1, -2)
        
        # 3. Depthwise-separable convolution: (B, dit_dim, S // 2)
        proj_x = self.proj(x)
        
        # 4. Channel attention: (B, dit_dim, S // 2)
        attn_x = self.eca(proj_x)
        
        # 5. Normalization: (B, S // 2, dit_dim)
        out = attn_x.transpose(-1, -2)
        out = self.layer_norm(out)
        
        return out
