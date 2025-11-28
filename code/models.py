import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class PerceiverIOEncoder(nn.Module):
    """PerceiverIO with optional modality masking"""
    def __init__(self, d_model: int = 64, num_heads: int = 4, num_latents: int = 4,
                 use_text: bool = True, use_audio: bool = True, use_video: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        
        # Input projections
        self.text_proj = nn.Sequential(
            nn.Linear(300, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(74, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(35, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Learnable latent array (bottleneck)
        self.latent = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        
        # Cross-attention: latent attends to inputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
    
    def forward(self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """Forward with modality masking support"""
        batch_size = text.size(0)
        
        # Project modalities
        inputs_list = []
        
        if self.use_text:
            z_text = self.text_proj(text.mean(dim=1, keepdim=True))
            inputs_list.append(z_text)
        
        if self.use_audio:
            z_audio = self.audio_proj(audio.mean(dim=1, keepdim=True))
            inputs_list.append(z_audio)
        
        if self.use_video:
            z_video = self.video_proj(video.mean(dim=1, keepdim=True))
            inputs_list.append(z_video)
        
        # If no modality, use zeros
        if not inputs_list:
            inputs = torch.zeros(batch_size, 1, self.d_model, device=text.device)
        else:
            inputs = torch.cat(inputs_list, dim=1)
        
        # PerceiverIO: latent attends to inputs
        latent = self.latent.expand(batch_size, -1, -1)
        
        # Cross-attention
        attn_out, _ = self.cross_attention(latent, inputs, inputs)
        latent = self.norm1(latent + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(latent)
        latent = self.norm2(latent + ffn_out)
        
        # Pool latents
        fused_features = latent.mean(dim=1)
        
        return fused_features


class BarlowTwinsRouter(nn.Module):
    """Router with direct supervision"""
    def __init__(self, d_model: int = 64, num_experts: int = 4, tau: float = 0.3):
        super().__init__()
        self.tau = tau
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Direct routing path
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_experts)
        )
        
        # Auxiliary projections for Barlow Twins
        self.proj1 = nn.Linear(d_model, d_model // 2)
        self.proj2 = nn.Linear(d_model, d_model // 2)
        
        # Xavier initialization
        for m in [self.router[0], self.router[-1], self.proj1, self.proj2]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, perceiver_features: torch.Tensor):
        """Forward without detach"""
        logits = self.router(perceiver_features)
        weights = F.softmax(logits, dim=-1)
        
        view1 = self.proj1(perceiver_features)
        view2 = self.proj2(perceiver_features)
        
        return weights, (view1, view2)


class LoRAExpert(nn.Module):
    """Single LoRA expert"""
    def __init__(self, d_model: int = 64, r: int = 8, alpha: int = 16):
        super().__init__()
        
        self.lora_down = nn.Linear(d_model, r, bias=False)
        self.lora_up = nn.Linear(r, d_model, bias=False)
        self.scaling = alpha / r
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_up(self.lora_down(features)) * self.scaling
        return features + lora_output


class PerceiverIO_MoE(nn.Module):
    """PerceiverIO + LoRA MoE"""
    def __init__(self, num_classes: int = 7, d_model: int = 64, 
                 num_experts: int = 4, r: int = 8, alpha: int = 16, 
                 tau: float = 0.3, num_latents: int = 4,
                 use_text: bool = True, use_audio: bool = True, use_video: bool = True):
        super().__init__()
        
        self.num_experts = num_experts
        self.d_model = d_model
        
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        
        # PerceiverIO encoder
        self.perceiver_encoder = PerceiverIOEncoder(
            d_model=d_model,
            num_heads=4,
            num_latents=num_latents,
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video
        )
        
        # LoRA experts
        self.experts = nn.ModuleList([
            LoRAExpert(d_model, r, alpha) for _ in range(num_experts)
        ])
        
        # Router
        self.router = BarlowTwinsRouter(d_model, num_experts, tau)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, text: torch.Tensor, audio: torch.Tensor, 
                video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        # Mask unused modalities
        if not self.use_text:
            text = torch.zeros_like(text)
        if not self.use_audio:
            audio = torch.zeros_like(audio)
        if not self.use_video:
            video = torch.zeros_like(video)
        
        # Encode
        perceiver_features = self.perceiver_encoder(text, audio, video)
        
        # Route
        expert_weights, (view1, view2) = self.router(perceiver_features)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(perceiver_features)
            expert_outputs.append(expert_out)
        
        # Fuse with einsum
        expert_outputs = torch.stack(expert_outputs, dim=1)
        fused = torch.einsum('bne,bn->be', expert_outputs, expert_weights)
        
        # Classify
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'expert_weights': expert_weights,
            'features': fused,
            'perceiver_features': perceiver_features,
            'router_views': (view1, view2)
        }


class BaselineModel(nn.Module):
    """Simple baseline"""
    def __init__(self, num_classes: int = 7, d_model: int = 64):
        super().__init__()
        
        self.text_proj = nn.Linear(300, d_model)
        self.audio_proj = nn.Linear(74, d_model)
        self.video_proj = nn.Linear(35, d_model)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, text: torch.Tensor, audio: torch.Tensor, 
                video: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_feat = self.text_proj(text.mean(dim=1))
        audio_feat = self.audio_proj(audio.mean(dim=1))
        video_feat = self.video_proj(video.mean(dim=1))
        
        fused = torch.cat([text_feat, audio_feat, video_feat], dim=-1)
        logits = self.fusion(fused)
        
        return {'logits': logits, 'features': fused}