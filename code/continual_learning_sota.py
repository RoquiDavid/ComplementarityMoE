"""
SOTA Baselines for Multimodal Continual Learning (Text+Audio+Video)
Adapted implementations of recent methods for fair comparison on CMU-MOSEI

References:
- D-MoLE: Dynamic Mixture of Curriculum LoRA Experts (ICML 2025)
- CL-MoE: Dual Momentum MoE (CVPR 2025)  
- ProgLoRA: Progressive LoRA (ACL 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from copy import deepcopy


class ContinualLearner:
    """Base class for continual learning strategies"""
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.task_id = 0
        self.class_weights = None
        
    def set_class_weights(self, train_loader):
        """Compute and set class weights for current task"""
        class_counts = torch.zeros(7, device=self.device)
        
        for batch in train_loader:
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            for label in labels:
                class_counts[label] += 1
        
        total = class_counts.sum()
        weights = total / (class_counts + 1.0)
        weights = torch.clamp(weights, max=10.0)
        weights = weights / weights.sum() * 7
        
        self.class_weights = weights
        print(f"  Class weights: {weights.cpu().numpy()}")
        
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        raise NotImplementedError
        
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        raise NotImplementedError


class DMoLE_Adapted(ContinualLearner):
    """
    Adaptation of D-MoLE (ICML 2025) for text+audio+video
    
    Key ideas from paper:
    - Dynamic layer-wise expert allocation based on gradient magnitude
    - Inter-modal curriculum learning: adjust modality update ratios by difficulty
    
    Adaptations for our setting:
    - Applied to PerceiverIO instead of MLLM
    - 3 modalities (text+audio+video) instead of 2 (vision+language)
    """
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                curriculum_alpha: float = 0.5):
        super().__init__(model, device)
        self.curriculum_alpha = curriculum_alpha
        self.modality_difficulties = {'text': 1.0, 'audio': 1.0, 'video': 1.0}
        
    def compute_modality_difficulty(self, train_loader):
        """Estimate modality difficulty based on gradient magnitude"""
        self.model.eval()
        
        grad_magnitudes = {'text': 0.0, 'audio': 0.0, 'video': 0.0}
        num_batches = 0
        
        for batch in train_loader:
            if num_batches >= 5:  # Sample only 5 batches for efficiency
                break
                
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            
            self.model.zero_grad()
            outputs = self.model(text, audio, video)
            loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
            loss.backward()
            
            # Measure gradient magnitude for each modality encoder
            if hasattr(self.model, 'perceiver_encoder'):
                grad_magnitudes['text'] += self.model.perceiver_encoder.text_proj[0].weight.grad.abs().mean().item()
                grad_magnitudes['audio'] += self.model.perceiver_encoder.audio_proj[0].weight.grad.abs().mean().item()
                grad_magnitudes['video'] += self.model.perceiver_encoder.video_proj[0].weight.grad.abs().mean().item()
            
            num_batches += 1
        
        # Normalize difficulties
        total = sum(grad_magnitudes.values())
        for k in grad_magnitudes:
            self.modality_difficulties[k] = grad_magnitudes[k] / (total + 1e-8)
        
        print(f"  Modality difficulties: {self.modality_difficulties}")
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        return nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)
        
        # Compute modality difficulties for curriculum
        self.compute_modality_difficulty(train_loader)
        
        # Dynamic expert allocation: assign expert to current task
        num_experts = len(self.model.experts)
        primary_expert = self.task_id % num_experts
        print(f"  Task {self.task_id} → Expert {primary_expert} (D-MoLE style)")
        
        # Freeze previous experts
        for i, expert in enumerate(self.model.experts):
            freeze = (i < self.task_id)
            for param in expert.parameters():
                param.requires_grad = not freeze
        
        # All other components trainable with modality-aware learning rates
        if hasattr(self.model, 'perceiver_encoder'):
            for param in self.model.perceiver_encoder.parameters():
                param.requires_grad = True
        
        for param in self.model.router.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Curriculum-based learning rates for modality encoders
        text_lr = lr * (1.0 + self.curriculum_alpha * self.modality_difficulties['text'])
        audio_lr = lr * (1.0 + self.curriculum_alpha * self.modality_difficulties['audio'])
        video_lr = lr * (1.0 + self.curriculum_alpha * self.modality_difficulties['video'])
        
        print(f"  Modality LRs: text={text_lr:.5f}, audio={audio_lr:.5f}, video={video_lr:.5f}")
        
        optimizer = torch.optim.AdamW([
            {'params': self.model.perceiver_encoder.text_proj.parameters(), 'lr': text_lr},
            {'params': self.model.perceiver_encoder.audio_proj.parameters(), 'lr': audio_lr},
            {'params': self.model.perceiver_encoder.video_proj.parameters(), 'lr': video_lr},
            {'params': self.model.router.parameters(), 'lr': lr},
            {'params': [p for e in self.model.experts for p in e.parameters() if p.requires_grad], 'lr': lr},
            {'params': self.model.classifier.parameters(), 'lr': lr}
        ], weight_decay=1e-4)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                loss = self.compute_loss(outputs, labels, self.task_id)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            print(f"    Epoch {epoch+1}/{epochs}: Loss={epoch_loss/num_batches:.4f}")
        
        self.task_id += 1


class CLMoE_Adapted(ContinualLearner):
    """
    Adaptation of CL-MoE (CVPR 2025) for text+audio+video
    
    Key ideas from paper:
    - Dual-Router: instance-level (soft) + task-level (hard) routing
    - Dynamic Momentum: EMA-based expert update
    
    Adaptations:
    - Applied to PerceiverIO features instead of LLaVA
    - 3 modalities instead of 2
    """
    def __init__(self, model: nn.Module, device: str = 'cuda',
                momentum: float = 0.9):
        super().__init__(model, device)
        self.momentum = momentum
        self.task_experts = {}  # Task-level routing
        self.ema_experts = None  # For momentum update
        
    def assign_expert(self, task_id: int):
        """Task-level router: hard assignment"""
        num_experts = len(self.model.experts)
        self.task_experts[task_id] = task_id % num_experts
        print(f"  Task {task_id} assigned to Expert {task_id % num_experts} (task-level)")
    
    def update_ema_experts(self):
        """Dynamic Momentum: update EMA of experts"""
        if self.ema_experts is None:
            self.ema_experts = [deepcopy(expert.state_dict()) for expert in self.model.experts]
        else:
            for i, expert in enumerate(self.model.experts):
                for key in self.ema_experts[i]:
                    self.ema_experts[i][key] = (
                        self.momentum * self.ema_experts[i][key] + 
                        (1 - self.momentum) * expert.state_dict()[key]
                    )
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
        
        # Instance-level routing loss: encourage diversity
        expert_weights = outputs['expert_weights']
        entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1).mean()
        
        # Task-level guidance: prefer assigned expert
        task_expert_id = self.task_experts.get(task_id, task_id % len(self.model.experts))
        task_loss = -torch.log(expert_weights[:, task_expert_id] + 1e-8).mean()
        
        return ce_loss + 0.5 * task_loss - 0.1 * entropy
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)
        self.assign_expert(self.task_id)
        
        expert_id = self.task_experts[self.task_id]
        
        # Only current expert trainable
        for i, expert in enumerate(self.model.experts):
            for param in expert.parameters():
                param.requires_grad = (i == expert_id)
        
        for param in self.model.router.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        if hasattr(self.model, 'perceiver_encoder'):
            for param in self.model.perceiver_encoder.parameters():
                param.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                loss = self.compute_loss(outputs, labels, self.task_id)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Update EMA
            self.update_ema_experts()
            
            print(f"    Epoch {epoch+1}/{epochs}: Loss={epoch_loss/num_batches:.4f}")
        
        self.task_id += 1


class ProgLoRA_Adapted(ContinualLearner):
    """
    Adaptation of ProgLoRA (ACL 2025) for text+audio+video
    
    Key ideas from paper:
    - Progressive LoRA: add new LoRA blocks per task
    - Task-relevance based allocation: reuse similar blocks
    
    Adaptations:
    - Applied to PerceiverIO+MoE architecture
    - Track task similarity via feature distance
    """
    def __init__(self, model: nn.Module, device: str = 'cuda',
                similarity_threshold: float = 0.7):
        super().__init__(model, device)
        self.similarity_threshold = similarity_threshold
        self.task_features = []  # Store representative features per task
        self.task_expert_map = {}  # Which expert(s) for which task
        
    def compute_task_similarity(self, train_loader):
        """Compute similarity with previous tasks"""
        if not self.task_features:
            return None
        
        # Extract features from current task
        self.model.eval()
        current_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 5:  # Sample only
                    break
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                
                outputs = self.model(text, audio, video)
                current_features.append(outputs['perceiver_features'].mean(0))
        
        current_feat = torch.stack(current_features).mean(0)
        
        # Compare with previous tasks
        similarities = []
        for prev_feat in self.task_features:
            sim = F.cosine_similarity(current_feat.unsqueeze(0), prev_feat.unsqueeze(0))
            similarities.append(sim.item())
        
        self.task_features.append(current_feat)
        
        return similarities
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        return nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)
        
        # Compute task similarity
        similarities = self.compute_task_similarity(train_loader)
        
        num_experts = len(self.model.experts)
        
        if similarities and max(similarities) > self.similarity_threshold:
            # Reuse most similar task's expert
            most_similar_task = similarities.index(max(similarities))
            allocated_expert = self.task_expert_map[most_similar_task]
            print(f"  Task {self.task_id} similar to Task {most_similar_task} → Reusing Expert {allocated_expert}")
        else:
            # Allocate new expert
            allocated_expert = self.task_id % num_experts
            print(f"  Task {self.task_id} → New Expert {allocated_expert}")
        
        self.task_expert_map[self.task_id] = allocated_expert
        
        # Only allocated expert trainable
        for i, expert in enumerate(self.model.experts):
            for param in expert.parameters():
                param.requires_grad = (i == allocated_expert)
        
        # Other components trainable
        for param in self.model.router.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        if hasattr(self.model, 'perceiver_encoder'):
            for param in self.model.perceiver_encoder.parameters():
                param.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                loss = self.compute_loss(outputs, labels, self.task_id)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            print(f"    Epoch {epoch+1}/{epochs}: Loss={epoch_loss/num_batches:.4f}")
        
        self.task_id += 1