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
        self.class_weights = None  # Will be set per task
        
    def set_class_weights(self, train_loader):
        """Compute and set class weights for current task"""
        class_counts = torch.zeros(7, device=self.device)
        
        for batch in train_loader:
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            for label in labels:
                class_counts[label] += 1
        
        # Inverse frequency weights WITH CAPPING
        total = class_counts.sum()
        weights = total / (class_counts + 1.0)  # Add smoothing
        weights = torch.clamp(weights, max=10.0)  # Cap at 10x max
        weights = weights / weights.sum() * 7
        
        self.class_weights = weights
        print(f"  Class weights: {weights.cpu().numpy()}")
        
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        raise NotImplementedError
        
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        raise NotImplementedError

class ComplementarityMoE_Simple:
    def __init__(self, model, device, lambda_barlow=0.01, tau=0.2):
        self.model = model
        self.device = device
        self.lambda_barlow = lambda_barlow
        self.tau = tau
        self.task_id = 0
        self.router_lr_mult = 3.0
        self.primary_loss_weight = 1.0
        
    def set_class_weights(self, train_loader):
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
    
    def train_task(self, train_loader, val_loader, epochs, lr):
        self.set_class_weights(train_loader)
        
        # Primary expert
        num_experts = len(self.model.experts)
        primary_expert = self.task_id % num_experts
        print(f"  Primary expert for Task {self.task_id}: Expert {primary_expert}")
        
        # FREEZE after Task 0
        if self.task_id > 0:
            for param in self.model.perceiver_encoder.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = False
            print(f"  [FROZEN] Perceiver and Classifier frozen")
        
        # Build optimizer without frozen params
        trainable_params = [
            {'params': self.model.router.parameters(), 'lr': lr * self.router_lr_mult},
            {'params': self.model.experts[primary_expert].parameters(), 'lr': lr}
        ]
        
        # Add perceiver and classifer only for task 0
        if self.task_id == 0:
            trainable_params.insert(0, {'params': self.model.perceiver_encoder.parameters(), 'lr': lr})
            trainable_params.append({'params': self.model.classifier.parameters(), 'lr': lr})
        
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
        
        # Training
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                
                # CE Loss
                ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(
                    outputs['logits'], labels
                )
                
                # Routing losses
                expert_weights = outputs['expert_weights']
                primary_loss = -torch.log(expert_weights[:, primary_expert] + 1e-8).mean()
                entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1).mean()
                
                # Barlow Twins
                view1, view2 = outputs['router_views']
                z1_norm = (view1 - view1.mean(0)) / (view1.std(0) + 1e-8)
                z2_norm = (view2 - view2.mean(0)) / (view2.std(0) + 1e-8)
                N = view1.size(0)
                c = (z1_norm.T @ z2_norm) / N
                on_diag = ((torch.diagonal(c) - self.tau) ** 2).sum()
                off_diag = (c ** 2).sum() - (torch.diagonal(c) ** 2).sum()
                bt_loss = on_diag + self.lambda_barlow * off_diag
                
                # Total loss
                total_loss_batch = (
                    ce_loss +
                    self.primary_loss_weight * primary_loss +
                    -0.1 * entropy +
                    self.lambda_barlow * bt_loss
                )
                
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if epoch % 2 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        self.task_id += 1

class EWC(ContinualLearner):
    """Elastic Weight Consolidation"""
    def __init__(self, model: nn.Module, device: str = 'cuda', lambda_ewc: float = 1000):
        super().__init__(model, device)
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}
        self.optpar_dict = {}
        
    def compute_fisher(self, train_loader):
        """Compute Fisher Information Matrix"""
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        for batch in train_loader:
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            
            labels_continuous = batch['labels'].to(self.device)
            labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
            
            self.model.zero_grad()
            outputs = self.model(text, audio, video)
            loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        for name in fisher:
            fisher[name] /= len(train_loader)
            
        return fisher
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        if self.class_weights is not None:
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
        else:
            ce_loss = nn.CrossEntropyLoss()(outputs['logits'], targets)
        
        if task_id == 0:
            return ce_loss
        
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                ewc_loss += (self.fisher_dict[name] * 
                           (param - self.optpar_dict[name]) ** 2).sum()
        
        return ce_loss + (self.lambda_ewc / 2) * ewc_loss
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        for epoch in range(epochs):
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
        
        self.fisher_dict = self.compute_fisher(train_loader)
        self.optpar_dict = {name: param.data.clone() 
                           for name, param in self.model.named_parameters() 
                           if param.requires_grad}
        self.task_id += 1


class MoELoRA_True(ContinualLearner):
    """TRUE MoELoRA baseline from SOTA"""
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)
        self.task_experts = {}
        
    def assign_expert(self, task_id: int):
        num_experts = len(self.model.experts)
        self.task_experts[task_id] = task_id % num_experts
        print(f"  Task {task_id} assigned to Expert {task_id % num_experts}")
        
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        if self.class_weights is not None:
            return nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
        else:
            return nn.CrossEntropyLoss()(outputs['logits'], targets)
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)
        self.assign_expert(self.task_id)
        expert_id = self.task_experts[self.task_id]
        
        for i, expert in enumerate(self.model.experts):
            for param in expert.parameters():
                param.requires_grad = (i == expert_id)
        
        for param in self.model.router.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        for epoch in range(epochs):
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
        
        self.task_id += 1


class KnowledgeDistillation(ContinualLearner):
    """Knowledge Distillation baseline"""
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                 lambda_kd: float = 0.5, temperature: float = 2.0):
        super().__init__(model, device)
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.previous_model = None
        
    def compute_cross_modal_similarity(self, features: torch.Tensor) -> torch.Tensor:
        features_norm = F.normalize(features, dim=-1)
        similarity = torch.mm(features_norm, features_norm.t())
        return similarity
    
    def distillation_loss(self, student_features: torch.Tensor, 
                         teacher_features: torch.Tensor) -> torch.Tensor:
        student_sim = self.compute_cross_modal_similarity(student_features)
        teacher_sim = self.compute_cross_modal_similarity(teacher_features)
        
        student_sim = student_sim / self.temperature
        teacher_sim = teacher_sim / self.temperature
        
        loss = F.kl_div(
            F.log_softmax(student_sim, dim=-1),
            F.softmax(teacher_sim, dim=-1),
            reduction='batchmean'
        )
        
        return loss
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        if self.class_weights is not None:
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
        else:
            ce_loss = nn.CrossEntropyLoss()(outputs['logits'], targets)
        
        if task_id == 0 or self.previous_model is None:
            return ce_loss
        
        kd_loss = 0
        if 'features' in outputs:
            with torch.no_grad():
                teacher_outputs = self.previous_model(
                    outputs.get('text_input'),
                    outputs.get('audio_input'), 
                    outputs.get('video_input')
                )
            
            kd_loss = self.distillation_loss(
                outputs['features'],
                teacher_outputs['features']
            )
        
        return ce_loss + self.lambda_kd * kd_loss
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                optimizer.zero_grad()
                outputs = self.model(text, audio, video)
                
                outputs['text_input'] = text
                outputs['audio_input'] = audio
                outputs['video_input'] = video
                
                loss = self.compute_loss(outputs, labels, self.task_id)
                loss.backward()
                optimizer.step()
        
        import copy
        self.previous_model = copy.deepcopy(self.model)
        for param in self.previous_model.parameters():
            param.requires_grad = False
        self.previous_model.eval()
        
        self.task_id += 1


class Naive(ContinualLearner):
    """Naive fine-tuning (no continual learning strategy)"""
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)
        
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        return nn.CrossEntropyLoss(weight=self.class_weights)(outputs['logits'], targets)
    
    def train_task(self, train_loader, val_loader, epochs: int, lr: float):
        self.set_class_weights(train_loader)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, weight_decay=1e-4
        )
        
        self.model.train()
        for epoch in range(epochs):
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
        
        self.task_id += 1


# Alias for backward compatibility
MoELoRA_Baseline = MoELoRA_True