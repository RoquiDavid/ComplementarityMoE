import sys
import os
import torch
import numpy as np
from pathlib import Path

#add path
sys.path.insert(0, os.path.expanduser('/data/roqui'))

from models import PerceiverIO_MoE
from continual_learning import ComplementarityMoE_Simple, EWC, Naive
from continual_learning_sota import DMoLE_Adapted, CLMoE_Adapted, ProgLoRA_Adapted
from mosei_dataset import load_mosei_continual

# Configuration
DATA_DIR = os.path.expanduser("/data/roqui/cmu_mosei_data")
NUM_TASKS = 2
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001
SEED = 42  # Seed fixe for reproducibility
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fixe all seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("="*80)
print("Test for all baslines")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Tasks: {NUM_TASKS}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print(f"Seed: {SEED}")
print("")

# Charger données avec seed
print("Dataloading")
train_loaders, val_loaders, test_loaders = load_mosei_continual(
    data_dir=DATA_DIR,
    num_tasks=NUM_TASKS,
    batch_size=BATCH_SIZE,
    seed=SEED
)
print(f"{NUM_TASKS} task loaded")


def evaluate_model(model, test_loaders, device):
    """Evaluate all tasks"""
    model.eval()
    accuracies = []
    
    for test_loader in test_loaders:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                video = batch['video'].to(device)
                labels_continuous = batch['labels'].to(device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                outputs = model(text, audio, video)
                preds = outputs['logits'].argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracies.append(correct / total)
    
    return accuracies


def run_experiment(method_name, learner_class, learner_kwargs=None):
    """Execute a full experiment with fixed seed"""
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    print("="*80)
    print(f"MÉTHODE: {method_name}")
    print("="*80)
    
    # Relaod data with fixed seed
    train_loaders, val_loaders, test_loaders = load_mosei_continual(
        data_dir=DATA_DIR,
        num_tasks=NUM_TASKS,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    
    # Model
    model = PerceiverIO_MoE(
        num_classes=7,
        d_model=64,
        num_experts=4,
        r=8,
        alpha=16,
        tau=0.3,
        num_latents=4
    ).to(DEVICE)
    
    #learner
    if learner_kwargs is None:
        learner_kwargs = {}
    learner = learner_class(model, DEVICE, **learner_kwargs)
    
    #Training on task
    all_accuracies = []
    
    for task_id in range(NUM_TASKS):
        print(f"\n--- Task {task_id} ---")
        
        
        learner.train_task(
            train_loaders[task_id],
            val_loaders[task_id],
            epochs=EPOCHS,
            lr=LR
        )
        
        #Evaluate
        task_accs = evaluate_model(model, test_loaders[:task_id+1], DEVICE)
        all_accuracies.append(task_accs)
        
        #Display
        for tid, acc in enumerate(task_accs):
            print(f"  Task {tid}: {acc*100:.2f}%")
    
    #Finals metrics
    final_accs = all_accuracies[-1]
    avg_acc = np.mean(final_accs)
    
    if len(all_accuracies) > 1:
        forgetting = []
        for i in range(len(all_accuracies) - 1):
            task_i_after_training = all_accuracies[i][i]
            task_i_final = all_accuracies[-1][i]
            forgetting.append(task_i_after_training - task_i_final)
        avg_forgetting = np.mean(forgetting)
    else:
        avg_forgetting = 0.0
    
    print(f"\n{method_name} - Resuls:")
    print(f"  Mean accuracy: {avg_acc*100:.2f}%")
    print(f"  Mean forgetting: {avg_forgetting*100:.2f}%")
    print("")
    
    return {
        'avg_accuracy': avg_acc,
        'avg_forgetting': avg_forgetting,
        'final_accuracies': final_accs,
        'all_accuracies': all_accuracies
    }


# ============================================================================
# Execute all test
# ============================================================================

results = {}

# 1. ComplementarityMoE 
results['ComplementarityMoE'] = run_experiment(
    'ComplementarityMoE (Yours)',
    ComplementarityMoE_Simple,  
    {}
)

# 2. D-MoLE
results['DMoLE'] = run_experiment(
    'D-MoLE (ICML 2025)',
    DMoLE_Adapted,
    {'curriculum_alpha': 0.5}
)

# 3. CL-MoE
results['CLMoE'] = run_experiment(
    'CL-MoE (CVPR 2025)',
    CLMoE_Adapted,
    {'momentum': 0.9}
)

# 4. ProgLoRA
results['ProgLoRA'] = run_experiment(
    'ProgLoRA (ACL 2025)',
    ProgLoRA_Adapted,
    {'similarity_threshold': 0.7}
)

# 5. EWC
results['EWC'] = run_experiment(
    'EWC (Baseline)',
    EWC,
    {'lambda_ewc': 1000}
)

# 6. Naive
results['Naive'] = run_experiment(
    'Naive Finetuning',
    Naive,
    {}
)


# ============================================================================
# Compare results
# ============================================================================

print("\n" + "="*80)
print("Benchmark")
print("="*80)
print(f"{'Model':<30} {'Accuracy':>12} {'Forgetting':>12}")
print("-"*80)

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)

for method_name, metrics in sorted_results:
    acc = metrics['avg_accuracy'] * 100
    forget = metrics['avg_forgetting'] * 100
    marker = "← YOURS" if method_name == 'ComplementarityMoE' else ""
    print(f"{method_name:<30} {acc:>11.2f}% {forget:>11.2f}% {marker}")

print("="*80)

# Compare our approach to others
your_acc = results['ComplementarityMoE']['avg_accuracy'] * 100
best_sota_acc = max([v['avg_accuracy'] for k, v in results.items() if k != 'ComplementarityMoE']) * 100

print("\nResult:")
if your_acc >= best_sota_acc:
    print(f"Our approach perfoms well ({your_acc:.1f}% vs {best_sota_acc:.1f}%)")
elif your_acc >= best_sota_acc - 3:
    print(f"Our approach performs really well ({your_acc:.1f}% vs {best_sota_acc:.1f}%)")
else:
    print(f"Our approach is under ({your_acc:.1f}% vs {best_sota_acc:.1f}%)")

print("")

# Save
import json
output_file = os.path.expanduser("/data/roqui/test_baselines_deterministic.json") #Replace by your own path
with open(output_file, 'w') as f:
    # Convert to serialisable format
    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            'avg_accuracy': float(v['avg_accuracy']),
            'avg_forgetting': float(v['avg_forgetting']),
            'final_accuracies': [float(a) for a in v['final_accuracies']]
        }
    json.dump(save_results, f, indent=2)

print(f"Results save: {output_file}")
print("\n" + "="*80)
print("Test finished")
print("="*80)
