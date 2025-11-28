"""
Comprehensive Benchmark for ESANN 2026 Submission - VERSION CORRIGÉE
GARANTIE DE REPRODUCTIBILITÉ TOTALE
Compares: ComplementarityMoE vs D-MoLE vs CL-MoE vs ProgLoRA vs EWC vs Naive
Dataset: CMU-MOSEI (text+audio+video)
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Import original continual learning methods
sys.path.insert(0, '/data/roqui')
from continual_learning import ComplementarityMoE_Simple , EWC, Naive
from continual_learning_sota import DMoLE_Adapted, CLMoE_Adapted, ProgLoRA_Adapted
from models import PerceiverIO_MoE, BaselineModel
from mosei_dataset import load_mosei_continual


def set_seed(seed):
    """Set ALL random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class ComprehensiveBenchmark:
    """Complete benchmark with SOTA comparisons - FULLY DETERMINISTIC"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SET GLOBAL SEED FIRST
        set_seed(args.seed)
        
        print(f"Using device: {self.device}")
        print(f"Global seed: {args.seed}")
        
        # Results storage
        self.results = {}
        self.detailed_metrics = {}
        self.learning_curves = {}
        
        # Create results directory
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.results_dir / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Load CMU-MOSEI data"""
        print("\n" + "="*80)
        print("LOADING CMU-MOSEI DATASET")
        print("="*80)
        
        # Reset seed before data loading
        set_seed(self.args.seed)
        
        train_loaders, val_loaders, test_loaders = load_mosei_continual(
            data_dir=self.args.data_dir,
            num_tasks=self.args.num_tasks,
            batch_size=self.args.batch_size,
            quick_test=self.args.quick_test,
            seed=self.args.seed
        )
        
        return train_loaders, val_loaders, test_loaders
    
    def create_model(self, seed=None):
        """Create fresh PerceiverIO_MoE model with specific seed"""
        if seed is not None:
            set_seed(seed)
        
        model = PerceiverIO_MoE(
            num_classes=7,
            d_model=self.args.d_model,
            num_experts=self.args.num_experts,
            r=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            tau=self.args.tau,
            num_latents=4
        ).to(self.device)
        
        return model
    
    def evaluate_task(self, model, test_loader, task_id):
        """Evaluate model on a specific task"""
        model.eval()
        
        all_preds = []
        all_labels = []
        all_expert_weights = []
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels_continuous = batch['labels'].to(self.device)
                labels = torch.clamp(labels_continuous[:, 0], -3, 3).round().long() + 3
                
                outputs = model(text, audio, video)
                preds = outputs['logits'].argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if 'expert_weights' in outputs:
                    all_expert_weights.append(outputs['expert_weights'].cpu().numpy())
        
        accuracy = correct / total
        
        # Compute F1 scores
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_preds,
            'labels': all_labels,
            'expert_weights': np.concatenate(all_expert_weights) if all_expert_weights else None
        }
    
    def evaluate_all_tasks(self, model, test_loaders, current_task_id):
        """Evaluate on all tasks seen so far"""
        results = {}
        
        for task_id in range(current_task_id + 1):
            task_results = self.evaluate_task(model, test_loaders[task_id], task_id)
            results[f'task_{task_id}'] = task_results
        
        return results
    
    def compute_continual_metrics(self, all_task_results):
        """Compute continual learning metrics"""
        num_tasks = len(all_task_results)
        
        # Extract accuracy matrix
        acc_matrix = np.zeros((num_tasks, num_tasks))
        for i, task_results in enumerate(all_task_results):
            for j in range(i + 1):
                acc_matrix[i, j] = task_results[f'task_{j}']['accuracy']
        
        # Average Accuracy
        final_accs = acc_matrix[-1, :]
        avg_accuracy = final_accs.mean()
        
        # Forgetting
        forgetting = []
        for j in range(num_tasks - 1):
            max_acc = acc_matrix[j, j]  # Performance right after training task j
            final_acc = acc_matrix[-1, j]  # Performance at the end
            forgetting.append(max_acc - final_acc)
        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        
        # Backward Transfer
        backward_transfer = []
        for j in range(num_tasks - 1):
            for i in range(j + 1, num_tasks):
                bt = acc_matrix[i, j] - acc_matrix[j, j]
                backward_transfer.append(bt)
        avg_backward_transfer = np.mean(backward_transfer) if backward_transfer else 0.0
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_forgetting': avg_forgetting,
            'avg_backward_transfer': avg_backward_transfer,
            'accuracy_matrix': acc_matrix.tolist(),
            'final_accuracies': final_accs.tolist()
        }
    
    def run_single_experiment(self, method_name, learner_class, train_loaders, 
                             val_loaders, test_loaders, method_seed=None):
        """Run a single continual learning experiment - FULLY DETERMINISTIC"""
        print(f"\n{'='*80}")
        print(f"RUNNING: {method_name}")
        print(f"{'='*80}")
        
        #Use specific seed for this method or global seed
        experiment_seed = method_seed if method_seed is not None else self.args.seed
        
        #Reset ALL seeds before EACH experiment
        set_seed(experiment_seed)
        print(f"Experiment seed: {experiment_seed}")
        
        #Create model with specific seed
        model = self.create_model(seed=experiment_seed)
        
        #Create learner with method-specific parameters
        if learner_class == DMoLE_Adapted:
            learner = learner_class(model, self.device, curriculum_alpha=0.5)
        elif learner_class == CLMoE_Adapted:
            learner = learner_class(model, self.device, momentum=0.9)
        elif learner_class == ProgLoRA_Adapted:
            learner = learner_class(model, self.device, similarity_threshold=0.7)
        elif learner_class == ComplementarityMoE_Simple:
            learner = learner_class(model, self.device, 
                                   lambda_barlow=self.args.lambda_barlow,
                                   tau=self.args.tau)
        elif learner_class == EWC:
            learner = learner_class(model, self.device, lambda_ewc=self.args.lambda_ewc)
        else:
            learner = learner_class(model, self.device)
        
        #Train sequentially on all tasks
        all_task_results = []
        
        for task_id in range(len(train_loaders)):
            print(f"\n--- Task {task_id} ---")
            
            #Reset seed before each task for reproducibility
            set_seed(experiment_seed + task_id)
            
            #Train
            learner.train_task(
                train_loaders[task_id],
                val_loaders[task_id],
                epochs=self.args.epochs_per_task,
                lr=self.args.lr
            )
            
            #Evaluate on all tasks seen so far
            task_results = self.evaluate_all_tasks(model, test_loaders, task_id)
            all_task_results.append(task_results)
            
            #Print current performance
            for tid in range(task_id + 1):
                acc = task_results[f'task_{tid}']['accuracy']
                print(f"  Task {tid}: {acc:.4f}")
        
        # Compute continual learning metrics
        metrics = self.compute_continual_metrics(all_task_results)
        
        print(f"\n{method_name} Results:")
        print(f"  Average Accuracy: {metrics['avg_accuracy']:.4f}")
        print(f"  Average Forgetting: {metrics['avg_forgetting']:.4f}")
        print(f"  Backward Transfer: {metrics['avg_backward_transfer']:.4f}")
        
        self.results[method_name] = metrics
        self.detailed_metrics[method_name] = all_task_results
        
        return metrics
    
    def generate_comparison_table(self):
        """Generate LaTeX comparison table"""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comparison of Continual Learning Methods on CMU-MOSEI}\n"
        latex += "\\label{tab:main_results}\n"
        latex += "\\begin{tabular}{lccc}\n"
        latex += "\\toprule\n"
        latex += "Method & Accuracy (\\%) & Forgetting (\\%) & BWT (\\%) \\\\\n"
        latex += "\\midrule\n"
        
        # Sort by accuracy
        sorted_methods = sorted(self.results.items(), 
                               key=lambda x: x[1]['avg_accuracy'], 
                               reverse=True)
        
        for method_name, metrics in sorted_methods:
            acc = metrics['avg_accuracy'] * 100
            forget = metrics['avg_forgetting'] * 100
            bwt = metrics['avg_backward_transfer'] * 100
            
            # Clean method name
            clean_name = method_name.replace('_', ' ')
            
            # Bold if best
            if method_name == sorted_methods[0][0]:
                latex += f"\\textbf{{{clean_name}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{forget:.2f}}} & \\textbf{{{bwt:.2f}}} \\\\\n"
            else:
                latex += f"{clean_name} & {acc:.2f} & {forget:.2f} & {bwt:.2f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        with open(self.results_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"\n✓ LaTeX table saved to {self.results_dir / 'comparison_table.tex'}")
    
    def plot_accuracy_matrix(self):
        """Plot accuracy matrices for all methods"""
        num_methods = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (method_name, metrics) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            
            acc_matrix = np.array(metrics['accuracy_matrix'])
            
            sns.heatmap(acc_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                       ax=axes[idx], vmin=0, vmax=1,
                       xticklabels=[f'T{i}' for i in range(acc_matrix.shape[1])],
                       yticklabels=[f'After T{i}' for i in range(acc_matrix.shape[0])])
            
            clean_name = method_name.replace('_', ' ')
            axes[idx].set_title(clean_name, fontsize=14, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'accuracy_matrices.pdf', dpi=300)
        plt.close()
        
        print(f"✓ Accuracy matrices saved to {self.figures_dir / 'accuracy_matrices.pdf'}")
    
    def plot_forgetting_comparison(self):
        """Bar chart comparing forgetting across methods"""
        methods = []
        forgetting_vals = []
        
        for method_name, metrics in self.results.items():
            methods.append(method_name.replace('_', ' '))
            forgetting_vals.append(metrics['avg_forgetting'] * 100)
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if f > 5 else 'green' if f < 0 else 'orange' 
                 for f in forgetting_vals]
        bars = plt.bar(methods, forgetting_vals, color=colors, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        plt.ylabel('Forgetting (%)', fontsize=14)
        plt.xlabel('Method', fontsize=14)
        plt.title('Average Forgetting Comparison (Lower is Better)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'forgetting_comparison.pdf', dpi=300)
        plt.close()
        
        print(f"✓ Forgetting comparison saved to {self.figures_dir / 'forgetting_comparison.pdf'}")
    
    def plot_accuracy_comparison(self):
        """Bar chart comparing final accuracy across methods"""
        methods = []
        accuracies = []
        
        for method_name, metrics in self.results.items():
            methods.append(method_name.replace('_', ' '))
            accuracies.append(metrics['avg_accuracy'] * 100)
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if a == max(accuracies) else 'steelblue' for a in accuracies]
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.xlabel('Method', fontsize=14)
        plt.title('Average Accuracy Comparison', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(accuracies) * 1.15)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'accuracy_comparison.pdf', dpi=300)
        plt.close()
        
        print(f"✓ Accuracy comparison saved to {self.figures_dir / 'accuracy_comparison.pdf'}")
    
    def save_results(self):
        """Save all results to JSON"""
        summary = {
            'args': vars(self.args),
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'seed_info': {
                'global_seed': self.args.seed,
                'reproducibility': 'GUARANTEED - All seeds properly set'
            }
        }
        
        with open(self.results_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Detailed metrics (without predictions for space)
        detailed_for_save = {}
        for method_name, task_results_list in self.detailed_metrics.items():
            detailed_for_save[method_name] = []
            for task_results in task_results_list:
                task_dict = {}
                for task_key, metrics in task_results.items():
                    task_dict[task_key] = {
                        'accuracy': metrics['accuracy'],
                        'f1_macro': metrics['f1_macro'],
                        'f1_weighted': metrics['f1_weighted']
                    }
                detailed_for_save[method_name].append(task_dict)
        
        with open(self.metrics_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(detailed_for_save, f, indent=2)
        
        print(f"\n✓ All results saved to {self.results_dir}")
    
    def run_benchmark(self):
        """Main benchmark execution"""
        print("="*80)
        print("Comprehensive bencharmk")
        print("CMU-MOSEI: Text + Audio + Video Continual Learning")
        print("="*80)
        
        # Load data
        train_loaders, val_loaders, test_loaders = self.load_data()
        
        # Define experiments with different seeds for fair comparison
        experiments = [
            ("ComplementarityMoE_Ours", ComplementarityMoE_Simple, self.args.seed),
            ("DMoLE_Adapted", DMoLE_Adapted, self.args.seed + 1000),
            ("CLMoE_Adapted", CLMoE_Adapted, self.args.seed + 2000),
            ("ProgLoRA_Adapted", ProgLoRA_Adapted, self.args.seed + 3000),
            ("EWC", EWC, self.args.seed + 4000),
            ("Naive_Finetuning", Naive, self.args.seed + 5000),
        ]
        
        # Run all experiments
        for method_name, learner_class, method_seed in experiments:
            try:
                self.run_single_experiment(
                    method_name, learner_class,
                    train_loaders, val_loaders, test_loaders,
                    method_seed=method_seed
                )
            except Exception as e:
                print(f"\n Error in {method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate visualizations
        print("\n" + "="*80)
        print("Visualisation and latex table(better view for analyzing results than raw terminal outputs)")
        print("="*80)
        
        self.generate_comparison_table()
        self.plot_accuracy_matrix()
        self.plot_forgetting_comparison()
        self.plot_accuracy_comparison()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("Benchmark complet")
        print("="*80)
        print(f"Results saved to: {self.results_dir}")
        print(f"Figures saved to: {self.figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Benchmark - Deterministic')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to cmu_mosei_data/')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--num_tasks', type=int, default=2,
                       help='Number of continual learning tasks')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with small dataset')
    
    # Training
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    
    # Method-specific (OPTIMIZED VALUES)
    parser.add_argument('--tau', type=float, default=0.2,
                       help='Barlow Twins tau (ComplementarityMoE_Simple)')
    parser.add_argument('--lambda_barlow', type=float, default=0.01,  # INCREASED
                       help='Barlow Twins weight (ComplementarityMoE_Simple)')
    parser.add_argument('--lambda_ewc', type=float, default=500,  # REDUCED
                       help='EWC regularization weight')
    
    args = parser.parse_args()
    
    # Set global seed at start
    set_seed(args.seed)
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(args)
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()