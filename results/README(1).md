# Results Directory

This directory contains experimental results from the paper.

## Expected Results Structure

After running experiments, you should see:

```
results/
├── complementarity_moe/           # Our method
│   ├── summary.json               # Overall metrics
│   ├── detailed_metrics.json      # Per-task breakdown
│   └── figures/
│       ├── accuracy_matrices.pdf
│       ├── forgetting_comparison.pdf
│       └── accuracy_comparison.pdf
├── benchmark/                     # All methods comparison
│   ├── summary.json
│   └── figures/
└── seed_experiments/              # Multi-seed validation
    ├── seed_100/
    ├── seed_80/
    ├── seed_90/
    ├── seed_70/
    └── seed_30/
```

## Main Results (Paper Table 1)

### ComplementarityMoE (Ours)
```
Accuracy:  51.54% ± 1.37%
Forgetting: 0.00%
BWT:        0.00% (Task 0: +1.2% improvement)
Seeds:     [100, 80, 90, 70, 30]
```

### Baselines
| Method | Accuracy (%) | Std (%) | Forgetting (%) |
|--------|-------------|---------|----------------|
| **ComplementarityMoE (Ours)** | **51.54** | **1.37** | **0.00** |
| DMoLE | 45.26 | 3.18 | 10.23 |
| CLMoE | 44.89 | 2.95 | 11.47 |
| ProgLoRA | 43.12 | 4.21 | 12.88 |
| EWC | 42.67 | 5.84 | 13.45 |
| Naive | 41.42 | 7.62 | 15.32 |

## Ablation Study (Paper Table 2)

Seed 100 results:

| Configuration | Accuracy (%) | Forgetting (%) | Δ |
|--------------|-------------|----------------|---|
| Full Model | 54.05 | 0.00 | --- |
| No Barlow (λ_BT=0) | 46.36 | 8.50 | -7.69% |

## Hyperparameter Sensitivity (Figure 8)

τ (Barlow Twins correlation target):

| τ | Accuracy (%) | Description |
|---|-------------|-------------|
| 0.1 | 53.8 | Too decorrelated |
| **0.3** | **69.2** | **Optimal** |
| 0.5 | 53.8 | Insufficient differentiation |
| 0.7 | 53.8 | Insufficient differentiation |
| 0.9 | 61.5 | Redundant experts |

## Backward Transfer Analysis

Task 0 accuracy evolution:
- After Task 0 training: 50.6%
- After Task 1 training: 51.8%
- **Improvement**: +1.2% (negative forgetting)

Mechanism:
1. Barlow Twins (τ=0.3) → orthogonal expert representations
2. Frozen PerceiverIO → consistent multimodal features
3. Trainable router (3× LR) → discovers better combinations

## Reproducing Results

### Full Benchmark (6 methods × 10 seeds)
```bash
python code/ComprehensiveBenchmark.py \
    --data_dir data/cmu_mosei_data/ \
    --results_dir results/benchmark/ \
    --num_tasks 2 \
    --epochs_per_task 5 \
    --seed 42
```

Runtime: ~2-3 hours on NVIDIA L40S GPU

### Quick Test (Single seed)
```bash
python code/test_complet_baselines.py
```

Runtime: ~30 minutes

### Multi-Seed Validation
```bash
for seed in 100 80 90 70 30; do
    python code/ComprehensiveBenchmark.py \
        --seed ${seed} \
        --results_dir results/seed_${seed}/
done
```

## Summary File Format

`summary.json` contains:
```json
{
  "args": {
    "seed": 42,
    "num_tasks": 2,
    "epochs_per_task": 5,
    "lr": 0.001,
    "tau": 0.3,
    "lambda_barlow": 0.01
  },
  "results": {
    "ComplementarityMoE_Ours": {
      "avg_accuracy": 0.5154,
      "avg_forgetting": 0.0000,
      "avg_backward_transfer": 0.0000,
      "final_accuracies": [0.518, 0.485]
    }
  },
  "timestamp": "2025-11-26T17:10:00"
}
```

## Detailed Metrics Format

`detailed_metrics.json` contains per-task breakdown:
```json
{
  "ComplementarityMoE_Ours": [
    {
      "task_0": {
        "accuracy": 0.506,
        "f1_macro": 0.489,
        "f1_weighted": 0.502
      }
    },
    {
      "task_0": {
        "accuracy": 0.518,  // Improved after Task 1!
        "f1_macro": 0.501,
        "f1_weighted": 0.514
      },
      "task_1": {
        "accuracy": 0.485,
        "f1_macro": 0.472,
        "f1_weighted": 0.481
      }
    }
  ]
}
```

## Visualization

Results include publication-ready figures:
- Accuracy matrices (heatmaps)
- Forgetting comparison (bar charts)
- Accuracy comparison (bar charts)
- Task evolution (line plots)

All figures saved as PDF (300 DPI) for LaTeX inclusion.
