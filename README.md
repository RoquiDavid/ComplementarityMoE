# ComplementarityMoE: Positive Backward Transfer in Multimodal Continual Learning

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of "Orthogonality-Constrained MoE: Achieving Positive Backward Transfer in Multimodal Continual Learning"

Submitted to IJCNN 2026 (under review)

**Authors:** David Roqui¹'², Nistor Grozavu¹, Ann Bourges³, Adèle Cormier³'⁴

¹ETIS, CY Cergy Paris Université | ²Fondation des Sciences du Patrimoine | ³C2RMF | ⁴EPITOPOS

## Results on CMU-MOSEI

| Method | Accuracy (%) | Forgetting (%) | BWT (%) |
|--------|--------------|----------------|---------|
| **ComplementarityMoE (Ours)** | **51.93±1.49** | **-8.45±2.31** | **+8.45±2.31** |
| D-MoLE | 45.26±3.12 | 9.87±2.45 | -9.87±2.45 |
| CL-MoE | 44.89±3.45 | 11.05±2.78 | -11.05±2.78 |
| ProgLoRA | 43.12±4.01 | 12.34±3.12 | -12.34±3.12 |
| EWC | 42.67±3.87 | 12.98±2.98 | -12.98±2.98 |
| Naive | 41.42±7.62 | 11.76±4.23 | -11.76±4.23 |

**Key findings:**
- First demonstration of positive backward transfer in multimodal continual learning
- 6.67% improvement over D-MoLE (previous state-of-the-art)
- 5x lower training variance compared to naive fine-tuning
- Task-ID-free routing (no task identifiers needed at inference)

## Method Overview

### Novel Contributions

1. **Complementarity-constrained Barlow Twins loss** (τ=0.5): Adapts self-supervised view-invariance to enforce orthogonal expert specializations
2. **Positive backward transfer**: Router discovers improved expert combinations for old tasks when learning new ones
3. **Parameter-efficient architecture**: 87.5% reduction via LoRA experts (1,024 vs 4,096 params per expert)
4. **Selective freezing strategy**: 77% of parameters frozen after Task 0, enabling stable continual learning

### Architecture
```
ComplementarityMoE = Frozen PerceiverIO + Barlow Router (3x LR) + 4 LoRA Experts
```

**Components:**
- PerceiverIO Encoder: 920K parameters (frozen after Task 0)
- 4 LoRA Experts: r=8, α=16, 8.2K parameters each
- Barlow Twins Router: τ=0.5 for complementarity
- Training: 5 epochs/task, LR=10⁻³, batch 32

## Repository Structure
```
ComplementarityMoE/
├── README.md                          
├── models.py                          # PerceiverIO_MoE architecture
├── continual_learning.py              # ComplementarityMoE + baselines (EWC, Naive)
├── continual_learning_sota.py         # SOTA baselines (D-MoLE, CL-MoE, ProgLoRA)
├── mosei_dataset.py                   # CMU-MOSEI data loader
├── ComprehensiveBenchmark.py          # Main benchmark script
├── test_complet_baselines.py          # Quick baseline test
└── launch.sh                          # Multi-seed experiments
```

## Installation

### Requirements
```bash
conda create -n complementarity python=3.10
conda activate complementarity
pip install torch==2.0.0 numpy scikit-learn matplotlib seaborn h5py
```

### Dataset

Download CMU-MOSEI from the [official source](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/). Extract files to `cmu_mosei_data/` directory.

Required files:
- CMU_MOSEI_TimestampedWordVectors.csd (text)
- CMU_MOSEI_COVAREP.csd (audio)
- CMU_MOSEI_VisualFacet42.csd (video)
- CMU_MOSEI_Labels.csd (labels)

## Reproducing Results

### Main Results (Table I in paper)

Run experiments on 5 seeds used in the paper:
```bash
for seed in 42 80 90 100 123; do
    python ComprehensiveBenchmark.py \
        --data_dir /path/to/cmu_mosei_data \
        --results_dir results/seed_${seed} \
        --num_tasks 2 \
        --batch_size 32 \
        --epochs_per_task 5 \
        --lr 0.001 \
        --seed ${seed} \
        --tau 0.5 \
        --lambda_barlow 0.01
done
```

Expected results (mean over 5 seeds):
- Accuracy: 51.93% ± 1.49%
- Forgetting: -8.45% ± 2.31%
- Backward Transfer: +8.45% ± 2.31%

### Ablation Studies

All ablations conducted on seed 100 for computational efficiency.

**Tau parameter ablation (Table II):**
```bash
for tau in 0.1 0.2 0.3 0.5; do
    python ComprehensiveBenchmark.py \
        --seed 100 \
        --tau ${tau} \
        --results_dir results/tau_${tau}
done
```

Expected: optimal at τ=0.5 (54.06% accuracy, +11.63% BWT)

**Old expert learning rate ablation (Table III):**
```bash
for old_lr in 0.0 0.001 0.01 0.1 1.0; do
    python ComprehensiveBenchmark.py \
        --seed 100 \
        --old_expert_lr ${old_lr} \
        --results_dir results/old_lr_${old_lr}
done
```

Expected: optimal at 0.01x (54.06% accuracy)

**Expert capacity ablation (Table IV):**
```bash
for num_experts in 2 4 6 8; do
    python ComprehensiveBenchmark.py \
        --seed 100 \
        --num_experts ${num_experts} \
        --results_dir results/experts_${num_experts}
done
```

Expected: optimal at 4 experts (54.06% accuracy)

## Quick Start

### Train ComplementarityMoE
```bash
python ComprehensiveBenchmark.py \
    --data_dir cmu_mosei_data/ \
    --results_dir results/complementarity/ \
    --num_tasks 2 \
    --epochs_per_task 5 \
    --lr 0.001 \
    --seed 42 \
    --tau 0.5 \
    --lambda_barlow 0.01
```

### Run All Baselines
```bash
python ComprehensiveBenchmark.py \
    --data_dir cmu_mosei_data/ \
    --results_dir results/benchmark/ \
    --num_tasks 2 \
    --epochs_per_task 5 \
    --seed 42
```

This runs our method plus all baselines:
- D-MoLE (ICML 2025 adapted)
- CL-MoE (CVPR 2025 adapted)
- ProgLoRA (ACL 2025 adapted)
- EWC (baseline)
- Naive fine-tuning (baseline)

### Quick Test
```bash
python test_complet_baselines.py
```

## Key Hyperparameters

Based on grid search and ablation studies:

- **τ (tau)**: 0.5 (Barlow Twins complementarity constraint)
- **Router LR multiplier**: 3x (elevated for exploration)
- **Old expert LR**: 0.01x (minimal plasticity)
- **Number of experts**: 4 (2x overprovisioning)
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Lambda Barlow**: 0.01
- **Learning rate**: 0.001
- **Batch size**: 32
- **Epochs per task**: 5

## Method Details

### Barlow Twins Adaptation

Standard Barlow Twins enforces view-invariance (τ=1.0) for self-supervised learning. We adapt it for complementarity-driven expert routing:
```python
# Standard: Perfect correlation on diagonal
L_BT = Σᵢ (Cᵢᵢ - 1.0)² + λ Σᵢ≠ⱼ Cᵢⱼ²

# Ours: Partial correlation (τ=0.5) for complementarity
L_BT = Σᵢ (Cᵢᵢ - 0.5)² + λ Σᵢ≠ⱼ Cᵢⱼ²
```

This creates experts that are orthogonal (off-diagonal → 0) yet maintain collaborative capacity (diagonal = 0.5).

### Training Strategy

**Task 0:** Train all components jointly

**Task t > 0:** Selective freezing with asymmetric learning rates
```
PerceiverIO:     frozen (0x LR)      # Stable features
Current expert:  trainable (1x LR)   # Full plasticity
Old experts:     limited (0.01x LR)  # Minimal refinement
Future experts:  frozen (0x LR)      # Reserved capacity
Router:          elevated (3x LR)    # Rapid exploration
Classifier:      trainable (1x LR)   # Adapt to distribution
```

**Total loss:**
```
L = L_CE + L_primary + 0.01·L_BT - 0.1·H(w)
```

where L_CE is cross-entropy, L_primary encourages primary expert usage, L_BT is Barlow Twins complementarity, and H(w) is routing entropy for load balancing.

### Mechanism of Backward Transfer

Positive BWT occurs through three factors:

1. **Frozen PerceiverIO**: Provides stable features without distribution shift
2. **Elevated router LR (3x)**: Enables exploration to discover better expert combinations
3. **Minimal old expert plasticity (0.01x)**: Allows refinement without interference

When training Task 1, the router learns Expert 1 provides complementary features that also benefit Task 0, yielding A₀¹ > A₀⁰ (backward transfer).

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{roqui2026complementarity,
  title={Orthogonality-Constrained MoE: Achieving Positive Backward Transfer in Multimodal Continual Learning},
  author={Roqui, David and Grozavu, Nistor and Bourges, Ann and Cormier, Ad{\`e}le},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026},
  note={Under review}
}
```

## Related Work

This work extends our prior research on multimodal heritage monitoring:
```bibtex
@inproceedings{roqui2025heritage,
  title={A Multimodal Approach to Heritage Preservation in the Context of Climate Change},
  author={Roqui, David and Cormier, Ad{\`e}le and Grozavu, Nistor and Bourges, Ann},
  booktitle={Computer Applications and Quantitative Methods in Archaeology (CAA)},
  year={2025}
}
```

Both works validate τ=0.5 for Barlow Twins across different domains:
- Heritage: sensor + image fusion for degradation (76.9% accuracy, n=37)
- Continual learning: text + audio + video for sentiment (51.93% accuracy, 2 tasks)

## Acknowledgments

Supported by Fondation des Sciences du Patrimoine (FSP), ETIS Laboratory (CY Cergy Paris Université), C2RMF, and EPITOPOS.

## Contact

David Roqui: david.roqui@ensea.fr

For questions about the paper or code, please open an issue on GitHub.

## License

MIT License - see LICENSE file for details.
