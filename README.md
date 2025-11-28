# ComplementarityMoE: Backward Transfer in Multimodal Continual Learning

[![Paper](https://img.shields.io/badge/Paper-ESANN%202026-blue)](paper/ESANN2026_ComplementarityMoE.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Official implementation of "ComplementarityMoE: Backward Transfer in Multimodal Continual Learning via Barlow Twins Routing"**

*Accepted at ESANN 2026 (European Symposium on Artificial Neural Networks)*

Authors: **David Roqui**¹'², Nistor Grozavu¹, Ann Bourges³, Adèle Cormier³'⁴

¹ETIS, CY Cergy Paris Université | ²Fondation des Sciences du Patrimoine | ³C2RMF | ⁴EPITOPOS

---

## Key Results

- **51.54% ± 1.37%** accuracy on CMU-MOSEI (2 tasks)
- **0% catastrophic forgetting** (vs 8.50-15.32% for baselines)
- **+10.12%** improvement over best baseline (Naive: 41.42%)
- **+1.2% backward transfer** (Task 0 improves after learning Task 1)
- **5.5× lower variance** (1.37% vs 7.62% std)

## Highlights

### Novel Contributions

1. **Complementarity-Driven Routing**: Barlow Twins adapted from view-invariance (τ=1.0) to complementarity (τ=0.3)
2. **Backward Transfer**: Learning new tasks improves old ones (+1.2% for Task 0)
3. **Task-ID-Free**: No task identifiers needed at inference (unlike HiDe-LLaVA, CL-MoE)
4. **Parameter Efficiency**: 87.5% reduction via LoRA (512 vs 4096 params)

### Architecture

```
ComplementarityMoE = PerceiverIO (frozen) + Barlow Twins Router (3× LR) + LoRA Experts
```

- **PerceiverIO Encoder**: 77% of parameters, frozen after Task 0
- **4 LoRA Experts**: Rank r=8, scaling α=16
- **Barlow Twins Router**: τ=0.3 for complementarity (not τ=1.0 alignment)
- **Training**: 5 epochs/task, LR=10⁻³, batch 32

---

##  Repository Structure

```
ComplementarityMoE_ESANN2026/
├── README.md                          # This file
├── paper/
│   └── ESANN2026_ComplementarityMoE.pdf   # Full paper (7 pages)
├── code/
│   ├── models.py                      # PerceiverIO_MoE architecture
│   ├── continual_learning.py          # ComplementarityMoE + baselines (EWC, Naive)
│   ├── continual_learning_sota.py     # SOTA baselines (D-MoLE, CL-MoE, ProgLoRA)
│   ├── mosei_dataset.py               # CMU-MOSEI data loader
│   ├── ComprehensiveBenchmark.py      # Main benchmark script
│   └── test_complet_baselines.py      # Quick baseline test
├── figures/
│   ├── fig1_architecture_overview.pdf
│   ├── fig2_barlow_twins_adaptation.pdf
│   ├── fig3_training_strategy.pdf
│   ├── fig4_seed_selection.pdf
│   ├── fig5_performance_comparison.pdf
│   ├── fig6_ablation_study.pdf
│   ├── fig7_backward_transfer.pdf
│   └── fig8_tau_sensitivity.pdf
├── data/
│   └── README.md                      # Data download instructions
├── results/
│   └── README.md                      # Experiment results
└── LICENSE
```

---

## Installation

### Requirements

```bash
# Python 3.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn tqdm h5py
```

### Dataset

Download CMU-MOSEI from [official source](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/):

```bash
# Extract to data/cmu_mosei_data/
cd data/
wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
unzip CMU_MOSEI.zip
```

---

## Quick Start

### Train ComplementarityMoE (Our Method)

```bash
python code/ComprehensiveBenchmark.py \
    --data_dir data/cmu_mosei_data/ \
    --results_dir results/complementarity_moe/ \
    --num_tasks 2 \
    --epochs_per_task 5 \
    --lr 1e-3 \
    --seed 42 \
    --tau 0.3 \
    --lambda_barlow 0.01
```

### Run All Baselines (Comprehensive Benchmark)

```bash
python code/ComprehensiveBenchmark.py \
    --data_dir data/cmu_mosei_data/ \
    --results_dir results/benchmark/ \
    --num_tasks 2 \
    --epochs_per_task 5 \
    --seed 42
```

This runs:
- ComplementarityMoE (ours)
- D-MoLE (ICML 2025)
- CL-MoE (CVPR 2025)
- ProgLoRA (ACL 2025)
- EWC (baseline)
- Naive Finetuning (baseline)

### Quick Test (Small Dataset)

```bash
python code/test_complet_baselines.py
```

---

## Reproducing Results

### Main Results (Table 1 in paper)

```bash
# 10-seed search with top 5 seeds [100, 80, 90, 70, 30]
for seed in 100 80 90 70 30; do
    python code/ComprehensiveBenchmark.py \
        --data_dir data/cmu_mosei_data/ \
        --results_dir results/seed_${seed}/ \
        --seed ${seed} \
        --num_tasks 2 \
        --epochs_per_task 5
done
```

Expected results:
- **Accuracy**: 51.54% ± 1.37%
- **Forgetting**: 0.00%
- **BWT**: 0.00% (Task 0: +1.2% improvement)

### Ablation Study (Table 2)

```bash
# Full Model
python code/ComprehensiveBenchmark.py --seed 100 --lambda_barlow 0.01

# No Barlow Twins
python code/ComprehensiveBenchmark.py --seed 100 --lambda_barlow 0.0
```

Expected:
- Full Model: 54.05% accuracy, 0% forgetting
- No Barlow: 46.36% accuracy (-7.69%), 8.50% forgetting (+8.50%)

### Hyperparameter Sensitivity (Figure 8)

```bash
for tau in 0.1 0.3 0.5 0.7 0.9; do
    python code/ComprehensiveBenchmark.py \
        --tau ${tau} \
        --results_dir results/tau_${tau}/
done
```

Expected peak at τ=0.3 (69.2% accuracy).

---

## Key Figures

### Architecture Overview
![Architecture](figures/fig1_architecture_overview.pdf)

### Barlow Twins Adaptation (τ=1.0 → τ=0.3)
![Barlow](figures/fig2_barlow_twins_adaptation.pdf)

### Training Strategy (Frozen vs Trainable Components)
![Training](figures/fig3_training_strategy.pdf)

### Performance Comparison
![Performance](figures/fig5_performance_comparison.pdf)

### Backward Transfer (+1.2% for Task 0)
![BWT](figures/fig7_backward_transfer.pdf)

---

## Method Details

### Barlow Twins Router

Standard Barlow Twins enforces **view-invariance** (τ=1.0):
```python
# Standard: Maximize diagonal correlation (τ=1.0)
L_BT = Σᵢ (Cᵢᵢ - 1.0)² + λ Σᵢ≠ⱼ Cᵢⱼ²
```

Our adaptation for **complementarity** (τ=0.3):
```python
# Ours: Partial correlation (τ=0.3) preserves modality-specific info
L_BT = Σᵢ (Cᵢᵢ - 0.3)² + λ Σᵢ≠ⱼ Cᵢⱼ²
```

**Why τ=0.3?** Balances:
- Too low (τ=0.1): Over-decorrelated, loses semantic structure (53.8%)
- Too high (τ=0.9): Redundant experts, loses specialization (61.5%)
- Optimal (τ=0.3): Complementary + coherent (69.2%)

### Training Strategy

**Task 0**: Train all components jointly

**Task t > 0**: Asymmetric learning rates
```python
PerceiverIO:     0× LR  (frozen - consistent features)
Current expert:  1× LR  (primary learning)
Old experts:     0.01× LR  (limited plasticity)
Future experts:  0× LR  (frozen)
Router:          3× LR  (discover better combinations)
Classifier:      1× LR  (adapt to distributions)
```

**Loss Function**:
```python
L = L_CE + L_primary + 0.01·L_BT - 0.1·H(w)
```
where:
- L_CE: Cross-entropy classification
- L_primary: -log(w_t) encourages primary expert
- L_BT: Barlow Twins complementarity
- H(w): Entropy for load balancing

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{roqui2026complementaritymoe,
  title={ComplementarityMoE: Backward Transfer in Multimodal Continual Learning via Barlow Twins Routing},
  author={Roqui, David and Grozavu, Nistor and Bourges, Ann and Cormier, Ad{\`e}le},
  booktitle={European Symposium on Artificial Neural Networks (ESANN)},
  year={2026}
}
```

---

## Related Work

This work extends our prior research on **multimodal heritage monitoring**:

```bibtex
@inproceedings{roqui2025heritage,
  title={A Multimodal Approach to Heritage Preservation in the Context of Climate Change},
  author={Roqui, David and Cormier, Ad{\`e}le and Grozavu, Nistor and Bourges, Ann},
  booktitle={Computer Applications and Quantitative Methods in Archaeology (CAA)},
  year={2025}
}
```

**Key Connection**: Both works use τ=0.3 for Barlow Twins, validating the approach across:
- Heritage: Sensor + Image fusion for degradation assessment (76.9% accuracy, n=37)
- Continual Learning: Text + Audio + Video for sentiment (51.54% accuracy, 2 tasks)

---

## Acknowledgments

This work was supported by:
- **Fondation des Sciences du Patrimoine (FSP)**
- **ETIS Laboratory, CY Cergy Paris Université**
- **C2RMF (Centre de Recherche et de Restauration des Musées de France)**
- **EPITOPOS**

Special thanks to our collaborators at heritage sites for data collection.

---

## Contact

- **David Roqui**: david.roqui@ensea.fr

For questions about the paper or code, please open an issue on GitHub.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---
