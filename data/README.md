# Data Directory

## CMU-MOSEI Dataset

This directory should contain the CMU-MOSEI dataset for multimodal sentiment analysis.

### Download Instructions

1. **Official Source**: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/

2. **Download Files**:
```bash
cd data/
wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
unzip CMU_MOSEI.zip
```

3. **Expected Structure**:
```
data/
└── cmu_mosei_data/
    ├── CMU_MOSEI_TimestampedWordVectors.csd    # Text (GloVe 300D)
    ├── CMU_MOSEI_COVAREP.csd                   # Audio (COVAREP 74D)
    ├── CMU_MOSEI_VisualFacet42.csd             # Video (FACET 35D)
    └── CMU_MOSEI_Labels.csd                    # 7-class sentiment labels
```

### Dataset Statistics

- **Total samples**: 23,453 video clips
- **Modalities**: Text (GloVe), Audio (COVAREP), Video (FACET)
- **Labels**: 7 classes (sentiment -3 to +3)
- **Splits**: 70% train, 15% validation, 15% test per task

### Continual Learning Setup (Paper)

- **Task 0**: 11,726 clips (first temporal half)
- **Task 1**: 11,727 clips (second temporal half)
- **Temporal split**: Prevents data leakage between tasks

### Citation

If you use CMU-MOSEI, please cite:

```bibtex
@inproceedings{zadeh2018multimodal,
  title={Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2018}
}
```

### Data Loading

The dataset is automatically loaded by `code/mosei_dataset.py`:

```python
from mosei_dataset import load_mosei_continual

train_loaders, val_loaders, test_loaders = load_mosei_continual(
    data_dir='data/cmu_mosei_data/',
    num_tasks=2,
    batch_size=32,
    seed=42
)
```

### Notes

- Files are in HDF5 format (.csd)
- Lazy loading for memory efficiency
- Automatic NaN/Inf cleaning
- Deterministic splits with fixed seed
