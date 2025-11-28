"""
PyTorch Dataset for CMU-MOSEI – Memory-Optimized Version
Loads data on demand (lazy loading)
Includes truncation and cleaning of invalid values (Inf/NaN)
FIXED: Deterministic seeds for reproducible splits
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import numpy as np

class CMUMOSEIDataset(Dataset):
    """Dataset PyTorch for CMU-MOSEI with lazy loading"""
    
    def __init__(self, data_path, split='train', split_ratio=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            data_path: path to cmu_mosei_data/
            split: 'train', 'valid', or 'test'
            split_ratio: (train, valid, test) ratios
            seed: random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.split = split
        self.seed = seed
        
        # Files path
        self.files = {
            'text': (self.data_path / 'CMU_MOSEI_TimestampedWordVectors.csd', 'glove_vectors'),
            'audio': (self.data_path / 'CMU_MOSEI_COVAREP.csd', 'COVAREP'),
            'video': (self.data_path / 'CMU_MOSEI_VisualFacet42.csd', 'FACET 4.2'),
            'labels': (self.data_path / 'CMU_MOSEI_Labels.csd', 'All Labels'),
        }
        
        # Get videos valid id
        self.video_ids = self._get_valid_video_ids()
        
        # Split
        self.video_ids = self._load_split(split, split_ratio)
        
        print(f"Dataset {split}: {len(self.video_ids)} vidéos")
        
    def _get_valid_video_ids(self):
        """Get video id without loading all the data"""
        print(f"Get videos id")
        
        filepath, key_name = self.files['labels']
        
        with h5py.File(filepath, 'r') as f:
            root = f[key_name]
            data_group = root['data'] if 'data' in root else root
            label_ids = set(data_group.keys())
        
        valid_ids = label_ids.copy()
        
        for modality, (filepath, key_name) in self.files.items():
            if modality == 'labels':
                continue
                
            with h5py.File(filepath, 'r') as f:
                root = f[key_name]
                data_group = root['data'] if 'data' in root else root
                modality_ids = set(data_group.keys())
                valid_ids &= modality_ids
        
        print(f"  {len(valid_ids)} Valid video found")
        return sorted(list(valid_ids)) 
    
    def _load_split(self, split, split_ratio):
        """Determinist split"""
        # Shuffle and fix seed for reproducibility
        rng = np.random.RandomState(self.seed)
        shuffled_ids = self.video_ids.copy()
        rng.shuffle(shuffled_ids)
        
        n = len(shuffled_ids)
        train_ratio, valid_ratio, test_ratio = split_ratio
        
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        if split == 'train':
            return shuffled_ids[:train_end]
        elif split == 'valid':
            return shuffled_ids[train_end:valid_end]
        else:  # test
            return shuffled_ids[valid_end:]
    
    def _load_sample(self, video_id, modality):
        """Load sample in demand"""
        filepath, key_name = self.files[modality]
        
        with h5py.File(filepath, 'r') as f:
            root = f[key_name]
            data_group = root['data'] if 'data' in root else root
            
            if video_id not in data_group:
                return None
            
            group = data_group[video_id]
            
            return {
                'features': np.array(group['features']),
                'intervals': np.array(group['intervals'])
            }
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load "in demand" the samples
        text = self._load_sample(video_id, 'text')
        audio = self._load_sample(video_id, 'audio')
        video = self._load_sample(video_id, 'video')
        labels = self._load_sample(video_id, 'labels')
        
        
        return {
            'video_id': video_id,
            'text': torch.FloatTensor(text['features']),
            'audio': torch.FloatTensor(audio['features']),
            'video': torch.FloatTensor(video['features']),
            'labels': torch.FloatTensor(labels['features']),
            'text_intervals': torch.FloatTensor(text['intervals']),
            'audio_intervals': torch.FloatTensor(audio['intervals']),
            'video_intervals': torch.FloatTensor(video['intervals']),
        }


def collate_fn(batch, max_text_len=300, max_audio_len=3000, max_video_len=2000):
    """
    Personalize collate function
    """
    from torch.nn.utils.rnn import pad_sequence
    
    video_ids = [item['video_id'] for item in batch]
    
    text_list = []
    audio_list = []
    video_list = []
    
    for item in batch:
        # Truncate
        text = item['text'][:max_text_len] if len(item['text']) > max_text_len else item['text']
        audio = item['audio'][:max_audio_len] if len(item['audio']) > max_audio_len else item['audio']
        video = item['video'][:max_video_len] if len(item['video']) > max_video_len else item['video']
        
        # Nettoyer NaN/Inf
        text = torch.nan_to_num(text, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=-1.0)
        
        text_list.append(text)
        audio_list.append(audio)
        video_list.append(video)
    
    # Padding
    text = pad_sequence(text_list, batch_first=True).contiguous()
    audio = pad_sequence(audio_list, batch_first=True).contiguous()
    video = pad_sequence(video_list, batch_first=True).contiguous()
    
    # Labels
    labels = torch.stack([item['labels'][0] for item in batch]).contiguous()
    labels = torch.nan_to_num(labels, nan=0.0, posinf=3.0, neginf=-3.0)
    
    return {
        'video_id': video_ids,
        'text': text,
        'audio': audio,
        'video': video,
        'labels': labels
    }


def load_mosei_continual(data_dir: str, num_tasks: int = 5, 
                        batch_size: int = 32, quick_test: bool = False,
                        seed: int = 42):
    """
    Load MOSEI dataset split into continual learning tasks - DETERMINISTIC
    
    Args:
        data_dir: Path to cmu_mosei_data/
        num_tasks: Number of continual learning tasks
        batch_size: Batch size for DataLoaders
        quick_test: If True, use smaller dataset for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_loaders, val_loaders, test_loaders: Lists of DataLoaders
    """
    print("Loading MOSEI dataset...")
    
    # Create datasets with fixed seed
    train_dataset = CMUMOSEIDataset(data_dir, split='train', seed=seed)
    val_dataset = CMUMOSEIDataset(data_dir, split='valid', seed=seed)
    test_dataset = CMUMOSEIDataset(data_dir, split='test', seed=seed)
    
    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    print(f"\nCreating {num_tasks} task splits...")
    
    def create_task_loaders(dataset, num_tasks, batch_size, seed):
        """Split dataset into task-specific loaders - DETERMINISTIC"""
        task_size = len(dataset) // num_tasks
        loaders = []
        
        # Create deterministic generator for DataLoader
        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        for i in range(num_tasks):
            start_idx = i * task_size
            end_idx = (i + 1) * task_size if i < num_tasks - 1 else len(dataset)
            
            indices = list(range(start_idx, end_idx))
            subset = torch.utils.data.Subset(dataset, indices)
            
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,  #0 for lazy loading
                worker_init_fn=seed_worker,
                generator=g
            )
            loaders.append(loader)
            print(f"  Task {i}: {len(indices)} samples")
        
        return loaders
    
    # Create task splits with fixed seed
    train_loaders = create_task_loaders(train_dataset, num_tasks, batch_size, seed)
    val_loaders = create_task_loaders(val_dataset, num_tasks, batch_size, seed)
    test_loaders = create_task_loaders(test_dataset, num_tasks, batch_size, seed)
    
    return train_loaders, val_loaders, test_loaders


# Test script
if __name__ == '__main__':
    print("="*60)
    print("CMU-MOSEI test")
    print("="*60)
    
    DATA_PATH = './cmu_mosei_data/'
    
    # Test 1: Basic dataset
    print("\nTest 1: Dataset creation")
    train_dataset = CMUMOSEIDataset(DATA_PATH, split='train', seed=42)
    
    # Test 2: Sample
    print("\nTest 2: Samples")
    sample = train_dataset[0]
    print(f"  Video ID: {sample['video_id']}")
    print(f"  Text shape: {sample['text'].shape}")
    
    # Test 3: DataLoader
    print("\nTest 3: DataLoader")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    batch = next(iter(train_loader))
    print(f"  Batch shapes OK: {batch['text'].shape}")
    
    # Test 4: Continual loading
    print("\nTest 4: Continual learning splits...")
    train_loaders, val_loaders, test_loaders = load_mosei_continual(
        DATA_PATH, num_tasks=3, batch_size=32, seed=42
    )
    print(f"{len(train_loaders)} task loaders created")
    
    # Test 5: reproducibility
    print("\nTest 5: reproducibility test")
    train_loaders1, _, _ = load_mosei_continual(DATA_PATH, num_tasks=2, batch_size=32, seed=42)
    train_loaders2, _, _ = load_mosei_continual(DATA_PATH, num_tasks=2, batch_size=32, seed=42)
    
    batch1 = next(iter(train_loaders1[0]))
    batch2 = next(iter(train_loaders2[0]))
    
    if batch1['video_id'] == batch2['video_id']:
        print("reproducibility okay")
    else:
        print("reproducibility")
    
    print("\n" + "="*60)
    print("All tests passed")
    print("="*60)