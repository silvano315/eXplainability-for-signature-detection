import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.utils.logger_setup import get_logger

logger = get_logger(__name__)

class CEDARDataset(Dataset):
    """
    PyTorch Dataset for CEDAR signature dataset with smart loading.
    """
    
    def __init__(self,
                 data_path: Path, 
                 metadata: Dict[str, Dict[str, Any]],
                 transform: Optional[transforms.Compose] = None,
                 split_filter: Optional[str] = None) -> None:
        """
        Initialize CEDAR dataset.
        
        Args:
            data_path: Path to dataset root directory
            metadata: Dataset metadata dictionary
            transform: Optional torchvision transforms
            split_filter: Filter by split ('train', 'val', 'test', or None for all)
        """
        self.data_path = data_path
        self.metadata = metadata
        self.transform = transform
        self.split_filter = split_filter

        if split_filter:
            self.filtered_metadata = {
                filename: info for filename, info in metadata.items()
                if info.get("split") == split_filter
            }
        else:
            self.filtered_metadata = metadata

        # filename list and label mapping
        self.filenames = list(self.filtered_metadata.keys())
        self.label_map = {"original": 0, "forgeries": 1}
        
        logger.info(f"Dataset initialized: {len(self.filenames)} images" + 
                   (f" (split: {split_filter})" if split_filter else ""))
        
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of images in the dataset
        """
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        filename = self.filenames[idx]
        info = self.filtered_metadata[filename]

        if info["type"] == "original":
            img_path = self.data_path / "full_org" / filename
        elif info["type"] == "forgeries":
            img_path = self.data_path / "full_forg" / filename
        else:
            raise ValueError(f"Unknown signature type: {info['type']}")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {filename}: {e}")
            # dummy image
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        label = self.label_map[info["type"]]
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get class distribution for current split."""
        counts = defaultdict(int)
        for info in self.filtered_metadata.values():
            counts[info["type"]] += 1
        return dict(counts)
    
    def get_subject_info(self) -> Dict[str, Any]:
        """Get subject information for current split."""
        subjects = set()
        for info in self.filtered_metadata.values():
            if info["subject_id"] != -1:
                subjects.add(info["subject_id"])
        
        return {
            "unique_subjects": len(subjects),
            "subject_ids": sorted(list(subjects))
        }
    
def create_balanced_splits(metadata: Dict[str, Dict[str, Any]], 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Create balanced train/val/test splits for signature verification.
    
    Args:
        metadata: Dataset metadata dictionary
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Updated metadata with split assignments
    """
    logger.info("Creating balanced splits for signature verification...")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Separate by signature type
    original_files = []
    forged_files = []
    
    for filename, info in metadata.items():
        if info["type"] == "original":
            original_files.append(filename)
        elif info["type"] == "forged":
            forged_files.append(filename)
    
    logger.info(f"Found {len(original_files)} original and {len(forged_files)} forged signatures")
    
    random.seed(random_seed)
    random.shuffle(original_files)
    random.shuffle(forged_files)
    
    # Calculate split points for each type
    def split_files(files, train_r, val_r):
        n = len(files)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return files[:train_end], files[train_end:val_end], files[val_end:]
    
    orig_train, orig_val, orig_test = split_files(original_files, train_ratio, val_ratio)
    forg_train, forg_val, forg_test = split_files(forged_files, train_ratio, val_ratio)
    
    # Update metadata with split assignments
    updated_metadata = metadata.copy()    
    split_assignment = {
        **{f: "train" for f in orig_train + forg_train},
        **{f: "val" for f in orig_val + forg_val},
        **{f: "test" for f in orig_test + forg_test}
    }
    
    for filename, split in split_assignment.items():
        updated_metadata[filename]["split"] = split
    
    split_counts = defaultdict(lambda: {"original": 0, "forgeries": 0})
    for filename, info in updated_metadata.items():
        split = info.get("split", "unknown")
        sig_type = info["type"]
        split_counts[split][sig_type] += 1
    
    logger.info("Split distribution:")
    for split in ["train", "val", "test"]:
        orig_count = split_counts[split]["original"]
        forg_count = split_counts[split]["forgeries"]
        total = orig_count + forg_count
        logger.info(f"  {split}: {total} images (original: {orig_count}, forged: {forg_count})")
    
    return updated_metadata

def get_default_transforms(input_size: Tuple[int, int] = (224, 224),
                          normalize: bool = True) -> Dict[str, transforms.Compose]:
    """
    Get default transforms for CEDAR dataset.
    
    Args:
        input_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Dictionary with train and val transforms
    """
    base_transforms = [
        transforms.Resize(input_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        base_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        )
    
    train_transforms = [
        transforms.Resize(input_size),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ]
    
    if normalize:
        train_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        )
    
    return {
        "train": transforms.Compose(train_transforms),
        "val": transforms.Compose(base_transforms),
        "test": transforms.Compose(base_transforms)
    }

def create_dataloaders(data_path: Path,
                      metadata: Dict[str, Dict[str, Any]], 
                      batch_size: int = 32,
                      num_workers: int = 4,
                      input_size: Tuple[int, int] = (224, 224),
                      normalize: bool = True) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders from metadata.
    
    Args:
        data_path: Path to dataset root directory
        metadata: Dataset metadata with split assignments
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        input_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Dictionary with train/val/test dataloaders
    """
    logger.info("Creating dataloaders...")
    
    transforms_dict = get_default_transforms(input_size, normalize)
    
    datasets = {}
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        datasets[split] = CEDARDataset(
            data_path=data_path,
            metadata=metadata,
            transform=transforms_dict[split],
            split_filter=split
        )
        
        shuffle = (split == "train")
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train")  # Drop last incomplete batch for training
        )
        
        class_counts = datasets[split].get_class_counts()
        subject_info = datasets[split].get_subject_info()
        
        logger.info(f"{split.upper()} SET:")
        logger.info(f"  Images: {len(datasets[split])}")
        logger.info(f"  Classes: {class_counts}")
        logger.info(f"  Subjects: {subject_info['unique_subjects']}")
        logger.info(f"  Batches: {len(dataloaders[split])}")
    
    return dataloaders

def save_split_metadata(metadata: Dict[str, Dict[str, Any]], 
                       output_path: Path) -> None:
    """
    Save metadata with split assignments.
    
    Args:
        metadata: Metadata with split assignments
        output_path: Path to save updated metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Split metadata saved to: {output_path}")

def load_split_metadata(metadata_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata with split assignments.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Loaded metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Split metadata loaded from: {metadata_path}")
    return metadata