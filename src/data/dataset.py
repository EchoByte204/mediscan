"""
Custom PyTorch Dataset for COVID-19 diagnosis
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class COVID19Dataset(Dataset):
    """
    Custom Dataset for COVID-19 chest X-ray images
    """
    
    def __init__(self, root_dir, transform=None, use_clahe=False):
        """
        Args:
            root_dir (str): Directory with all the images organized by class
            transform (callable, optional): Optional transform to be applied on images
            use_clahe (bool): Whether to apply CLAHE preprocessing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_clahe = use_clahe
        
        # Get class names from directory structure
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
        
        print(f"✅ Loaded {len(self.images)} images from {len(self.classes)} classes")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of classes in dataset"""
        from collections import Counter
        label_counts = Counter(self.labels)
        
        print(f"\nClass Distribution in {self.root_dir}:")
        print("-" * 50)
        for cls, idx in self.class_to_idx.items():
            count = label_counts[idx]
            percentage = (count / len(self.labels)) * 100
            print(f"  {cls:20} : {count:5} ({percentage:5.2f}%)")
        print("-" * 50)
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE if requested
        if self.use_clahe:
            from src.data.preprocessing import MedicalImagePreprocessor
            image = MedicalImagePreprocessor.apply_clahe(image)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced dataset
        Useful for weighted loss or weighted sampling
        
        Returns:
            torch.Tensor: Weights for each class
        """
        from collections import Counter
        
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weights (inverse frequency)
        weights = []
        for idx in range(len(self.classes)):
            count = label_counts[idx]
            weight = total_samples / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """
        Calculate sample weights for weighted random sampling
        
        Returns:
            list: Weight for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        
        return sample_weights


class DataLoaderFactory:
    """
    Factory class to create DataLoaders with proper configuration
    """
    
    @staticmethod
    def create_dataloaders(
        train_dir,
        val_dir,
        test_dir,
        train_transform,
        val_transform,
        batch_size=32,
        num_workers=4,
        use_weighted_sampling=True,
        use_clahe=False
    ):
        """
        Create train, validation, and test DataLoaders
        
        Args:
            train_dir (str): Path to training data
            val_dir (str): Path to validation data
            test_dir (str): Path to test data
            train_transform: Training transformations
            val_transform: Validation/test transformations
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            use_weighted_sampling (bool): Use weighted sampling to handle class imbalance
            use_clahe (bool): Apply CLAHE preprocessing
            
        Returns:
            tuple: (train_loader, val_loader, test_loader, class_names)
        """
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = COVID19Dataset(train_dir, transform=train_transform, use_clahe=use_clahe)
        val_dataset = COVID19Dataset(val_dir, transform=val_transform, use_clahe=use_clahe)
        test_dataset = COVID19Dataset(test_dir, transform=val_transform, use_clahe=use_clahe)
        
        # Create samplers for handling class imbalance
        if use_weighted_sampling:
            print("\n✅ Using weighted random sampling for training")
            from torch.utils.data import WeightedRandomSampler
            
            sample_weights = train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # Validation and test loaders (no sampling needed)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\n✅ DataLoaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, train_dataset.classes


# Test the module
# Test the module
if __name__ == "__main__":
    from preprocessing import DataAugmentation
    
    # Paths (Windows absolute paths)
    base_dir = r'C:\Users\Asus\Desktop\New folder\mediscan-covid19'
    train_dir = os.path.join(base_dir, 'data', 'processed', 'train')
    val_dir = os.path.join(base_dir, 'data', 'processed', 'val')
    test_dir = os.path.join(base_dir, 'data', 'processed', 'test')
    
    # Transforms
    train_transform = DataAugmentation.get_train_transform()
    val_transform = DataAugmentation.get_val_transform()
    
    # Create loaders
    train_loader, val_loader, test_loader, classes = DataLoaderFactory.create_dataloaders(
        train_dir, val_dir, test_dir,
        train_transform, val_transform,
        batch_size=16,
        num_workers=2
    )
    
    print(f"\n✅ Classes: {classes}")
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"✅ Batch shape: {images.shape}")
    print(f"✅ Labels shape: {labels.shape}")