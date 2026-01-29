"""
Data preprocessing and augmentation for COVID-19 diagnosis
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class DataAugmentation:
    """
    Data augmentation strategies for medical images
    """
    
    @staticmethod
    def get_train_transform():
        """
        Training data augmentation
        Includes aggressive augmentation for better generalization
        """
        return transforms.Compose([
            # Resize
            transforms.Resize((224, 224)),
            
            # Geometric transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            
            # Color/Intensity augmentation
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            
            # Random perspective
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Normalize (ImageNet statistics)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Random erasing (simulates occlusions)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    
    @staticmethod
    def get_val_transform():
        """
        Validation/Test data transformation
        No augmentation, only preprocessing
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_test_transform():
        """
        Test data transformation (same as validation)
        """
        return DataAugmentation.get_val_transform()


class MedicalImagePreprocessor:
    """
    Preprocessing utilities for medical images
    """
    
    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Improves contrast in medical images
        """
        import cv2
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img_gray)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(img_rgb)
    
    @staticmethod
    def denoise_image(image, strength=10):
        """
        Apply denoising to reduce noise in medical images
        """
        import cv2
        
        img_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_np, None, strength, strength, 7, 21)
        
        return Image.fromarray(denoised)


# Quick test
if __name__ == "__main__":
    print("Testing data augmentation...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (300, 300), color='gray')
    
    # Test transforms
    train_transform = DataAugmentation.get_train_transform()
    val_transform = DataAugmentation.get_val_transform()
    
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)
    
    print(f"✅ Train transform output shape: {train_tensor.shape}")
    print(f"✅ Val transform output shape: {val_tensor.shape}")
    print("✅ Data augmentation module working!")