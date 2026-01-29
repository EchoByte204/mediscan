"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualizes which regions of the image influence the model's decision
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image


class GradCAM:
    """
    Implements Grad-CAM for visualizing CNN decisions
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained PyTorch model
            target_layer: Layer to visualize (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor (1, 3, 224, 224)
            target_class: Target class index (None = predicted class)
            
        Returns:
            cam: Activation map as numpy array
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get prediction
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
        if target_class is None:
            target_class = predicted_class.item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class.item(), confidence.item()
    
    def overlay_heatmap(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM heatmap on original image
        
        Args:
            image: Original image as numpy array (H, W, 3)
            cam: Class activation map
            alpha: Transparency of heatmap
            colormap: OpenCV colormap
            
        Returns:
            overlay: Image with heatmap overlay
        """
        # Resize CAM to image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = heatmap * alpha + image * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay, heatmap
    
    def remove_hooks(self):
        """Remove registered hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM based on model architecture
    
    Args:
        model: PyTorch model
        model_name: Name of model architecture
        
    Returns:
        target_layer: Layer to visualize
    """
    if model_name == 'resnet50':
        # Last convolutional layer in ResNet50
        return model.backbone.layer4[-1]
    elif model_name == 'densenet121':
        # Last layer in DenseNet121
        return model.backbone.features[-1]
    elif model_name == 'efficientnet_b0':
        # Last convolutional layer in EfficientNet
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_gradcam(
    model,
    image_path,
    transform,
    device,
    class_names,
    model_name='resnet50',
    save_path=None
):
    """
    Complete Grad-CAM visualization pipeline
    
    Args:
        model: Trained model
        image_path: Path to input image
        transform: Image preprocessing transform
        device: Device (cuda/cpu)
        class_names: List of class names
        model_name: Model architecture name
        save_path: Optional path to save visualization
        
    Returns:
        dict: Results including prediction, confidence, and visualizations
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, predicted_class, confidence = gradcam.generate_cam(input_tensor)
    
    # Create overlay
    overlay, heatmap = gradcam.overlay_heatmap(original_np, cam, alpha=0.4)
    
    # Clean up
    gradcam.remove_hooks()
    
    # Prepare results
    results = {
        'predicted_class': predicted_class,
        'predicted_label': class_names[predicted_class],
        'confidence': confidence * 100,
        'original_image': original_np,
        'heatmap': heatmap,
        'overlay': overlay,
        'cam': cam
    }
    
    # Save if requested
    if save_path:
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure with all visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Overlay\n{results["predicted_label"]}: {results["confidence"]:.2f}%',
            fontsize=12, fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


# Test module
if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("Available classes:")
    print("  - GradCAM: Main Grad-CAM implementation")
    print("  - get_target_layer: Get target layer for different architectures")
    print("  - visualize_gradcam: Complete visualization pipeline")