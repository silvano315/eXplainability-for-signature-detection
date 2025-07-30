import numpy as np
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger_setup import get_logger


logger = get_logger()

class SignatureExplainer:
    """
    Explainable AI module for signature verification models.
    """
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        Initialize explainer with model and device.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
        """
        self.model = model.eval()
        self.device = device
        self.model = self.model.to(self.device)

        self.gradients = {}
        self.activations = {}
        self.hooks = []

        logger.info(f"SignatureExplainer initialized with model: {model.__class__.__name__}")

    def _register_hooks(self, target_layer: str) -> None:
        """
        Register forward and backward hooks for Grad-CAM.
        
        Args:
            target_layer: Name of target layer for Grad-CAM
        """
        def forward_hook(module, input, output):
            self.activations[target_layer] = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients[target_layer] = grad_output[0]

        target_module = None
        for name, module in self.model.named_modules():
            if name == target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer {target_layer} not found in model.")
        
        # Register hooks
        fh = target_module.register_forward_hook(forward_hook)
        bh = target_module.register_full_backward_hook(backward_hook)
        self.hooks.extend([fh, bh])

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()

    def _get_last_conv_layer(self) -> str:
        """
        Get the name of the last convolutional layer.
        
        Returns:
            Name of last conv layer
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                conv_layers.append(name)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in model")
        
        last_layer = conv_layers[-1]
        logger.info(f"Using last conv layer: {last_layer}")
        return last_layer
    
    def grad_cam(self, 
                 image: torch.Tensor, 
                 target_class: Optional[int] = None,
                 target_layer: Optional[str] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            target_layer: Target layer name (None for last conv layer)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        if target_layer is None:
            target_layer = self._get_last_conv_layer()

        self._register_hooks(target_layer)

        try:
            # Forward pass
            image = image.to(self.device)
            image.requires_grad_(True)
            
            outputs = self.model(image)
            
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            target_score = outputs[0, target_class]
            target_score.backward()
            
            # Gradients and activations
            gradients = self.gradients[target_layer]
            activations = self.activations[target_layer]
            
            # Calculate weights: global average pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Generate cam
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)  # Only positive influence
            
            # Resize to input size
            cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
            
            # Normalize
            cam = cam.squeeze().cpu().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            logger.info(f"Grad-CAM generated for class {target_class} using layer {target_layer}")
            return cam
            
        finally:
            self._remove_hooks()
    
    def integrated_gradients(self, 
                           image: torch.Tensor,
                           target_class: Optional[int] = None,
                           steps: int = 50,
                           baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            steps: Number of integration steps
            baseline: Baseline image (None for zero baseline)
            
        Returns:
            Attribution map as numpy array
        """
        image = image.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        else:
            baseline = baseline.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(image)
                target_class = outputs.argmax(dim=1).item()
        
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        gradients = []
        for alpha in alphas:
            # Interpolated image
            interpolated = (baseline + alpha * (image - baseline)).detach().requires_grad_(True) #TBT
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(interpolated)
            target_score = outputs[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward()
            
            # Store gradients
            gradients.append(interpolated.grad.clone())
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (image - baseline) * avg_gradients
        
        # Sum across channels and normalize
        attribution = integrated_gradients.sum(dim=1).squeeze().cpu().detach().numpy()
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        logger.info(f"Integrated Gradients generated for class {target_class} with {steps} steps")
        return attribution
    
    def occlusion_map(self, 
                     image: torch.Tensor,
                     target_class: Optional[int] = None,
                     patch_size: int = 15,
                     stride: int = 8) -> np.ndarray:
        """
        Generate occlusion sensitivity map.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            patch_size: Size of occlusion patch
            stride: Stride for sliding window
            
        Returns:
            Occlusion sensitivity map as numpy array
        """
        image = image.to(self.device)
        
        with torch.no_grad():
            original_output = self.model(image)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_score = original_output[0, target_class].item()
        
        _, _, height, width = image.shape
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros((height, width))
        
        # Slide occlusion patch
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Create occluded image
                occluded_image = image.clone()
                occluded_image[:, :, y:y+patch_size, x:x+patch_size] = 0
                
                # Get prediction for occluded image
                with torch.no_grad():
                    occluded_output = self.model(occluded_image)
                    occluded_score = occluded_output[0, target_class].item()
                
                # Calculate sensitivity
                sensitivity = original_score - occluded_score
                
                # Assign sensitivity to patch region
                sensitivity_map[y:y+patch_size, x:x+patch_size] = max(
                    sensitivity_map[y:y+patch_size, x:x+patch_size].max(), 
                    sensitivity
                )
        
        # Normalize
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-8)
        
        logger.info(f"Occlusion map generated for class {target_class} with patch size {patch_size}")
        return sensitivity_map
    
    def compare_explanations(self, 
                           image: torch.Tensor,
                           target_class: Optional[int] = None,
                           target_layer: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate all explanation methods for comparison.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Dictionary with all explanation maps
        """
        logger.info("Generating all explanation methods...")
        
        explanations = {}

        #TBT
        image_gc = image.clone().detach()
        image_ig = image.clone().detach()
        image_occ = image.clone().detach()
        
        # Grad-CAM
        explanations['grad_cam'] = self.grad_cam(image_gc, target_class, target_layer)
        
        # Integrated Gradients
        explanations['integrated_gradients'] = self.integrated_gradients(image_ig, target_class)
        
        # Occlusion Map
        explanations['occlusion'] = self.occlusion_map(image_occ, target_class)
        
        return explanations
    
    def get_model_layers(self) -> List[str]:
        """
        Get list of all layer names in the model.
        
        Returns:
            List of layer names
        """
        return [name for name, _ in self.model.named_modules()]
    
    def get_conv_layers(self) -> List[str]:
        """
        Get list of convolutional layer names.
        
        Returns:
            List of conv layer names
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                conv_layers.append(name)
        return conv_layers