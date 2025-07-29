import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.utils.logger_setup import get_logger

logger = get_logger()

class XAIVisualizer:
    """
    Visualization utilities for XAI explanations.
    """
    
    def __init__(self, class_names: List[str] = None) -> None:
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or ["Original", "Forgeries"]
        
        # Custom colormap for heatmaps
        self.heatmap_cmap = LinearSegmentedColormap.from_list(
            'custom', ['blue', 'cyan', 'yellow', 'red'], N=256
        )
        
        logger.info("XAIVisualizer initialized")
    
    def denormalize_image(self, 
                         tensor: torch.Tensor, 
                         mean: List[float] = [0.485, 0.456, 0.406],
                         std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Denormalize tensor image for visualization.
        
        Args:
            tensor: Normalized image tensor (C, H, W)
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Denormalized image array (H, W, C)
        """
        image = tensor.clone()
        
        # Denormalize
        for i, (m, s) in enumerate(zip(mean, std)):
            image[i] = image[i] * s + m
        
        # Convert to numpy and transpose
        image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        return image
    
    def plot_single_explanation(self, 
                              image: torch.Tensor,
                              heatmap: np.ndarray,
                              method_name: str,
                              prediction: Optional[Tuple[int, float]] = None,
                              save_path: Optional[Path] = None,
                              alpha: float = 0.4) -> None:
        """
        Plot single explanation with original image and heatmap overlay.
        
        Args:
            image: Original image tensor (C, H, W)
            heatmap: Explanation heatmap
            method_name: Name of explanation method
            prediction: Tuple of (predicted_class, confidence)
            save_path: Optional path to save plot
            alpha: Transparency for heatmap overlay
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img_np = self.denormalize_image(image)
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        im1 = axes[1].imshow(heatmap, cmap=self.heatmap_cmap)
        axes[1].set_title(f'{method_name} Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(img_np)
        im2 = axes[2].imshow(heatmap, cmap=self.heatmap_cmap, alpha=alpha)
        axes[2].set_title(f'{method_name} Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        if prediction:
            pred_class, confidence = prediction
            class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class {pred_class}"
            fig.suptitle(f'Prediction: {class_name} (Confidence: {confidence:.3f})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'{method_name} Explanation', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Single explanation plot saved to {save_path}")
        
        plt.show()
    
    def plot_comparison(self, 
                       image: torch.Tensor,
                       explanations: Dict[str, np.ndarray],
                       prediction: Optional[Tuple[int, float]] = None,
                       save_path: Optional[Path] = None,
                       alpha: float = 0.4) -> None:
        """
        Plot comparison of multiple explanation methods.
        
        Args:
            image: Original image tensor (C, H, W)
            explanations: Dictionary of explanation maps
            prediction: Tuple of (predicted_class, confidence)
            save_path: Optional path to save plot
            alpha: Transparency for heatmap overlay
        """
        n_methods = len(explanations)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        # Denormalize image
        img_np = self.denormalize_image(image)
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].axis('off')
        
        for i, (method_name, heatmap) in enumerate(explanations.items(), 1):
            # Heatmap only
            im1 = axes[0, i].imshow(heatmap, cmap=self.heatmap_cmap)
            axes[0, i].set_title(f'{method_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Overlay
            axes[1, i].imshow(img_np)
            axes[1, i].imshow(heatmap, cmap=self.heatmap_cmap, alpha=alpha)
            axes[1, i].set_title('Overlay', fontsize=12)
            axes[1, i].axis('off')
        
        if prediction:
            pred_class, confidence = prediction
            class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class {pred_class}"
            fig.suptitle(f'XAI Methods Comparison - Prediction: {class_name} (Confidence: {confidence:.3f})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('XAI Methods Comparison', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_analysis(self, 
                           correct_explanations: List[Dict[str, np.ndarray]],
                           wrong_explanations: List[Dict[str, np.ndarray]],
                           method_name: str = 'grad_cam',
                           save_path: Optional[Path] = None) -> None:
        """
        Plot comparison between correct and wrong predictions.
        
        Args:
            correct_explanations: List of explanation dicts for correct predictions
            wrong_explanations: List of explanation dicts for wrong predictions
            method_name: Which explanation method to use
            save_path: Optional path to save plot
        """
        # Calculate average heatmaps
        correct_avg = np.mean([exp[method_name] for exp in correct_explanations], axis=0)
        wrong_avg = np.mean([exp[method_name] for exp in wrong_explanations], axis=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Correct predictions average
        im1 = axes[0].imshow(correct_avg, cmap=self.heatmap_cmap)
        axes[0].set_title(f'Correct Predictions\n(n={len(correct_explanations)})', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Wrong predictions average
        im2 = axes[1].imshow(wrong_avg, cmap=self.heatmap_cmap)
        axes[1].set_title(f'Wrong Predictions\n(n={len(wrong_explanations)})', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference map
        diff_map = correct_avg - wrong_avg
        im3 = axes[2].imshow(diff_map, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title('Difference\n(Correct - Wrong)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        fig.suptitle(f'Error Analysis - {method_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_specific_patterns(self, 
                                   original_explanations: List[Dict[str, np.ndarray]],
                                   forged_explanations: List[Dict[str, np.ndarray]],
                                   method_name: str = 'grad_cam',
                                   save_path: Optional[Path] = None) -> None:
        """
        Plot average explanation patterns for each class.
        
        Args:
            original_explanations: List of explanations for original signatures
            forged_explanations: List of explanations for forged signatures
            method_name: Which explanation method to use
            save_path: Optional path to save plot
        """
        # Calculate average heatmaps
        original_avg = np.mean([exp[method_name] for exp in original_explanations], axis=0)
        forged_avg = np.mean([exp[method_name] for exp in forged_explanations], axis=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original signatures average
        im1 = axes[0].imshow(original_avg, cmap=self.heatmap_cmap)
        axes[0].set_title(f'Original Signatures\n(n={len(original_explanations)})', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Forged signatures average
        im2 = axes[1].imshow(forged_avg, cmap=self.heatmap_cmap)
        axes[1].set_title(f'Forged Signatures\n(n={len(forged_explanations)})', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference map
        diff_map = original_avg - forged_avg
        im3 = axes[2].imshow(diff_map, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title('Difference\n(Original - Forged)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        fig.suptitle(f'Class-Specific Patterns - {method_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class-specific patterns plot saved to {save_path}")
        
        plt.show()
    
    def plot_layer_comparison(self, 
                            image: torch.Tensor,
                            layer_explanations: Dict[str, np.ndarray],
                            save_path: Optional[Path] = None) -> None:
        """
        Plot Grad-CAM from different layers for comparison.
        
        Args:
            image: Original image tensor (C, H, W)
            layer_explanations: Dictionary mapping layer names to heatmaps
            save_path: Optional path to save plot
        """
        n_layers = len(layer_explanations)
        n_cols = min(4, n_layers + 1)  # Max 4 columns
        n_rows = (n_layers + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Denormalize image
        img_np = self.denormalize_image(image)
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot each layer
        for i, (layer_name, heatmap) in enumerate(layer_explanations.items(), 1):
            row = i // n_cols
            col = i % n_cols
            
            im = axes[row, col].imshow(heatmap, cmap=self.heatmap_cmap)
            axes[row, col].set_title(f'{layer_name}', fontsize=10)
            axes[row, col].axis('off')
        
        for i in range(len(layer_explanations) + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle('Grad-CAM Layer Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Layer comparison plot saved to {save_path}")
        
        plt.show()