import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)

class TestImageSelector:
    """
    Utility to select and analyze test images with model predictions.
    """
    
    def __init__(self, 
                 data_path: Path,
                 metadata: Dict[str, Dict[str, Any]], 
                 model: torch.nn.Module,
                 device: torch.device,
                 transform: Optional[torch.nn.Module] = None) -> None:
        """
        Initialize selector with test data and model.
        
        Args:
            data_path: Path to dataset root
            metadata: Dataset metadata with splits
            model: Trained model
            device: Device for inference
            transform: Transform for preprocessing
        """
        self.data_path = data_path
        self.model = model.eval()
        self.device = device
        self.transform = transform
        
        self.test_metadata = {
            fname: info for fname, info in metadata.items() 
            if info.get("split") == "test"
        }
        
        self.subjects_data = self._group_by_subject()
        
        logger.info(f"TestImageSelector initialized with {len(self.test_metadata)} test images")
    
    def _group_by_subject(self) -> Dict[int, Dict[str, List[str]]]:
        """Group test images by subject ID and signature type."""
        subjects = {}
        
        for filename, info in self.test_metadata.items():
            subject_id = info["subject_id"]
            sig_type = info["type"]
            
            if subject_id not in subjects:
                subjects[subject_id] = {"original": [], "forgeries": []}
            
            subjects[subject_id][sig_type].append(filename)
        
        return subjects
    
    def get_random_image(self, signature_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get random test image, optionally filtered by type.
        
        Args:
            signature_type: Filter by 'original' or 'forgeries' (None for any)
            
        Returns:
            Tuple of (filename, metadata)
        """
        if signature_type:
            filtered_files = [
                fname for fname, info in self.test_metadata.items()
                if info["type"] == signature_type
            ]
        else:
            filtered_files = list(self.test_metadata.keys())
        
        if not filtered_files:
            raise ValueError(f"No test images found for type: {signature_type}")
        
        filename = random.choice(filtered_files)
        return filename, self.test_metadata[filename]
    
    def get_image_by_subject(self, subject_id: int, signature_type: str, sample_idx: int = 0) -> Tuple[str, Dict[str, Any]]:
        """
        Get specific image by subject ID and type.
        
        Args:
            subject_id: Subject ID
            signature_type: 'original' or 'forged'
            sample_idx: Index of sample for that subject (default: 0)
            
        Returns:
            Tuple of (filename, metadata)
        """
        if subject_id not in self.subjects_data:
            raise ValueError(f"Subject {subject_id} not found in test set")
        
        if signature_type not in self.subjects_data[subject_id]:
            raise ValueError(f"Signature type '{signature_type}' not valid")
        
        files = self.subjects_data[subject_id][signature_type]
        if not files:
            raise ValueError(f"No {signature_type} signatures for subject {subject_id}")
        
        if sample_idx >= len(files):
            raise ValueError(f"Sample index {sample_idx} out of range for subject {subject_id}")
        
        filename = files[sample_idx]
        return filename, self.test_metadata[filename]
    
    def find_original_for_forged(self, forged_filename: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find corresponding original signature for a forged one.
        
        Args:
            forged_filename: Filename of forged signature
            
        Returns:
            Tuple of (original_filename, metadata) or None if not found
        """
        forged_info = self.test_metadata.get(forged_filename)
        if not forged_info or forged_info["type"] != "forgeries":
            return None
        
        subject_id = forged_info["subject_id"]
        
        original_files = self.subjects_data.get(subject_id, {}).get("original", [])
        
        if original_files:
            # Return first original signature for this subject
            original_filename = original_files[0]
            return original_filename, self.test_metadata[original_filename]
        
        return None
    
    def load_and_predict(self, filename: str) -> Tuple[Image.Image, torch.Tensor, int, float]:
        """
        Load image and get model prediction.
        
        Args:
            filename: Image filename
            
        Returns:
            Tuple of (PIL_image, preprocessed_tensor, predicted_class, confidence)
        """
        info = self.test_metadata[filename]
        
        if info["type"] == "original":
            img_path = self.data_path / "full_org" / filename
        else:
            img_path = self.data_path / "full_forg" / filename
        
        pil_image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        else:
            from torchvision import transforms
            basic_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = basic_transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        return pil_image, tensor.squeeze(0), predicted_class, confidence_score
    
    def show_image_analysis(self, filename: str, show_comparison: bool = True) -> Dict[str, Any]:
        """
        Show complete analysis of selected image.
        
        Args:
            filename: Image filename to analyze
            show_comparison: Whether to show original comparison for forged images
            
        Returns:
            Analysis results dictionary
        """
        info = self.test_metadata[filename]
        
        pil_image, tensor, pred_class, confidence = self.load_and_predict(filename)
        
        class_names = ["Original", "Forged"]
        true_class = 0 if info["type"] == "original" else 1
        
        is_correct = pred_class == true_class
        
        print("="*60)
        print(f"IMAGE ANALYSIS: {filename}")
        print("="*60)
        print(f"Subject ID: {info['subject_id']}")
        print(f"Sample ID: {info['sample_id']}")
        print(f"True Class: {class_names[true_class]} ({true_class})")
        print(f"Predicted Class: {class_names[pred_class]} ({pred_class})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Prediction: {'CORRECT :)' if is_correct else 'WRONG :()'}")
        print()
        
        # Visualize comparison
        comparison_image = None
        comparison_info = None
        
        if info["type"] == "forgeries" and show_comparison:
            original_result = self.find_original_for_forged(filename)
            if original_result:
                orig_filename, orig_info = original_result
                orig_pil, _, orig_pred, orig_conf = self.load_and_predict(orig_filename)
                comparison_image = orig_pil
                comparison_info = {
                    "filename": orig_filename,
                    "predicted_class": orig_pred,
                    "confidence": orig_conf,
                    "info": orig_info
                }
                
                print(f"COMPARISON WITH ORIGINAL:")
                print(f"Original file: {orig_filename}")
                print(f"Original prediction: {class_names[orig_pred]} (conf: {orig_conf:.4f})")
                print()
        
        if comparison_image:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Main image (forgeries)
            axes[0].imshow(pil_image)
            axes[0].set_title(f'FORGED Signature\nPred: {class_names[pred_class]} ({confidence:.3f})', 
                             fontsize=12, fontweight='bold', 
                             color='red' if not is_correct else 'green')
            axes[0].axis('off')
            
            # Comparison image (original)
            axes[1].imshow(comparison_image)
            axes[1].set_title(f'ORIGINAL Signature\nPred: {class_names[comparison_info["predicted_class"]]} ({comparison_info["confidence"]:.3f})', 
                             fontsize=12, fontweight='bold',
                             color='green' if comparison_info["predicted_class"] == 0 else 'red')
            axes[1].axis('off')
            
            fig.suptitle(f'Subject {info["subject_id"]} - Signature Comparison', 
                        fontsize=14, fontweight='bold')
        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(pil_image)
            plt.title(f'{info["type"].upper()} Signature\nPred: {class_names[pred_class]} (Confidence: {confidence:.3f})', 
                     fontsize=14, fontweight='bold',
                     color='green' if is_correct else 'red')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        results = {
            "filename": filename,
            "metadata": info,
            "pil_image": pil_image,
            "tensor": tensor,
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": confidence,
            "is_correct": is_correct,
            "comparison": comparison_info
        }
        
        return results
    
    def list_available_subjects(self) -> None:
        """Print list of available subjects in test set."""
        print("AVAILABLE SUBJECTS IN TEST SET:")
        print("-" * 40)
        
        for subject_id, data in sorted(self.subjects_data.items()):
            orig_count = len(data["original"])
            forg_count = len(data["forgeries"])
            print(f"Subject {subject_id:2d}: {orig_count} original, {forg_count} forged")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get test set statistics."""
        stats = {
            "total_images": len(self.test_metadata),
            "original_count": sum(1 for info in self.test_metadata.values() if info["type"] == "original"),
            "forged_count": sum(1 for info in self.test_metadata.values() if info["type"] == "forgeries"),
            "unique_subjects": len(self.subjects_data),
            "subjects_with_both": sum(1 for data in self.subjects_data.values() 
                                    if data["original"] and data["forgeries"])
        }
        
        return stats