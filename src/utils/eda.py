import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter, defaultdict
from PIL import Image
import pandas as pd

from src.utils.logger_setup import get_logger

logger = get_logger(__name__)

def print_dataset_statistics(metadata: Dict[str, Dict[str, Any]]) -> None:
    """
    Print comprehensive dataset statistics.
    
    Args:
        metadata: Dataset metadata dictionary
    """
    logger.info("Generating dataset statistics...")
    
    total_images = len(metadata)
    type_counts = Counter(info["type"] for info in metadata.values())
    
    subjects_by_type = defaultdict(set)
    samples_per_subject = defaultdict(lambda: defaultdict(list))
    
    for info in metadata.values():
        if info["type"] != "unknown":
            subjects_by_type[info["type"]].add(info["subject_id"])
            samples_per_subject[info["type"]][info["subject_id"]].append(info["sample_id"])
    
    print("="*60)
    print("CEDAR DATASET STATISTICS")
    print("="*60)
    print(f"Total Images: {total_images}")
    print(f"Original Signatures: {type_counts.get('original', 0)}")
    print(f"Forged Signatures: {type_counts.get('forgeries', 0)}")
    print(f"Balance Ratio: {type_counts.get('original', 0) / type_counts.get('forgeries', 1):.2f}")
    print()
    
    print("SUBJECT ANALYSIS")
    print("-" * 30)
    for sig_type in ["original", "forgeries"]:
        if sig_type in subjects_by_type:
            subjects = subjects_by_type[sig_type]
            sample_counts = [len(samples_per_subject[sig_type][s]) for s in subjects]
            
            print(f"{sig_type.capitalize()} signatures:")
            print(f"  Unique subjects: {len(subjects)}")
            print(f"  Subject range: {min(subjects)} - {max(subjects)}")
            print(f"  Samples per subject: {min(sample_counts)} - {max(sample_counts)} (avg: {np.mean(sample_counts):.1f})")
            print()

def plot_dataset_distribution(metadata: Dict[str, Dict[str, Any]], 
                            save_path: Optional[Path] = None) -> None:
    """
    Create visualization plots for dataset distribution.
    
    Args:
        metadata: Dataset metadata dictionary  
        save_path: Optional path to save plots
    """
    logger.info("Creating distribution plots...")
    
    subjects_by_type = defaultdict(set)
    samples_per_subject = defaultdict(lambda: defaultdict(int))
    
    for info in metadata.values():
        if info["type"] != "unknown":
            subjects_by_type[info["type"]].add(info["subject_id"])
            samples_per_subject[info["type"]][info["subject_id"]] += 1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CEDAR Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    type_counts = Counter(info["type"] for info in metadata.values())
    axes[0,0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Signature Type Distribution')
    
    all_sample_counts = []
    types = []
    for sig_type in ["original", "forgeries"]:
        sample_counts = list(samples_per_subject[sig_type].values())
        all_sample_counts.extend(sample_counts)
        types.extend([sig_type] * len(sample_counts))
    
    df = pd.DataFrame({'samples': all_sample_counts, 'type': types})
    sns.histplot(data=df, x='samples', hue='type', ax=axes[0,1], bins=20)
    axes[0,1].set_title('Samples per Subject Distribution')
    axes[0,1].set_xlabel('Number of Samples')
    
    for i, sig_type in enumerate(["original", "forgeries"]):
        subjects = sorted(list(subjects_by_type[sig_type]))
        sample_counts = [samples_per_subject[sig_type][s] for s in subjects]
        
        color = 'blue' if sig_type == 'original' else 'red'
        axes[1,0].scatter(subjects, sample_counts, alpha=0.6, label=sig_type, color=color)
    
    axes[1,0].set_xlabel('Subject ID')
    axes[1,0].set_ylabel('Number of Samples')
    axes[1,0].set_title('Samples per Subject by ID')
    axes[1,0].legend()
    
    sns.boxplot(data=df, x='type', y='samples', ax=axes[1,1])
    axes[1,1].set_title('Samples per Subject Box Plot')
    axes[1,1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "dataset_distribution.png", dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plots saved to {save_path}")
    
    plt.show()

def show_sample_images(data_path: Path, metadata: Dict[str, Dict[str, Any]], 
                      num_subjects: int = 3, samples_per_subject: int = 2,
                      save_path: Optional[Path] = None) -> None:
    """
    Display sample images from the dataset.
    
    Args:
        data_path: Path to dataset root directory
        metadata: Dataset metadata dictionary
        num_subjects: Number of subjects to show
        samples_per_subject: Number of samples per subject to show
        save_path: Optional path to save the plot
    """
    logger.info(f"Showing sample images: {num_subjects} subjects, {samples_per_subject} samples each")
    
    subjects_original = set()
    subjects_forged = set()
    
    for info in metadata.values():
        if info["type"] == "original":
            subjects_original.add(info["subject_id"])
        elif info["type"] == "forgeries":
            subjects_forged.add(info["subject_id"])
    
    common_subjects = list(subjects_original.intersection(subjects_forged))
    selected_subjects = sorted(common_subjects)[:num_subjects]
    
    if len(selected_subjects) == 0:
        logger.warning("No subjects found with both original and forged signatures")
        return
    
    rows = num_subjects
    cols = samples_per_subject * 2  # original + forgeries
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sample Signatures from CEDAR Dataset', fontsize=16, fontweight='bold')
    
    for row, subject_id in enumerate(selected_subjects):
        # Get files for this subject
        subject_files = {
            "original": [],
            "forgeries": []
        }
        
        for filename, info in metadata.items():
            if info["subject_id"] == subject_id:
                subject_files[info["type"]].append(filename)
        
        for sig_type in ["original", "forgeries"]:
            subject_files[sig_type] = sorted(
                subject_files[sig_type], 
                key=lambda x: metadata[x]["sample_id"]
            )
        
        col_idx = 0
        
        for i in range(min(samples_per_subject, len(subject_files["original"]))):
            filename = subject_files["original"][i]
            img_path = data_path / "full_org" / filename
            
            if img_path.exists():
                img = Image.open(img_path)
                axes[row, col_idx].imshow(img, cmap='gray')
                axes[row, col_idx].set_title(f'Subject {subject_id} - Original {i+1}', fontsize=10)
                axes[row, col_idx].axis('off')
            col_idx += 1
        
        for i in range(min(samples_per_subject, len(subject_files["forgeries"]))):
            filename = subject_files["forgeries"][i]
            img_path = data_path / "full_forg" / filename
            
            if img_path.exists():
                img = Image.open(img_path)
                axes[row, col_idx].imshow(img, cmap='gray')
                axes[row, col_idx].set_title(f'Subject {subject_id} - Forged {i+1}', fontsize=10)
                axes[row, col_idx].axis('off')
            col_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "sample_images.png", dpi=300, bbox_inches='tight')
        logger.info(f"Sample images saved to {save_path}")
    
    plt.show()

def analyze_image_properties(data_path: Path, metadata: Dict[str, Dict[str, Any]], 
                           sample_size: int = 100) -> Dict[str, Any]:
    """
    Analyze image properties like dimensions, file sizes, etc.
    
    Args:
        data_path: Path to dataset root directory
        metadata: Dataset metadata dictionary
        sample_size: Number of images to sample for analysis
        
    Returns:
        Dictionary with image property statistics
    """
    logger.info(f"Analyzing image properties from {sample_size} samples...")
    
    filenames = list(metadata.keys())
    sample_files = np.random.choice(filenames, min(sample_size, len(filenames)), replace=False)
    
    properties = {
        "widths": [],
        "heights": [], 
        "file_sizes": [],
        "channels": []
    }
    
    for filename in sample_files:
        info = metadata[filename]
        
        if info["type"] == "original":
            img_path = data_path / "full_org" / filename
        elif info["type"] == "forgeries":
            img_path = data_path / "full_forg" / filename
        else:
            continue
            
        if img_path.exists():
            try:
                # Image properties
                img = Image.open(img_path)
                properties["widths"].append(img.width)
                properties["heights"].append(img.height)
                properties["channels"].append(len(img.getbands()))
                
                # File size
                file_size = img_path.stat().st_size / 1024  # KB
                properties["file_sizes"].append(file_size)
                
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
    
    stats = {}
    for prop, values in properties.items():
        if values:
            stats[prop] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "unique_values": len(set(values))
            }
    
    print("\nIMAGE PROPERTIES ANALYSIS")
    print("-" * 40)
    for prop, stat in stats.items():
        print(f"{prop.capitalize()}:")
        print(f"  Range: {stat['min']:.1f} - {stat['max']:.1f}")
        print(f"  Mean ± Std: {stat['mean']:.1f} ± {stat['std']:.1f}")
        print(f"  Unique values: {stat['unique_values']}")
        print()
    
    return stats

def generate_eda_report(data_path: Path, metadata: Dict[str, Dict[str, Any]], 
                       output_path: Optional[Path] = None) -> None:
    """
    Generate complete EDA report with all analyses.
    
    Args:
        data_path: Path to dataset root directory
        metadata: Dataset metadata dictionary
        output_path: Optional path to save outputs
    """
    logger.info("Generating complete EDA report...")
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Run all analyses
    print_dataset_statistics(metadata)
    plot_dataset_distribution(metadata, output_path)
    show_sample_images(data_path, metadata, save_path=output_path)
    analyze_image_properties(data_path, metadata)
    
    logger.info("EDA report generation complete!")