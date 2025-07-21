import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

def extract_info_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from CEDAR filename.
    
    Args:
        filename: Image filename (e.g., 'original_10_1.png')
        
    Returns:
        Dictionary with extracted information
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    if len(parts) != 3:
        logger.warning(f"Unexpected filename format: {filename}")
        return {"type": "unknown", "subject_id": -1, "sample_id": -1, "split": None}
    
    return {
        "type": parts[0],
        "subject_id": int(parts[1]),
        "sample_id": int(parts[2]),
        "split": None  # To be filled during split assignment (train/val/test)
    }

def scan_directory(directory_path: Path, expected_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan directory and extract metadata from all images.
    
    Args:
        directory_path: Path to directory containing images
        expected_type: Expected type ('original' or 'forgeries')
        
    Returns:
        Dictionary mapping filename to metadata
    """
    metadata = {}
    image_files = list(directory_path.glob("*.png"))
    
    logger.info(f"Found {len(image_files)} images in {directory_path}")

    for img_path in image_files:
        filename = img_path.name
        info = extract_info_from_filename(filename)
        
        if info["type"] != expected_type and info["type"] != "unknown":
            logger.warning(f"Type mismatch in {filename}: expected {expected_type}, got {info['type']}")
        
        metadata[filename] = info
    
    return metadata

def create_dataset_metadata(data_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive metadata for CEDAR dataset.
    
    Args:
        data_path: Path to dataset root (containing full_org and full_forg)
        
    Returns:
        Complete metadata dictionary
    """
    logger.info(f"Creating metadata for dataset at {data_path}")

    org_dir = data_path / "full_org"
    forg_dir = data_path / "full_forg"

    if not org_dir.exists():
        raise FileNotFoundError(f"Original signatures directory not found: {org_dir}")
    if not forg_dir.exists():
        raise FileNotFoundError(f"Forged signatures directory not found: {forg_dir}")
    
    metadata = {}
    metadata.update(scan_directory(org_dir, "original"))
    metadata.update(scan_directory(forg_dir, "forgeries"))

    logger.info(f"Created metadata for {len(metadata)} total images")
    return metadata

def validate_dataset_consistency(metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate dataset structure and return comprehensive statistics.
    
    Args:
        metadata: Dataset metadata dictionary
        
    Returns:
        Validation results and statistics
    """
    logger.info("Validating dataset consistency...")

    # Group metadata by type and subject
    type_counts = Counter(info["type"] for info in metadata.values())
    subjects_by_type = defaultdict(set)
    samples_per_subject = defaultdict(lambda: defaultdict(list))
    
    for filename, info in metadata.items():
        if info["type"] != "unknown":
            subjects_by_type[info["type"]].add(info["subject_id"])
            samples_per_subject[info["type"]][info["subject_id"]].append(info["sample_id"])

    stats = {
        "total_images": len(metadata),
        "type_distribution": dict(type_counts),
        "subjects_per_type": {t: len(subjects) for t, subjects in subjects_by_type.items()},
        "total_unique_subjects": len(set().union(*subjects_by_type.values())),
        "samples_per_subject_stats": {}
    }

    for sig_type in subjects_by_type.keys():
        sample_counts = [len(samples) for samples in samples_per_subject[sig_type].values()]
        stats["samples_per_subject_stats"][sig_type] = {
            "min": min(sample_counts) if sample_counts else 0,
            "max": max(sample_counts) if sample_counts else 0,
            "mean": sum(sample_counts) / len(sample_counts) if sample_counts else 0,
            "total_subjects": len(sample_counts)
        }

    validation_results = {
        "consistency_checks": {
            "all_types_valid": all(info["type"] in ["original", "forged"] for info in metadata.values()),
            "subjects_match_across_types": subjects_by_type.get("original", set()) == subjects_by_type.get("forged", set()),
            "no_missing_samples": True  # updated below
        }
    }

    missing_samples = []
    for sig_type in subjects_by_type.keys():
        for subject_id, samples in samples_per_subject[sig_type].items():
            expected_samples = set(range(1, max(samples) + 1))
            actual_samples = set(samples)
            if expected_samples != actual_samples:
                missing = expected_samples - actual_samples
                missing_samples.append(f"{sig_type}_subject_{subject_id}: missing samples {missing}")
    
    validation_results["consistency_checks"]["no_missing_samples"] = len(missing_samples) == 0
    validation_results["missing_samples"] = missing_samples

    logger.info(f"Dataset validation complete:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Type distribution: {stats['type_distribution']}")
    logger.info(f"  Unique subjects: {stats['total_unique_subjects']}")
    logger.info(f"  Consistency checks passed: {sum(validation_results['consistency_checks'].values())}/3")
    
    if missing_samples:
        logger.warning(f"Found {len(missing_samples)} missing sample issues")
    
    return {**stats, **validation_results}

def save_metadata(metadata: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Dataset metadata
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {output_path}")

def load_metadata(metadata_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Path to JSON metadata file
        
    Returns:
        Loaded metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Metadata loaded from: {metadata_path}")
    return metadata

def get_subject_list(metadata: Dict[str, Dict[str, Any]]) -> List[int]:
    """
    Get sorted list of unique subject IDs.
    
    Args:
        metadata: Dataset metadata
        
    Returns:
        Sorted list of subject IDs
    """
    subjects = set()
    for info in metadata.values():
        if info["subject_id"] != -1:
            subjects.add(info["subject_id"])
    
    return sorted(list(subjects))