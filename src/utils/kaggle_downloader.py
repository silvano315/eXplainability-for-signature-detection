import os
from pathlib import Path
import yaml
import logging
from typing import Optional
import kaggle
from zipfile import ZipFile

class KaggleDataDownloader:
    """Download and manage datasets from Kaggle."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize downloader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._check_kaggle_credentials()

        self.download_path = Path(self.config['kaggle']['download_path'])
        self.download_path.mkdir(parents=True, exist_ok=True)

    def _check_kaggle_credentials(self) -> None:
        """
        Verify Kaggle API credentials are properly set up.
        
        You need to place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and 
        KAGGLE_KEY environment variables.
        """
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            if not (os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')):
                raise ValueError(
                    "Kaggle API credentials not found. Please follow these steps:\n"
                    "1. Go to https://kaggle.com/account\n"
                    "2. Click on 'Create API Token'\n"
                    "3. Move the downloaded kaggle.json to ~/.kaggle/\n"
                    "   OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
                )
            
    def download_dataset(self) -> Path:
        """
        Download CEDAR dataset from Kaggle.
        
        Returns:
            Path: Path to downloaded dataset
        """
        dataset_name = self.config['kaggle']['dataset_name']

        try:
            if self._check_dataset_exists():
                self.logger.info("Dataset already downloaded")
                return self.download_path
            
            self.logger.info(f"Downloading dataset: {dataset_name}")

            kaggle.api.dataset_download_files(
                dataset_name,
                path = self.download_path,
                unzip = True
            )

            self.logger.info(f"Dataset downloaded successfully to {self.download_path}")

            self._validate_download()
            
            return self.download_path
        
        except Exception as e:
            self.logger.info(f"Error downloading dataset: {str(e)}")
            raise

    def _check_dataset_exists(self) -> bool:
        """
        Check if dataset is already downloaded.
        
        Returns:
            bool: True if dataset exists
        """
        expected_files = ['cedar_dataset.zip', 'cedar_paper.pdf']
        return all((self.download_path / f).exists() for f in expected_files)
    
    def _validate_download(self) -> None:
        """
        Validate downloaded dataset structure and content.
        
        Expected structure:
        data/raw/cedardataset/
        └── cedar_dataset/
            ├── full_forg/
            └── full_org/
        
        Raises:
            ValueError: If dataset structure is invalid
        """
        self.logger.info("Starting dataset validation...")
        
        dataset_folder = next(self.download_path.glob("cedar_dataset"), None)
        if not dataset_folder:
            raise ValueError("Dataset folder 'cedar_dataset' not found")
        
        splits = ['full_forg', 'full_org']
        for split in splits:
            split_path = dataset_folder / split
            if not split_path.exists():
                raise ValueError(f"Missing {split} directory in dataset")
                
            if not any(split_path.rglob('*.png')):
                raise ValueError(f"No image files found in {split} directory")
            
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            if len(classes) != self.config['dataset']['num_classes']:
                raise ValueError(
                    f"Expected {self.config['dataset']['num_classes']} classes in {split} split, "
                    f"but found {len(classes)}"
                )
        
        self.logger.info("Dataset validation successful")
        self.logger.info(f"Found all required splits: {splits}")
        self.logger.info(f"Number of classes per split: {self.config['dataset']['num_classes']}")
        
    
def setup_dataset():
    """
    Utility function to download and setup the dataset.
    
    Returns:
        Path: Path to dataset
    """
    downloader = KaggleDataDownloader()
    return downloader.download_dataset()

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    setup_dataset()