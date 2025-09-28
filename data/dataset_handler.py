"""
Module for handling the animal dataset from Kaggle.
"""
import kagglehub
import os
from typing import Optional
from config import DATASET_PATH


class DatasetHandler:
    """Handles downloading and managing the animal dataset."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the dataset handler.
        
        Args:
            dataset_path: Path to store the dataset (defaults to config value)
        """
        self.dataset_path = dataset_path or DATASET_PATH
        self.dataset_url = "iamsouravbanerjee/animal-image-dataset-90-different-animals"
        
    def download_dataset(self) -> str:
        """
        Download the animal dataset from Kaggle.
        
        Returns:
            Path to the downloaded dataset
        """
        print("Downloading animal dataset from Kaggle...")
        path = kagglehub.dataset_download(self.dataset_url)
        print(f"Dataset downloaded to: {path}")
        return path
    
    def get_dataset_info(self, dataset_path: str) -> dict:
        """
        Get information about the dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "path": dataset_path,
            "animal_classes": [],
            "total_images": 0
        }
        
        # Count animal classes and images
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    info["animal_classes"].append(item)
                    # Count images in each class folder
                    if os.path.exists(item_path):
                        images = [f for f in os.listdir(item_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        info["total_images"] += len(images)
        
        info["num_classes"] = len(info["animal_classes"])
        return info