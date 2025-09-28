"""
Preprocessing script to convert Kaggle animal dataset to YOLO format.
"""
import os
import sys
import shutil
import random
from pathlib import Path
import kagglehub
from typing import List, Tuple

# Handle imports for different directory structures
try:
    from config import DATASET_PATH
except ImportError:
    try:
        from config import DATASET_PATH
    except ImportError:
        # Fallback to a default path
        DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class YOLOPreprocessor:
    """Preprocess Kaggle animal dataset for YOLO training."""
    
    def __init__(self, dataset_path: str = None, output_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_path: Path to the Kaggle dataset
            output_path: Path to output YOLO formatted dataset
        """
        # Ensure dataset_path is always a valid string
        self.dataset_path = dataset_path or DATASET_PATH or os.getcwd()
        # Ensure self.dataset_path is a valid path string
        if not isinstance(self.dataset_path, str) or not self.dataset_path:
            self.dataset_path = os.path.join(os.getcwd(), "data")
            
        # Ensure output_path is always a valid string
        if output_path:
            self.output_path = output_path
        else:
            # Use a default path based on dataset_path, ensuring dirname won't return None
            dataset_dir = os.path.dirname(self.dataset_path) or os.getcwd()
            self.output_path = os.path.join(dataset_dir, "yolo_dataset")
        
        # Create output directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images", "test"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels", "test"), exist_ok=True)
        
    def download_dataset(self) -> str:
        """
        Download the Kaggle animal dataset.
        
        Returns:
            Path to the downloaded dataset
        """
        print("Downloading animal dataset from Kaggle...")
        path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
        print(f"Dataset downloaded to: {path}")
        
        # Handle the nested structure of this specific dataset
        # The actual animal folders are in path/animals/animals/
        nested_animals_path = os.path.join(path, "animals", "animals")
        if os.path.exists(nested_animals_path):
            return nested_animals_path
            
        # Fallback to original path
        return path
    
    def get_animal_classes(self, dataset_path: str) -> List[str]:
        """
        Get list of animal classes from the dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            List of animal class names
        """
        classes = []
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                # Only include directories that are likely to be animal classes
                # Filter out hidden directories, cache directories, and non-directories
                if (os.path.isdir(item_path) and 
                    not item.startswith('.') and 
                    item != '__pycache__' and
                    item != 'yolo_dataset'):
                    classes.append(item)
        return sorted(classes)
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate that the dataset has the expected structure.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            True if dataset is valid, False otherwise
        """
        if not os.path.exists(dataset_path):
            return False
            
        classes = self.get_animal_classes(dataset_path)
        if len(classes) == 0:
            return False
            
        # Check if at least some classes have images
        valid_classes = 0
        for animal_class in classes:
            class_path = os.path.join(dataset_path, animal_class)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 0:
                valid_classes += 1
                
        return valid_classes > 0
    
    def create_yaml_config(self, classes: List[str]) -> str:
        """
        Create YOLO dataset configuration file.
        
        Args:
            classes: List of animal classes
            
        Returns:
            Path to the YAML configuration file
        """
        yaml_path = os.path.join(self.output_path, "animal_dataset.yaml")
        
        # Create class mapping
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        
        # Write YAML file with relative paths
        with open(yaml_path, 'w') as f:
            # Use . for path to indicate current directory
            f.write("path: .\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write("names:\n")
            for cls, idx in class_mapping.items():
                f.write(f"  {idx}: {cls}\n")
        
        print(f"YOLO configuration file created at: {yaml_path}")
        return yaml_path
    
    def process_dataset(self, train_split: float = 0.7, val_split: float = 0.15) -> Tuple[str, List[str]]:
        """
        Process the dataset and convert to YOLO format.
        
        Args:
            train_split: Proportion of data to use for training
            val_split: Proportion of data to use for validation
            
        Returns:
            Tuple of (yaml_config_path, list_of_classes)
        """
        # Download dataset if not already present
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path {self.dataset_path} not found. Downloading...")
            self.dataset_path = self.download_dataset()
        else:
            print(f"Using existing dataset at: {self.dataset_path}")
            
        # Validate dataset structure
        if not self.validate_dataset(self.dataset_path):
            print("WARNING: Dataset structure appears to be invalid or empty!")
            print("Expected structure: dataset_path/animal_name/images.jpg")
            print("Current dataset path:", self.dataset_path)
            print("Contents of dataset path:")
            if os.path.exists(self.dataset_path):
                for item in os.listdir(self.dataset_path):
                    item_path = os.path.join(self.dataset_path, item)
                    if os.path.isdir(item_path):
                        images = [f for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        print(f"  {item}/ ({len(images)} images)")
                    else:
                        print(f"  {item} (file)")
            print("\nPlease ensure you have:")
            print("1. Set up Kaggle API credentials")
            print("2. Accepted the dataset terms on Kaggle")
            print("3. Run the script to automatically download the dataset")
            raise ValueError("Invalid dataset structure. No valid animal classes with images found.")
        
        # Get animal classes
        classes = self.get_animal_classes(self.dataset_path)
        print(f"Found {len(classes)} animal classes")
        
        if len(classes) == 0:
            raise ValueError("No animal classes found in dataset. Please check the dataset structure.")
        
        # Create YOLO configuration
        yaml_path = self.create_yaml_config(classes)
        
        # Process each animal class
        for class_idx, animal_class in enumerate(classes):
            print(f"Processing {animal_class} ({class_idx + 1}/{len(classes)})")
            self.process_animal_class(animal_class, class_idx, train_split, val_split)
        
        print("Dataset processing completed!")
        return yaml_path, classes
    
    def process_animal_class(self, animal_class: str, class_idx: int, train_split: float, val_split: float):
        """
        Process a single animal class.
        
        Args:
            animal_class: Name of the animal class
            class_idx: Index of the class
            train_split: Proportion of data to use for training
            val_split: Proportion of data to use for validation
        """
        class_path = os.path.join(self.dataset_path, animal_class)
        if not os.path.exists(class_path):
            return
            
        # Get all images for this class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Skip if no images found
        if not images:
            print(f"Warning: No images found for class {animal_class}")
            return
            
        print(f"  Found {len(images)} images")
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train, validation, and test
        train_idx = int(len(images) * train_split)
        val_idx = train_idx + int(len(images) * val_split)
        
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]
        
        # Ensure we have at least one image in each split
        if not train_images and (val_images or test_images):
            if val_images:
                train_images = val_images[:1]
                val_images = val_images[1:]
            elif test_images:
                train_images = test_images[:1]
                test_images = test_images[1:]
        if not val_images and (train_images or test_images):
            if train_images:
                val_images = train_images[:1]
                train_images = train_images[1:]
            elif test_images:
                val_images = test_images[:1]
                test_images = test_images[1:]
        if not test_images and (train_images or val_images):
            if train_images:
                test_images = train_images[:1]
                train_images = train_images[1:]
            elif val_images:
                test_images = val_images[:1]
                val_images = val_images[1:]
        
        # Process training images
        for img_name in train_images:
            self.process_image(img_name, class_path, class_idx, "train")
        
        # Process validation images
        for img_name in val_images:
            self.process_image(img_name, class_path, class_idx, "val")
        
        # Process test images
        for img_name in test_images:
            self.process_image(img_name, class_path, class_idx, "test")
    
    def process_image(self, img_name: str, class_path: str, class_idx: int, split: str):
        """
        Process a single image and create corresponding label file.
        
        Args:
            img_name: Name of the image file
            class_path: Path to the class directory
            class_idx: Index of the class
            split: Either "train", "val", or "test"
        """
        # Source and destination paths
        src_img_path = os.path.join(class_path, img_name)
        dst_img_path = os.path.join(self.output_path, "images", split, img_name)
        
        # Copy image
        shutil.copy2(src_img_path, dst_img_path)
        
        # Create label file for classification (class index only)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.output_path, "labels", split, label_name)
        
        # For classification, the label file contains just the class index
        with open(label_path, 'w') as f:
            f.write(f"{class_idx}\n")


def main():
    """Main function to preprocess the dataset."""
    print("Preprocessing Kaggle animal dataset for YOLO training...")
    
    # Initialize preprocessor
    preprocessor = YOLOPreprocessor()
    
    # Process dataset with train/val/test splits
    yaml_path, classes = preprocessor.process_dataset(train_split=0.7, val_split=0.15)
    
    print(f"\nDataset preprocessing completed!")
    print(f"YOLO config file: {yaml_path}")
    print(f"Number of classes: {len(classes)}")
    print("Classes:", ", ".join(classes[:10]), "..." if len(classes) > 10 else "")


if __name__ == "__main__":
    main()