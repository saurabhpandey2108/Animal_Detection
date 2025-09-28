"""
Training script for custom YOLO model on animal dataset.
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
from typing import Optional

# Handle imports for different directory structures
try:
    from config import MODEL_PATH
except ImportError:
    try:
        from config import MODEL_PATH
    except ImportError:
        # Fallback to a default path
        MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class YOLOTrainer:
    """Train a custom YOLO model for animal detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to save the trained model
        """
        self.model_path = model_path or os.path.join(MODEL_PATH, "animal_detector.pt")
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    def train_model(self, data_yaml: str, epochs: int = 100, imgsz: int = 640, 
                   batch_size: int = 16, pretrained_model: str = "yolov8n-cls.pt") -> str:
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to YOLO dataset configuration file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
            pretrained_model: Pretrained model to start from
            
        Returns:
            Path to the trained model
        """
        print(f"Training YOLO model with {epochs} epochs...")
        print(f"Data config: {data_yaml}")
        print(f"Pretrained model: {pretrained_model}")
        
        try:
            # Load a pretrained YOLO model (classification)
            model = YOLO(pretrained_model)
            
            # Train the model
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                name="animal_detection_training",
                exist_ok=True
            )
            
            # Save the trained model
            model_save_path = self.model_path
            model.export(format="pt", save_dir=MODEL_PATH)
            
            print(f"Model trained and saved to: {model_save_path}")
            return model_save_path
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def validate_model(self, model_path: str, data_yaml: str) -> dict:
        """
        Validate the trained model.
        
        Args:
            model_path: Path to the trained model
            data_yaml: Path to YOLO dataset configuration file
            
        Returns:
            Validation results
        """
        print("Validating trained model...")
        
        try:
            # Load the trained model
            model = YOLO(model_path)
            
            # Validate the model
            results = model.val(data=data_yaml)
            
            print("Validation completed!")
            return {
                "accuracy": results.results_dict.get("metrics/accuracy_top1", 0),
                "precision": results.results_dict.get("metrics/precision", 0),
                "recall": results.results_dict.get("metrics/recall", 0),
                "f1_score": results.results_dict.get("metrics/f1", 0)
            }
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise
    
    def evaluate_on_test_set(self, model_path: str, data_yaml: str) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Args:
            model_path: Path to the trained model
            data_yaml: Path to YOLO dataset configuration file
            
        Returns:
            Test set evaluation results
        """
        print("Evaluating model on test set...")
        
        try:
            # Load the trained model
            model = YOLO(model_path)
            
            # Evaluate the model on the test set
            results = model.val(data=data_yaml, split="test")
            
            print("Test set evaluation completed!")
            return {
                "accuracy": results.results_dict.get("metrics/accuracy_top1", 0),
                "precision": results.results_dict.get("metrics/precision", 0),
                "recall": results.results_dict.get("metrics/recall", 0),
                "f1_score": results.results_dict.get("metrics/f1", 0),
                "top5_accuracy": results.results_dict.get("metrics/accuracy_top5", 0)
            }
            
        except Exception as e:
            print(f"Error during test set evaluation: {str(e)}")
            raise


def main():
    """Main function to train the YOLO model."""
    print("Training custom YOLO model for animal detection...")
    
    # Check if dataset YAML file is provided
    if len(sys.argv) < 2:
        print("Usage: python yolo_trainer.py <path_to_dataset_yaml> [epochs]")
        print("Example: python yolo_trainer.py ./yolo_dataset/animal_dataset.yaml 50")
        return
    
    data_yaml = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    # Initialize trainer
    trainer = YOLOTrainer()
    
    # Train model
    try:
        model_path = trainer.train_model(data_yaml, epochs=epochs)
        print(f"Model training completed! Model saved at: {model_path}")
        
        # Validate model
        print("Starting model validation...")
        metrics = trainer.validate_model(model_path, data_yaml)
        print("Validation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Evaluate on test set
        print("Starting test set evaluation...")
        test_metrics = trainer.evaluate_on_test_set(model_path, data_yaml)
        print("Test Set Evaluation Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()