
"""
Training script to train a YOLO model on the existing dataset.
This script is meant to be run from within the yolo_classification_dataset directory.
"""

import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try both import paths
try:
    from models.yolo_trainer import YOLOTrainer
except ImportError:
    # If that fails, try importing directly
    try:
        from models.yolo_trainer import YOLOTrainer
    except ImportError:
        # If both fail, add the animal_detection directory to path
        from models.yolo_trainer import YOLOTrainer


def train_model():
    """Main training function."""
    print("Training YOLO model on existing dataset...")
    
    # Change this path to point to the new location of your dataset.yaml
    dataset_yaml = os.path.join(parent_dir, 'datasets', 'dataset.yaml')
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset YAML file not found at {dataset_yaml}")
        print("Please ensure the dataset is properly organized.")
        sys.exit(1)
    
    print(f"Using dataset configuration: {dataset_yaml}")
    
    # Initialize trainer
    trainer = YOLOTrainer()
    
    # Train model with 5 epochs for pretraining
    try:
        print("Starting training...")
        model_path = trainer.train_model(
            data_yaml=dataset_yaml,
            epochs=5,
            imgsz=224,  # Smaller image size for faster training
            batch_size=16
        )
        print(f"Training completed! Model saved to: {model_path}")
        
        # Validate model
        print("Starting validation...")
        metrics = trainer.validate_model(model_path, dataset_yaml)
        print("Validation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        return model_path
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_on_test_set(model_path):
    """Evaluate the trained model on the test set."""
    print("Evaluating model on test set...")
    
    try:
        # Import YOLO here to avoid import issues
        from ultralytics import YOLO
        
        # Verify test directory exists
        if not os.path.exists("test"):
            print("Error: test directory not found")
            return
            
        # Count classes in test set
        test_classes = [d for d in os.listdir("test") if os.path.isdir(os.path.join("test", d))]
        print(f"Found {len(test_classes)} classes in test directory")
        
        # Load the trained model
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        model = YOLO(model_path)
        
        # Evaluate the model on the test set
        print("Starting evaluation on test set...")
        results = model.val(
            data="dataset.yaml",
            split="test",
            imgsz=224,
            batch=16,
            verbose=True
        )
        
        print("Test Set Evaluation Results:")
        print(f"  Top-1 Accuracy: {results.results_dict.get('metrics/accuracy_top1', 0):.4f}")
        print(f"  Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 0):.4f}")
        print(f"  Precision: {results.results_dict.get('metrics/precision', 0):.4f}")
        print(f"  Recall: {results.results_dict.get('metrics/recall', 0):.4f}")
        print(f"  F1-Score: {results.results_dict.get('metrics/f1', 0):.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to train and evaluate the model."""
    # Train the model
    model_path = train_model()
    
    # Evaluate on test set if training was successful
    if model_path and os.path.exists(model_path):
        print("\n" + "="*50)
        evaluate_on_test_set(model_path)
        print("="*50)


if __name__ == "__main__":
    main()