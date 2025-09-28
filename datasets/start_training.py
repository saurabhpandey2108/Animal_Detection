# D:\Project\Animal_Detection\datasets\start_training.py

import os
from ultralytics import YOLO

def train_model_directly():
    """
    This is the definitive training script.
    It tells YOLO to use the current directory as the dataset,
    bypassing all the faulty YAML path-finding logic.
    """
    print("🚀 Starting the definitive YOLO training process...")
    
    # --- Training Parameters ---
    EPOCHS = 50
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    
    # Load the pretrained model
    model = YOLO("yolov8n-cls.pt")
    
    try:
        # --- Start Training ---
        # This is the critical change. We pass '.' which means "this current folder".
        # The library will automatically find the train, val, and test subdirectories.
        model.train(
            data='.',      
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project="training_runs",
            name="final_animal_classifier",
            exist_ok=True,
            verbose=True
        )
        print("\n✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Sanity check to make sure we are in the right place
    if not (os.path.exists("train") and os.path.exists("val") and os.path.exists("test")):
        print("❌ Error: Make sure you run this script from the 'datasets' directory,")
        print("   which contains the 'train', 'val', and 'test' folders.")
    else:
        train_model_directly()