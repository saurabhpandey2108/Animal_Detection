# Animal Detection using Deep Learning with YOLO

A real-time animal detection system using YOLO (You Only Look Once) for efficient object detection that can identify animals in video streams and provide information about them.

## Features

- Real-time animal detection using YOLOv8
- Recognition of common animal species
- Provides detailed information about detected animals
- High performance for real-time applications
- Both CLI and GUI interfaces
- Support for custom model training on 90+ animal classes
- Simple user interface with OpenCV

## Prerequisites

### Hardware Requirements
- Any computer with a camera (webcam, USB camera, etc.)
- For Raspberry Pi deployment (optional): Raspberry Pi 4 (recommended) or Raspberry Pi 3
- For training: GPU recommended (optional but faster)

### Software Requirements
- Python 3.10 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd animal_detection
   ```

2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses the "Animal Image Dataset (90 Different Animals)" from Kaggle:
https://www.kaggle.com/iamsouravbanerjee/animal-image-dataset-90-different-animals

The dataset information is used to provide detailed features about detected animals.

## Usage

### CLI Mode (Default)
To run the animal detection system in CLI mode:
```bash
python main.py
```

### GUI Mode
To run the animal detection system in GUI mode:
```bash
python main.py --gui
```

### Controls (CLI Mode)
- Press `q` to quit the application
- Press `i` to display detailed information about the last detected animal

### Controls (GUI Mode)
- Click "Start Detection" to begin animal detection
- Click "Stop Detection" to stop the detection process
- Click "Animal Info" to display detailed information about the last detected animal
- Click "Exit" to close the application

## Training Custom Model

To achieve better accuracy on the 90 animal classes from the Kaggle dataset, you can train a custom YOLO model:

1. Download the Kaggle dataset:
   ```bash
   # The dataset will be automatically downloaded when you run the training script
   ```

2. Run the training pipeline:
   ```bash
   python train_animal_detector.py --epochs 50
   ```

3. Optional training parameters:
   ```bash
   python train_animal_detector.py --epochs 100 --batch-size 32 --img-size 640
   ```

The trained model will be automatically used by the detection system, providing much better accuracy on the specific animal classes.

## How It Works

1. The system initializes the camera and loads a YOLOv8 model
2. It captures video frames in real-time
3. Each frame is processed through the YOLO model to detect animals
4. When an animal is detected with sufficient confidence, it displays:
   - The name of the detected animal
   - Confidence level of the detection
   - Frames per second (FPS) performance metric
5. Detailed animal information can be accessed through the interface

## YOLO Model

This implementation supports two models:

1. **Pre-trained COCO Model** (default fallback):
   - Uses YOLOv8 nano model pre-trained on COCO dataset
   - Detects common animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
   - Fast inference but limited animal classes

2. **Custom Trained Model** (recommended):
   - Trained specifically on the 90+ animal classes from the Kaggle dataset
   - Much better accuracy for wildlife animals
   - Automatically used when available

## GUI Features

The graphical user interface provides:
- Intuitive control buttons
- Real-time video display
- Detection information panel
- Detailed animal information window
- Status bar with system information

## Performance Optimization

The system includes several optimizations:
- Frame skipping to reduce computational load
- Efficient YOLOv8 nano model for faster inference
- Reduced resolution processing when needed
- Threading for responsive UI

## Supported Animals

With the custom trained model, the system can detect all 90+ animals from the Kaggle dataset, including:
- Mammals: antelope, bat, beaver, bison, boar, buffalo, camel, cat, cow, deer, dog, dolphin, donkey, elephant, fox, goat, gorilla, hamster, hedgehog, hippopotamus, horse, kangaroo, koala, leopard, lion, mouse, opossum, orangutan, otter, panda, pig, rabbit, raccoon, rat, rhinoceros, sheep, squirrel, tiger, whale, wolf, zebra
- Birds: eagle, flamingo, goose, hornbill, owl, parrot, pelecaniformes, pelican, penguin, sandpiper, sparrow, swan, woodpecker
- Marine life: crab, dolphin, jellyfish, lobster, octopus, oyster, pufferfish, seahorse, seal, shark, squid, starfish, whale
- Insects: bee, beetle, butterfly, caterpillar, cockroach, dragonfly, fly, grasshopper, ladybugs, mosquito, moth
- Reptiles: crocodile, lizard, snake, turtle
- Others: chimpanzee, crab, gorilla, okapi, porcupine, prawn

## Project Structure

```
animal_detection/
├── main.py                   # Main application entry point
├── train_animal_detector.py  # Training pipeline script
├── requirements.txt          # Project dependencies
├── README.md                 # This file
├── animal_detection/         # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset_handler.py
│   │   └── yolo_preprocessor.py
│   ├── models/               # Model training modules
│   │   ├── __init__.py
│   │   └── yolo_trainer.py
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── detection.py
│   └── ui/                   # GUI application
│       ├── __init__.py
│       └── gui_app.py
```

## Customization

You can customize the system by modifying the following:

1. **Confidence Threshold**: Adjust the minimum confidence required for detections in [config.py](animal_detection/config.py)
2. **Animal Information**: Update animal features in [config.py](animal_detection/config.py)
3. **Model Training**: Adjust training parameters in [train_animal_detector.py](train_animal_detector.py)

## Future Improvements

1. **Extended Animal Detection**: Continue improving the custom model with more training data
2. **Raspberry Pi Optimization**: Further optimize for edge deployment
3. **Improved Classification**: Add more detailed animal information and characteristics

## Troubleshooting

### Camera Issues
If you encounter camera problems:
1. Ensure the camera is properly connected
2. Check that no other applications are using the camera

### Performance Issues
If the system is running slowly:
1. Increase the frame skip interval in [main.py](main.py)
2. Reduce the camera resolution

### GUI Issues
If the GUI doesn't start:
1. Ensure tkinter is installed (usually included with Python)
2. Try running in CLI mode instead

### Training Issues
If training fails:
1. Ensure you have enough disk space
2. Check that the dataset path is correct
3. For faster training, use a GPU if available

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Animal Image Dataset provided by Sourav Banerjee on Kaggle
- Ultralytics for the YOLO implementation
- PyTorch for the deep learning framework
- OpenCV for computer vision utilities