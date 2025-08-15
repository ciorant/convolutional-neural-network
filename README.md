# Fashion-MNIST CNN Classifier

A **PyTorch** implementation of a Convolutional Neural Network for **Fashion-MNIST** image classification using object-oriented design principles.

## Overview
This project implements a CNN classifier for the Fashion-MNIST dataset, featuring a clean object-oriented architecture with a comprehensive training pipeline, including validation, early stopping, and model persistence.

## Features
- **Object-oriented** CNN implementation with modular design  
- Automatic feature map size calculation for dynamic architecture adaptation  
- Comprehensive training pipeline with validation monitoring  
- Early stopping to prevent overfitting  
- Model persistence with save/load functionality  
- Clean separation of concerns between data loading, model definition, and training  

## Architecture
The CNN consists of:

1. **Two convolutional blocks** with ReLU activation and max pooling  
2. **Dynamic classifier** that automatically adapts to feature map dimensions  
3. **Configurable hidden units** and training parameters  

    ```
    Input (1×28×28) 
        ↓
    Conv Block 1: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
        ↓
    Conv Block 2: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d  
        ↓
    Classifier: Flatten → Linear → Output (10 classes)
    ```
    
## Model Performance
The model achieves competitive performance on Fashion-MNIST:

- **Training accuracy:** Typically 85–90% after a few epochs  
- **Validation accuracy:** Typically 80–85%  
- **Fast training:** ~30 seconds per epoch on GPU  

## Project Structure
```
├── data.py           # Data loading and preprocessing
├── oop_cnn.py        # CNN model implementation
├── main.py           # Training script
└── README.md         # This file
```

## Key Implementation Details

### Dynamic Architecture
The model automatically calculates the flattened feature size after convolutional layers:
```python
with torch.no_grad():
    dummy = torch.zeros(1, input_channels, *image_size)
    dummy_out = self.block2(self.block1(dummy))
    flattened_size = dummy_out.numel()
```

### Training Features
- Automatic device selection (CUDA/CPU)  
- Comprehensive metrics tracking (loss, accuracy)  
- Early stopping with configurable patience  
- Validation monitoring at specified intervals  

### Model Persistence
Models are saved with complete configuration for reproducibility:
```python
model.save_model("model.pth")
loaded_model = Convolutional.load_model("model.pth")
```

### Methods
The `Convolutional` class provides a complete training interface:

- `setup_training()` – Configure optimizer and device  
- `fit()` – Train with validation and early stopping  
- `validate()` – Evaluate on validation set  
- `predict()` – Generate predictions  
- `save_model()` / `load_model()` – Model persistence  

## Dataset
Fashion-MNIST consists of **70,000 grayscale images** (28×28 pixels) across 10 clothing categories:

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Data split:
- 80% train  
- 20% validation  
- Separate test set  

## Requirements
- Python 3.7+  
- PyTorch 1.9+  
- torchvision  
- NumPy  

## Technical Highlights
- Clean OOP design with separation of concerns  
- Type hints for better code maintainability  
- Automatic GPU/CPU handling  
- Robust error handling in model loading  
- Flexible architecture supporting different image sizes  

## Future Enhancements
Potential improvements could include:

- Configuration file support  
- Additional CNN architectures  
- Data augmentation  
- Learning rate scheduling  
- Advanced regularization techniques  
