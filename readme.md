# Traffic Sign Recognition Neural Network

This project implements a convolutional neural network (CNN) to classify German traffic signs using TensorFlow/Keras. It includes two main scripts: a basic trainer (`traffic.py`) and an advanced hyperparameter optimizer (`traffic_optimize.py`).

## 📁 Project Overview

The system can identify 43 different types of German traffic signs from images, including stop signs, speed limits, yield signs, and more. It uses computer vision and deep learning to achieve high accuracy classification.

**What the AI learns:**
- **Low-level features**: Edges, lines, curves, basic shapes
- **Mid-level features**: Geometric patterns, text regions, color boundaries  
- **High-level features**: Complete traffic sign concepts (circular vs triangular, red vs blue, etc.)
- **Final classification**: Specific sign identification (30 km/h vs 50 km/h speed limit)

## 🗂️ Data Structure

Your data directory should be organized like this:
```
gtsrb/
├── 0/          # Speed limit (20km/h)
│   ├── image1.ppm
│   ├── image2.ppm
│   └── ...
├── 1/          # Speed limit (30km/h)
│   ├── image1.ppm
│   └── ...
├── 2/          # Speed limit (50km/h)
└── ...
├── 42/         # Right-of-way at intersection
```

Each numbered folder (0-42) represents a different traffic sign category containing multiple image examples.

---

## 🚀 Script 1: traffic.py (Basic Training)

### What it does

`traffic.py` trains a convolutional neural network with predefined architecture to classify traffic signs. Think of it as the "standard recipe" approach.

### How it works

1. **Data Loading**: Reads all images from your dataset
   - Resizes each image to 30x30 pixels (standardization)
   - Normalizes pixel values from 0-255 to 0-1 (helps training)
   - Creates labels matching folder numbers to sign categories

2. **Neural Network Architecture**:
   ```
   Input (30x30x3 image)
          ↓
   Conv Layer 1 (32 filters, 3x3) → Detects edges, lines
          ↓
   Max Pooling (2x2) → Reduces size, keeps important features
          ↓
   Conv Layer 2 (32 filters, 3x3) → Detects shapes, patterns
          ↓
   Max Pooling (2x2) → Further size reduction
          ↓
   Flatten → Converts 2D to 1D for dense layers
          ↓
   Dense Layer (128 neurons) → Combines features
          ↓
   Dropout (50%) → Prevents overfitting
          ↓
   Output Layer (43 neurons) → Final classification
   ```

3. **Training Process**:
   - Splits data: 60% training, 40% testing
   - Trains for 10 epochs (complete passes through data)
   - Uses Adam optimizer with categorical crossentropy loss
   - Evaluates final accuracy on test set

### Command Line Usage

**Basic training (no model saving):**
```bash
python traffic.py gtsrb
```

**Training with model saving:**
```bash
python traffic.py gtsrb my_traffic_model.h5
```

**Parameters:**
- `gtsrb`: Directory containing your traffic sign dataset
- `my_traffic_model.h5`: (Optional) Filename to save trained model

### Example Output
```
running load_data ... function completed, returning populated lists images[] and labels[]
running get_model ... function completed, returning model
Epoch 1/10
486/486 ━━━━━━━━━━━━━━━━━━━━ 6s - loss: 3.7139 - accuracy: 0.1545
Epoch 2/10
486/486 ━━━━━━━━━━━━━━━━━━━━ 5s - loss: 2.0086 - accuracy: 0.4082
...
Epoch 10/10
486/486 ━━━━━━━━━━━━━━━━━━━━ 5s - loss: 0.2497 - accuracy: 0.9256
333/333 - 3s - loss: 0.1616 - accuracy: 0.9535
running main ... Model saved to my_traffic_model.h5
```

### Key Functions

- **`load_data(data_dir)`**: Loads and preprocesses all images
- **`get_model()`**: Creates the neural network architecture
- **`build_conv_pooling_layers()`**: Builds convolutional feature detection layers
- **`build_hidden_layers()`**: Builds dense classification layers

---

## 🔬 Script 2: traffic_optimize.py (Hyperparameter Optimization)

### What it does

`traffic_optimize.py` uses advanced Bayesian optimization to automatically find the best neural network configuration. Instead of using a fixed "recipe," it intelligently tests different combinations to maximize accuracy.

### How it works

**Think of it like a smart chef** who tries different ingredient combinations and cooking times to perfect a recipe, learning from each attempt to make better choices next time.

#### 1. Bayesian Optimization Process

**Optuna (the optimization engine) works like this:**
- **Trial 1**: Try random architecture → Get 94% accuracy
- **Trial 2**: Based on Trial 1, try similar but modified architecture → Get 96% accuracy  
- **Trial 3**: Learning from both, try more targeted changes → Get 95% accuracy
- **Continue**: Each trial informs better choices for the next trial

#### 2. What Gets Optimized

**Convolutional Layers:**
- `num_layers_conv`: 1-3 layers (more layers = detect more complex features)
- `nodes_per_conv_layer`: 16, 32, 64, 128, 256, or 512 filters
- `kernel_size`: 3x3, 3x5, 5x3, or 5x5 filters (different feature detectors)
- `pool_size`: 2x2 or 3x3 downsampling

**Hidden Layers:**
- `num_layers_hidden`: 1-5 layers
- `first_hidden_layer_nodes`: 64, 128, 256, or 512 neurons
- `subsequent_hidden_layer_nodes_decrease`: How much to shrink each layer (0.25-0.75)
- `dropout`: How much to randomly disable neurons (prevents overfitting)

**Training Parameters:**
- `epochs`: 5-15 training cycles
- `learning_rate`: 0.0001 to 0.01 (how fast the AI learns)

#### 3. Smart Features

**Early Stopping**: If a model isn't improving after 3 epochs, stop training early

**Pruning**: If a trial is performing much worse than others after 3 epochs, abandon it and try something else

**Validation Split**: Uses 80% for training, 20% for validation during optimization (never touches the final test set until the end)

### Command Line Usage

**Run optimization:**
```bash
python traffic_optimize.py gtsrb optimized_model.h5
```

**Parameters:**
- `gtsrb`: Directory containing your traffic sign dataset  
- `optimized_model.h5`: Filename for the best model found

### Example Output

```
Getting baseline accuracy...
Baseline accuracy: 0.9697

Starting Bayesian optimization with 10 trials...
This will take several hours. Go get coffee! ☕

Trial 0: Val Accuracy = 0.9498
  Conv layers: 2, nodes: 16
  Hidden layers: 2, nodes: 512
  Kernel: (3,3), Epochs: 6

Trial 1: Val Accuracy = 0.9681
  Conv layers: 2, nodes: 32  
  Hidden layers: 2, nodes: 128
  Kernel: (3,5), Epochs: 7

Trial 4 failed: Negative dimension size (architecture issue)
Trial 6 failed: Trial was pruned at epoch 3 (poor performance)

============================================================
OPTIMIZATION COMPLETE!
============================================================

Best validation accuracy: 0.9681
Best parameters:
  num_layers_conv: 2
  nodes_per_conv_layer: 32
  kernel_size_x: 3
  kernel_size_y: 5
  pool_size: 2
  num_layers_hidden: 2
  first_hidden_layer_nodes: 128
  epochs: 7
  learning_rate: 0.0021137059440645744

Final test accuracy: 0.9705
Improvement over baseline: 0.0009
```

### Key Functions

- **`objective(trial)`**: Function that Optuna calls to test each architecture
- **`build_optimized_model(params)`**: Builds model with Optuna-suggested parameters  
- **`run_hyperparameter_optimization(n_trials)`**: Manages the entire optimization process

---

## 🔍 Understanding the Results

### What the numbers mean

**Accuracy**: Percentage of traffic signs correctly identified
- 0.95 = 95% accuracy = 95 out of 100 signs correctly identified
- For traffic signs, 95%+ is excellent (safety critical!)

**Loss**: How "wrong" the predictions are
- Lower is better
- Starts high (model is guessing randomly)
- Decreases as model learns

### Why some trials fail

**Dimension Errors**: 
```
Trial 4 failed: Negative dimension size
```
- Too many convolutional layers shrunk the image too small
- Like trying to cut a 2x2 piece of paper with 3x3 scissors

**Pruning**:
```
Trial 6 failed: Trial was pruned at epoch 3
```
- Model was performing poorly compared to others
- Optuna stopped it early to save time (smart!)

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install tensorflow opencv-python scikit-learn optuna
```

### Dataset Download

**The dataset is not included in this repository due to its large size (>80MB). You must download it separately:**

1. **Download the German Traffic Sign Recognition Benchmark (GTSRB) dataset:**
   ```bash
   wget https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
   ```

2. **Extract the dataset:**
   ```bash
   unzip gtsrb.zip
   ```

3. **Move the dataset to your project directory:**
   ```bash
   mv gtsrb/ /path/to/your/project/
   ```

**Alternative download method:**
- Visit: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
- Download manually and extract to your project directory

### File Structure
```
project/
├── traffic.py
├── traffic_optimize.py  
├── gtsrb/              # Downloaded dataset (not in repo)
│   ├── 0/              # Speed limit (20km/h)
│   ├── 1/              # Speed limit (30km/h)
│   ├── 2/              # Speed limit (50km/h)
│   └── ...
│   └── 42/             # Right-of-way at intersection
├── README.md
└── instructions_traffic.pdf
```

**Important**: The `gtsrb/` folder contains thousands of traffic sign images (~39,000 total) and is excluded from version control due to size constraints.

---

## 🧠 Neural Network Architecture Explained

### For Beginners: How CNNs Work

**Think of identifying a stop sign:**

1. **Convolutional Layers** (Feature Detectors):
   - **Layer 1**: Detects basic features like edges, lines
     - "I see a circular edge here"
     - "I see a straight line there"
   
   - **Layer 2**: Combines basic features into shapes
     - "These edges form an octagon"
     - "This area has red color"

2. **Pooling Layers** (Simplification):
   - Reduces image size while keeping important information
   - Like making a thumbnail that preserves key details

3. **Dense Layers** (Decision Making):
   - **Hidden Layer**: "Red octagon with white border = probably a stop sign"
   - **Output Layer**: "99% confident this is a stop sign"

### Architecture Comparison

**traffic.py (Fixed)**:
```
Input → Conv(32) → Pool → Conv(32) → Pool → Dense(128) → Output(43)
```

**traffic_optimize.py (Optimized Example)**:
```
Input → Conv(32) → Pool → Conv(32) → Pool → Dense(128) → Dense(63) → Output(43)
```

The optimizer found that:
- 2 convolutional layers work best for 30x30 images
- 32 filters per layer is optimal
- 3x5 kernels detect horizontal features better than 3x3
- 2 hidden layers with decreasing size works well

---

## 🔧 Customization Options

### Modifying traffic.py

**Change image size:**
```python
IMG_WIDTH = 64   # Instead of 30
IMG_HEIGHT = 64  # Instead of 30
```

**Adjust training:**
```python
EPOCHS = 20      # Train longer
TEST_SIZE = 0.3  # Use more data for training
```

**Modify architecture in `get_model()`:**
```python
conv_layers = build_conv_pooling_layers(
    num_layers_conv=3,           # More layers
    nodes_per_conv_layer=64,     # More filters
    kernel_size=(5, 5),          # Larger kernels
)
```

### Modifying traffic_optimize.py

**Change optimization range:**
```python
'num_layers_conv': trial.suggest_int('num_layers_conv', 1, 4),  # Try up to 4 layers
'nodes_per_conv_layer': trial.suggest_categorical('nodes_per_conv_layer', [8, 16, 32, 64]),  # Smaller range
```

**More trials:**
```python
study, best_model = run_hyperparameter_optimization(n_trials=50)  # Takes much longer!
```

---

## 📊 Performance Tips

### For Better Results

1. **More Data**: More images = better accuracy
2. **Data Augmentation**: Rotate, flip, zoom images to create variations
3. **Longer Training**: Increase epochs if accuracy is still improving
4. **Ensemble Methods**: Combine multiple models' predictions

### For Faster Training

1. **GPU**: Use CUDA-enabled GPU (10x faster)
2. **Smaller Images**: Reduce IMG_WIDTH/IMG_HEIGHT  
3. **Fewer Trials**: Start with 10 trials for optimization
4. **Early Stopping**: Already implemented in optimizer

---

## 🚨 Troubleshooting

### Common Issues

**"Could not find cuda drivers"**:
- Normal warning if you don't have a GPU
- Training will use CPU (slower but works)

**"Negative dimension size"**:
- Too many convolutional layers for small images
- Reduce `num_layers_conv` or increase image size

**"Out of memory"**:
- Reduce batch size or image dimensions
- Use fewer/smaller layers

**Low accuracy (<90%)**:
- Check data quality and labels
- Try more epochs or different architecture
- Ensure images are properly preprocessed

### Getting Help

1. Check that your dataset follows the expected folder structure
2. Verify all images can be loaded (check for corrupted files)
3. Monitor training logs for unusual patterns
4. Start with `traffic.py` before trying optimization

---

## 🎯 Example Workflow

### Quick Start (30 minutes):
```bash
# 1. Basic training
python traffic.py gtsrb basic_model.h5

# 2. Check results, then optimize if needed
python traffic_optimize.py gtsrb optimized_model.h5
```

### Full Workflow (Several Hours):
```bash
# 1. Start with basic model
python traffic.py gtsrb baseline.h5

# 2. Run extensive optimization  
python traffic_optimize.py gtsrb best_model.h5

# 3. Compare results and choose best model
```

This comprehensive system gives you both a reliable baseline (`traffic.py`) and the ability to push performance boundaries (`traffic_optimize.py`) for your traffic sign recognition project!