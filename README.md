# MADDoSNet: Multi-Head Attention-based DDoS Detection Network

## Overview
MADDoSNet is a **deep learning-based model** designed to detect **Distributed Denial-of-Service (DDoS) attacks** using a combination of CNN, Bi-GRU, and Multi-Head Attention mechanisms. It processes network traffic data and identifies malicious patterns with high accuracy.

## Features
- **Multi-Modal Architecture:** Combines **CNN**, **Bidirectional GRU**, and **Multi-Head Attention** for feature extraction.
- **Efficient Training & Inference:** Optimized with **AdamW optimizer** and **learning rate scheduling**.
- **Robust Preprocessing:** Standardizes network traffic features and removes unnecessary attributes.
- **Comprehensive Evaluation:** Computes **accuracy, precision, recall, F1-score, AUC, and confusion matrices**.

## Dataset
The model is trained on a dataset containing **normal and attack network traffic samples**, stored in CSV files:
- `dataset_normal.csv`: Contains legitimate traffic data.
- `dataset_attack.csv`: Contains DDoS attack traffic.
The datasets can be found here:
```
https://gitlab.com/santhisenan/ids_iscx_2012_dataset
```

## Installation
### Clone the repository
```sh
git clone https://github.com/yourusername/MADDoSNet.git
cd MADDoSNet
```
### Install dependencies
```sh
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn
```

## Data Preprocessing
- Loads the dataset using `pandas`
- Drops unnecessary columns (`ip.src`, `ip.dst`, `frame.protocols`)
- Standardizes features using `StandardScaler`
- Converts labels into binary format (0 = attack, 1 = normal)
- Creates **25-length sequences** for training using a rolling window

## Model Architecture
The MADDoSNet model consists of:
1. **Convolutional Layers:** Extract spatial features from network traffic sequences.
2. **Residual Connections:** Enhances gradient flow and prevents degradation.
3. **Bidirectional GRU:** Captures sequential dependencies in the traffic data.
4. **Multi-Head Attention:** Enhances focus on important traffic patterns.
5. **Global Pooling Layers:** Reduces feature dimensionality.
6. **Dense Layers:** Final classification with dropout regularization.
7. **Sigmoid Activation:** Outputs probability of attack detection.

## Training the Model
Run the script to train the model:
```sh
python train.py
```
- Uses **binary cross-entropy loss** and **AdamW optimizer**.
- Learning rate scheduling with `ReduceLROnPlateau`.
- Runs for **500 epochs** with **20% validation split**.

## Evaluation Metrics
The trained model is evaluated on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUC (Area Under the Curve)**
- **Confusion Matrix** (Heatmap visualization)



## Visualizations
- **Accuracy and Loss Curves** are saved as `Model Accuracy.png` and `Model Loss.png`.
- **Confusion Matrix Heatmap** for attack vs. normal classification.

## Future Improvements
- Extend to **multi-class attack detection**.
- Optimize model for **real-time intrusion detection systems**.
- Integrate with **edge computing solutions** for low-latency detection.

## License
This project is licensed under the **MIT License**.




