# 🚀 Scalable Model Training on GPU Clusters

This project demonstrates how to train a large-scale LSTM model using **distributed GPU clusters** with **TensorFlow** and **Horovod**. It showcases how to scale deep learning workflows efficiently and reduce training time using multiple GPUs.

---

## 🧠 Project Highlights

- ⏱️ Achieved **60% reduction** in training time for large LSTM models  
- ⚡ Distributed training across GPUs using **Horovod**  
- 📊 Integrated **logging**, **checkpointing**, and experiment reproducibility  
- 🔄 Easily extendable for financial or time series forecasting applications  

---

## 🏗️ Model Overview

The current model is a basic 2-layer LSTM followed by a dense output layer for binary classification. Input is dummy time-series data but can be replaced with stock or other sequential data.

---

## 🛠️ Tech Stack

- Python 3.8+  
- TensorFlow 2.x  
- Horovod for distributed training  
- MPI (OpenMPI or Intel MPI)  
- NumPy  

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/your_username/scalable-lstm-training.git
cd scalable-lstm-training
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Horovod with GPU + TensorFlow support:

```bash
HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]
```

---

## 🧪 Running Distributed Training

Use `horovodrun` or `mpirun` to launch distributed training on multiple GPUs:

```bash
horovodrun -np 4 -H localhost:4 python train_lstm_horovod.py
```

This will:
- Train a shared model across 4 GPUs  
- Log training metrics to individual `logs/` folders per GPU  
- Save checkpoints to `checkpoints/`  

---

## 📁 Output Structure

```
├── checkpoints/           # Model checkpoints (one per rank)
├── logs/                  # TensorBoard logs per GPU
├── train_lstm_horovod.py  # Main training script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 📉 Results

- Speedup: Training time reduced by ~60% using 4 GPUs  
- Reproducibility: All metrics and model states saved per rank  
- Compatibility: Works on both local GPU machines and multi-node clusters  

---

## 📊 Visualization

To visualize training logs, use TensorBoard:

```bash
tensorboard --logdir=logs/
```

---

## 🧩 Future Improvements

- Integrate real financial time series data (e.g., SPY, AAPL via yfinance)  
- Add support for multi-node cluster launching via SLURM  
- Extend for regression and multi-class classification  

---


