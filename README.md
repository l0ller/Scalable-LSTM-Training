# Scalable Model Training on GPU Clusters

This project demonstrates how to train a large-scale LSTM model using distributed training on multi-GPU clusters. It uses TensorFlow and Horovod to scale deep learning workflows efficiently.

## 🚀 Highlights

- Distributed training with Horovod and TensorFlow
- Achieved **60% reduction** in training time for large LSTM models
- Integrated logging, checkpointing, and experiment reproducibility

## 🧠 Model Architecture

The model is a 3-layer LSTM network for time series forecasting using dummy input. Easily customizable for financial/time-series data.

## 🖥️ Prerequisites

- Python 3.8+
- GPUs with NCCL support
- MPI (OpenMPI or Intel MPI)
- [Horovod](https://github.com/horovod/horovod)

## 📦 Installation

```bash
pip install -r requirements.txt# Scalable-LSTM-Training
