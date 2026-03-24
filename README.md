# Alzheimer's Disease Detection using Deep Learning 🧠

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![GPU](https://img.shields.io/badge/Optimized_for-RTX_5060-76B900?logo=nvidia&logoColor=white)](https://nvidia.com)

## 📋 Project Overview
This repository implements an automated system for classifying brain MRI scans into three categories: **Mild Impairment**, **Moderate Impairment**, and **Normal**. The research features an **Optimized Sequential CNN** and a **Transfer Learning (Inception v3)** implementation presented at the **MLIP-2025** conference.

## 📂 Repository Structure
The project is organized to show the evolution from architectural optimization to clinical evaluation:

* **`/notebooks`**: 
    * `01_Optimized_Sequential_CNN.ipynb`: Custom architecture with Batch Normalization and L2 Regularization.
    * `02_InceptionV3_Paper_Implementation.ipynb`: Core implementation for the MLIP-2025 research paper.
    * `03_Final_Model_Evaluation_Report.ipynb`: Deep-dive analysis of Precision, Recall, and F1-scores.
* **`/docs`**: Full research manuscript: *"Deep Learning based detection of Alzheimer's Disease"*.

## 🛠️ Technical Stack & Hardware Optimization
This project is optimized for **WSL2 (Ubuntu 22.04 LTS)** and high-performance **Blackwell-architecture** GPUs.

### **Blackwell GPU (RTX 5060) Stability**
To prevent Autotuner conflicts in WSL2, the following environment variables are enforced at runtime:
```python
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
```

## 📂 Dataset & Data Handling
The model is trained on the **Alzheimer's Combined Dataset**. 

> **Note on Data Access:** Due to the large file size, the dataset is hosted externally. Please refer to the **Google Drive link** provided below in description to download the required image folders.

## 📊 Performance Results
The model demonstrates high diagnostic reliability on the test set:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **92%** |
| **Moderate Impairment F1-Score** | **1.00** |
| **No Impairment (Normal) F1-Score** | **0.89** |
| **Mild Impairment F1-Score** | **0.87** |

## 📂 Dataset & Resources
To run the training or inference, please download the following resources:
- **Dataset:** .https://drive.google.com/file/d/1YAAUR4U1yWXoWWnIUeRyRBy4Ctfuaef7/view?usp=drive_link
- **Manuel Inference** .https://drive.google.com/file/d/1i5xg-Cwx8PM92D2i9C4VbhurkkbUANmh/view?usp=drive_link

## 👨‍🔬 About the Author
**Sampara Emmanuel Arther George**
* **B.Tech in CSE (AI & Machine Learning)**, Karunya Institute of Technology and Sciences.
* **Research**: This project was presented at the **MLIP-2025** conference in February 2025.
* **Connect**: [GitHub: Arther-Codes](https://github.com/Arther-Codes)
