# Pose-Pressure Ulcer AI

Real-time Pressure Ulcer Prevention System using Pose Estimation and Pressure Sensing

<p align="center">
  <img src="results/images/system_demo.png" width="900">
</p>

---

## Overview

Pose-Pressure Ulcer AI is a real-time pressure ulcer prevention system that integrates human pose estimation and pressure sensing technologies.

The proposed system simultaneously analyzes patient posture and pressure distribution to monitor prolonged pressure concentration regions and visualize ulcer risk in real time.

### Key Features

* YOLO-based real-time pose estimation
* Residual GCN-based keypoint refinement
* Pressure sensing and pressure map generation
* Pose & pressure spatial-temporal alignment
* Real-time monitoring dashboard

---

## 1. Setup

### 1-1. Environment Setup

#### Requirements

* Python 3.10
* PyTorch
* CUDA Toolkit
* OpenCV
* Flask

---

### Installation

```bash
git clone https://github.com/JH-Kxx/Pose-Pressure-Ulcer-AI.git

cd Pose-Pressure-Ulcer-AI

pip install -r requirements.txt
```

---

### 1-2. Hardware Setup

#### Components

| Component            | Description                           |
| -------------------- | ------------------------------------- |
| Camera               | Real-time patient posture acquisition |
| Pressure Sensor      | Pressure distribution measurement     |
| Arduino              | Sensor data transmission              |
| PC / GPU Workstation | Real-time inference and visualization |

---

#### Arduino Connection

<p align="center">
  <img src="docs/hardware/arduino_connection.png" width="700">
</p>

---

#### Pressure Sensor Bed Setup

<p align="center">
  <img src="docs/hardware/pressure_sensor_bed.png" width="700">
</p>

---

### 1-3. Dataset Preparation

#### Datasets

* SLP Dataset

---

#### Dataset Structure

```text
datasets/
├── images/
├── labels/
└── pressure/
```

---
