# Pose-Pressure Ulcer AI

Real-time Pressure Ulcer Prevention System using Pose Estimation and Pressure Sensing

---

## 0. Project Overview

### Overview

This project proposes a real-time pressure ulcer prevention system based on human pose estimation and pressure sensing.
The system integrates camera-based posture analysis with pressure distribution data to monitor patient posture, analyze pressure concentration areas, and visualize ulcer risk in real time.

The proposed pipeline consists of:

* Pose Estimation using YOLO-based models
* GCN-based keypoint refinement
* Pressure sensing and pressure map generation
* Pose & Pressure spatial-temporal alignment
* Real-time monitoring dashboard

---

## 1. Demo Video

### System Demonstration

<div align="center">

Add demonstration GIF or YouTube link here

</div>

---

## 2. Setup

### 2-1. Environment Setup

#### Requirements

* Python 3.10
* PyTorch
* CUDA Toolkit
* OpenCV
* Flask

#### Installation

```bash
pip install -r requirements.txt
```

---

### 2-2. Hardware Setup

#### Components

| Component       | Description                           |
| --------------- | ------------------------------------- |
| Camera          | Real-time patient posture acquisition |
| Pressure Sensor | Pressure distribution measurement     |
| Arduino         | Sensor data transmission              |
| PC / GPU Server | Real-time inference and visualization |

---

### 2-3. Dataset Preparation

#### Datasets

* SLP Dataset
* OCHuman Dataset

#### Dataset Structure

```text
datasets/
├── images/
├── labels/
└── pressure/
```

---

## 3. System Pipeline

### 3-1. Pose Estimation Pipeline

The pose estimation pipeline performs real-time human posture analysis using YOLO-based pose estimation models.
To improve robustness under occlusion and noisy environments, GCN-based keypoint refinement is additionally applied.

#### Pipeline

```text
Input Image
    ↓
YOLO Pose Estimation
    ↓
17-to-14 Keypoint Mapping
    ↓
GCN Refinement
    ↓
Refined Pose Output
```

---

### 3-2. Pressure Sensing Pipeline

The pressure sensing pipeline acquires pressure distribution data from the pressure sensor system and generates pressure maps for risk analysis.

#### Pipeline

```text
Pressure Sensor
    ↓
Arduino Serial Communication
    ↓
Pressure Data Acquisition
    ↓
Pressure Map Generation
```

---

### 3-3. Pose & Pressure Alignment

Pose and pressure information are spatially and temporally synchronized to analyze posture-dependent pressure concentration regions.

#### Core Features

* Spatial alignment
* Temporal synchronization
* Pressure concentration analysis
* Risk region visualization

---

### 3-4. Dashboard

The dashboard visualizes real-time pose estimation results, pressure maps, and ulcer risk analysis outputs.

#### Features

* Real-time monitoring
* Pressure map visualization
* Risk score visualization
* Web-based dashboard

---

## 4. Future Work

### Infrared Camera for Low-light Environments

Future work includes the application of infrared cameras to improve pose estimation performance under low-light or nighttime hospital environments.

### Temperature Sensor Fusion

Additional temperature sensing modalities will be integrated to improve ulcer risk analysis by considering localized temperature changes.

### Edge AI Optimization

The system will be optimized for lightweight real-time inference in practical hospital environments and edge devices.
