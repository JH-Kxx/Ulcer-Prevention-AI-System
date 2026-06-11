# Pose-Pressure Ulcer AI

Real-Time Pressure Ulcer Prevention System using Pose Estimation and Pressure Sensing.

---

## 1. Project Overview

Pressure ulcers can occur when patients remain in the same posture for long periods of time.  
This project proposes a real-time monitoring system that combines:

- YOLO-based pose estimation
- GCN-based keypoint correction
- pressure sensing
- pose-pressure alignment
- real-time dashboard visualization

The goal is to monitor posture and pressure concentration simultaneously for early ulcer-risk analysis.

---

## 2. Pipeline

<p align="center">
  <img src="docs/pipline/system_pipeline.png" width="1000">
</p>

- #### Pose Estimation Pipeline
  - Capture real-time RGB images and estimate body keypoints, followed by GCN-based keypoint refinement.

- #### Pressure Sensing Pipeline
  - Acquire pressure sensor data and generate a smoothed body pressure heatmap through interpolation.

- #### Pose & Pressure Alignment
  - Temporally and spatially align pose keypoints with pressure maps for integrated risk analysis.

---

## 3. Setting
### 3-1. Full System

### 3-2. GCN Training

---

## 4. Usage
### 4-1. Project Structure

### 4-2. Full System Execution

### 4-3. GCN Training 

---

## 5. Results
### 5-1. Perfomance Comparison

### 5-2. GCN Refinement Result

### 5-3. Demonstration

<p align="center">
  <img src="docs/dashboard/system_demo.png" width="950">
</p>

#### Real-Time Features
- patient pose estimation
- pressure heatmap visualization
- synchronized pose-pressure monitoring
- real-time dashboard system
