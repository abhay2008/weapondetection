# 🛡️ Real-Time Edge Computer Vision Weapon Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV DNN](https://img.shields.io/badge/OpenCV-DNN_Module-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv3](https://img.shields.io/badge/YOLO-Darknet-00FFFF?style=for-the-badge)](https://pjreddie.com/darknet/yolo/)
[![Firebase](https://img.shields.io/badge/Firebase-RTDB_%26_Storage-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An intelligent edge security surveillance engine that detects firearms, knives, and weapons in real-time video feeds using deep neural networks (YOLOv3 / Darknet via OpenCV DNN). Upon threat detection, it instantly triggers security alarms, stores image evidence in Firebase Cloud Storage, and syncs incident logs to Firebase Realtime Database.

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[CCTV / Webcam Stream] --> B[OpenCV Video Capture 30+ FPS]
    B --> C[Blob Preprocessing: 416x416 Normalization]
    C --> D[YOLOv3 Darknet DNN Inference Engine]
    D --> E[Non-Maximum Suppression NMS Threshold: 0.4]
    E -->|Confidence > 0.5| F{Weapon Detected?}
    F -->|No| B
    F -->|Yes| G[Draw Bounding Boxes & Confidence Labels]
    G --> H[Upload Evidence Snapshot to Firebase Storage]
    G --> I[Push Incident Alert to Firebase Realtime DB]
    G --> J[Trigger Local Security Alarm / Siren]
```

---

## ✨ Features

- **⚡ Real-Time Object Detection:** Deep neural network inference utilizing OpenCV's optimized C++ DNN module with GPU/CPU acceleration.
- **🎯 Non-Maximum Suppression (NMS):** Eliminates overlapping bounding boxes and suppresses low-confidence background noise.
- **☁️ Cloud Incident Synchronization:** Integrates with `Pyrebase` to automatically stream incident timestamps, camera IDs, and high-res evidence snapshots to Firebase.
- **🔔 Edge Alerting:** Sub-second latency from detection to cloud alert broadcast for home security dashboards and mobile clients.

---

## 📂 Repository Contents

```
weapondetection/
├── weapon_detection.py    # Main detection loop, inference engine & Firebase sync
├── yolov3_testing.cfg     # Darknet YOLOv3 network topology & anchor specifications
├── img.png                # Sample validation and inference snapshot
└── README.md              # Project documentation
```

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.8+
- Webcam or IP Camera stream (RTSP/HTTP)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/abhay2008/weapondetection.git
cd weapondetection

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy pyrebase4
```

### 2. Download Model Weights
Download the pre-trained `yolov3.weights` or custom-trained `yolov3_weapon.weights` file and place it in the project root:
```bash
# Example weight file download
wget https://pjreddie.com/media/files/yolov3.weights
```

### 3. Run Weapon Detection

```bash
python weapon_detection.py
```

---

## 👨‍💻 Author

**Abhay Kashyap**
- GitHub: [@abhay2008](https://github.com/abhay2008)
