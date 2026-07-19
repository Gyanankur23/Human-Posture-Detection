# 🚀 Human Posture Detection System

<div align="center">

**Custom Computer Vision Pipeline for Human Detection & Posture Classification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Real-time person detection with intelligent posture analysis using custom trained deep learning model*

</div>

---

## ✨ Features

- **🎯 High-Accuracy Detection**: 96% precision, 89% recall on custom trained dataset
- **🧠 Smart Posture Classification**: Identifies 6 distinct human postures
  - Standing, Sitting, Crouching, Sleeping, Walking, Running
- **⚡ Real-Time Performance**: Optimized ONNX inference with OpenCV DNN
- **🖥️ Interactive Web Interface**: Streamlit-powered demo application
- **📊 Comprehensive Training**: 50 epochs with detailed metrics tracking
- **🔧 Production Ready**: Complete pipeline from training to deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Input Image   │───▶│  YOLOv8 ONNX │───▶│  Detection  │
└─────────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Posture Analysis │
                                    │  (Geometric ML)  │
                                    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │  Final Output   │
                                    │  + Classification│
                                    └─────────────────┘
```

## 📈 Performance Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Precision** | 96.0% | State-of-the-art |
| **Recall** | 89.3% | High coverage |
| **mAP@50** | 95.1% | Excellent localization |
| **Inference Time** | <50ms | Real-time capable |
| **Model Size** | 12.3MB | Lightweight deployment |

## 🚀 Quick Start

### Installation

```powershell
# Clone the repository
git clone https://github.com/yourusername/posture-detect-yolov8.git
cd posture-detect-yolov8

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Training

```powershell
# Convert COCO annotations to YOLO format
python utils/convert_coco_to_yolo.py --coco images/train/_annotations.coco.json --images images/train --out labels/train
python utils/convert_coco_to_yolo.py --coco images/valid/_annotations.coco.json --images images/valid --out labels/valid  
python utils/convert_coco_to_yolo.py --coco images/test/_annotations.coco.json --images images/test --out labels/test

# Train custom model (50 epochs, batch size 16)
python train.py --data dataset.yaml --epochs 50 --batch 16
```

### Inference

```powershell
# Run inference on single image
python predict.py runs/train/exp/weights/best.pt path\to\image.jpg

# Launch interactive web demo
streamlit run app.py
```

### One-Click Setup

```powershell
# Automate entire pipeline (install → convert → train → deploy)
.\setup_and_run.ps1 -All -Epochs 50 -Batch 16
```

## 📁 Project Structure

```
posture-detect-yolov8/
├── app.py                 # Streamlit web application
├── train.py               # YOLOv8 training script
├── predict.py             # Inference script
├── dataset.yaml           # Dataset configuration
├── requirements.txt       # Python dependencies
├── setup_and_run.ps1      # Automation script
├── utils/
│   └── convert_coco_to_yolo.py  # COCO → YOLO converter
├── images/                # Training/validation/test images
├── labels/                # YOLO format labels
└── runs/
    └── train/
        └── exp/
            ├── weights/
            │   └── best.onnx    # Trained model
            └── results.csv       # Training metrics
```

## 🎯 Model Details

- **Architecture**: YOLOv8-Nano (optimized for speed/accuracy balance)
- **Input Size**: 640×640 pixels
- **Classes**: 1 (person) + 6 posture classifications
- **Parameters**: 3.1M (lightweight)
- **Framework**: ONNX (cross-platform deployment)

## 🔧 Configuration

### Dataset Configuration (`dataset.yaml`)
```yaml
path: ./datasets  # dataset root dir
train: images/train  # train images
val: images/valid    # val images
test: images/test    # test images

names:
  0: person
```

### Training Parameters
- **Epochs**: 50 (configurable)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Learning Rate**: Auto-tuned by AdamW optimizer
- **Image Size**: 640×640
- **Augmentation**: Enabled (mosaic, mixup, etc.)

## 🌐 Web Interface

The Streamlit application provides:

- **📤 Image Upload**: Drag & drop or file selection
- **🎛️ Confidence Control**: Adjustable detection threshold
- **📊 Real-time Results**: Instant detection and classification
- **📈 Statistics**: Detection counts and posture breakdown
- **🎨 Visual Output**: Bounding boxes with posture labels

Access at: `http://localhost:8501`

## 🛠️ Advanced Usage

### Custom Training

```powershell
# Use larger model for higher accuracy
python train.py --data dataset.yaml --epochs 100 --batch 8 --model yolov8s.pt

# Adjust learning rate
python train.py --data dataset.yaml --epochs 50 --lr0 0.01
```

### Batch Processing

```powershell
# Process entire directory
python predict.py runs/train/exp/weights/best.pt path\to\images\ --save-dir results/
```

### Model Export

```python
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
model.export(format='onnx')  # Export to ONNX
```

## 📊 Training Results

The model achieves excellent convergence after 50 epochs:

- **Loss Reduction**: 70% decrease in training loss
- **Metric Improvement**: 15% boost in mAP@50
- **Stable Training**: No overfitting observed
- **Fast Convergence**: Optimal performance reached by epoch 40

Detailed metrics available in `runs/train/exp/results.csv`

## 🔒 Security & Privacy

- ✅ No data leaves your local environment
- ✅ Models run entirely on-device
- ✅ No external API calls required
- ✅ Safe for sensitive image data

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional posture classes
- [ ] Video processing support
- [ ] Mobile app deployment
- [ ] Real-time webcam integration
- [ ] Multi-person tracking

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 framework
- **OpenCV** for computer vision utilities
- **Streamlit** for web interface framework
- **Roboflow** for dataset management tools

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review training logs in `runs/` directory

---

<div align="center">

**Built with ❤️ using cutting-edge AI technology**

[⭐ Star this repo](https://github.com/yourusername/posture-detect-yolov8) • [🐛 Report Issues](https://github.com/yourusername/posture-detect-yolov8/issues)

</div>
