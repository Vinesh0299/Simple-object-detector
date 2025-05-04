# Simple Object Detector

This project is a simple object detection pipeline that downloads videos, processes them into labeled datasets, and trains a YOLO model for object detection.

## Features

- **YouTube Video Downloader**: Downloads videos from YouTube using `yt-dlp`.
- **Dataset Creator**: Converts videos into labeled image datasets with bounding boxes.
- **YOLO Model Training**: Trains a YOLO model using the processed dataset.
- **Logging**: Provides detailed logs for debugging and monitoring.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vinesh0299/Simple-object-detector.git
   cd Simple-object-detector
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python training.py
   ```
