# Eyes Track - Anti-Cheating System for Online Exams

## Overview

This script is a proof-of-concept for an anti-cheating system designed for online exams. It leverages computer vision and machine learning techniques to monitor and detect suspicious activities during an exam. The system can:

- Count the number of times the user's eyes look right or left.
- Detect if the user's eyes are away from the screen.
- Count the number of people looking at the screen.
- Detect if a phone is being held by the user.
- Control the mouse cursor, always moving it back to the initial position in the open window.

**Note**: This script is for academic purposes only. It requires further development to be suitable for production environments and could be implemented in other programming languages for better performance and scalability.

## Requirements

Before running the script, ensure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- NumPy (`numpy`)
- YOLOv3 weights and configuration files (`yolov3.weights`, `yolov3.cfg`)
- COCO names file (`coco.names`)

You can install the required Python packages using pip:

```bash
pip install opencv-python mediapipe pyautogui numpy
.

#How to run it 
Run the Script:

Navigate to the directory containing the script and run it with Python:
```bash
python eyestrack.py
