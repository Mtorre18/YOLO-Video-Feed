# Inference Service

Handles real-time video inference by integrating preprocessing, object detection, 
and Non-Maximum Suppression (NMS) filtering.

This service continuously captures video frames, applies object detection, 
filters results using NMS, and outputs predictions.

## **How to Run**

Follow these steps to run the inference service:

### **1. Prerequisites**
Install FFmpeg for you system

https://ffmpeg.org/download.html

ensure it is added to your PATH

Install dependencies using pip:

```bash
pip install opencv-python numpy
```

### **2. Set Up YOLO Model Files**
Ensure the following model files are in `storage/yolo_models` directory:
- **Weights file:** `yolov4-tiny-logistics_size_416_1.weights`
- **Configuration file:** `yolov4-tiny-logistics_size_416_1.cfg`
- **Class names file:** `logistics.names`

### **3. Additional settings** 
- **Frame Drop Rate:**
  - Adjust the `drop_rate` parameter in the `Preprocessing` class to control how many frames are skipped:
    ```python
    stream = Preprocessing(VIDEO_SOURCE, drop_rate=60)  
    ```
- **Processed Frames Save Directory:**
  - The processed frames with bounding boxes will be saved in the `output/` directory by default.
  - You can change the save_dir path if you want it elsewhere

### **4. Run the Script**
Run the Python script with the following command:
```bash
python app.py
```

### **5. Run the video feed**
- **Video Source:** Using a separate terminal, run the following, change video path as needed:
  ```bash
    ffmpeg -re -i worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000 
  ```

 Press `q` to stop the script.

