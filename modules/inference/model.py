import cv2
import numpy as np
from typing import List, Tuple
import time

class Detector:
    """
    A class that represents an object detection model using OpenCV's DNN module
    with a YOLO-based architecture.
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str, score_threshold: float=.5) -> None:
        """
        Initializes the YOLO model by loading the pre-trained network and class labels.

        :param weights_path: Path to the pre-trained YOLO weights file.
        :param config_path: Path to the YOLO configuration file.
        :param class_path: Path to the file containing class labels.

        :ivar self.net: The neural network model loaded from weights and config files.
        :ivar self.classes: A list of class labels loaded from the class_path file.
        :ivar self.img_height: Height of the input image/frame.
        :ivar self.img_width: Width of the input image/frame.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load class labels
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.img_height: int = 0
        self.img_width: int = 0

        self.score_threshold = score_threshold

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """
        Runs the YOLO model on a single input frame and returns raw predictions.

        :param preprocessed_frame: A single image frame that has been preprocessed 
                                   for YOLO model inference (e.g., resized and normalized).

        :return: A list of NumPy arrays containing the raw output from the YOLO model.
                 Each output consists of multiple detections with bounding boxes, 
                 confidence scores, and class probabilities.

        :ivar self.img_height: The height of the input image/frame.
        :ivar self.img_width: The width of the input image/frame.

        **YOLO Output Format:**
        Each detection in the output contains:
        - First 4 values: Bounding box center x, center y, width, height.
        - 5th value: Confidence score.
        - Remaining values: Class probabilities for each detected object.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        self.img_height, self.img_width = preprocessed_frame.shape[:2]

        # Give the configuration and weight files for the model and load the network.

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        ln = self.net.getUnconnectedOutLayersNames()  
        print(f"layers: {ln}")


        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(preprocessed_frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)

        t0 = time.time()
        outputs = self.net.forward(ln)
        t = time.time()

        print(f'forward propagation time={t-t0}')

        # Return model outputs:
        return outputs

    def post_process(
        self, predict_output: List[np.ndarray]
    ) -> Tuple[List[List[int]], List[int], List[float], List[np.ndarray]]:
        """
        Processes the raw YOLO model predictions and filters out low-confidence detections.

        :param predict_output: A list of NumPy arrays containing raw predictions 
                               from the YOLO model.
        :param score_threshold: Minimum confidence score required for a detection 
                                to be considered valid.

        :return: A tuple containing:
            - **bboxes (List[List[int]])**: List of bounding boxes as `[x, y, width, height]`, 
              where (x, y) represents the top-left corner.
            - **class_ids (List[int])**: List of detected object class indices.
            - **confidence_scores (List[float])**: List of confidence scores for each detection.
            - **class_scores (List[np.ndarray])**: List of all class-specific confidence scores.


        **Bounding Box Conversion:**
        YOLO outputs bounding box coordinates in the format:
        ```
        center_x, center_y, width, height
        ```
        This function converts them to:
        ```
        x, y, width, height
        ```
        where (x, y) is the top-left corner.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        


        bboxes = []
        confidence_scores = []
        class_ids = []
        class_scores = []

        for output in predict_output:
            for detection in output:
                scores = detection[5:]  
                classID = np.argmax(scores)
                confidence = detection[4]

                if confidence > self.score_threshold:  
                    x, y, w, h = detection[:4] * np.array(
                        [self.img_width, self.img_height, self.img_width, self.img_height]
                    )

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    width = int(w)
                    height = int(h)
                    bbox = [x1, y1, width, height]

                    bboxes.append(bbox)
                    confidence_scores.append(confidence)
                    class_ids.append(classID)
                    class_scores.append(scores)  

            # Return these variables in order:
            return bboxes, class_ids, confidence_scores, class_scores

"""
EXAMPLE USAGE:
model = Detector()

# Perform object detection on the current frame
predictions = self.detector.predict(frame)

# Extract bounding boxes, class IDs, confidence scores, and class-specific scores
bboxes, class_ids, confidence_scores, class_scores = self.detector.post_process(
    predictions
)
"""
