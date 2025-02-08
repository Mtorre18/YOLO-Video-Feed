import numpy as np
from typing import List, Tuple


class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.

        :ivar self.score_threshold: The threshold below which detections are discarded.
        :ivar self.nms_iou_threshold: The IoU threshold that determines whether two boxes 
                                      are considered redundant.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[np.ndarray],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.

        :return: A tuple containing:
            - **filtered_bboxes (List[List[int]])**: The final bounding boxes after NMS.
            - **filtered_class_ids (List[int])**: The class IDs of retained bounding boxes.
            - **filtered_scores (List[float])**: The confidence scores of retained bounding boxes.
            - **filtered_class_scores (List[float])**: The class-specific scores of retained boxes.

        **How NMS Works:**
        - The function selects the bounding box with the highest confidence.
        - It suppresses any boxes that have a high IoU (overlapping area) with this selected box.
        - This process is repeated until all valid boxes are retained.

        **Example Usage:**
        ```python
        nms_processor = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        final_bboxes, final_class_ids, final_scores, final_class_scores = nms_processor.filter(
            bboxes, class_ids, scores, class_scores
        )
        ```
        """
        
        if len(bboxes) == 0:
            return [], [], [], []



        bboxes = np.array(bboxes)
        confidence_scores = np.array(scores)
        class_ids = np.array(class_ids)
        class_scores = np.array(class_scores)


        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = x1 + bboxes[:, 2]
        y2 = y1 + bboxes[:, 3]

        indices = np.argsort(confidence_scores)[::-1] 

        filtered_bboxes = []  
        filtered_class_ids=[]
        filtered_scores=[]
        filtered_class_scores=[]

        while len(indices) > 0:
            best_index = indices[0]  
            filtered_bboxes.append(bboxes[best_index].tolist())
            filtered_class_ids.append(class_ids[best_index])
            filtered_scores.append(confidence_scores[best_index])
            filtered_class_scores.append(class_scores[best_index].tolist())



            x1_best, y1_best, x2_best, y2_best = x1[best_index], y1[best_index], x2[best_index], y2[best_index]

            x1_remain, y1_remain, x2_remain, y2_remain = x1[indices], y1[indices], x2[indices], y2[indices]

            #intersection area
            inter_x1 = np.maximum(x1_best, x1_remain)
            inter_y1 = np.maximum(y1_best, y1_remain)
            inter_x2 = np.minimum(x2_best, x2_remain)
            inter_y2 = np.minimum(y2_best, y2_remain)

            inter_width = np.maximum(0, inter_x2 - inter_x1)
            inter_height = np.maximum(0, inter_y2 - inter_y1)
            intersection_area = inter_width * inter_height

            # areas of each box
            best_area = (x2_best - x1_best) * (y2_best - y1_best)
            remain_areas = (x2_remain - x1_remain) * (y2_remain - y1_remain)

            # calc IoUs
            union_area = best_area + remain_areas - intersection_area
            ious = intersection_area / union_area

            new_indices = []

            for i in indices[1:]:  
                if (i - 1) < len(ious):
                    if ious[i - 1] < self.nms_iou_threshold:
                        new_indices.append(i)

            indices = new_indices
  

        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores

