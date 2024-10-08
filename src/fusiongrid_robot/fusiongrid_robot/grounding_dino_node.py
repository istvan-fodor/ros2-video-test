import sys
print("Curent Python: ", sys.executable)


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import torch
import numpy as np
from PIL import Image as PILImage
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline



@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def annotate(image: Union[PILImage.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    image_cv2 = np.array(image) if isinstance(image, PILImage.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    seed = 734546733
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        np.random.seed(seed)
        color = np.random.randint(150, 256, size=3)
        seed = seed + 1

        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def load_image(image_str: str) -> PILImage.Image:
    if image_str.startswith("http"):
        import requests
        image = PILImage.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = PILImage.open(image_str).convert("RGB")
    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(image: PILImage.Image, labels: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]
    return results

def segment(image: PILImage.Image, detection_results: List[Dict[str, Any]], polygon_refinement: bool = False) -> List[DetectionResult]:
    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(image: Union[PILImage.Image, str], labels: List[str], threshold: float = 0.3, polygon_refinement: bool = False) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold)

    if len(detections) > 0:
        detections = segment(image, detections, polygon_refinement)

    return np.array(image), detections

class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')

        self.subscription = self.create_subscription(
            CompressedImage,
            'camera/image',
            self.image_callback,
            1
        )

        self.sensor_publisher = self.create_publisher(CompressedImage, 'sensor/image/compressed', 10)

        self.bridge = CvBridge()

        self.latest_image = None
        self.processing = False
        
    def image_callback(self, msg):
        if self.processing:
            self.get_logger().info('Still processing the previous image, skipping new message...')
            return

        self.processing = True

        try:
            self.get_logger().info('Received a new compressed image, processing...')
            self.latest_image = msg

            # Convert the ROS Compressed Image message to a NumPy array
            np_arr = np.frombuffer(msg.data, np.uint8)
    
            # Decode the image using OpenCV
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = PILImage.fromarray(frame)
            
            sensor_image = ai_model_segment_image(image, labels=['cellphone', 'headphone', 'toy'], threshold=0.4)
            
            if sensor_image is not None:
                self.get_logger().info('Sensor detected! Publishing segmented image...')
                
                sensor_msg = self.bridge.cv2_to_compressed_imgmsg(sensor_image)

                self.sensor_publisher.publish(sensor_msg)
            else:
                self.get_logger().info('No sensor detected.')

        except Exception as e:
            self.get_logger().error(f'Error during image processing: {e}')

        finally:
            self.processing = False


def ai_model_segment_image(image, labels, threshold):
    image, detections = grounded_segmentation(
        image=image,
        labels=labels,
        threshold=threshold,    
        polygon_refinement=True,
    )
    image = annotate(image, detections)  

    return image

detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"
labels = ['face']

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print('device =', device)

segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
processor = AutoProcessor.from_pretrained(segmenter_id)
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)


def main(args=None):
    rclpy.init(args=args)
    camera_receiver = CameraReceiver()
    rclpy.spin(camera_receiver)
    camera_receiver.destroy_node()
    rclpy.shutdown()
