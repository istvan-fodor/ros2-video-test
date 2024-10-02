import sys
print("Curent Python: ", sys.executable)


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, PILImage.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    seed = 734546733
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        np.random.seed(seed)
        color = np.random.randint(150, 256, size=3)
        seed = seed + 1
        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
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


def detect(
    image: PILImage.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: PILImage.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[PILImage.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)

    if len(detections) > 0:
        detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections

class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')

        # Create a subscription to the 'camera/image' topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            0
        )

        # Create a publisher to send the segmented part to a new topic 'sensor/image'
        self.sensor_publisher = self.create_publisher(Image, 'sensor/image', 10)

        # Initialize CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()

        # Store the latest image in a class attribute
        self.latest_image = None

        self.processing = False
        
    def image_callback(self, msg):
        # Check if the system is already processing an image
        if self.processing:
            self.get_logger().info('Still processing the previous image, skipping new message...')
            return

        # Set the processing flag to True, indicating that we're now processing an image
        self.processing = True

        try:
            # Store the latest image message
            self.get_logger().info('Received a new image, processing...')
            self.latest_image = msg

            # Convert the ROS Compressed Image message to an OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
    
            # Decode the image using OpenCV
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #frame = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
            image = PILImage.fromarray(frame)
            
            # Call the AI stub to simulate detecting a sensor
            sensor_image = ai_model_segment_image(image, labels=['face'], threshold=0.4)
            
            if sensor_image is not None:
                self.get_logger().info('Sensor detected! Publishing segmented image...')
                
                # Convert the processed sensor image back to a ROS Image message
                sensor_msg = self.bridge.cv2_to_imgmsg(sensor_image, encoding='bgr8')

                # Publish the segmented sensor image to the 'sensor/image' topic
                self.sensor_publisher.publish(sensor_msg)
            else:
                self.get_logger().info('No sensor detected.')

        except Exception as e:
            # Log the exception
            self.get_logger().error(f'Error during image processing: {e}')

        finally:
            # Ensure the processing flag is reset even if an error occurs
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
