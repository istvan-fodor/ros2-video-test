from ultralytics import YOLO


model = None

def load_detection_model(model_path='best.pt'):
    global model
    model = YOLO(model_path)  # load model

def detection_gauge_face(img):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''

    results = model(img)  # run inference, detects gauge face and needle

    # get list of detected boxes, already sorted by confidence
    boxes = results[0].boxes

    if len(boxes) == 0:
        raise Exception("No gauge detected in image")

    # get highest confidence box which is of a gauge face
    gauge_face_box = boxes[0]

    box_list = []
    for box in boxes:
        box_list.append(box.xyxy[0].int())

    return gauge_face_box.xyxy[0].int(), box_list
