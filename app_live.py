import numpy as np
import matplotlib as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import time

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
CAMERA_WIDTH= 640
CAMERA_HEIGHT =400



def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

#step
print("Loading model...")
BASE_DIR = Path(__file__).resolve().parent                           #base directory of the current file
model_path = BASE_DIR / "Models" / "efficientdet_lite0.tflite"       # full path to the model file
base_options = python.BaseOptions(model_asset_path=str(model_path)) 

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
print("Model loaded successfully.")

print("Starting webcam...")
capture = cv2.VideoCapture(0)
time.sleep(2)
if not capture.isOpened():
    print('Error: Could not open webcam')
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print('Error: Could not read frame')
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result =  detector.detect(mp_image)
    annotated_image = visualize(image.copy(), detection_result)


    cv2.putText(annotated_image, 'Press "q" to exit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Object Detection (Live App)', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
       print('Exiting...')
       break

capture.release()
cv2.destroyAllWindows()
detector.close()
print("Webcam released and all resources cleaned up.")





