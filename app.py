import numpy as np
import matplotlib as plt
import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

st.title('Mediapipe Object Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])




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
BASE_DIR = Path(__file__).resolve().parent                           #base directory of the current file
model_path = BASE_DIR / "Models" / "efficientdet_lite0.tflite"       # full path to the model file
base_options = python.BaseOptions(model_asset_path=str(model_path)) 

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

if uploaded_file is not None:
# #way 1
    # #st.write(np.asarray(bytearray(uploaded_file.read()))) #read() returns bytes, so we convert it to a bytearray and then to a numpy array
    file_array = np.asarray(bytearray(uploaded_file.read()))
    image = cv2.imdecode(file_array, 1) # decode the image from the numpy array
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the
    # st.write(image.shape)
    # st.write(image)

#way 2
    # image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)


    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result =  detector.detect(mp_image)
    annotated_image = visualize(image, detection_result)
    #rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    #st.write(detection_result)
    st.image(annotated_image)