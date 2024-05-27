import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO, checks
from PIL import Image
import numpy as np
import easyocr
import cv2

# Check dependencies
checks()

# Initialize YOLO model
model = YOLO("best1.pt")

# Initialize OCR reader
reader = easyocr.Reader(['en'])

def score_frame(frame):
    results = model(frame, save=True, conf=0.15)
    if not results[0]:
        return [0], 0, 0
    result = results[0].boxes
    labels = result.cls
    cord = results[0].boxes.xyxyn[0]
    conf = result.conf
    return labels, cord, conf

def plot_boxes(results, frame):
    labels, cord, conf = results
    n = len(labels)
    x_shape, y_shape = np.array(frame).shape[1], np.array(frame).shape[0]

    for i in range(n):
        row = cord
        if conf >= 0.15:
            x_min = int(row[0].item() * x_shape)
            y_min = int(row[1].item() * y_shape)
            x_max = int(row[2].item() * x_shape)
            y_max = int(row[3].item() * y_shape)

            frame_array = np.array(frame)
            cv2.rectangle(img=frame_array, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 255, 0), thickness=2)

            plate_region = frame_array[y_min:y_max, x_min:x_max]
            plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
            plate_array = np.array(plate_image)

            sub_image = cv2.getRectSubPix(frame_array, (x_max - x_min + 30, y_max - y_min + 30), (x_min - 30, y_min - 30))
            (height, width) = sub_image.shape[0:2]
            center = (width // 2, height // 2)

            M = cv2.getRotationMatrix2D(center, -90, 0.4)
            rotated = cv2.warpAffine(plate_array, M, (width, height))
            rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(rotated_rgb, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 1, 30, 35)
            edged = cv2.Canny(bfilter, 30, 200)

            plate_number = reader.readtext(bfilter)
            concat_number = ' '.join(number[1] for number in plate_number)
            number_conf = np.mean([number[2] for number in plate_number])

            cv2.putText(img=edged, text=concat_number + f"(Conf: {number_conf})",
                        org=(x_min, y_min),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 255, 0),
                        thickness=1)
            return bfilter
        else:
            return frame

# Streamlit UI
st.title("YOLO and OCR Application")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image)
    results = score_frame(image)
    processed_frame = plot_boxes(results, image)

    st.image(frame, caption='Uploaded Image', use_column_width=True)
    st.image(processed_frame, caption='Processed Image', use_column_width=True)
