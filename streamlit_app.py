import matplotlib.pyplot as plt
from   ultralytics       import YOLO
from   PIL               import Image
import numpy             as np
import easyocr
import cv2
from IPython.display     import display
import os

from ultralytics import YOLO, checks, hub
checks()  # checks

# YOLO
model = YOLO("/home/unthinkable-lap/Desktop/streamlit-app/best1.pt")

#ocr
reader = easyocr.Reader(['en'])


def score_frame(frame):
  results = model(frame, save = True, conf = 0.15)

  if not results[0]:
    # raise ValueError("No results generated")
    return [0],0,0

  # print(results[0].boxes)

  result = results[0].boxes
  labels = result.cls
  # temp = results[0].boxes.xyxy[0]
  # if temp.numel() == 0:
  #   cord = torch.tensor([0,0,0,0])


  cord = results[0].boxes.xyxyn[0]
  conf = result.conf
  return labels, cord, conf


def plot_boxes(results, frame):

  labels, cord, conf = results
  n = len(labels)

  # if n == 0:
  #   print("No labels generated")
  #   return frame

  # if cord[0].item() == 0 and cord[1].item() == 0 and cord[2].item() == 0 and cord[3].item() == 0:
  #   print("No coordinates generated")
  #   return frame

  # frame_array = np.array(frame)

  x_shape, y_shape = np.array(frame).shape[1], np.array(frame).shape[0]


  for i in range(n):
    row = cord
    if conf >= 0.15:

      x_min = int(row[0].item() * x_shape)
      y_min = int(row[1].item() * x_shape)
      x_max = int(row[2].item() * y_shape)
      y_max = int(row[3].item() * y_shape)

      # print(x_min, y_min, x_max, y_max)
      frame_array = np.array(frame)
      cv2.rectangle(img = frame_array, pt1 = (x_min, y_min), pt2 = (x_max, y_max), color = (0, 255, 0), thickness = 2) # this draws rectangle


      plate_region = frame_array[y_min:y_max, x_min:x_max] #printing
      # print("Plate region", plate_region)
      # plt.imshow(plate_region)
      # display(plate_region)

      plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)) # printing

      plate_array = np.array(plate_image) # printing
      # print(plate_array.shape)
      # plt.imshow(plate_array)

      # pre-process
      sub_image = cv2.getRectSubPix(frame_array, (x_max - x_min + 30, y_max - y_min + 30), (x_min - 30, y_min - 30))
      # (height, width) = plate_array.shape[0:2]
      (height, width) = sub_image.shape[0:2]
      center = (width // 2, height // 2)

      M = cv2.getRotationMatrix2D(center, -90, 0.4)
      rotated = cv2.warpAffine(plate_array, M, (width, height))
      rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

      gray = cv2.cvtColor(rotated_rgb, cv2.COLOR_BGR2GRAY)
      bfilter = cv2.bilateralFilter(gray, 1, 30, 35)
      edged = cv2.Canny(bfilter, 30, 200) # image, strong edges, weak edges
      bfilter_store = bfilter.copy()
      plt.imshow(bfilter_store)

      # plt.imshow(rotated)
      # plt.imshow(bfilter)
      # display_image(edged)

      plate_number = reader.readtext(bfilter)
      # print('this is plate number ', plate_number)

      concat_number = ' '.join(number[1] for number in plate_number)
      number_conf = np.mean([number[2] for number in plate_number])
      print(concat_number)

      cv2.putText(img = edged, text = concat_number + f"(Conf: {number_conf})",
      org = (x_min, y_min),
      fontFace = cv2.FONT_HERSHEY_SIMPLEX,
      fontScale = 0.7,
      color = (0, 255, 0),
      thickness = 1)

      return bfilter

    else:
      return frame



image_path = '/content/drive/MyDrive/Tyre_testing/Original_image.jpg'
image = Image.open(image_path)
frame = np.array(image)
# results = score_frame(frame)
results = score_frame(image) # without converting the source image to np.array()
frame = plot_boxes(results, image)
# plt.imshow(frame)
# Use tire center logic to get the text in horizontal direction