import cv2
import numpy as np
from yoloseg import YOLOSeg

# load image
img = cv2.imread('data/test2.jpg')

img = cv2.resize(img, (640, 640))

yoloseg = YOLOSeg('models/best.onnx')

yoloseg(img)

result = yoloseg.draw_masks(img)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
