from utils import getGridLines
import cv2


img = cv2.imread('data/12.jpg')

lines = getGridLines(img)

x_axios = [0, 0, img.shape[1], 0]
y_axios = [0, 0, 0, img.shape[0]]

for line in lines:
    cv2.line(img, line[0: 2], line[2:], (0, 255, 0), 2)


cv2.line(img, x_axios[0:2], x_axios[2:], (0, 0, 255), 2)
cv2.line(img, y_axios[0:2], y_axios[2:], (0, 0, 255), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
