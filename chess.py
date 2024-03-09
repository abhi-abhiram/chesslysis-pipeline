from unittest import result
import cv2
from networkx import draw
import numpy as np
from yolo import YOLOSeg, YOLOObj

# load image
img = cv2.imread('data/68 copy.jpg')

yoloseg = YOLOSeg('models/best.onnx')

yoloseg(img)

if len(yoloseg.boxes) == 0:
    print('No chessboard detected')
    exit()

x1, y1, x2, y2 = yoloseg.boxes[0].astype(int)

mask_map = yoloseg.mask_maps[0]

mask = np.zeros_like(mask_map, dtype=np.uint8)

mask[mask_map > 0.5] = 255

filled = np.zeros_like(img, dtype=np.uint8)

filled[mask == 255] = img[mask == 255]

contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

largest_contour = np.array(largest_contour).squeeze()

approx = cv2.approxPolyDP(largest_contour, 0.07 *
                          cv2.arcLength(largest_contour, True), True)

approx = approx.squeeze()


# perspective transform
pts1 = np.float32(approx)  # type: ignore
pts2 = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])  # type: ignore
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # type: ignore
board = cv2.warpPerspective(img, matrix, (640, 640))  # type: ignore
board_copy = board.copy()


# chess pieces detection
yoloobj = YOLOObj('./models/chess_pieces.onnx', official_nms=True)

yoloobj(board)

obj_detection = yoloobj.draw_detections(board)

pieces = yoloobj.boxes

# draw pieces
for piece in pieces:
    cv2.rectangle(board_copy, (int(piece[0]), int(piece[1])),
                  (int(piece[2]), int(piece[3])), (0, 255, 0), 2)


# plot the positions of the pieces
positions = np.zeros((8, 8), dtype=np.uint8)

for piece in pieces:
    x1, y1, x2, y2 = piece

    x = (x1 + x2) / 2

    y = (y1 + y2) / 2

    i = int(x // 80)

    j = int(y // 80)

    positions[j, i] = 1


# cv2.imshow('img', img)
cv2.imshow('mask', board_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
