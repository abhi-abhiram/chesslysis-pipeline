from turtle import position
import cv2
import numpy as np
from yolo import YOLOSeg, YOLOObj
import argparse
import os


def draw_points(img, points):
    for i in range(len(points)):
        cv2.circle(img, (points[i][0], points[i][1]), 5, (0, 0, 255), -1)
        cv2.line(img, (points[i][0], points[i][1]), (points[(
            i + 1) % 4][0], points[(i + 1) % 4][1]), (0, 0, 255), 2)
    return img


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='data/14.jpg')
parser.add_argument('--show-results', type=bool, default=False)
parser.add_argument('--save-results', type=bool, default=False)

args = parser.parse_args()


yoloseg = YOLOSeg('models/chessseg-v3.onnx')


def get_board(image):
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yoloseg(img_gray)

    if len(yoloseg.boxes) == 0:
        print('No chessboard detected')
        return None

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

    sorted_indices = np.argsort(approx[:, 1])

    sorted_points = approx[sorted_indices]

    if sorted_points[0, 0] > sorted_points[1, 0]:
        temp = sorted_points[0].copy()
        sorted_points[0] = sorted_points[1]
        sorted_points[1] = temp

    if sorted_points[2, 0] < sorted_points[3, 0]:
        temp = sorted_points[2].copy()
        sorted_points[2] = sorted_points[3]
        sorted_points[3] = temp

    approx = sorted_points

    if len(approx) != 4:
        print('No chessboard detected')
        return None

    # perspective transform
    pts1 = np.float32(approx)  # type: ignore
    pts2 = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])  # type: ignore

    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # type: ignore
    board = cv2.warpPerspective(img, matrix, (640, 640))  # type: ignore

    return board


def detect_pieces(image):
    board = get_board(image)

    if board is None:
        return

    # chess pieces detection
    yoloobj = YOLOObj('./models/chess_pieces.onnx')

    yoloobj(board)

    obj_detection = yoloobj.draw_detections(board)

    pieces = zip(yoloobj.boxes, yoloobj.scores, yoloobj.class_ids)

    # plot the positions of the pieces
    positions = np.chararray((8, 8), itemsize=2, unicode=True)

    plot = np.zeros((640, 640, 3), dtype=np.uint8)

    for i in range(8):
        for j in range(8):
            # draw the rectangles
            if (i + j) % 2 == 0:
                cv2.rectangle(plot, (i * 80, j * 80), (i * 80 + 80, j * 80 + 80),
                              (255, 255, 255), 2)

    for (box, _, classId) in pieces:

        x1, y1, x2, y2 = box.astype(int)

        x = (x1 + x2) / 2

        y = (y1 + y2) / 2

        i = int(x // 80)

        j = int(y // 80)

        positions[j, i] = yoloobj.class_names[classId]

    return positions


if os.path.exists(args.image):
    # read all the images in the directory
    if os.path.isdir(args.image):
        images = [os.path.join(args.image, f)
                  for f in os.listdir(args.image) if f.endswith('.jpg')]
    else:
        images = [args.image]

    for image in images:
        if not args.save_results:
            positions = detect_pieces(image)
            print(positions)
        else:
            board = get_board(image)
            if args.save_results:
                if not os.path.exists('./results'):
                    os.makedirs('./results')

                if board is not None:
                    cv2.imwrite(f'./results/{os.path.basename(image)}', board)
                else:
                    os.makedirs('./results/undetected')
                    cv2.imwrite(
                        f'./results/undetected/{os.path.basename(image)}', cv2.imread(image))

if args.show_results:
    # cv2.imshow('mask', board)
    # cv2.imshow('plot', plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
