import cv2
import json

dataset = "./dataset.coco/"

with open(dataset + "train/_annotations.coco.json", mode="r") as f:
    data = json.load(f)


x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False


def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False


def drawChar(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(img, text, pos, font, fontScale, fontColor, lineType)


categories = {category.get("id"): category.get("name")
              for category in data.get("categories")}


for item in data.get("images"):
    img = cv2.imread(dataset + "train/" + item.get("file_name"))
    cv2.imshow("image", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        continue

    if key == ord("c"):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        while True:
            i = img.copy()
            cv2.rectangle(i, (x_start, y_start),
                          (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("image", i)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                break

        if x_start > x_end:
            x_start, x_end = x_end, x_start

        if y_start > y_end:
            y_start, y_end = y_end, y_start

        roi = img[y_start:y_end, x_start:x_end]

        isSave = input("Do you want to save this image? (y/n): ")

        if isSave == "n":
            continue

        if isSave == "y":
            cv2.imwrite(dataset + "/changed/" + item.get("file_name"), roi)
            for annotation in data.get("annotations"):
                if annotation.get("image_id") == item.get("id"):
                    bbox = annotation.get("bbox")
                    new_bbox = [bbox[0] - x_start, bbox[1] - y_start,
                                bbox[2], bbox[3]]
                    cv2.rectangle(roi, (int(new_bbox[0]), int(new_bbox[1])),
                                  (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3])), (0, 255, 0), 2)

                    drawChar(roi, categories.get(annotation.get("category_id")),
                             (int(new_bbox[0]), int(new_bbox[1])))

                    annotation['bbox'] = new_bbox

            cv2.imshow("image", roi)
            cv2.waitKey(0)


# write annotations to file
with open(dataset + "/changed/_annotations.new.coco.json", mode="w") as f:
    json.dump(data, f)

cv2.destroyAllWindows()
