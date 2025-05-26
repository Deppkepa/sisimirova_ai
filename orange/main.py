import cv2
import numpy as np
from ultralytics import YOLO

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

model_path = "facial_best.pt"
oranges = cv2.imread("oranges.png")
hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

# Определяем границы для маски апельсинов (цветовой диапазон)
lower = np.array((10, 240, 200))
upper = np.array((15, 255, 255))
mask = cv2.inRange(hsv_oranges, lower, upper)
mask = cv2.dilate(mask, np.ones((7, 7)))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Сортируем контуры по площади и берем самый большой
sorted_contours = sorted(contours, key=cv2.contourArea)
m = cv2.moments(sorted_contours[-1])
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])
bbox = cv2.boundingRect(sorted_contours[-1])
x, y, w, h = bbox


model = YOLO(model_path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    result = model(frame)[0]
    masks = result.masks

    if not masks:
        cv2.imshow("Image", oranges)
        cv2.imshow("Mask", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        continue

    global_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for mask in masks:
        mask_data = mask.data.numpy()[0, :, :]
        mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0])).astype("uint8")
        global_mask = cv2.bitwise_or(global_mask, mask_data)

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    global_mask = cv2.GaussianBlur(global_mask, (5, 5), 0)
    global_mask = cv2.dilate(global_mask, struct)
    global_mask = global_mask.reshape(frame.shape[0], frame.shape[1], 1)

    parts = (frame * global_mask).astype("uint8")

    pos = np.where(global_mask > 0)
    if len(pos[0]) > 0 and len(pos[1]) > 0:
        min_y, max_y = int(np.min(pos[0]) * 0.9), int(np.max(pos[0]) * 1.1)
        min_x, max_x = int(np.min(pos[1]) * 0.9), int(np.max(pos[1]) * 1.1)
        min_y, max_y = max(0, min_y), min(frame.shape[0], max_y)
        min_x, max_x = max(0, min_x), min(frame.shape[1], max_x)

        global_mask_cropped = global_mask[min_y:max_y, min_x:max_x]
        parts_cropped = parts[min_y:max_y, min_x:max_x]

        resized_parts = cv2.resize(parts_cropped, (w, h))
        resized_mask = cv2.resize(global_mask_cropped, (w, h)) * 255

        roi = oranges[y:y + h, x:x + w]
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask))
        fg = cv2.bitwise_and(resized_parts, resized_parts, mask=resized_mask)
        combined = cv2.add(bg, fg)
        oranges_copy = oranges.copy()
        oranges_copy[y:y + h, x:x + w] = combined

        cv2.imshow("Image", oranges_copy)
        cv2.imshow("Mask", parts)
    else:
        cv2.imshow("Image", oranges)
        cv2.imshow("Mask", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
