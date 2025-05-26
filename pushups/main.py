from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import time

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def angle(a, b, c):  # Считает угол между тремя точками
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_ = d - e
    if angle_ < 0:
        angle_ = angle_ + 360
    return np.rad2deg(360 - angle_ if angle_ > 180 else angle_)

def process(image, keypoints):  # Вычисляет угол в локте
    left_visible_ear = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_visible_ear = keypoints[4][0] > 0 and keypoints[4][1] > 0
    # плечи
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    # локти
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    # запястья
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    # тазобедренные суставы
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    try:
        # Проверяем, с какой стороны лучше видно ухо чтобы понять какая рука
        if left_visible_ear and not right_visible_ear:
            angle_elbow = angle(left_shoulder, left_elbow, left_wrist)
            x, y = int(left_elbow[0]), int(left_elbow[1])
        else:
            angle_elbow = angle(right_shoulder, right_elbow, right_wrist)
            x, y = int(right_elbow[0]), int(right_elbow[1])
        cv2.putText(image, f"{int(angle_elbow)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)
        # Проверяем горизонтальное положение тела (для отжиманий)
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        body_angle = abs(shoulder_y - hip_y)  # Разница по Y между плечами и бедрами
        return angle_elbow, body_angle
    except ZeroDivisionError:
        return None, None

model_path = "yolo11n-pose.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
last_time = time.time()  # Для подсчета FPS (кадров в секунду)
flag = False  # Состояние (согнуты/разогнуты руки), нужно для фиксации отжимания
count = 0  # Счетчик отжиманий
writer = cv2.VideoWriter("out.mp4", -1, 10, (640, 480))  # Запись видео

time_not_pushing_up = time.time()  # Когда последний раз руки были разогнуты (для сброса счетчика)
while cap.isOpened():
    ret, frame = cap.read()
    cur_time = time.time()
    writer.write(frame)  # Записывает кадр на диск
    last_time = cur_time
    cv2.imshow('YOLO', frame)
    results = model(frame)  # Прогоняет кадр через модель (находит ключевые точки позы).

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if not results:  # Если модель ничего не вернула, переходим к следующему кадру
        continue

    result = results[0]  # Берет первый результат из предсказаний
    keypoints = result.keypoints.xy.tolist()  # Берет ключевые точки в виде списка координат
    if not keypoints:
        continue
    keypoints = keypoints[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)  # Рисует точки и скелет позы на кадре
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angle_, body_angle = process(annotated, keypoints)

    if count > 0 and time.time() - time_not_pushing_up > 10:  # Сброс счетчика после 5 секунд бездействия
        count = 0
        time_not_pushing_up = time.time()

    if angle_ is not None and body_angle is not None:
        # Проверяем, что тело находится в горизонтальном положении
        if body_angle < 50:
            if flag and angle_ > 150:  # Руки прямые (завершение отжимания)
                time_not_pushing_up = time.time()
                count += 1
                flag = False
            elif angle_ < 100:  # Руки согнуты (начало отжимания)
                flag = True

    cv2.putText(frame, f"Push-up Count: {count}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    cv2.imshow("Pose", annotated)

writer.release()
cap.release()
cv2.destroyAllWindows()