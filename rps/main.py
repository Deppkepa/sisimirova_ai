import cv2
import time
from pathlib import Path
from ultralytics import YOLO

model_path = "best.pt"

model = YOLO(model_path)

window_name = "Camera"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
state = "idle" # wait, result
prev_time = 0
curr_time = 0
player1_hand = ""
player2_hand = ""
timer = 4
game_result = ""

win_map = {
    "rock": "scissors", # камень
    "scissors": "paper", # ножницы
    "paper": "rock" # бумага,
}

while cap.isOpened():
    ret, frame = cap.read()
    cv2.putText(frame, f"{state} - {5 - timer:.1f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)
    results = model(frame)
    result = results[0]
    if not result:
        continue
    if len(result.boxes.xyxy) == 2: # если нашли две руки
        labels = []
        for label, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            x1, y1, x2, y2 = xyxy.numpy().astype("int")
            name = result.names[label.item()].lower()
            labels.append(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{name}", (x1 + 20, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        player1_hand, player2_hand = labels
        if player1_hand == "rock" and player2_hand == "rock" and state == "idle":
            state = "wait"
            prev_time = time.time()

    if state == 'wait':
        timer = round(time.time() - prev_time, 1)
    if timer >= 5:
        timer = 5
        if state == "wait":
            state = "result"
            if player1_hand == player2_hand:
                game_result = "draw"
            elif win_map[player1_hand] == player2_hand:
                game_result = "player 1 win"
            else:
                game_result = "player 2 win"

    cv2.putText(frame, f"{game_result}", (250, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('r'):
        state = "idle"

cap.release()
cv2.destroyAllWindows()