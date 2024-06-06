import cv2 as cv
import numpy as np
import os
from djitellopy import Tello
import time

CHESS_BOARD_DIM = (8, 6)
n = 0  # image counter
image_dir_path = "images"

# Check if the images directory exists, if not create it
if not os.path.exists(image_dir_path):
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already exists')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret

def main():
    tello = Tello()
    tello.connect()
    tello.streamon()
    n = 1
    while True:
        frame = tello.get_frame_read().frame
        copyFrame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
        cv.putText(frame, f"saved_img : {n}", (30, 40), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow("frame", frame)

        key = cv.waitKey(1)
        if key == ord('s') and board_detected:
            cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
            print(f"Saved image number {n}")
            n += 1  # increment the image counter
        
        if n >= 50 or key == ord('q'):
            break

    tello.streamoff()
    cv.destroyAllWindows()
    print("Total saved images:", n)

if __name__ == "__main__":
    main()
