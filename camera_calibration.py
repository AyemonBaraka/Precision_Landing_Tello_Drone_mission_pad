import cv2 as cv
import os
import numpy as np

# Checkerboard dimensions
CHESS_BOARD_DIM = (8, 6)

# Size of a square in the checkerboard (in millimeters)
SQUARE_SIZE = 25

# Termination criteria for corner sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Directory for calibration data
calib_data_path = "../calib_data"
if not os.path.exists(calib_data_path):
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')
else:
    print(f'"{calib_data_path}" Directory already exists')

# Prepare object points based on the known dimensions of the checkerboard
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE

# Arrays to store object points and image points from all the images
obj_points_3D = []  # 3D points in real world space
img_points_2D = []  # 2D points in image plane

# Directory for images
image_dir_path = "images"

# Iterate over the set of images
for i in range(1, 50):
    imagePath = os.path.join(image_dir_path, f"image{i}.png")
    print(f"Processing {imagePath}")

    image = cv.imread(imagePath)
    if image is None:
        print(f"Failed to load image {imagePath}")
        continue

    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grayScale, CHESS_BOARD_DIM, None)

    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)
        cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)
        cv.imshow('Chessboard', image)
        cv.waitKey(500)

cv.destroyAllWindows()

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)

print("Calibration successful")

# Save the calibration results
np.savez(
    os.path.join(calib_data_path, "MultiMatrix"),
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("Calibration data saved")

# Load the saved calibration data to verify
data = np.load(os.path.join(calib_data_path, "MultiMatrix.npz"))

camMatrix = data["camMatrix"]
distCoef = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("Loaded calibration data successfully")
print(f"Camera Matrix:\n{camMatrix}")
print(f"Distortion Coefficients:\n{distCoef}")
