### Author: Ayemon Baraka


import cv2
import numpy as np
from djitellopy import Tello
from droneblocks.DroneBlocksTello import DroneBlocksTello
import time
from cv2 import aruco
import math

# Constants
TARGET_DISTANCE = 40  # Target distance from the ArUco marker in cm
SPEED = 20  # Movement speed
MARKER_SIZE = 12  # Size of the ArUco marker in cm
MAX_SPEED = 25  #Maaximum Speed of tello drone
MIN_DISTANCE = 50
TOLERANCE_X_AXIS = 5
TOLERANCE_Y_AXIS = 5
POSITIONING_ITERATION_COUNT = 5
DEFAULT_TELLO_SPEED = 25
MARKER_DETECTED = True

## aruco x for left right
## aruco y for up down
## aruco z for forward backward

# Load camera calibration parameters
calib_data = np.load("../calib_data/MultiMatrix.npz")
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

dbtello = DroneBlocksTello()

# Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()
cap = tello.get_frame_read()
time.sleep(2)


##This idea of tello_position_near_mpad is taken from https://github.com/sislam14/DJITello
'''
This function tries to postion the drone near the current mission pad
@param tolerance_x    : This is absolute tolerance for mission pad X axis
@param tolerance_y    : This is absolute tolerance for mission pad Y axis
@param mpad_id        : This is the current mission pad id
@param iteration_count: This is number of iteration that the function will try to
                        position the drone near mission pad within tolerance
'''
def tello_position_near_mpad(tolerance_x, tolerance_y,mpad_id, iteration_count):
    y_min = 0-tolerance_y
    y_max = tolerance_y
    x_min = 0-tolerance_x
    x_max = tolerance_x
    loop_count = iteration_count

    x = tello.get_mission_pad_distance_x()
    y = tello.get_mission_pad_distance_y()
    z = tello.get_mission_pad_distance_z()

    for i in range(0,loop_count):
        if y >= y_min and y <= y_max:
            break
        else:
            tello.go_xyz_speed_mid(x,1,z,DEFAULT_TELLO_SPEED,mpad_id)
            time.sleep(1)
            x = tello.get_mission_pad_distance_x()
            y = tello.get_mission_pad_distance_y()
            z = tello.get_mission_pad_distance_z()

    for i in range(0,loop_count):
        if x >= x_min and x <= x_max:
            break
        else:
            tello.go_xyz_speed_mid(2,y,z,DEFAULT_TELLO_SPEED,mpad_id)
            time.sleep(1)
            x = tello.get_mission_pad_distance_x()
            y = tello.get_mission_pad_distance_y()
            z = tello.get_mission_pad_distance_z()

# Function to calculate distance to the marker
def calculate_distance(tvec):
    return np.linalg.norm(tvec)

# Function to move the drone based on the marker's position
def move_drone(tello, x_distance,y_distance,z_distance, tvec):
    x, y, z = x_distance,y_distance,z_distance

    distance = calculate_distance(tvec)
    print(f"Distance to marker: {distance:.2f} cm")

    if x < -11:
        tello.move_left(20)
    elif x > 11:
        tello.move_right(20)

    if z > TARGET_DISTANCE + 10:
        tello.move_forward(20)
    else: # z < TARGET_DISTANCE:
        #MARKER_DETECTED = False
        tello.move_down(40)
        time.sleep(1)
        mpad_id = tello.get_mission_pad_id()
        time.sleep(2)

        if mpad_id < 0:
            print("#######  No Mission Pad detected  #######")
            time.sleep(1)
            tello.land()

        else:
            print("#######  Mission Pad detected  #######")
            tello_position_near_mpad(TOLERANCE_X_AXIS, TOLERANCE_Y_AXIS, mpad_id, POSITIONING_ITERATION_COUNT)
            dbtello.display_down_arrow(display_color=DroneBlocksTello.PURPLE)
            tello.land()
            time.sleep(2)
            dbtello.send_command_with_return("EXT mled g 000000000000000b000000bb00000bb0b000bb00bb0bb0000bbb000000b00000")
            

dbtello.clear_everything()
dbtello.display_up_arrow(display_color=DroneBlocksTello.BLUE)



battery = tello.get_battery()

if battery <= 20:
    print("[WARNING]: Battery is low.", battery, "%")
else:
    print("Battery: ", battery, "%")

#enable mission pad detectiond only in downward
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(0)

# ArUco dictionary and parameters
marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
param_markers = aruco.DetectorParameters_create()

# Main loop
try:
    tello.takeoff()
    tello.move_up(40)
    time.sleep(2)

    while True:
        frame = cap.frame
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(frame, marker_dict, parameters=param_markers)

        if ids is not None:

            print("Marker detected")
            dbtello.display_smile(display_color=DroneBlocksTello.PURPLE)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)

            # Draw markers and axes
            aruco.drawDetectedMarkers(frame, corners)
            for rvec, tvec in zip(rvecs, tvecs):
                aruco.drawAxis(frame, cam_mat, dist_coef, rvec, tvec, 10)

                x_distance = tvec[0][0]
                y_distance = tvec[0][1]
                z_distance = tvec[0][2]

                # Print the position of the marker
                str_position = f"MARKER Position x={x_distance:.2f} y={y_distance:.2f} z={z_distance:.2f}"
                cv2.putText(frame, str_position, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(str_position)

                distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
                distance_str = f"distance  ={distance:.2f}"
                cv2.putText(frame, distance_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Move the drone based on the marker's position
                move_drone(tello, x_distance,y_distance,z_distance,tvec)
        else:
            if(MARKER_DETECTED == True):
                tello.rotate_clockwise(30)
                print("Marker Not detected")
                dbtello.display_sad(display_color=DroneBlocksTello.RED)
                time.sleep(2)

        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Land the drone and clean up
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
