import csv
import numpy as np
import cv2
from cv2 import aruco

def read_csv(csv_name):
    lat_lon = {}
    with open(csv_name, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        lat_lon[first_row[0]] = [first_row[1], first_row[2]]
        for row in reader:
            lat_lon[row[0]] = [row[1], row[2]]
    return lat_lon

def write_csv(loc, csv_name):
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['lat', 'lon'])
        writer.writerow(loc)

def detect_ArUco_details(image):
    ArUco_details_dict = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    arucoParam = aruco.DetectorParameters()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            corner_points = corners[i][0]

            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))

            ArUco_details_dict[marker_id] = [center_x, center_y]

    return ArUco_details_dict

def find_nearest_marker(aruco_details_dict, reference_marker_id):
    reference_marker_coords = aruco_details_dict[reference_marker_id]

    min_distance = float('inf')  # Initialize minimum distance to infinity
    nearest_marker_id = None

    for marker_id, coords in aruco_details_dict.items():
        if marker_id != reference_marker_id:
            distance = np.sqrt((reference_marker_coords[0] - coords[0])**2 + (reference_marker_coords[1] - coords[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_marker_id = marker_id

    return nearest_marker_id

def main():
    lat_lon = read_csv('lat_long.csv')
    reference_marker_id = 70
    csv_initialized = False

    cap = cv2.VideoCapture(0)  # Start the camera

    while True:
        ret, img = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Resize the image to 920x920
        img = cv2.resize(img, (960, 990))

        ArUco_details_dict = detect_ArUco_details(img)

        if reference_marker_id in ArUco_details_dict:
            nearest_marker_id = find_nearest_marker(ArUco_details_dict, reference_marker_id)

            if nearest_marker_id is not None:
                try:
                    nearest_marker_coords = lat_lon[str(nearest_marker_id)]
                    write_csv(nearest_marker_coords, 'live_location.csv')
                    csv_initialized = True
                except KeyError:
                    pass

        # Show the resized webcam feed
        cv2.imshow('Arena Feed', img)

        # Check for 'q' key press or window close event
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Arena Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()